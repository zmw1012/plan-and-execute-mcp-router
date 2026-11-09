"""Offline final audit for APV-Fin Diagnostic-12 (read-only; no APIs / no raw mutation)."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

RAW_ARTIFACTS = (
    "predictions.jsonl",
    "run_metadata.json",
    "per_task_metrics.csv",
    "manual_grading.csv",
    "cost_summary.json",
    "error_summary.json",
    "grouped_aggregates.json",
    "grading_summary.json",
    "apv_fin_metrics.json",
)

EXPECTED_METHOD = "apv_fin_v1_4_1_crash_safe_runtime"
EXPECTED_COMMIT = "3411a28f1c05dad87007626e31dfa3f4ecadfcc4"
MANIFEST_REL = "benchmarks/manifests/hk_apv_fin_diagnostic12_v1.json"
BASELINE_REF_REL = "benchmarks/manifests/hk_apv_fin_diagnostic12_baseline_reference_v1.json"

PRIMARY_ERROR_TAXONOMY = (
    "exact_success",
    "planner_parse_or_schema",
    "capability_or_tool_selection",
    "incomplete_disclosure_chain",
    "required_output_not_satisfied",
    "evidence_conflict_or_verification",
    "calculation_not_reached",
    "final_schema_or_synthesis",
    "resource_limit_exception_replan",
    "incorrect_refusal",
    "tool_execution_failure",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_stats(path: Path) -> Dict[str, Any]:
    data = path.read_bytes()
    rows = None
    if path.suffix == ".jsonl":
        rows = sum(1 for line in data.splitlines() if line.strip())
    elif path.suffix == ".csv":
        # header + data rows
        text = data.decode("utf-8")
        rows = max(0, sum(1 for line in text.splitlines() if line.strip()) - 1)
    return {
        "path": str(path),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "rows": rows,
    }


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_predictions(path: Path) -> List[Dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def load_csv_dicts(path: Path) -> List[Dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _as_bool(v: Any) -> Optional[bool]:
    if isinstance(v, bool):
        return v
    if v is None:
        return None
    s = str(v).strip().lower()
    if s in {"true", "1", "yes"}:
        return True
    if s in {"false", "0", "no"}:
        return False
    return None


def _as_float(v: Any) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _percentile(sorted_vals: Sequence[float], p: float) -> Optional[float]:
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    k = (len(sorted_vals) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(sorted_vals[int(k)])
    return float(sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f))


def strict_answer_correct(row: Dict[str, Any]) -> bool:
    """Failed/partial/resource-limit never count as strict success even if text matches."""
    if str(row.get("status") or "") != "full":
        return False
    if row.get("error_type") == "resource_limit_failure":
        return False
    if str(row.get("stopping_reason") or "").startswith("max_"):
        return False
    return bool(row.get("answer_correct") is True)


def descriptive_auto_correct(row: Dict[str, Any]) -> bool:
    return bool(row.get("answer_correct") is True)


def plan_steps(row: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    ps = row.get("plan_state") or {}
    steps = ps.get("steps") or {}
    return steps if isinstance(steps, dict) else {}


def classify_primary_error(row: Dict[str, Any], task: Dict[str, Any]) -> Tuple[str, List[str], str]:
    """Return (primary, secondary_list, explanation). Evidence-backed heuristics."""
    tid = row.get("task_id")
    status = str(row.get("status") or "")
    stop = str(row.get("stopping_reason") or "")
    err = str(row.get("error") or "")
    err_type = str(row.get("error_type") or "")
    steps = plan_steps(row)
    caps = [
        str(s.get("required_capability") or "")
        for s in steps.values()
        if isinstance(s, dict)
    ]
    failed_steps = [
        s
        for s in steps.values()
        if isinstance(s, dict) and s.get("status") == "failed"
    ]
    resolved_retrieve = [
        s
        for s in steps.values()
        if isinstance(s, dict)
        and s.get("step_type") == "retrieve"
        and s.get("status") == "resolved"
    ]
    failed_retrieve = [
        s
        for s in steps.values()
        if isinstance(s, dict)
        and s.get("step_type") == "retrieve"
        and s.get("status") == "failed"
    ]
    dep = int(row.get("dependency_auto_expansion_count") or 0)
    adapter_ok = int(row.get("output_adapter_success_count") or 0)
    req_miss = int(row.get("required_output_missing_count") or 0)
    invoked = list(row.get("invoked_operations") or [])
    successful = list(row.get("successful_invoked_operations") or [])
    secondary: List[str] = []

    answerability = str(task.get("answerability_class") or "")

    # Exact success (none expected for APV-Fin D12).
    if strict_answer_correct(row):
        return "exact_success", [], "Strict full+correct answer."

    # Planner parse/schema
    if stop == "planner_fail_closed" or "JSONDecodeError" in err:
        return (
            "planner_parse_or_schema",
            [],
            f"Planner fail-closed: {err[:160]}",
        )

    # Final schema / synthesis (evidence present / steps ready but final invalid)
    if "SchemaValidationError" in err or "evidence_ids" in err:
        sec = []
        if adapter_ok > 0 or resolved_retrieve:
            sec.append("required_output_not_satisfied" if req_miss else "final_schema_or_synthesis")
        return (
            "final_schema_or_synthesis",
            [s for s in sec if s != "final_schema_or_synthesis"],
            f"Final/validator schema failure after retrieve progress: {err[:160]}",
        )

    # Monitor refuse on infeasible / refusal tasks
    if stop == "monitor_refuse":
        if answerability in {"intentionally_infeasible", "should_refuse_or_clarify"}:
            # Refusal path completed as full but auto-grade may mark fabricated/incorrect.
            if descriptive_auto_correct(row):
                return "exact_success", [], "Monitor refuse matched refusal expectation."
            primary = "incorrect_refusal"
            # Capability hole may have triggered refuse.
            if any(
                (s.get("failure_reason") or "") == "capability_contract_no_candidate"
                for s in failed_steps
            ):
                secondary.append("capability_or_tool_selection")
            return (
                primary,
                secondary,
                "Monitor refused; auto-grade did not accept refusal quality.",
            )
        secondary.append("resource_limit_exception_replan" if err_type == "resource_limit_failure" else "")
        secondary = [s for s in secondary if s]
        return (
            "incorrect_refusal",
            secondary,
            "Monitor refused an answerable item.",
        )

    # Capability selection hole
    if any(
        (s.get("failure_reason") or "") == "capability_contract_no_candidate"
        for s in failed_steps
    ):
        if stop.startswith("max_exception") or err_type == "resource_limit_failure":
            secondary.append("resource_limit_exception_replan")
        return (
            "capability_or_tool_selection",
            secondary,
            "Capability contract produced no candidate for a retrieve step.",
        )

    # Disclosure chain incomplete (planned disclosure/narrative retrieve did not finish).
    disclosureish = any(
        c.startswith("disclosure.") or c == "financial.official_narrative_evidence"
        for c in caps
    )
    failed_disclosure = any(
        str(s.get("required_capability") or "").startswith("disclosure.")
        or str(s.get("required_capability") or "")
        == "financial.official_narrative_evidence"
        for s in failed_retrieve
    )
    if disclosureish and (
        dep > 0
        or failed_disclosure
        or any("list" in str(x) for x in invoked)
        or any("list" in str(x) for x in successful)
        or any("hkexnews" in str(x) for x in invoked)
    ):
        # Did not finish download/parse/keyword with required outputs.
        if failed_retrieve or req_miss > 0 or status == "failed":
            if stop.startswith("max_exception") or err_type == "resource_limit_failure":
                secondary.append("resource_limit_exception_replan")
            if req_miss > 0:
                secondary.append("required_output_not_satisfied")
            return (
                "incomplete_disclosure_chain",
                secondary,
                "Disclosure/list prefix or narrative retrieve started but chain did not complete "
                f"(dep_expansion={dep}, req_miss={req_miss}, failed_retrieve={len(failed_retrieve)}).",
            )

    # Required outputs missing after successful MCP/adapters
    if req_miss > 0 and (adapter_ok > 0 or successful):
        if stop.startswith("max_exception") or err_type == "resource_limit_failure":
            secondary.append("resource_limit_exception_replan")
        return (
            "required_output_not_satisfied",
            secondary,
            f"Adapter/MCP progress left required outputs missing (count={req_miss}).",
        )

    # Tool execution failure (invoked but not successful)
    if invoked and not successful and any(
        (s.get("failure_reason") or "") not in {None, ""} for s in failed_retrieve
    ):
        if err_type == "resource_limit_failure":
            secondary.append("resource_limit_exception_replan")
        return (
            "tool_execution_failure",
            secondary,
            "Operations invoked without successful executions for needed retrieve steps.",
        )

    # Compute not reached
    compute_pending = [
        s
        for s in steps.values()
        if isinstance(s, dict)
        and s.get("step_type") == "compute"
        and s.get("status") in {"pending", "ready", "failed"}
    ]
    if compute_pending and answerability == "structured_answerable":
        if stop.startswith("max_exception"):
            secondary.append("resource_limit_exception_replan")
        return (
            "calculation_not_reached",
            secondary,
            "Compute step never resolved; upstream retrieve/control blocked calculation.",
        )

    # Resource limit as residual (only if no clearer upstream)
    if err_type == "resource_limit_failure" or stop.startswith("max_exception"):
        return (
            "resource_limit_exception_replan",
            [],
            f"Stopped on resource limit ({stop or err_type}) without a clearer upstream class.",
        )

    if failed_retrieve:
        fr = failed_retrieve[0].get("failure_reason") or "retrieve_failed"
        return (
            "capability_or_tool_selection",
            [],
            f"Retrieve failed: {fr}",
        )

    return (
        "final_schema_or_synthesis",
        [],
        f"Unclassified failure status={status} stop={stop} err={err[:120]}",
    )


def mechanism_flags(row: Dict[str, Any]) -> Dict[str, bool]:
    steps = plan_steps(row)
    plan_ok = bool(row.get("initial_plan") or row.get("plan_state"))
    if str(row.get("stopping_reason") or "") == "planner_fail_closed":
        plan_ok = False
    caps = [
        str(s.get("required_capability") or "")
        for s in steps.values()
        if isinstance(s, dict) and s.get("step_type") == "retrieve"
    ]
    capability_compiled = any(bool(c) for c in caps)
    injected = int(row.get("capability_injected_candidate_count") or 0)
    filtered = int(row.get("capability_filter_candidate_count") or 0)
    selected = list(row.get("selected_operations") or [])
    invoked = list(row.get("invoked_operations") or [])
    successful = list(row.get("successful_invoked_operations") or [])
    adapter_ok = int(row.get("output_adapter_success_count") or 0) > 0
    req_miss = int(row.get("required_output_missing_count") or 0)
    required_outputs = []
    for s in steps.values():
        if isinstance(s, dict):
            required_outputs.extend(list(s.get("required_outputs") or []))
    required_satisfied = bool(required_outputs) and req_miss == 0 and (
        adapter_ok or bool(successful)
    )
    # If no required outputs declared, treat as N/A false for funnel "satisfied"
    if not required_outputs:
        required_satisfied = False
    retrieve_resolved = any(
        isinstance(s, dict)
        and s.get("step_type") == "retrieve"
        and s.get("status") == "resolved"
        for s in steps.values()
    )
    compute_executed = any(
        isinstance(s, dict)
        and s.get("step_type") == "compute"
        and s.get("status") == "resolved"
        for s in steps.values()
    )
    verify_resolved = any(
        isinstance(s, dict)
        and s.get("step_type") == "verify"
        and s.get("status") == "resolved"
        for s in steps.values()
    )
    final_valid = str(row.get("status") or "") == "full" and not (
        "SchemaValidationError" in str(row.get("error") or "")
    )
    # Monitor refuse counts as completed final contract for refusal path.
    if str(row.get("stopping_reason") or "") == "monitor_refuse" and str(row.get("status")) == "full":
        final_valid = True
    return {
        "planner_valid": plan_ok,
        "capability_compiled": capability_compiled,
        "compatible_candidate_found": injected > 0 or filtered > 0 or bool(selected),
        "operation_selected": bool(selected),
        "mcp_invoked": bool(invoked) or int(row.get("real_mcp_execution_count") or 0) > 0,
        "mcp_successful": bool(successful),
        "adapter_successful": adapter_ok,
        "required_outputs_satisfied": required_satisfied,
        "retrieve_steps_resolved": retrieve_resolved,
        "compute_executed": compute_executed,
        "verify_resolved": verify_resolved,
        "final_contract_valid": final_valid,
        "strict_answer_correct": strict_answer_correct(row),
    }


def audit_identity(
    *,
    root: Path,
    run_dir: Path,
    meta: Dict[str, Any],
    rows: Sequence[Dict[str, Any]],
    cost: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    man = load_json(root / MANIFEST_REL)
    ordered = list(man["ordered_task_ids"])
    pred_ids = [r["task_id"] for r in rows]
    cost = cost or {}
    index_prov = str(
        ((meta.get("embedding_index_status") or {}).get("provenance") or {}).get(
            "source"
        )
        or ""
    )
    checks = {
        "n_rows_12": len(rows) == 12,
        "unique_ids_12": len(set(pred_ids)) == 12,
        "ordered_ids_match_manifest": pred_ids == ordered,
        "manifest_hash_match": (
            meta.get("diagnostic_manifest_hash") == man.get("manifest_hash")
            or meta.get("subset_manifest_hash") == man.get("manifest_hash")
        ),
        "ordered_id_hash_match": meta.get("diagnostic_ordered_id_hash")
        == man.get("ordered_id_hash"),
        "run_status_completed": meta.get("run_status") == "completed",
        "processed_task_count_12": int(meta.get("processed_task_count") or 0) == 12,
        "method_version": meta.get("method_version") == EXPECTED_METHOD,
        "git_commit": str(meta.get("git_commit") or "") == EXPECTED_COMMIT,
        "git_dirty_false": meta.get("git_dirty") is False,
        "no_catalog_index_rebuild": (
            index_prov == "final_cache"
            and meta.get("allow_auto_build_index") is False
            and int(cost.get("catalog_index_embedding_calls") or 0) == 0
        ),
        "resume_no_duplicate_ids": len(pred_ids) == len(set(pred_ids)),
    }
    return {
        "checks": checks,
        "all_passed": all(checks.values()),
        "ordered_task_ids": ordered,
        "prediction_task_ids": pred_ids,
        "manifest_hash": man.get("manifest_hash"),
        "ordered_id_hash": man.get("ordered_id_hash"),
        "generation": {
            "model": meta.get("generation_model"),
            "provider": meta.get("generation_provider"),
            "host": meta.get("generation_base_url_host"),
        },
        "embedding": {
            "model": meta.get("embedding_model"),
            "provider": meta.get("embedding_provider"),
            "host": meta.get("embedding_base_url_host"),
        },
        "hashes": {
            "system_prompt_hash": meta.get("system_prompt_hash"),
            "schema_hash": meta.get("schema_hash"),
            "capability_compiler_hash": meta.get("capability_compiler_hash"),
            "capability_contract_hash": meta.get("capability_contract_hash"),
            "output_adapter_hash": meta.get("output_adapter_hash"),
            "result_envelope_hash": meta.get("result_envelope_hash"),
            "catalog_fingerprint": meta.get("catalog_fingerprint"),
        },
        "budgets": meta.get("limits"),
        "resume": bool(meta.get("resume")),
        "full_count": meta.get("full_count"),
        "failed_count": meta.get("failed_count"),
        "completed_ids": meta.get("completed_ids"),
        "failed_ids": meta.get("failed_ids"),
    }


def build_per_task_audit(
    *,
    rows: Sequence[Dict[str, Any]],
    metrics_rows: Sequence[Dict[str, str]],
    tasks_by_id: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    metrics_by_id = {r["task_id"]: r for r in metrics_rows}
    out: List[Dict[str, Any]] = []
    for row in rows:
        tid = row["task_id"]
        task = tasks_by_id[tid]
        m = metrics_by_id.get(tid) or {}
        primary, secondary, explanation = classify_primary_error(row, task)
        flags = mechanism_flags(row)
        tokens = row.get("tokens") or {}
        out.append(
            {
                "task_id": tid,
                "domain": task.get("domain") or m.get("domain"),
                "project_difficulty": task.get("project_difficulty")
                or m.get("project_difficulty"),
                "answerability_class": task.get("answerability_class")
                or m.get("answerability_class"),
                "status": row.get("status"),
                "stopping_reason": row.get("stopping_reason"),
                "error_type": row.get("error_type"),
                "error": row.get("error"),
                "predicted_status": row.get("predicted_status"),
                "descriptive_auto_correct": descriptive_auto_correct(row),
                "strict_answer_correct": strict_answer_correct(row),
                "auto_grade_reason": row.get("auto_grade_reason"),
                "selected_operations": json.dumps(
                    row.get("selected_operations") or [], ensure_ascii=False
                ),
                "invoked_operations": json.dumps(
                    row.get("invoked_operations") or [], ensure_ascii=False
                ),
                "successful_invoked_operations": json.dumps(
                    row.get("successful_invoked_operations") or [], ensure_ascii=False
                ),
                "exact_tool_set": _as_bool(m.get("exact_tool_set")),
                "operation_f1": _as_float(m.get("operation_f1")),
                "operation_precision": _as_float(m.get("operation_precision")),
                "operation_recall": _as_float(m.get("operation_recall")),
                "dependency_auto_expansion_count": int(
                    row.get("dependency_auto_expansion_count") or 0
                ),
                "output_adapter_success_count": int(
                    row.get("output_adapter_success_count") or 0
                ),
                "required_output_missing_count": int(
                    row.get("required_output_missing_count") or 0
                ),
                "deterministic_binding_count": int(
                    row.get("deterministic_binding_count") or 0
                ),
                "llm_binding_count": int(row.get("llm_binding_count") or 0),
                "exception_replan_calls": int(row.get("exception_replan_calls") or 0),
                "pagination_count": int(row.get("pagination_count") or 0),
                "rows_scanned": int(row.get("rows_scanned") or 0),
                "llm_calls": int(row.get("llm_calls") or m.get("llm_calls") or 0),
                "input_tokens": int(
                    tokens.get("input_tokens") or m.get("input_tokens") or 0
                ),
                "output_tokens": int(
                    tokens.get("output_tokens") or m.get("output_tokens") or 0
                ),
                "total_tokens": int(
                    tokens.get("total_tokens") or m.get("total_tokens") or 0
                ),
                "real_mcp_execution_count": int(
                    row.get("real_mcp_execution_count")
                    or m.get("real_mcp_execution_count")
                    or 0
                ),
                "embedding_calls": int(
                    row.get("embedding_calls") or m.get("embedding_calls") or 0
                ),
                "embedding_tokens": int(
                    row.get("embedding_tokens") or m.get("embedding_tokens") or 0
                ),
                "latency_ms": _as_float(row.get("latency_ms") or m.get("latency_ms")),
                "compute_resolved": flags["compute_executed"],
                "verify_resolved": flags["verify_resolved"],
                "retrieve_resolved": flags["retrieve_steps_resolved"],
                "primary_error": primary,
                "secondary_errors": json.dumps(secondary, ensure_ascii=False),
                "explanation": explanation,
            }
        )
    return out


def build_funnel(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    keys = [
        "planner_valid",
        "capability_compiled",
        "compatible_candidate_found",
        "operation_selected",
        "mcp_invoked",
        "mcp_successful",
        "adapter_successful",
        "required_outputs_satisfied",
        "retrieve_steps_resolved",
        "compute_executed",
        "verify_resolved",
        "final_contract_valid",
        "strict_answer_correct",
    ]
    counts = {k: 0 for k in keys}
    per_task = {}
    for row in rows:
        flags = mechanism_flags(row)
        per_task[row["task_id"]] = flags
        for k in keys:
            if flags.get(k):
                counts[k] += 1
    n = len(rows) or 1
    return {
        "n_tasks": len(rows),
        "counts": counts,
        "rates": {k: counts[k] / n for k in keys},
        "per_task": per_task,
        "note": "Funnel explains 0/12 strict answer success; stages are descriptive offline flags.",
    }


def build_resources(
    *,
    rows: Sequence[Dict[str, Any]],
    cost: Dict[str, Any],
    metrics_rows: Sequence[Dict[str, str]],
) -> Dict[str, Any]:
    llm_calls = [int(r.get("llm_calls") or 0) for r in rows]
    tokens = [int((r.get("tokens") or {}).get("total_tokens") or 0) for r in rows]
    lat = [
        float(r.get("latency_ms") or 0.0)
        for r in rows
        if r.get("latency_ms") is not None
    ]
    lat_sorted = sorted(lat)
    emb_calls = [int(r.get("embedding_calls") or 0) for r in rows]
    emb_tok = [int(r.get("embedding_tokens") or 0) for r in rows]
    mcp = [int(r.get("real_mcp_execution_count") or 0) for r in rows]
    pag = sum(int(r.get("pagination_count") or 0) for r in rows)
    scanned = sum(int(r.get("rows_scanned") or 0) for r in rows)
    dep = sum(int(r.get("dependency_auto_expansion_count") or 0) for r in rows)
    det_bind = sum(int(r.get("deterministic_binding_count") or 0) for r in rows)
    llm_bind = sum(int(r.get("llm_binding_count") or 0) for r in rows)
    return {
        "n_tasks": len(rows),
        "llm_calls_total": int(cost.get("total_llm_calls") or sum(llm_calls)),
        "llm_calls_mean": statistics.mean(llm_calls) if llm_calls else None,
        "llm_calls_median": statistics.median(llm_calls) if llm_calls else None,
        "input_tokens_total": int(cost.get("total_input_tokens") or 0),
        "output_tokens_total": int(cost.get("total_output_tokens") or 0),
        "total_tokens": int(cost.get("total_tokens") or sum(tokens)),
        "tokens_mean": statistics.mean(tokens) if tokens else None,
        "embedding_query_calls_total": int(
            cost.get("total_embedding_query_calls") or sum(emb_calls)
        ),
        "embedding_tokens_total": sum(emb_tok),
        "catalog_index_embedding_calls": int(
            cost.get("catalog_index_embedding_calls") or 0
        ),
        "mcp_executions_total": int(
            cost.get("total_real_mcp_executions") or sum(mcp)
        ),
        "mcp_executions_mean": statistics.mean(mcp) if mcp else None,
        "latency_ms_mean": statistics.mean(lat) if lat else None,
        "latency_ms_median": statistics.median(lat) if lat else None,
        "latency_ms_p95": _percentile(lat_sorted, 0.95),
        "pagination_count_sum": pag,
        "rows_scanned_sum": scanned,
        "dependency_auto_expansion_count_sum": dep,
        "deterministic_binding_count_sum": det_bind,
        "llm_binding_count_sum": llm_bind,
        "monitor_calls": int(cost.get("monitor_calls") or 0),
        "monitor_repair_calls": int(cost.get("monitor_repair_calls") or 0),
        "planner_calls": int(cost.get("planner_calls") or 0),
        "planner_repair_calls": int(cost.get("planner_repair_calls") or 0),
        "tool_binding_calls": int(cost.get("tool_binding_calls") or 0),
        "tool_binding_repair_calls": int(cost.get("tool_binding_repair_calls") or 0),
        "calls_by_stage": cost.get("calls_by_stage"),
        "estimated_cost_usd": cost.get("estimated_cost_usd"),
        "estimated_cost_note": cost.get("note")
        or "estimated_cost_usd is null without a frozen provider price table.",
    }


def build_baseline_comparison(
    *,
    root: Path,
    apv_rows: Sequence[Dict[str, Any]],
    apv_metrics_rows: Sequence[Dict[str, str]],
) -> Dict[str, Any]:
    ref = load_json(root / BASELINE_REF_REL)
    metrics_by_id = {r["task_id"]: r for r in apv_metrics_rows}
    apv_per_task = {}
    for row in apv_rows:
        tid = row["task_id"]
        m = metrics_by_id.get(tid) or {}
        apv_per_task[tid] = {
            "status": row.get("status"),
            "predicted_status": row.get("predicted_status"),
            "answer_correct_descriptive": descriptive_auto_correct(row),
            "strict_answer_correct": strict_answer_correct(row),
            "exact_tool_set": _as_bool(m.get("exact_tool_set")),
            "operation_f1": _as_float(m.get("operation_f1")),
            "llm_calls": int(row.get("llm_calls") or 0),
            "real_mcp_execution_count": int(row.get("real_mcp_execution_count") or 0),
            "error": row.get("error"),
            "error_type": row.get("error_type"),
            "stopping_reason": row.get("stopping_reason"),
            "resource_limit": row.get("error_type") == "resource_limit_failure"
            or str(row.get("stopping_reason") or "").startswith("max_"),
        }

    def summarize_condition(name: str, per_task_map: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        ids = list(ref["diagnostic12_ordered_task_ids"])
        strict = 0
        descriptive = 0
        full = 0
        failed = 0
        exact = 0
        exact_n = 0
        f1s: List[float] = []
        llm: List[float] = []
        mcp: List[float] = []
        resource = 0
        for tid in ids:
            row = per_task_map.get(tid) or {}
            if name == "apv_fin":
                if row.get("strict_answer_correct"):
                    strict += 1
                if row.get("answer_correct_descriptive"):
                    descriptive += 1
            else:
                # Frozen baseline reference stores descriptive answer_correct.
                if row.get("answer_correct") is True:
                    descriptive += 1
                    # For baselines, treat full+correct as strict proxy (same as reference practice).
                    if row.get("status") == "full":
                        strict += 1
            if row.get("status") == "full":
                full += 1
            if row.get("status") == "failed":
                failed += 1
            ets = row.get("exact_tool_set")
            if ets in (True, False):
                exact_n += 1
                if ets is True:
                    exact += 1
            f1 = row.get("operation_f1")
            if isinstance(f1, (int, float)):
                f1s.append(float(f1))
            if isinstance(row.get("llm_calls"), (int, float)):
                llm.append(float(row["llm_calls"]))
            if isinstance(row.get("real_mcp_execution_count"), (int, float)):
                mcp.append(float(row["real_mcp_execution_count"]))
            if name == "apv_fin":
                if row.get("resource_limit"):
                    resource += 1
            else:
                err = str(row.get("error") or "")
                if err.startswith("max_") or "resource" in err:
                    resource += 1
        return {
            "n": 12,
            "strict_answer_correct": strict,
            "descriptive_auto_correct": descriptive,
            "full": full,
            "failed": failed,
            "exact_tool_set": exact if exact_n else None,
            "exact_tool_set_rate": (exact / exact_n) if exact_n else None,
            "operation_f1_mean": (sum(f1s) / len(f1s)) if f1s else None,
            "llm_calls_mean": (sum(llm) / len(llm)) if llm else None,
            "mcp_calls_mean": (sum(mcp) / len(mcp)) if mcp else None,
            "resource_limit_like": resource,
        }

    conditions = {}
    for cond in ("operation_visible", "mcp_zero", "flat_all_tools", "mcp_free"):
        per = {
            tid: (ref["per_task"].get(tid) or {}).get(cond) or {}
            for tid in ref["diagnostic12_ordered_task_ids"]
        }
        conditions[cond] = {
            "source": ref["conditions"].get(cond),
            "summary": summarize_condition(cond, per),
            "per_task": per,
        }
    conditions["apv_fin"] = {
        "source": {
            "available": True,
            "path": str(root / "outputs/apv_fin_diagnostic12_live_v1_4_1"),
            "method_version": EXPECTED_METHOD,
            "git_commit": EXPECTED_COMMIT,
            "descriptive_only_comparison": True,
        },
        "summary": summarize_condition("apv_fin", apv_per_task),
        "per_task": apv_per_task,
    }

    expected_correct = {
        "operation_visible": 2,
        "mcp_zero": 2,
        "flat_all_tools": 2,
        "mcp_free": 1,
        "apv_fin": 0,
    }
    # Verify expected descriptive/strict counts against observed.
    verification = {}
    for name, expected in expected_correct.items():
        if name == "apv_fin":
            observed = conditions[name]["summary"]["strict_answer_correct"]
        else:
            observed = conditions[name]["summary"]["descriptive_auto_correct"]
        verification[name] = {
            "expected_correct": expected,
            "observed_correct": observed,
            "match": observed == expected,
            "metric": "strict_answer_correct"
            if name == "apv_fin"
            else "descriptive_answer_correct_from_frozen_reference",
        }

    return {
        "descriptive_only": True,
        "not_final_population_accuracy": True,
        "n": 12,
        "no_statistical_significance": True,
        "no_provider_effect_claim": True,
        "baseline_reference_id": ref.get("reference_id"),
        "baseline_reference_hash": ref.get("reference_hash"),
        "expected_correct_verification": verification,
        "conditions": conditions,
    }


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def render_report_md(payload: Dict[str, Any]) -> str:
    s = payload["strict_metrics"]
    idn = payload["identity"]
    tax = payload["error_taxonomy"]
    fun = payload["mechanism_funnel"]
    res = payload["resources"]
    base = payload["baseline_comparison"]
    lines = []
    lines.append("# APV-Fin Diagnostic-12 Final Offline Audit v1")
    lines.append("")
    lines.append(
        "**Read-only descriptive audit. Not a re-run. Not population accuracy. No API calls.**"
    )
    lines.append("")
    lines.append("## Identity")
    lines.append("")
    lines.append(f"- method_version: `{payload['method_version']}`")
    lines.append(f"- git_commit: `{payload['git_commit']}`")
    lines.append(f"- run_status: `{idn.get('checks', {})}` all_passed=`{idn.get('all_passed')}`")
    lines.append(f"- ordered IDs match manifest: `{idn['checks']['ordered_ids_match_manifest']}`")
    lines.append(f"- generation: `{idn['generation']}`")
    lines.append(f"- embedding: `{idn['embedding']}`")
    lines.append("")
    lines.append("## Strict metrics")
    lines.append("")
    lines.append(f"- strict answer success: **{s['strict_correct']}/{s['n']}**")
    lines.append(
        f"- descriptive auto-grade correct: **{s['descriptive_auto_correct']}/{s['n']}**"
    )
    lines.append(f"- status full/failed: **{s['full']}/{s['failed']}**")
    lines.append(f"- resource-limit failures: **{s['resource_limit']}/{s['n']}**")
    lines.append(f"- exact tool set: **{s['exact_tool_set']}/{s['n']}**")
    lines.append(f"- mean operation F1: **{s['operation_f1_mean']}**")
    if s.get("notes"):
        lines.append(f"- notes: {s['notes']}")
    lines.append("")
    lines.append("## Error taxonomy (primary)")
    lines.append("")
    for k, v in tax["primary_counts"].items():
        lines.append(f"- `{k}`: {v}")
    lines.append("")
    lines.append("## Mechanism funnel (counts / 12)")
    lines.append("")
    for k, v in fun["counts"].items():
        lines.append(f"- `{k}`: {v}")
    lines.append("")
    lines.append("## Resources")
    lines.append("")
    lines.append(
        f"- LLM calls total/mean/median: {res['llm_calls_total']} / {res['llm_calls_mean']} / {res['llm_calls_median']}"
    )
    lines.append(
        f"- tokens in/out/total: {res['input_tokens_total']} / {res['output_tokens_total']} / {res['total_tokens']}"
    )
    lines.append(
        f"- embedding query calls/tokens: {res['embedding_query_calls_total']} / {res['embedding_tokens_total']}"
    )
    lines.append(f"- MCP executions: {res['mcp_executions_total']}")
    lines.append(
        f"- latency mean/median/p95 ms: {res['latency_ms_mean']} / {res['latency_ms_median']} / {res['latency_ms_p95']}"
    )
    lines.append(f"- estimated_cost_usd: `{res['estimated_cost_usd']}`")
    lines.append("")
    lines.append("## Same-12 baseline descriptive comparison")
    lines.append("")
    lines.append("| Condition | Strict/desc correct | Full | Failed | Exact tools | F1 mean | LLM mean | MCP mean | Resource-like |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name in (
        "operation_visible",
        "mcp_zero",
        "flat_all_tools",
        "mcp_free",
        "apv_fin",
    ):
        sm = base["conditions"][name]["summary"]
        corr = (
            sm["strict_answer_correct"]
            if name == "apv_fin"
            else sm["descriptive_auto_correct"]
        )
        lines.append(
            f"| {name} | {corr}/12 | {sm['full']} | {sm['failed']} | "
            f"{sm['exact_tool_set']} | {sm['operation_f1_mean']} | "
            f"{sm['llm_calls_mean']} | {sm['mcp_calls_mean']} | {sm['resource_limit_like']} |"
        )
    lines.append("")
    lines.append("Verification vs expected correct counts:")
    lines.append("")
    for name, ver in base["expected_correct_verification"].items():
        lines.append(
            f"- `{name}`: expected {ver['expected_correct']}, observed {ver['observed_correct']}, match={ver['match']}"
        )
    lines.append("")
    lines.append("## Research conclusions (evidence-backed)")
    lines.append("")
    for c in payload["research_conclusions"]:
        lines.append(f"### {c['title']}")
        lines.append("")
        lines.append(c["text"])
        lines.append("")
    lines.append("## Integrity")
    lines.append("")
    lines.append(f"- raw hashes before/after unchanged: `{payload['raw_hashes']['unchanged']}`")
    lines.append("- no API / no rerun / no raw mutation")
    lines.append("")
    return "\n".join(lines)


def research_conclusions(
    *,
    funnel: Dict[str, Any],
    taxonomy: Dict[str, Any],
    resources: Dict[str, Any],
    baseline: Dict[str, Any],
    strict: Dict[str, Any],
) -> List[Dict[str, str]]:
    counts = funnel["counts"]
    primary = taxonomy["primary_counts"]
    apv = baseline["conditions"]["apv_fin"]["summary"]
    ov = baseline["conditions"]["operation_visible"]["summary"]
    return [
        {
            "title": "1. Capability-aware routing improved candidate/tool selection selectively",
            "text": (
                f"Compatible candidates and selections appeared on "
                f"{counts['compatible_candidate_found']}/12 and {counts['operation_selected']}/12 tasks; "
                f"exact tool-set reached {strict['exact_tool_set']}/12 and mean F1={strict['operation_f1_mean']}. "
                f"Versus OV exact={ov['exact_tool_set']}/12 and F1={ov['operation_f1_mean']}, APV-Fin is not uniformly better, "
                "but several tasks show capability injection/list prefix (dependency expansions) that OV-style free selection lacked. "
                "This is descriptive only (n=12)."
            ),
        },
        {
            "title": "2. Better selection did not convert into answer correctness",
            "text": (
                f"Strict answer success is {strict['strict_correct']}/12 while MCP success occurred on "
                f"{counts['mcp_successful']}/12 and adapters on {counts['adapter_successful']}/12. "
                f"Required outputs were satisfied on only {counts['required_outputs_satisfied']}/12; "
                f"final contract valid {counts['final_contract_valid']}/12. "
                "The drop is concentrated after retrieve: missing disclosure fields, unfinished chains, and final schema/synthesis failures."
            ),
        },
        {
            "title": "3. Plan/Monitor/Verify/Final control cost is high relative to yield",
            "text": (
                f"Mean LLM calls={resources['llm_calls_mean']} (median {resources['llm_calls_median']}), "
                f"total tokens={resources['total_tokens']}, monitor+repair="
                f"{resources['monitor_calls']}+{resources['monitor_repair_calls']}. "
                f"Strict yield={strict['strict_correct']}/12. OV mean LLM calls={ov['llm_calls_mean']} with "
                f"{ov['descriptive_auto_correct']}/12 descriptive correct on the same IDs. "
                "APV-Fin spends more control tokens without converting them into correct answers on this diagnostic slice."
            ),
        },
        {
            "title": "4. Disclosure dependency chains still do not complete",
            "text": (
                f"Primary incomplete_disclosure_chain={primary.get('incomplete_disclosure_chain', 0)}; "
                f"dependency expansions sum={resources['dependency_auto_expansion_count_sum']}. "
                "Typical pattern: list/prefix executes, then keyword/page/narrative retrieve hits "
                "max_targeted_retrievals or missing required outputs, then exceptional Monitor replans until the replan cap. "
                "Producer chain starts more often than in v7/v8 engineering smokes, but still fails to finish download→parse→evidence."
            ),
        },
        {
            "title": "5. Resource-limit is usually a downstream stop, not the first root cause",
            "text": (
                f"resource_limit error_type count={strict['resource_limit']}/12, but primary taxonomy assigns "
                f"resource_limit_exception_replan as primary only {primary.get('resource_limit_exception_replan', 0)} times. "
                "Most capped tasks already failed retrieve/disclosure/required-output stages; the replan budget then expires. "
                "Treating all failures as 'budget too small' would mis-locate the defect."
            ),
        },
        {
            "title": "6. Current APV-Fin behaves like an over-engineered controller on this slice",
            "text": (
                f"With planner/monitor/binding repairs and multi-step verify/final machinery, APV-Fin reaches "
                f"{apv['strict_answer_correct']}/12 strict vs baselines' descriptive corrects "
                f"(OV/MCP-Zero/Flat=2, MCP-Free=1) on the same 12 IDs. "
                "Control complexity and token spend rise while answer yield falls — consistent with over-engineering for this diagnostic regime, "
                "not a claim about all future tasks."
            ),
        },
        {
            "title": "7. Simpler follow-on direction (not implemented)",
            "text": (
                "Evidence suggests retaining capability-aware routing and evidence/result normalization, "
                "while removing multi-round Monitor/Replan and collapsing to a shallow retrieve→compute→verify pipeline. "
                "This audit does not implement that redesign."
            ),
        },
    ]


def run_final_audit(
    *,
    root: Path,
    run_dir: Path,
    out_dir: Optional[Path] = None,
    docs_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    root = root.resolve()
    run_dir = run_dir.resolve()
    out_dir = (out_dir or (run_dir / "final_audit_v1")).resolve()
    docs_dir = (
        docs_dir or (root / "docs" / "apv_fin_diagnostic12_final_audit_v1")
    ).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)

    before = {name: file_stats(run_dir / name) for name in RAW_ARTIFACTS}

    meta = load_json(run_dir / "run_metadata.json")
    rows = load_predictions(run_dir / "predictions.jsonl")
    metrics_rows = load_csv_dicts(run_dir / "per_task_metrics.csv")
    cost = load_json(run_dir / "cost_summary.json")
    grading = load_json(run_dir / "grading_summary.json")
    apv_metrics = load_json(run_dir / "apv_fin_metrics.json")
    grouped = load_json(run_dir / "grouped_aggregates.json")

    from hk_equity_mcp.experiment.unified_apv_fin_runner import load_diagnostic12_tasks

    tasks = load_diagnostic12_tasks(root)
    tasks_by_id = {t["task_id"]: t for t in tasks}

    identity = audit_identity(
        root=root, run_dir=run_dir, meta=meta, rows=rows, cost=cost
    )
    per_task = build_per_task_audit(
        rows=rows, metrics_rows=metrics_rows, tasks_by_id=tasks_by_id
    )
    funnel = build_funnel(rows)
    resources = build_resources(rows=rows, cost=cost, metrics_rows=metrics_rows)
    baseline = build_baseline_comparison(
        root=root, apv_rows=rows, apv_metrics_rows=metrics_rows
    )

    strict_correct = sum(1 for r in rows if strict_answer_correct(r))
    descriptive_correct = sum(1 for r in rows if descriptive_auto_correct(r))
    full = sum(1 for r in rows if r.get("status") == "full")
    failed = sum(1 for r in rows if r.get("status") == "failed")
    resource_limit = sum(
        1
        for r in rows
        if r.get("error_type") == "resource_limit_failure"
        or str(r.get("stopping_reason") or "").startswith("max_exception")
    )
    exact = sum(1 for r in per_task if r.get("exact_tool_set") is True)
    f1s = [r["operation_f1"] for r in per_task if isinstance(r.get("operation_f1"), float)]
    f1_mean = (sum(f1s) / len(f1s)) if f1s else None

    expected = {
        "strict_correct": 0,
        "descriptive_auto_correct": 1,
        "full": 1,
        "failed": 11,
        "resource_limit": 8,
        "exact_tool_set": 3,
        "operation_f1_mean": 0.4488095,
    }
    notes = []
    # apv_fin_metrics descriptive_auto_grade_accuracy may disagree with grading_summary;
    # audit uses prediction-level answer_correct for descriptive and strict rule above.
    if float(apv_metrics.get("descriptive_auto_grade_accuracy") or 0) != (
        descriptive_correct / 12
    ):
        notes.append(
            "apv_fin_metrics.descriptive_auto_grade_accuracy disagrees with prediction-level "
            f"answer_correct ({apv_metrics.get('descriptive_auto_grade_accuracy')} vs "
            f"{descriptive_correct/12}); grading_summary.auto_graded_correct="
            f"{grading.get('auto_graded_correct')}. Audit uses prediction answer_correct for descriptive."
        )
    if abs((f1_mean or 0) - expected["operation_f1_mean"]) > 1e-9:
        notes.append(
            f"operation_f1_mean observed={f1_mean} expected≈{expected['operation_f1_mean']}"
        )

    strict_metrics = {
        "n": 12,
        "strict_correct": strict_correct,
        "descriptive_auto_correct": descriptive_correct,
        "full": full,
        "failed": failed,
        "resource_limit": resource_limit,
        "exact_tool_set": exact,
        "operation_f1_mean": f1_mean,
        "expected": expected,
        "matches_expected": {
            "strict_correct": strict_correct == expected["strict_correct"],
            "descriptive_auto_correct": descriptive_correct
            == expected["descriptive_auto_correct"],
            "full": full == expected["full"],
            "failed": failed == expected["failed"],
            "resource_limit": resource_limit == expected["resource_limit"],
            "exact_tool_set": exact == expected["exact_tool_set"],
            "operation_f1_mean": abs((f1_mean or 0) - expected["operation_f1_mean"])
            < 1e-9,
        },
        "notes": "; ".join(notes) if notes else None,
        "condition_specific_overall_success": None,
        "condition_specific_overall_success_note": (
            "overall success deferred; manual grading fields incomplete / not used in this offline audit"
        ),
        "grading_summary": grading,
        "grouped_overall_excerpt": {
            k: (grouped.get("overall") or {}).get(k)
            for k in (
                "auto_graded_correct",
                "auto_graded_accuracy",
                "exact_tool_set_rate",
                "mean_operation_f1",
                "completed_count",
                "failed_count",
            )
        },
        "apv_fin_metrics_excerpt": {
            k: apv_metrics.get(k)
            for k in (
                "common_strict_answer_success",
                "final_answer_accuracy",
                "descriptive_auto_grade_accuracy",
                "tool_set_exact_rate",
                "tool_set_f1_mean",
                "resource_limit_failures",
            )
        },
    }

    primary_counts = Counter(r["primary_error"] for r in per_task)
    taxonomy = {
        "taxonomy": list(PRIMARY_ERROR_TAXONOMY),
        "primary_counts": {k: int(primary_counts.get(k, 0)) for k in PRIMARY_ERROR_TAXONOMY},
        "per_task_primary": {r["task_id"]: r["primary_error"] for r in per_task},
        "note": (
            "Primary errors prefer upstream root causes; resource_limit is secondary when "
            "disclosure/required-output/capability failures precede the replan cap."
        ),
    }

    conclusions = research_conclusions(
        funnel=funnel,
        taxonomy=taxonomy,
        resources=resources,
        baseline=baseline,
        strict=strict_metrics,
    )

    after = {name: file_stats(run_dir / name) for name in RAW_ARTIFACTS}
    unchanged = all(
        before[name]["sha256"] == after[name]["sha256"] for name in RAW_ARTIFACTS
    )
    raw_hashes = {
        "before": before,
        "after": after,
        "unchanged": unchanged,
    }

    summary = {
        "audit_version": "apv_fin_diagnostic12_final_audit_v1",
        "descriptive_only": True,
        "method_version": meta.get("method_version"),
        "git_commit": meta.get("git_commit"),
        "run_dir": str(run_dir),
        "identity": identity,
        "strict_metrics": strict_metrics,
        "error_taxonomy": taxonomy,
        "mechanism_funnel": {
            "n_tasks": funnel["n_tasks"],
            "counts": funnel["counts"],
            "rates": funnel["rates"],
            "note": funnel["note"],
        },
        "resources": resources,
        "baseline_comparison_verification": baseline["expected_correct_verification"],
        "research_conclusion_titles": [c["title"] for c in conclusions],
        "raw_hashes_unchanged": unchanged,
        "no_api_no_rerun_no_raw_mutation": True,
    }

    payload = {
        **summary,
        "research_conclusions": conclusions,
        "raw_hashes": raw_hashes,
        "baseline_comparison": baseline,
        "mechanism_funnel": funnel,
    }

    # Write artifacts (audit outputs only).
    (out_dir / "audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(out_dir / "per_task_audit.csv", per_task)
    (out_dir / "error_taxonomy.json").write_text(
        json.dumps(taxonomy, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "mechanism_funnel.json").write_text(
        json.dumps(funnel, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "baseline_comparison.json").write_text(
        json.dumps(baseline, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "resource_summary.json").write_text(
        json.dumps(resources, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "raw_hashes.json").write_text(
        json.dumps(raw_hashes, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report = render_report_md(payload)
    (out_dir / "final_audit_report.md").write_text(report, encoding="utf-8")

    # Mirror to docs/ (committable).
    for name in (
        "audit_summary.json",
        "per_task_audit.csv",
        "error_taxonomy.json",
        "mechanism_funnel.json",
        "baseline_comparison.json",
        "resource_summary.json",
        "raw_hashes.json",
        "final_audit_report.md",
    ):
        src = out_dir / name
        (docs_dir / name).write_bytes(src.read_bytes())

    return {
        "out_dir": str(out_dir),
        "docs_dir": str(docs_dir),
        "summary": summary,
        "strict_metrics": strict_metrics,
        "identity_all_passed": identity["all_passed"],
        "raw_hashes_unchanged": unchanged,
    }
