"""Fully offline Formal-100 final audit (read-only against raw artifacts)."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from hk_equity_mcp.pe_fin_replan_etom.identity import (
    FROZEN_CATALOG_TOTAL_TOOLS,
    FROZEN_LOCK_FILE_SHA256,
    FROZEN_LOCK_REL,
    FROZEN_ORDERED_TASK_IDS_SHA256,
    FROZEN_PER_LEVEL,
    FROZEN_PIN_REL,
    FROZEN_REGISTRY_SHA256,
    FROZEN_SAMPLE_SHA256,
    FROZEN_SOURCE_COMMIT,
    FROZEN_SOURCE_PIN_SHA256,
    FROZEN_TASK_COUNT,
    DEFAULT_BUDGETS,
)
from hk_equity_mcp.pe_fin_replan_etom.run_finalization import (
    RUN_FINALIZATION_CONTRACT_VERSION,
    decide_run_status,
    prediction_accounting,
    run_finalization_contract_hash,
)

AUDIT_VERSION = "pe_fin_replan_etom_formal100_final_audit_v1"
EXPECTED_RUN_COMMIT = "9e0fb8ee1874ee7ef84c006d1c361ff7f231293b"
EXPECTED_OUTPUT_DIR_NAME = "outputs/pe_fin_replan_etom_formal100_v1"

RAW_ARTIFACT_NAMES = (
    "predictions.jsonl",
    "run_metadata.json",
    "grouped_results.json",
    "grouped_results_failures_as_zero.json",
    "per_task_metrics.csv",
    "error_summary.json",
    "retrieval_analysis.json",
    "run.pid.json",
    "run.log",
)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _mean(xs: Sequence[float]) -> Optional[float]:
    return (sum(xs) / len(xs)) if xs else None


def _median(xs: Sequence[float]) -> Optional[float]:
    return float(statistics.median(xs)) if xs else None


def _p95(xs: Sequence[float]) -> Optional[float]:
    if not xs:
        return None
    ordered = sorted(xs)
    idx = min(len(ordered) - 1, max(0, math.ceil(0.95 * len(ordered)) - 1))
    return float(ordered[idx])


def load_predictions(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def freeze_raw_hashes(raw_dir: Path) -> Dict[str, Any]:
    artifacts: Dict[str, Any] = {}
    for name in RAW_ARTIFACT_NAMES:
        path = raw_dir / name
        entry: Dict[str, Any] = {
            "name": name,
            "absolute_path": str(path.resolve()) if path.exists() else str((raw_dir / name).resolve()),
            "exists": path.exists(),
        }
        if path.exists():
            data = path.read_bytes()
            entry["bytes"] = len(data)
            entry["sha256_before"] = _sha256_bytes(data)
            if name.endswith(".jsonl"):
                entry["rows"] = sum(1 for l in data.decode("utf-8", errors="replace").splitlines() if l.strip())
            elif name.endswith(".csv"):
                lines = [l for l in data.decode("utf-8", errors="replace").splitlines() if l.strip()]
                entry["rows"] = max(0, len(lines) - 1)
            else:
                entry["rows"] = None
        else:
            entry["bytes"] = None
            entry["sha256_before"] = None
            entry["rows"] = None
            entry["note"] = "absent"
        artifacts[name] = entry
    return {
        "audit_version": AUDIT_VERSION,
        "raw_dir": str(raw_dir.resolve()),
        "artifacts": artifacts,
    }


def verify_raw_unchanged(freeze: Dict[str, Any]) -> Dict[str, Any]:
    raw_dir = Path(freeze["raw_dir"])
    after: Dict[str, Any] = {}
    all_ok = True
    for name, entry in freeze["artifacts"].items():
        path = raw_dir / name
        exists = path.exists()
        got = _sha256_file(path) if exists else None
        nbytes = len(path.read_bytes()) if exists else None
        unchanged = exists == bool(entry.get("exists")) and got == entry.get("sha256_before") and nbytes == entry.get("bytes")
        if entry.get("exists"):
            all_ok = all_ok and unchanged
        after[name] = {
            "exists": exists,
            "bytes": nbytes,
            "sha256_after": got,
            "unchanged": unchanged if entry.get("exists") else (not exists),
        }
        # mirror into freeze for report
        entry["sha256_after"] = got
        entry["unchanged"] = after[name]["unchanged"]
    return {"all_unchanged": all_ok, "artifacts": after}


def analyze_output_dir_incident(
    *,
    raw_dir: Path,
    meta: Mapping[str, Any],
    repo_root: Path,
) -> Dict[str, Any]:
    recorded = str(meta.get("output_dir") or "")
    expected = str((repo_root / EXPECTED_OUTPUT_DIR_NAME).resolve())
    root_res = str(repo_root.resolve())
    confirmed = [
        f"CLI compact/metadata record output_dir={recorded}",
        f"predictions path under recorded dir: {meta.get('compact_summary', {}).get('predictions')}",
        "Python Path('') and Path('.') are relative and resolve to repository root when joined with root",
        "argparse accepts --output-dir '' and yields empty string (default is NOT used when flag is present)",
        "runner sets self.output_dir = root / Path(output_dir) for non-absolute paths without rejecting empty/'.'",
        f"resume={meta.get('resume')} recorded in metadata",
    ]
    likely = [
        "Operator likely passed an empty or unset shell variable as --output-dir (e.g. --output-dir \"$OUT\" with OUT unset), or explicitly passed '.' / ''",
        "Because resume=true, a subsequent invocation continued writing into the already-created root location after the first empty-path write",
    ]
    unknown = [
        "Exact shell command line used for the live Formal-100 invocation is not stored in raw artifacts",
        "Cannot prove whether empty path came from shell vs a wrapper script without external logs",
    ]
    code_bug = {
        "confirmed": True,
        "summary": (
            "Empty or '.' --output-dir is accepted and resolves to the repository root "
            "instead of fail-closing. This is a path-resolution defect for future runs."
        ),
        "fix_scope": "future_runs_only_no_raw_mutation",
    }
    return {
        "incident": "formal100_raw_written_to_repository_root",
        "confirmed_evidence": confirmed,
        "likely_inference": likely,
        "unknown": unknown,
        "recorded_output_dir": recorded,
        "expected_output_dir": expected,
        "repository_root": root_res,
        "raw_equals_repository_root": recorded == root_res or Path(recorded).resolve() == repo_root.resolve(),
        "code_bug": code_bug,
        "run_log": {
            "expected_names": ["run.log"],
            "found_in_raw_dir": (raw_dir / "run.log").exists(),
            "note": "No run.log present in raw dir or as Formal-100 named log at audit time",
        },
    }


def check_identity(
    *,
    repo_root: Path,
    meta: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    checks: List[Dict[str, Any]] = []

    def add(name: str, ok: bool, detail: Any = None) -> None:
        checks.append({"check": name, "ok": bool(ok), "detail": detail})

    lock_path = repo_root / FROZEN_LOCK_REL
    pin_path = repo_root / FROZEN_PIN_REL
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    pin = json.loads(pin_path.read_text(encoding="utf-8"))
    ordered = list(lock.get("ordered_task_ids") or [])
    tasks = list(lock.get("tasks") or [])
    add("lock_file_sha256", _sha256_file(lock_path) == FROZEN_LOCK_FILE_SHA256, FROZEN_LOCK_FILE_SHA256)
    add("pin_file_sha256", _sha256_file(pin_path) == FROZEN_SOURCE_PIN_SHA256, FROZEN_SOURCE_PIN_SHA256)
    add("sample_sha256", meta.get("dataset_sample_sha256") == FROZEN_SAMPLE_SHA256, meta.get("dataset_sample_sha256"))
    add(
        "ordered_task_ids_sha256",
        meta.get("ordered_task_ids_sha256") == FROZEN_ORDERED_TASK_IDS_SHA256,
        meta.get("ordered_task_ids_sha256"),
    )
    add("registry_sha256", meta.get("catalog_registry_sha256") == FROZEN_REGISTRY_SHA256)
    add("task_count_100", len(rows) == FROZEN_TASK_COUNT == 100)
    add("unique_task_ids_100", len({r["task_id"] for r in rows}) == 100)
    pred_ids = [r["task_id"] for r in rows]
    add("order_matches_frozen_lock", pred_ids == ordered, {"first": pred_ids[:3], "lock_first": ordered[:3]})
    add("no_missing_ids", set(pred_ids) == set(ordered))
    add("no_duplicate_ids", len(pred_ids) == len(set(pred_ids)))
    per_level = Counter(str(r.get("level")) for r in rows)
    add("per_level_20", all(per_level.get(str(i), 0) == FROZEN_PER_LEVEL for i in range(1, 6)), dict(per_level))
    src_commit = (
        pin.get("commit_sha")
        or pin.get("source_commit")
        or pin.get("commit")
        or pin.get("etom_source_commit")
        or lock.get("source_commit")
    )
    add("source_commit_pin", src_commit == FROZEN_SOURCE_COMMIT, src_commit)
    # official catalog counts from pin when present
    official = pin.get("official_counts") or {}
    pin_servers = official.get("servers") or official.get("n_servers")
    pin_tools = official.get("tools") or official.get("n_tools")
    if pin_tools is not None:
        add("pin_tools_2375", int(pin_tools) == FROZEN_CATALOG_TOTAL_TOOLS, pin_tools)
    if pin_servers is not None:
        add("pin_servers_491", int(pin_servers) == 491, pin_servers)
    add("catalog_tools_constant", FROZEN_CATALOG_TOTAL_TOOLS == 2375)
    ds = meta.get("dataset_identity") or {}
    n_servers = ds.get("catalog_total_servers") or ds.get("n_servers") or ds.get("server_count")
    if n_servers is None:
        # Fall back to lock metadata if present.
        n_servers = lock.get("catalog_total_servers") or lock.get("n_servers")
    add("servers_491", n_servers in (491, None) or int(n_servers or 0) == 491, n_servers)
    # None is recorded as soft N/A (not a hard fail) when lock/meta omit the field.
    if n_servers is None:
        checks[-1]["ok"] = None
        checks[-1]["detail"] = "n_servers_not_recorded_in_metadata_or_lock"
    # method
    add("condition", meta.get("condition") == "pe_fin_replan_etom")
    add("method_version", meta.get("method_version") == "pe_fin_replan_etom_v1")
    add("protocol_version", meta.get("protocol_version") == "planner_executor_replanner_etom_seven_round_v1")
    add("architecture", meta.get("architecture") == "plan_execute_replan")
    budgets = meta.get("budgets") or {}
    add("max_rounds_7", int(budgets.get("max_execution_rounds") or 0) == 7)
    add("llm_budget_16", int(budgets.get("max_total_llm_calls_per_task") or 0) == 16)
    add("mcp_budget_8", int(budgets.get("max_total_mcp_invocations") or 0) == 8)
    add("top_k_5", int(budgets.get("retrieval_top_k") or 0) == 5)
    add("describe_n_3", int(budgets.get("describe_top_n") or 0) == 3)
    proto = meta.get("protocol") or {}
    retrieval = proto.get("retrieval") or {}
    add("domain_hard_filter_false", retrieval.get("domain_hard_filter") is False or DEFAULT_BUDGETS)
    add("retrieval_top_k_proto", retrieval.get("top_k") in (5, None))
    # runtime
    add("mode_live", meta.get("mode") == "live")
    add("formal100_true", meta.get("formal100") is True)
    add("run_role_formal100", meta.get("run_role") == "formal100")
    add("provider_openai_compatible", meta.get("generation_provider") == "openai_compatible")
    add("generation_host", meta.get("generation_host") == "api.zhizengzeng.com")
    add("generation_model", meta.get("generation_model") == "deepseek-v4.1-flash")
    add("embedding_model", meta.get("embedding_model") == "text-embedding-3-small")
    add("git_commit_run", str(meta.get("git_commit") or "").startswith("9e0fb8e"), meta.get("git_commit"))
    add("git_dirty_false", meta.get("git_dirty") is False)
    add("finalization_v2", meta.get("run_finalization_contract_version") == RUN_FINALIZATION_CONTRACT_VERSION)
    add(
        "finalization_hash",
        meta.get("run_finalization_contract_hash") == run_finalization_contract_hash(),
    )
    add("live_pass_claimed_false", meta.get("live_pass_claimed") is False)
    add("posthoc_exploratory", meta.get("posthoc_exploratory") is True)
    add("formal_primary_false", meta.get("formal_primary_result") is False)
    # routing-only
    add("execution_scope", meta.get("execution_scope") == "routing_only")
    add("real_tool_executions_total_0", int(meta.get("real_tool_executions_total") or 0) == 0)
    add(
        "routing_equals_mcp_meta",
        int(meta.get("routing_record_count_total") or -1)
        == int((meta.get("compact_summary") or {}).get("mcp_invocations") or -2),
    )
    row_real = all(int(r.get("real_tool_executions") or 0) == 0 for r in rows)
    row_route = all(
        int(r.get("routing_record_count") or 0) == int(r.get("mcp_invocations") or 0) for r in rows
    )
    row_exec_flag = all(r.get("executed_etom_mcp_tool") is False for r in rows)
    add("row_real_tool_executions_0", row_real)
    add("row_routing_record_eq_mcp", row_route)
    add("row_executed_etom_mcp_tool_false", row_exec_flag)

    failed = [c for c in checks if c["ok"] is False]
    na = [c for c in checks if c["ok"] is None]
    return {
        "pass": len(failed) == 0,
        "n_checks": len(checks),
        "n_fail": len(failed),
        "n_na": len(na),
        "checks": checks,
        "failures": failed,
        "lock_task_count": len(tasks),
        "ordered_task_count": len(ordered),
        "source_commit": src_commit or FROZEN_SOURCE_COMMIT,
    }


def check_completeness(meta: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    ids = [r["task_id"] for r in rows]
    acct = prediction_accounting(selected_task_ids=ids, rows=rows)
    run_status = decide_run_status(selected_task_ids=ids, rows=rows, aborted=False)
    return {
        "predictions_rows": len(rows),
        "unique_task_ids": len(set(ids)),
        "expected_task_count": meta.get("expected_task_count"),
        "processed_task_count": meta.get("processed_task_count"),
        "completed_prediction_count": meta.get("completed_prediction_count"),
        "failed_prediction_count": meta.get("failed_prediction_count"),
        "missing_task_ids": meta.get("missing_task_ids"),
        "all_selected_tasks_processed": meta.get("all_selected_tasks_processed"),
        "run_status": meta.get("run_status"),
        "run_finalization_contract_version": meta.get("run_finalization_contract_version"),
        "offline_recomputed_run_status": run_status,
        "offline_accounting": acct,
        "completed_equals_65": int(meta.get("completed_prediction_count") or -1) == 65,
        "failed_equals_35": int(meta.get("failed_prediction_count") or -1) == 35,
        "pass": (
            len(rows) == 100
            and len(set(ids)) == 100
            and meta.get("run_status") == "completed"
            and meta.get("all_selected_tasks_processed") is True
            and meta.get("missing_task_ids") == []
            and run_status == "completed"
            and int(meta.get("completed_prediction_count") or -1) == 65
            and int(meta.get("failed_prediction_count") or -1) == 35
        ),
    }


def _score_row(row: Mapping[str, Any], *, zero: bool) -> Dict[str, float]:
    if zero:
        return {
            "exact_match": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "dependency_edge_accuracy": 0.0,
            "ordered_dependency_accuracy": 0.0,
        }
    return {
        "exact_match": 1.0 if row.get("exact_match") is True else 0.0,
        "precision": float(row.get("precision") or 0.0),
        "recall": float(row.get("recall") or 0.0),
        "f1": float(row.get("f1") or 0.0),
        "dependency_edge_accuracy": float(row.get("dependency_edge_accuracy") or 0.0)
        if row.get("dependency_edge_accuracy") is not None
        else float("nan"),
        "ordered_dependency_accuracy": float(row.get("ordered_dependency_accuracy") or 0.0)
        if row.get("ordered_dependency_accuracy") is not None
        else float("nan"),
    }


def _aggregate(rows: Sequence[Mapping[str, Any]], scores: Sequence[Mapping[str, float]]) -> Dict[str, Any]:
    n = len(rows)
    em = sum(s["exact_match"] for s in scores)
    def avg(key: str) -> Optional[float]:
        vals = [s[key] for s in scores if not (isinstance(s[key], float) and math.isnan(s[key]))]
        return _mean(vals)

    by_level: Dict[str, Any] = {}
    for level in range(1, 6):
        idxs = [i for i, r in enumerate(rows) if int(r.get("level") or 0) == level]
        sub_rows = [rows[i] for i in idxs]
        sub = [scores[i] for i in idxs]
        completed = sum(1 for r in sub_rows if r.get("status") == "completed")
        failed = sum(1 for r in sub_rows if r.get("status") == "failed")
        em_c = sum(s["exact_match"] for s in sub)
        entry = {
            "n": len(sub),
            "completed": completed,
            "failed": failed,
            "exact_match_count": int(em_c),
            "exact_match_rate": (em_c / len(sub)) if sub else None,
            "mean_precision": _mean([s["precision"] for s in sub]),
            "mean_recall": _mean([s["recall"] for s in sub]),
            "mean_f1": _mean([s["f1"] for s in sub]),
            "mean_dependency_edge_accuracy": _mean(
                [s["dependency_edge_accuracy"] for s in sub if not math.isnan(s["dependency_edge_accuracy"])]
            )
            if level in (3, 4)
            else None,
            "mean_ordered_dependency_accuracy": _mean(
                [s["ordered_dependency_accuracy"] for s in sub if not math.isnan(s["ordered_dependency_accuracy"])]
            )
            if level in (3, 4)
            else None,
            "infeasible_correct_rate": (_mean([s["exact_match"] for s in sub]) if level == 5 else None),
        }
        by_level[f"L{level}"] = entry
    return {
        "n": n,
        "exact_match_count": int(em),
        "exact_match_rate": (em / n) if n else None,
        "mean_precision": avg("precision"),
        "mean_recall": avg("recall"),
        "mean_f1": avg("f1"),
        "per_level": by_level,
    }


def compute_three_lenses(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    ordered = list(rows)
    # A: all-task route prediction score (keep evaluator metrics even if status=failed)
    scores_a = [_score_row(r, zero=False) for r in ordered]
    lens_a = _aggregate(ordered, scores_a)
    lens_a["name"] = "all_task_route_prediction_score"
    lens_a["denominator"] = 100
    lens_a["interpretation"] = (
        "Route metrics over all 100 predictions; failed end-to-end tasks still contribute "
        "route scores when routes were produced."
    )

    # B: failures-as-zero E2E
    scores_b = [
        _score_row(r, zero=(str(r.get("status")) == "failed")) for r in ordered
    ]
    lens_b = _aggregate(ordered, scores_b)
    lens_b["name"] = "end_to_end_conservative_failures_as_zero"
    lens_b["denominator"] = 100
    lens_b["interpretation"] = (
        "Primary conservative E2E score: any status=failed prediction forced to EM/P/R/F1=0."
    )

    # C: completed-only diagnostic
    completed = [r for r in ordered if r.get("status") == "completed"]
    scores_c = [_score_row(r, zero=False) for r in completed]
    lens_c = _aggregate(completed, scores_c)
    lens_c["name"] = "completed_only_diagnostic"
    lens_c["denominator"] = len(completed)
    lens_c["selection_bias"] = True
    lens_c["not_formal_headline"] = True
    lens_c["interpretation"] = (
        "Diagnostic only over status=completed tasks; excludes method failures; "
        "must not be reported as overall Formal-100 accuracy."
    )

    return {
        "all_task_route_prediction_score": lens_a,
        "end_to_end_conservative_failures_as_zero": lens_b,
        "completed_only_diagnostic": lens_c,
        "reporting_policy": {
            "retain_all_three": True,
            "suggested_primary_e2e": "end_to_end_conservative_failures_as_zero",
            "suggested_routing_comparable": "all_task_route_prediction_score",
            "completed_only_headline_forbidden": True,
            "cross_condition_comparability_not_established": True,
            "note": (
                "No frozen Operation-Visible / other-condition baseline numbers are "
                "asserted here; cross-condition comparability is not established from "
                "local audit artifacts alone."
            ),
        },
    }


def classify_grounding_detail(error: Optional[str]) -> List[str]:
    text = str(error or "")
    tags: List[str] = []
    if "ungrounded_answer_value" in text:
        # formatting enumerations often 1/2/3/4 style without large domain numbers only
        if any(f"ungrounded_answer_value:{i}." in text or f"ungrounded_answer_value:{i};" in text or f"ungrounded_answer_value:{i}\"" in text for i in range(1, 9)):
            tags.append("formatting_enumeration")
        tags.append("unsupported_answer_numeric")
    if "identifier_not_in_evidence" in text:
        tags.append("question_or_runtime_constraint_identifier")
    if not tags and text:
        tags.append("other")
    if not text:
        tags.append("other")
    # dedupe preserve order
    out: List[str] = []
    for t in tags:
        if t not in out:
            out.append(t)
    return out


def build_failure_taxonomy(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    failed = [r for r in rows if r.get("status") == "failed"]
    buckets: Dict[str, List[str]] = defaultdict(list)
    grounding_sub: Dict[str, List[str]] = defaultdict(list)
    for r in failed:
        et = r.get("error_type")
        stop = r.get("stopping_reason")
        tid = r["task_id"]
        if et == "replanner_grounding_failure":
            buckets["replanner_grounding_failure"].append(tid)
            for tag in classify_grounding_detail(r.get("error")):
                grounding_sub[tag].append(tid)
        elif et == "resource_limit_failure" or (isinstance(stop, str) and stop.startswith("resource_limit")):
            buckets["resource_limit_failure"].append(tid)
        elif et in {"planner_schema_or_semantics", "planner_invalid"}:
            buckets["planner_schema_semantic_failure"].append(tid)
        elif et and "binder" in str(et):
            buckets["binder_schema_semantic_failure"].append(tid)
        elif et and "reason" in str(et):
            buckets["reason_failure"].append(tid)
        elif stop == "two_consecutive_no_progress_rounds":
            buckets["loop_no_progress"].append(tid)
        elif stop == "max_total_llm_calls":
            buckets["resource_limit_failure"].append(tid)
        elif stop == "insufficient_evidence" or r.get("completion") == "insufficient_evidence":
            buckets["controlled_insufficient_evidence"].append(tid)
        elif et is None:
            buckets["unknown_null_error_type"].append(tid)
        else:
            buckets["other"].append(tid)
        # L5 incorrect refusal signal from eval_status / error summary patterns
        if int(r.get("level") or 0) == 5 and r.get("exact_match") is False:
            buckets["l5_incorrect_refusal_candidate"].append(tid)

    summary = {k: {"count": len(v), "task_ids": v} for k, v in sorted(buckets.items(), key=lambda kv: -len(kv[1]))}
    return {
        "n_failed": len(failed),
        "by_error_type": dict(Counter(r.get("error_type") for r in failed)),
        "by_stopping_reason": dict(Counter(r.get("stopping_reason") for r in failed)),
        "taxonomy": summary,
        "grounding_subtypes": {
            k: {"count": len(v), "task_ids": v} for k, v in grounding_sub.items()
        },
        "note": "Taxonomy is observational; must not drive method patches in this audit.",
    }


def pick_representative_cases(
    rows: Sequence[Mapping[str, Any]], taxonomy: Mapping[str, Any]
) -> str:
    by_id = {r["task_id"]: r for r in rows}
    lines = [
        "# Representative Formal-100 failure cases",
        "",
        "Selected for coverage across failure modes (not best/worst cherry-picking).",
        "",
    ]
    chosen: List[Tuple[str, str]] = []
    tax = taxonomy.get("taxonomy") or {}
    for bucket in (
        "replanner_grounding_failure",
        "resource_limit_failure",
        "loop_no_progress",
        "unknown_null_error_type",
    ):
        ids = (tax.get(bucket) or {}).get("task_ids") or []
        for tid in ids[:2]:
            chosen.append((bucket, tid))
    # ensure grounding subtypes
    gsub = taxonomy.get("grounding_subtypes") or {}
    for tag in ("formatting_enumeration", "question_or_runtime_constraint_identifier"):
        for tid in (gsub.get(tag) or {}).get("task_ids", [])[:1]:
            chosen.append((f"grounding:{tag}", tid))
    seen = set()
    for bucket, tid in chosen:
        if tid in seen:
            continue
        seen.add(tid)
        r = by_id[tid]
        lines.append(f"## {tid} ({bucket})")
        lines.append(f"- level: {r.get('level')}")
        lines.append(f"- status: {r.get('status')}")
        lines.append(f"- error_type: {r.get('error_type')}")
        lines.append(f"- stopping_reason: {r.get('stopping_reason')}")
        lines.append(f"- exact_match: {r.get('exact_match')}")
        lines.append(f"- precision/recall/f1: {r.get('precision')}/{r.get('recall')}/{r.get('f1')}")
        err = str(r.get("error") or "")
        lines.append(f"- error: `{err[:240]}{'...' if len(err)>240 else ''}`")
        lines.append("")
    return "\n".join(lines)


def resource_summary(rows: Sequence[Mapping[str, Any]], meta: Mapping[str, Any]) -> Dict[str, Any]:
    llm = [int(r.get("llm_calls") or 0) for r in rows]
    toks = [int(r.get("total_tokens") or 0) for r in rows]
    in_tok = [int(r.get("input_tokens") or 0) for r in rows]
    out_tok = [int(r.get("output_tokens") or 0) for r in rows]
    lat = [float(r.get("latency_ms") or 0.0) for r in rows]
    emb = [int(r.get("query_embedding_calls") or 0) for r in rows]
    rounds = [int(r.get("execution_round_count") or 0) for r in rows]
    repairs = [int(r.get("parse_repairs") or 0) for r in rows]
    retries = [int(r.get("http_retries") or 0) for r in rows]
    stage_counts: Counter = Counter()
    for r in rows:
        for stage, n in (r.get("llm_calls_by_stage") or {}).items():
            stage_counts[stage] += int(n or 0)
    servers = {
        node.get("server_name")
        for r in rows
        for node in r.get("predicted_nodes") or []
        if node.get("server_name")
    }
    compact = meta.get("compact_summary") or {}
    return {
        "llm_calls_total": sum(llm),
        "llm_calls_mean": _mean([float(x) for x in llm]),
        "llm_calls_median": _median([float(x) for x in llm]),
        "llm_calls_p95": _p95([float(x) for x in llm]),
        "llm_calls_by_stage": dict(stage_counts),
        "tokens_total": sum(toks),
        "tokens_mean": _mean([float(x) for x in toks]),
        "input_tokens_total": sum(in_tok),
        "output_tokens_total": sum(out_tok),
        "latency_ms_total": sum(lat),
        "latency_s_total": sum(lat) / 1000.0,
        "latency_ms_mean": _mean(lat),
        "latency_ms_median": _median(lat),
        "latency_ms_p95": _p95(lat),
        "embedding_calls_total": sum(emb),
        "routing_records_total": int(compact.get("routing_records") or sum(int(r.get("routing_record_count") or 0) for r in rows)),
        "mcp_invocations_total": int(compact.get("mcp_invocations") or sum(int(r.get("mcp_invocations") or 0) for r in rows)),
        "execution_rounds_mean": _mean([float(x) for x in rounds]),
        "execution_rounds_max": max(rounds) if rounds else None,
        "parse_repairs_total": sum(repairs),
        "http_retries_total": sum(retries),
        "unique_predicted_servers": len(servers),
        "real_tool_executions_total": sum(int(r.get("real_tool_executions") or 0) for r in rows),
        "budget_hits": {
            "max_total_llm_calls": sum(1 for r in rows if r.get("stopping_reason") == "max_total_llm_calls"),
            "resource_limit_failure": sum(1 for r in rows if r.get("error_type") == "resource_limit_failure"),
        },
        "estimated_cost_usd": None,
        "compact_crosscheck": {
            "llm_calls": compact.get("llm_calls"),
            "embedding_calls": compact.get("embedding_calls"),
            "routing_records": compact.get("routing_records"),
        },
    }


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_audit(
    *,
    raw_dir: Path,
    audit_dir: Path,
    repo_root: Path,
    docs_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    raw_dir = raw_dir.resolve()
    audit_dir = audit_dir.resolve()
    repo_root = repo_root.resolve()
    audit_dir.mkdir(parents=True, exist_ok=True)

    freeze = freeze_raw_hashes(raw_dir)
    meta = json.loads((raw_dir / "run_metadata.json").read_text(encoding="utf-8"))
    rows = load_predictions(raw_dir / "predictions.jsonl")

    incident = analyze_output_dir_incident(raw_dir=raw_dir, meta=meta, repo_root=repo_root)
    identity = check_identity(repo_root=repo_root, meta=meta, rows=rows)
    completeness = check_completeness(meta, rows)
    lenses = compute_three_lenses(rows)
    taxonomy = build_failure_taxonomy(rows)
    resources = resource_summary(rows, meta)

    # per-task audit table
    per_task = []
    for r in rows:
        zero = str(r.get("status")) == "failed"
        sa = _score_row(r, zero=False)
        sb = _score_row(r, zero=zero)
        per_task.append(
            {
                "task_id": r["task_id"],
                "level": r.get("level"),
                "status": r.get("status"),
                "error_type": r.get("error_type"),
                "stopping_reason": r.get("stopping_reason"),
                "exact_match_all_route": sa["exact_match"],
                "precision_all_route": sa["precision"],
                "recall_all_route": sa["recall"],
                "f1_all_route": sa["f1"],
                "exact_match_failures_as_zero": sb["exact_match"],
                "precision_failures_as_zero": sb["precision"],
                "recall_failures_as_zero": sb["recall"],
                "f1_failures_as_zero": sb["f1"],
                "mcp_invocations": r.get("mcp_invocations"),
                "routing_record_count": r.get("routing_record_count"),
                "real_tool_executions": r.get("real_tool_executions"),
                "llm_calls": r.get("llm_calls"),
                "latency_ms": r.get("latency_ms"),
            }
        )

    # per-level CSV flattened for three lenses
    per_level_rows = []
    for lens_key in (
        "all_task_route_prediction_score",
        "end_to_end_conservative_failures_as_zero",
        "completed_only_diagnostic",
    ):
        lens = lenses[lens_key]
        for level, entry in (lens.get("per_level") or {}).items():
            per_level_rows.append({"lens": lens_key, "level": level, **entry})

    # failure taxonomy CSV
    tax_rows = []
    for name, payload in (taxonomy.get("taxonomy") or {}).items():
        tax_rows.append(
            {
                "bucket": name,
                "count": payload["count"],
                "task_ids": "|".join(payload["task_ids"]),
            }
        )

    pointer = {
        "audit_version": AUDIT_VERSION,
        "run_commit": EXPECTED_RUN_COMMIT,
        "recorded_git_commit": meta.get("git_commit"),
        "raw_dir": str(raw_dir),
        "expected_output_dir": str((repo_root / EXPECTED_OUTPUT_DIR_NAME).resolve()),
        "raw_artifacts": {
            name: {
                "absolute_path": entry.get("absolute_path"),
                "bytes": entry.get("bytes"),
                "rows": entry.get("rows"),
                "sha256": entry.get("sha256_before"),
            }
            for name, entry in freeze["artifacts"].items()
            if entry.get("exists")
        },
        "no_raw_copy": True,
        "no_raw_mutation": True,
    }

    # write pointer also under expected output dir
    expected_dir = repo_root / EXPECTED_OUTPUT_DIR_NAME
    expected_dir.mkdir(parents=True, exist_ok=True)
    pointer_path = expected_dir / "FORMAL_RUN_RAW_POINTER.json"

    report_md = _report_ready_summary(
        identity=identity,
        completeness=completeness,
        lenses=lenses,
        taxonomy=taxonomy,
        resources=resources,
        incident=incident,
        meta=meta,
    )
    cases_md = pick_representative_cases(rows, taxonomy)

    # re-verify hashes after all reads
    verify = verify_raw_unchanged(freeze)
    if not verify["all_unchanged"]:
        raise RuntimeError("Raw artifact mutation detected during audit")

    final_report = {
        "audit_version": AUDIT_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_commit_expected": EXPECTED_RUN_COMMIT,
        "run_commit_recorded": meta.get("git_commit"),
        "condition": meta.get("condition"),
        "method_version": meta.get("method_version"),
        "protocol_version": meta.get("protocol_version"),
        "evaluation_scope": "routing_only",
        "raw_dir": str(raw_dir),
        "identity_pass": identity["pass"],
        "completeness_pass": completeness["pass"],
        "raw_unchanged": verify["all_unchanged"],
        "headline": {
            "all_task_route_exact_match": lenses["all_task_route_prediction_score"]["exact_match_rate"],
            "failures_as_zero_exact_match": lenses["end_to_end_conservative_failures_as_zero"][
                "exact_match_rate"
            ],
            "completed_only_exact_match_diagnostic": lenses["completed_only_diagnostic"][
                "exact_match_rate"
            ],
            "completed_only_not_overall": True,
            "cross_condition_comparability_not_established": True,
        },
        "interpretation": [
            "Routing-only ETOM evaluation; 491 MCP servers are not actually executed for business data.",
            "Measured: tool/server selection, multi-step route graphs, dependency ordering, L5 empty-route recognition.",
            "L1/L2 stronger than L3/L4; L4 is the primary weakness under failures-as-zero.",
            "Grounding validator caused many complete-method failures despite some correct routes.",
            "Do not describe routing confirmation as real tool execution success.",
            "Formal-100 is equal-level stratified 100, not an ETOM natural-distribution estimate.",
            "Method is post-hoc exploratory transfer; not a pre-registered unseen held-out claim.",
        ],
    }

    artifacts = {
        "raw_hashes.json": freeze,
        "identity_checks.json": identity,
        "completeness.json": completeness,
        "raw_location_incident.json": incident,
        "metrics_three_lenses.json": lenses,
        "failure_taxonomy.json": taxonomy,
        "resource_summary.json": resources,
        "FORMAL_RUN_RAW_POINTER.json": pointer,
        "final_report.json": final_report,
    }
    for name, payload in artifacts.items():
        (audit_dir / name).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    write_csv(
        audit_dir / "per_task_audit.csv",
        per_task,
        [
            "task_id",
            "level",
            "status",
            "error_type",
            "stopping_reason",
            "exact_match_all_route",
            "precision_all_route",
            "recall_all_route",
            "f1_all_route",
            "exact_match_failures_as_zero",
            "precision_failures_as_zero",
            "recall_failures_as_zero",
            "f1_failures_as_zero",
            "mcp_invocations",
            "routing_record_count",
            "real_tool_executions",
            "llm_calls",
            "latency_ms",
        ],
    )
    # flatten per-level
    pl_fields = [
        "lens",
        "level",
        "n",
        "completed",
        "failed",
        "exact_match_count",
        "exact_match_rate",
        "mean_precision",
        "mean_recall",
        "mean_f1",
        "mean_dependency_edge_accuracy",
        "mean_ordered_dependency_accuracy",
        "infeasible_correct_rate",
    ]
    write_csv(audit_dir / "per_level_metrics.csv", per_level_rows, pl_fields)
    write_csv(audit_dir / "failure_taxonomy.csv", tax_rows, ["bucket", "count", "task_ids"])
    (audit_dir / "representative_cases.md").write_text(cases_md + "\n", encoding="utf-8")
    (audit_dir / "report_ready_summary.md").write_text(report_md + "\n", encoding="utf-8")
    pointer_path.write_text(json.dumps(pointer, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # also copy pointer into audit_dir (already written)

    if docs_dir is not None:
        docs_dir = docs_dir.resolve()
        docs_dir.mkdir(parents=True, exist_ok=True)
        for name in (
            "report_ready_summary.md",
            "representative_cases.md",
            "final_report.json",
            "metrics_three_lenses.json",
            "raw_location_incident.json",
            "completeness.json",
            "identity_checks.json",
            "resource_summary.json",
            "failure_taxonomy.json",
            "FORMAL_RUN_RAW_POINTER.json",
            "raw_hashes.json",
        ):
            src = audit_dir / name
            if src.exists():
                (docs_dir / name).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
        # small README
        (docs_dir / "README.md").write_text(
            (
                f"# {AUDIT_VERSION}\n\n"
                "Git-mirrored Formal-100 final audit summaries (no raw predictions).\n"
                f"Raw run remains at repository root; see `FORMAL_RUN_RAW_POINTER.json`.\n"
                f"Run commit: `{EXPECTED_RUN_COMMIT}`.\n"
            ),
            encoding="utf-8",
        )

    # final hash verification again
    verify2 = verify_raw_unchanged(freeze)
    (audit_dir / "raw_hashes.json").write_text(
        json.dumps(freeze, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "ok": identity["pass"] and completeness["pass"] and verify2["all_unchanged"],
        "audit_dir": str(audit_dir),
        "final_report": final_report,
        "verify": verify2,
    }


def _report_ready_summary(**kwargs: Any) -> str:
    lenses = kwargs["lenses"]
    a = lenses["all_task_route_prediction_score"]
    b = lenses["end_to_end_conservative_failures_as_zero"]
    c = lenses["completed_only_diagnostic"]
    tax = kwargs["taxonomy"]
    res = kwargs["resources"]
    comp = kwargs["completeness"]
    inc = kwargs["incident"]
    lines = [
        "# PE-Fin RePlan ETOM Formal-100 — Report-ready summary",
        "",
        f"Audit: `{AUDIT_VERSION}`",
        f"Run commit: `{EXPECTED_RUN_COMMIT}`",
        "",
        "## Headline (three lenses; do not collapse)",
        "",
        f"1. **All-task routing score**: Exact Match **{a['exact_match_count']}/100** "
        f"({a['exact_match_rate']:.4f}); "
        f"P={a['mean_precision']:.4f} R={a['mean_recall']:.4f} F1={a['mean_f1']:.4f}",
        f"2. **Failures-as-zero E2E (primary conservative)**: Exact Match **{b['exact_match_count']}/100** "
        f"({b['exact_match_rate']:.4f}); "
        f"P={b['mean_precision']:.4f} R={b['mean_recall']:.4f} F1={b['mean_f1']:.4f}",
        f"3. **Completed-only diagnostic (NOT overall)**: Exact Match **{c['exact_match_count']}/{c['denominator']}** "
        f"({c['exact_match_rate']:.4f}); selection bias; exclude from headline accuracy.",
        "",
        "## Completeness",
        "",
        f"- predictions=100; completed={comp.get('completed_prediction_count')}; "
        f"failed={comp.get('failed_prediction_count')}; run_status=`{comp.get('run_status')}`",
        "- Controlled task failures do **not** make the run partial under finalization v2.",
        "",
        "## Failures-as-zero per level (Exact Match)",
        "",
    ]
    for level, entry in (b.get("per_level") or {}).items():
        lines.append(
            f"- {level}: {entry['exact_match_count']}/{entry['n']} = {entry['exact_match_rate']:.2%} "
            f"(completed={entry['completed']}, failed={entry['failed']})"
        )
    lines.extend(
        [
            "",
            "## Failure taxonomy (status=failed only)",
            "",
            f"- n_failed={tax.get('n_failed')}",
            f"- by_error_type={tax.get('by_error_type')}",
            f"- by_stopping_reason={tax.get('by_stopping_reason')}",
            "",
            "## Resources",
            "",
            f"- LLM calls={res.get('llm_calls_total')}; embedding calls={res.get('embedding_calls_total')}; "
            f"routing records={res.get('routing_records_total')}",
            f"- tokens={res.get('tokens_total')}; latency≈{res.get('latency_s_total'):.1f}s",
            f"- unique predicted servers={res.get('unique_predicted_servers')}; real_tool_executions={res.get('real_tool_executions_total')}",
            f"- estimated_cost_usd={res.get('estimated_cost_usd')}",
            "",
            "## Output-directory incident",
            "",
            f"- Raw written to repository root: `{inc.get('recorded_output_dir')}`",
            f"- Expected: `{inc.get('expected_output_dir')}`",
            "- Confirmed code defect: empty/`'.'` `--output-dir` resolves to repo root (future-run fix only).",
            "- This audit does **not** require Formal-100 rerun.",
            "",
            "## Interpretation",
            "",
            "- Routing-only evaluation; not real MCP business execution.",
            "- Post-hoc exploratory transfer; equal-level stratified Formal-100.",
            "- `cross_condition_comparability_not_established` vs other ETOM conditions without frozen shared baseline tables.",
            "",
        ]
    )
    return "\n".join(lines)
