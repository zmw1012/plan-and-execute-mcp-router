"""Fail-closed gates and zero-network audit for Planner cache (R1–R3 / formula)."""

from __future__ import annotations

import hashlib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


class IncompletePlannerCacheError(RuntimeError):
    pass


def summarize_planner_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    task_ids = [str(r.get("task_id") or "") for r in rows]
    unique = sorted({t for t in task_ids if t})
    failed_ids = sorted(
        str(r.get("task_id")) for r in rows if not r.get("ok") and r.get("task_id")
    )
    ok_ids = sorted(
        str(r.get("task_id")) for r in rows if r.get("ok") and r.get("task_id")
    )
    dupes = sorted({t for t in task_ids if t and task_ids.count(t) > 1})
    return {
        "n_rows": len(rows),
        "n_unique_task_ids": len(unique),
        "n_ok": len(ok_ids),
        "n_failed": len(failed_ids),
        "failed_task_ids": failed_ids,
        "ok_task_ids": ok_ids,
        "duplicate_task_ids": dupes,
        "complete": len(rows) == 155
        and len(unique) == 155
        and len(failed_ids) == 0
        and not dupes,
        "rows_complete_155": len(rows) == 155 and len(unique) == 155 and not dupes,
    }


def assert_planner_cache_for_step_pipeline(
    *,
    rows: Sequence[Dict[str, Any]],
    allow_incomplete: bool,
    formal: bool,
    stage: str,
) -> Dict[str, Any]:
    """Require complete 155-ok Planner cache unless diagnostic override.

    Formal formula selection always refuses incomplete caches.
    """
    summary = summarize_planner_rows(rows)
    if formal and not summary["complete"]:
        raise IncompletePlannerCacheError(
            f"Formal {stage} forbidden with incomplete Planner cache: "
            f"rows={summary['n_rows']} unique={summary['n_unique_task_ids']} "
            f"failed={summary['n_failed']} ids={summary['failed_task_ids'][:20]}"
        )
    if not summary["rows_complete_155"]:
        raise IncompletePlannerCacheError(
            f"{stage}: Planner rows must be 155 unique task IDs "
            f"(got rows={summary['n_rows']} unique={summary['n_unique_task_ids']} "
            f"dupes={summary.get('duplicate_task_ids')})"
        )
    if summary["n_failed"] and not allow_incomplete:
        raise IncompletePlannerCacheError(
            f"{stage}: incomplete Planner cache "
            f"(failed={summary['n_failed']}). Pass "
            "--allow-incomplete-planner-cache for diagnostic ranking only; "
            "formal formula selection remains forbidden."
        )
    return summary


def _latency_stats(values: Sequence[float]) -> Dict[str, Optional[float]]:
    xs = [float(x) for x in values if x is not None]
    if not xs:
        return {"mean": None, "median": None, "p95": None, "n": 0}
    xs_sorted = sorted(xs)
    p95_idx = min(len(xs_sorted) - 1, max(0, int(round(0.95 * (len(xs_sorted) - 1)))))
    return {
        "mean": float(statistics.mean(xs_sorted)),
        "median": float(statistics.median(xs_sorted)),
        "p95": float(xs_sorted[p95_idx]),
        "n": len(xs_sorted),
    }


def audit_planner_cache_zero_network(
    *,
    output_dir: Path,
    expected_task_ids: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Zero-network Planner-only gate/audit (no new 12-checkpoint system)."""
    out = Path(output_dir)
    planner_path = out / "planner_cache" / "planner_outputs.jsonl"
    identity_path = out / "planner_cache" / "planner_cache_identity.json"
    meta_path = out / "planner_cache" / "run_metadata.json"

    if not planner_path.exists():
        raise FileNotFoundError(
            f"Missing {planner_path}; run --planner-only first (future live step)"
        )

    rows: List[Dict[str, Any]] = []
    with planner_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    blob = planner_path.read_bytes()
    outputs_sha = hashlib.sha256(blob).hexdigest()
    summary = summarize_planner_rows(rows)

    expected = list(expected_task_ids or [])
    missing = sorted(set(expected) - set(summary.get("ok_task_ids") or []) - set(summary.get("failed_task_ids") or [])) if expected else []
    if expected:
        present = {str(r.get("task_id")) for r in rows if r.get("task_id")}
        missing = sorted(set(expected) - present)
        extra = sorted(present - set(expected))
    else:
        extra = []

    step_type_counts: Counter = Counter()
    domain_step_counts: Counter = Counter()
    plan_class_counts: Counter = Counter()
    repair_total = 0
    llm_calls = 0
    tokens_in = 0
    tokens_out = 0
    tokens_total = 0
    latencies: List[float] = []

    for row in rows:
        repair_total += int(row.get("repair_count") or 0)
        usage = row.get("usage") or row.get("llm_usage") or {}
        llm_calls += int(
            row.get("llm_calls")
            or usage.get("calls")
            or (1 if row.get("ok") is not None else 0)
        )
        tokens_in += int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
        tokens_out += int(usage.get("output_tokens") or usage.get("completion_tokens") or 0)
        tokens_total += int(
            usage.get("total_tokens")
            or (
                int(usage.get("input_tokens") or 0)
                + int(usage.get("output_tokens") or 0)
            )
        )
        lat = row.get("latency_ms") or row.get("latency_sec")
        if lat is not None:
            # normalize to ms if looks like seconds
            lat_f = float(lat)
            if lat_f < 100 and "latency_sec" in row:
                lat_f *= 1000.0
            latencies.append(lat_f)

        plan = row.get("plan") or {}
        steps = plan.get("steps") or row.get("retrieve_steps") or []
        if plan.get("refusal") or plan.get("clarify") or plan.get("no_tool"):
            if plan.get("refusal"):
                plan_class_counts["refusal"] += 1
            if plan.get("clarify") or plan.get("clarification"):
                plan_class_counts["clarification"] += 1
            if plan.get("no_tool"):
                plan_class_counts["no_tool"] += 1
        answerability = str(
            plan.get("answerability_class")
            or row.get("answerability_class")
            or ""
        ).lower()
        if "no_tool" in answerability or answerability == "unanswerable_no_tool":
            plan_class_counts["no_tool_plan"] += 1

        for step in steps:
            if not isinstance(step, dict):
                continue
            st = str(step.get("type") or "retrieve").lower()
            step_type_counts[st] += 1
            if st == "retrieve":
                domain_step_counts[str(step.get("domain") or "unknown")] += 1

    identity = {}
    if identity_path.exists():
        identity = json.loads(identity_path.read_text(encoding="utf-8"))
    run_meta = {}
    if meta_path.exists():
        run_meta = json.loads(meta_path.read_text(encoding="utf-8"))

    # Systemic failure heuristic: repeated identical error types dominate failures.
    fail_reasons = Counter(
        str(r.get("error_type") or r.get("error") or "unknown")
        for r in rows
        if not r.get("ok")
    )
    systemic = False
    if summary["n_failed"] >= 5:
        top = fail_reasons.most_common(1)
        if top and top[0][1] >= max(3, int(0.5 * summary["n_failed"])):
            reason = top[0][0].lower()
            if any(
                k in reason
                for k in ("schema", "protocol", "validation", "canonical")
            ):
                systemic = True

    audit = {
        "zero_network": True,
        "planner_outputs_path": str(planner_path),
        "planner_outputs_sha256": outputs_sha,
        "rows": summary["n_rows"],
        "unique_task_ids": summary["n_unique_task_ids"],
        "missing_task_ids": missing,
        "duplicate_task_ids": summary.get("duplicate_task_ids") or [],
        "extra_task_ids": extra,
        "ok": summary["n_ok"],
        "failed": summary["n_failed"],
        "failed_task_ids": summary["failed_task_ids"],
        "repair_count": repair_total,
        "step_type_counts": dict(step_type_counts),
        "retrieve_compute_reason_counts": {
            "retrieve": int(step_type_counts.get("retrieve") or 0),
            "compute": int(step_type_counts.get("compute") or 0),
            "reason": int(step_type_counts.get("reason") or 0),
        },
        "plan_class_counts": dict(plan_class_counts),
        "per_domain_retrieve_step_counts": dict(domain_step_counts),
        "llm_calls": llm_calls,
        "tokens": {
            "input": tokens_in,
            "output": tokens_out,
            "total": tokens_total,
        },
        "latency_ms": _latency_stats(latencies),
        "identity": {
            "kind": identity.get("kind"),
            "protocol_version": identity.get("protocol_version"),
            "planner_prompt_hash": identity.get("planner_prompt_hash")
            or identity.get("prompt_hash"),
            "planner_schema_hash": identity.get("planner_schema_hash")
            or identity.get("schema_hash"),
            "canonicalizer_hash": identity.get("canonicalizer_hash"),
            "capability_profile_hash": identity.get("capability_profile_hash"),
            "identity_sha256": identity.get("identity_sha256"),
        },
        "run_metadata_embedding_backend": run_meta.get("embedding_backend"),
        "failure_reason_counts": dict(fail_reasons),
        "systemic_schema_or_protocol_failure": systemic,
        "gate_recommendation": (
            "stop_and_inspect"
            if systemic
            else (
                "proceed_with_report"
                if summary["n_failed"] == 0
                else "proceed_with_sparse_failures_report_only"
            )
        ),
        "complete_155_ok": bool(summary["complete"]),
        "formal_step_pipeline_allowed": bool(summary["complete"]),
    }

    audit_dir = out / "planner_cache_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    (audit_dir / "planner_cache_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return audit
