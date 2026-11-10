"""Fail-closed gates for Planner cache completeness (R1–R3 / formula)."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence


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
    return {
        "n_rows": len(rows),
        "n_unique_task_ids": len(unique),
        "n_ok": len(ok_ids),
        "n_failed": len(failed_ids),
        "failed_task_ids": failed_ids,
        "ok_task_ids": ok_ids,
        "complete": len(rows) == 155
        and len(unique) == 155
        and len(failed_ids) == 0,
        "rows_complete_155": len(rows) == 155 and len(unique) == 155,
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
            f"(got rows={summary['n_rows']} unique={summary['n_unique_task_ids']})"
        )
    if summary["n_failed"] and not allow_incomplete:
        raise IncompletePlannerCacheError(
            f"{stage}: incomplete Planner cache "
            f"(failed={summary['n_failed']}). Pass "
            "--allow-incomplete-planner-cache for diagnostic ranking only; "
            "formal formula selection remains forbidden."
        )
    return summary
