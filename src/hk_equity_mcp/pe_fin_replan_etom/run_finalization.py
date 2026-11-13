"""Run finalization contract for PE-Fin RePlan-ETOM experiments.

Separates *run* completion (all selected tasks processed into prediction rows)
from *task* outcomes (row status completed/failed). Controlled task failures
must not force ``run_status=partial``.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

RUN_FINALIZATION_CONTRACT_VERSION = "pe_fin_replan_etom_run_finalization_v2"

# Terminal prediction-row statuses that count as "processed" for resume skip.
TERMINAL_PREDICTION_STATUSES = frozenset({"completed", "failed"})

RUN_FINALIZATION_RULES = (
    "run_completed_iff_all_selected_tasks_have_prediction_rows",
    "task_status_completed_or_failed_does_not_set_run_partial",
    "run_partial_for_missing_selected_task_rows_or_incomplete_finalization",
    "run_aborted_on_keyboardinterrupt_systemexit_or_uncaught_runner_error",
    "resume_skips_terminal_prediction_rows_including_controlled_failures",
    "processed_means_prediction_row_present",
    "completed_failed_mean_row_method_outcome",
    "engineering_smoke_valid_without_all_tasks_status_completed",
    "live_pass_claimed_remains_false_for_engineering_smoke",
)


def run_finalization_contract_hash() -> str:
    blob = json.dumps(
        {
            "version": RUN_FINALIZATION_CONTRACT_VERSION,
            "terminal_prediction_statuses": sorted(TERMINAL_PREDICTION_STATUSES),
            "rules": list(RUN_FINALIZATION_RULES),
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _latest_by_task(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Mapping[str, Any]]:
    latest: Dict[str, Mapping[str, Any]] = {}
    for row in rows:
        tid = row.get("task_id")
        if tid is None:
            continue
        latest[str(tid)] = row
    return latest


def prediction_accounting(
    *,
    selected_task_ids: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Compute processed/completed/failed/missing counters (latest row wins)."""
    expected = [str(t) for t in selected_task_ids]
    expected_set = list(dict.fromkeys(expected))  # preserve order, unique
    latest = _latest_by_task(rows)
    processed_ids = [tid for tid in expected_set if tid in latest]
    completed_ids = [
        tid
        for tid in processed_ids
        if str(latest[tid].get("status") or "") == "completed"
    ]
    failed_ids = [
        tid for tid in processed_ids if str(latest[tid].get("status") or "") == "failed"
    ]
    # Non-terminal / unknown statuses still count as processed if a row exists.
    other_ids = [
        tid
        for tid in processed_ids
        if tid not in completed_ids and tid not in failed_ids
    ]
    missing_ids = [tid for tid in expected_set if tid not in latest]
    unique_prediction_count = len(latest)
    return {
        "expected_task_count": len(expected_set),
        "expected_task_ids": expected_set,
        "processed_task_count": len(processed_ids),
        "processed_task_ids": processed_ids,
        "completed_prediction_count": len(completed_ids),
        "failed_prediction_count": len(failed_ids),
        "completed_task_ids": completed_ids,
        "failed_task_ids": failed_ids,
        "other_prediction_task_ids": other_ids,
        "missing_task_ids": missing_ids,
        "unique_prediction_count": unique_prediction_count,
        "all_selected_tasks_processed": len(missing_ids) == 0 and len(expected_set) > 0,
        "duplicate_task_ids_in_selection": len(expected) != len(expected_set),
    }


def decide_run_status(
    *,
    selected_task_ids: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
    aborted: bool = False,
    finalization_ok: bool = True,
) -> str:
    """Return ``completed`` | ``partial`` | ``aborted``.

    ``completed`` requires every selected task id to have a prediction row and
    successful artifact finalization. Row-level ``failed`` is allowed.
    """
    if aborted:
        return "aborted"
    if not finalization_ok:
        return "aborted"
    acct = prediction_accounting(selected_task_ids=selected_task_ids, rows=rows)
    if acct["duplicate_task_ids_in_selection"]:
        return "partial"
    if acct["all_selected_tasks_processed"]:
        return "completed"
    return "partial"


def should_skip_on_resume(prior_row: Optional[Mapping[str, Any]]) -> bool:
    """Skip re-execution when a terminal prediction row already exists."""
    if not prior_row:
        return False
    return str(prior_row.get("status") or "") in TERMINAL_PREDICTION_STATUSES


def engineering_smoke_gate(
    *,
    accounting: Mapping[str, Any],
    run_status: str,
    routing_record_count_total: int,
    mcp_invocations_total: int,
    real_tool_executions_total: int,
    live_pass_claimed: bool,
    artifacts_complete: bool,
    runner_crashed: bool,
) -> Dict[str, Any]:
    """Engineering smoke validity (not formal accuracy / not all-task success)."""
    reasons: List[str] = []
    if runner_crashed:
        reasons.append("runner_crashed")
    if run_status != "completed":
        reasons.append(f"run_status={run_status}")
    if int(accounting.get("expected_task_count") or 0) != 5:
        reasons.append("expected_task_count!=5")
    if int(accounting.get("processed_task_count") or 0) != 5:
        reasons.append("processed_task_count!=5")
    if not accounting.get("all_selected_tasks_processed"):
        reasons.append("missing_task_rows")
    if int(accounting.get("unique_prediction_count") or 0) != 5:
        reasons.append("unique_prediction_count!=5")
    if int(routing_record_count_total) != int(mcp_invocations_total):
        reasons.append("routing_record_count!=mcp_invocations")
    if int(real_tool_executions_total) != 0:
        reasons.append("real_tool_executions!=0")
    if live_pass_claimed:
        reasons.append("live_pass_claimed")
    if not artifacts_complete:
        reasons.append("artifacts_incomplete")
    return {
        "engineering_smoke_valid": len(reasons) == 0,
        "requires_all_tasks_status_completed": False,
        "requires_exact_match_threshold": False,
        "live_pass_claimed": False,
        "reasons": reasons,
        "contract_version": RUN_FINALIZATION_CONTRACT_VERSION,
        "contract_hash": run_finalization_contract_hash(),
    }


def describe_run_finalization() -> Dict[str, Any]:
    return {
        "version": RUN_FINALIZATION_CONTRACT_VERSION,
        "hash": run_finalization_contract_hash(),
        "rules": list(RUN_FINALIZATION_RULES),
    }
