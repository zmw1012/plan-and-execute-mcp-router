"""Crash-safe runtime policies for APV-Fin v1.4.1 (no method redesign)."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Optional, Sequence, Set

from hk_equity_mcp.apv_fin.llm_budget import LlmCallBudgetExhausted
from hk_equity_mcp.apv_fin.validator import controlled_failure_payload

RUNTIME_EXCEPTION_POLICY_VERSION = "apv_fin_runtime_exception_policy_v1_4_1"
RUNNER_FINALIZATION_POLICY_VERSION = "apv_fin_runner_finalization_policy_v1_4_1"
DISCLOSURE_PREFIX_POLICY_VERSION = "apv_fin_disclosure_prefix_policy_v1_4_1"

# Expected task-level exceptions that must not abort the batch.
TASK_ISOLATED_EXCEPTION_TYPES = (
    LlmCallBudgetExhausted,
)

CRASH_SAFE_PROTOCOL_CHANGE = (
    "crash_safe_runtime_before_valid_method_evaluation"
)


def runtime_exception_policy_hash() -> str:
    blob = json.dumps(
        {
            "version": RUNTIME_EXCEPTION_POLICY_VERSION,
            "stages": [
                "planner",
                "planner_repair",
                "tool_binding",
                "tool_binding_repair",
                "monitor",
                "monitor_repair",
                "reason",
                "verify",
                "final",
                "final_repair",
            ],
            "convert_to": {
                "status": "failed",
                "error_type": "resource_limit_failure",
                "stopping_reason": "llm_call_budget_exhausted",
            },
            "do_not_propagate_to_runner": True,
        },
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def runner_finalization_policy_hash() -> str:
    blob = json.dumps(
        {
            "version": RUNNER_FINALIZATION_POLICY_VERSION,
            "on_all_predictions": "completed",
            "on_unknown_exception": "aborted",
            "task_isolation": True,
            "preserve_written_predictions": True,
            "never_leave_running": True,
        },
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def disclosure_prefix_policy_hash() -> str:
    blob = json.dumps(
        {
            "version": DISCLOSURE_PREFIX_POLICY_VERSION,
            "trigger": "narrative_or_local_key_consumer_without_ledger_identity",
            "producer_chain": [
                "disclosure.list",
                "disclosure.download",
                "disclosure.parse",
                "disclosure.keyword_search",
                "disclosure.page_read",
            ],
            "apply_after_capability_compiler": True,
            "no_task_id_branching": True,
            "increment_dependency_auto_expansion_count": True,
        },
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def structured_llm_budget_failure(
    exc: LlmCallBudgetExhausted,
    *,
    plan_state: Any = None,
    trace: Optional[Sequence[Dict[str, Any]]] = None,
    ledger: Any = None,
    selected_operations: Optional[Sequence[str]] = None,
    invoked_operations: Optional[Sequence[str]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Convert LlmCallBudgetExhausted into a controlled failed task result."""
    payload = controlled_failure_payload(reason="llm_call_budget_exhausted")
    out: Dict[str, Any] = {
        "status": "failed",
        "error_type": "resource_limit_failure",
        "stopping_reason": "llm_call_budget_exhausted",
        "error": str(exc),
        "llm_budget_exhausted": True,
        "llm_calls_used": exc.used,
        "llm_call_budget": exc.limit,
        "llm_budget_stage": exc.stage,
        "final_answer": payload.get("final_answer"),
        "final_payload": payload,
        "predicted_status": "unavailable",
        "plan_state": plan_state.to_dict() if hasattr(plan_state, "to_dict") else plan_state,
        "trace": list(trace or []),
        "evidence_ledger": ledger.to_list() if hasattr(ledger, "to_list") else ledger,
        "selected_operations": list(selected_operations or []),
        "invoked_operations": list(invoked_operations or []),
    }
    if extra:
        out.update(extra)
    return out


def summarize_prediction_outcomes(
    *,
    expected_ids: Sequence[str],
    predictions_by_id: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Derive run-level completion counters from written predictions."""
    expected: Set[str] = {str(x) for x in expected_ids}
    completed_ids = []
    partial_ids = []
    failed_ids = []
    for tid in expected_ids:
        row = predictions_by_id.get(tid)
        if not row:
            continue
        status = str(row.get("status") or "")
        if status == "full":
            completed_ids.append(tid)
        elif status == "partial":
            partial_ids.append(tid)
        else:
            failed_ids.append(tid)
    processed = [tid for tid in expected_ids if tid in predictions_by_id]
    missing_ids = [tid for tid in expected_ids if tid not in predictions_by_id]
    return {
        "processed_task_count": len(processed),
        "full_count": len(completed_ids),
        "partial_count": len(partial_ids),
        "failed_count": len(failed_ids),
        "completed_ids": completed_ids,
        "full_task_ids": list(completed_ids),
        "completed_task_ids": list(completed_ids) + list(partial_ids),
        "partial_ids": partial_ids,
        "partial_task_ids": list(partial_ids),
        "failed_ids": failed_ids,
        "failed_task_ids": list(failed_ids),
        "missing_ids": missing_ids,
        "missing_task_ids": list(missing_ids),
        "expected_task_count": len(expected),
        "all_predictions_written": len(processed) == len(expected) and not any(
            tid not in predictions_by_id for tid in expected
        ),
    }
