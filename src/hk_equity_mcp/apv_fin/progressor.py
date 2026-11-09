"""Deterministic normal-state progress controller (no Monitor LLM on success)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from hk_equity_mcp.apv_fin.dependency_expansion import (
    build_followup_retrieve_step,
    extract_provided_values,
    list_only_should_not_resolve_narrative,
    next_capability_operations,
)
from hk_equity_mcp.apv_fin.plan_state import PlanState, PlanStep


EXCEPTION_TRIGGERS = (
    "mcp_execution_failed",
    "required_output_missing",
    "constraint_conflict",
    "no_ready_mandatory_unresolved",
    "capability_unsupported",
    "critical_ambiguity",
    "validator_inconsistency",
    "missing_compute_operands",
)


def success_criteria_fields_present(
    *,
    step: PlanStep,
    projection: Optional[Dict[str, Any]],
    task_spec: Optional[Dict[str, Any]] = None,
) -> bool:
    """Lightweight check that projected observation supports the step objective."""
    if not projection:
        return False
    proj = projection.get("projected") or projection
    rows = proj.get("rows") or []
    if step.step_type == "retrieve":
        if not rows and not proj.get("document_identified"):
            return False
        criteria = (step.success_criteria or "").lower()
        objective = (step.objective or "").lower()
        blob = " ".join(
            [
                criteria,
                objective,
                " ".join(str(m) for m in (task_spec or {}).get("metrics") or []),
            ]
        )
        if "close" in blob or "ohlc" in blob:
            # Need at least one close-like field.
            for row in rows:
                if isinstance(row, dict) and any(
                    str(k).lower() == "close" for k in row.keys()
                ):
                    return True
            return False
        return True
    if step.step_type == "compute":
        return bool(projection.get("ok") or projection.get("value") is not None)
    return True


def classify_after_step(
    *,
    step: PlanStep,
    execution_ok: bool,
    projection: Optional[Dict[str, Any]],
    task_spec: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
) -> Dict[str, Any]:
    """Decide whether to auto-resolve or raise an exceptional replan trigger."""
    if not execution_ok:
        return {
            "action": "exception",
            "trigger": "mcp_execution_failed",
            "reason": error or "execution_failed",
        }
    if list_only_should_not_resolve_narrative(
        operation_id=str(
            (projection or {}).get("operation_id")
            or step.selected_operation
            or step.invoked_operation
            or ""
        ),
        objective=step.objective,
        query_hint=step.retrieval_query_hint,
    ):
        return {
            "action": "expand_dependency",
            "trigger": None,
            "reason": "list_identifies_document_not_final_narrative_evidence",
        }
    if not success_criteria_fields_present(
        step=step, projection=projection, task_spec=task_spec
    ):
        return {
            "action": "exception",
            "trigger": "required_output_missing",
            "reason": "projected_result_missing_required_fields",
        }
    return {"action": "resolve", "trigger": None, "reason": "success_criteria_met"}


def apply_deterministic_progress(
    *,
    plan_state: PlanState,
    step: PlanStep,
    classification: Dict[str, Any],
    observation_ids: Optional[Sequence[str]] = None,
    projection: Optional[Dict[str, Any]] = None,
    metadata_by_id: Optional[Dict[str, Dict[str, Any]]] = None,
    available_deps: Optional[Dict[str, Any]] = None,
    already_invoked: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Mutate plan_state for normal progress / dependency expansion."""
    action = classification.get("action")
    out: Dict[str, Any] = {
        "action": action,
        "resolved": False,
        "new_steps": [],
        "dependency_auto_expansion_count": 0,
        "exception_trigger": classification.get("trigger"),
    }
    if action == "resolve":
        plan_state.mark_resolved(step.step_id, observation_ids=observation_ids)
        out["resolved"] = True
        return out

    if action == "expand_dependency":
        metadata_by_id = metadata_by_id or {}
        available = dict(available_deps or {})
        extracted = extract_provided_values(
            projection or {},
            provides=["local_key", "news_id", "release_date", "page_number"],
        )
        available.update(extracted)
        suggestions = next_capability_operations(
            metadata_by_id=metadata_by_id,
            available=available,
            already_invoked=already_invoked or [],
        )
        # Mark current list step resolved as "document identified" only if we can
        # unlock a follow-up; otherwise exceptional.
        if not suggestions:
            out["action"] = "exception"
            out["exception_trigger"] = "capability_unsupported"
            plan_state.mark_failed(
                step.step_id, "disclosure_list_without_followup_capability"
            )
            return out
        plan_state.mark_resolved(step.step_id, observation_ids=observation_ids)
        out["resolved"] = True
        # Insert at most one follow-up retrieve step.
        follow = build_followup_retrieve_step(
            parent_step_id=step.step_id,
            operation_hint=suggestions[0],
            available=available,
            index=1,
        )
        if follow["step_id"] not in plan_state.steps:
            from hk_equity_mcp.apv_fin.plan_state import PlanStep as PS

            plan_state.steps[follow["step_id"]] = PS.from_dict(
                follow, allow_runtime=False
            )
            plan_state._recompute_ready()
            out["new_steps"].append(follow["step_id"])
            out["dependency_auto_expansion_count"] = 1
        return out

    # exception — leave step failed for replan
    plan_state.mark_failed(
        step.step_id, classification.get("reason") or "exceptional_state"
    )
    return out


def needs_exceptional_replan(
    *,
    plan_state: PlanState,
    last_classification: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    if last_classification and last_classification.get("action") == "exception":
        return last_classification.get("trigger") or "exception"
    if plan_state.ready_steps():
        return None
    cov = plan_state.mandatory_coverage()
    if cov["mandatory_unresolved"]:
        return "no_ready_mandatory_unresolved"
    return None
