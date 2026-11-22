"""Deterministic normal-state progress controller (v1.2 required_outputs + workflows)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from hk_equity_mcp.apv_fin.dependency_expansion import (
    build_followup_retrieve_step,
    extract_provided_values,
    list_only_should_not_resolve_narrative,
    next_capability_operations,
    next_disclosure_capability,
)
from hk_equity_mcp.apv_fin.output_adapters import required_outputs_present
from hk_equity_mcp.apv_fin.plan_state import PlanState, PlanStep


EXCEPTION_TRIGGERS = (
    "mcp_execution_failed",
    "required_output_missing",
    "constraint_conflict",
    "no_ready_mandatory_unresolved",
    "capability_unsupported",
    "capability_contract_no_candidate",
    "critical_ambiguity",
    "validator_inconsistency",
    "missing_compute_operands",
    "unknown_capability",
)


def classify_after_step(
    *,
    step: PlanStep,
    execution_ok: bool,
    projection: Optional[Dict[str, Any]],
    task_spec: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
    adapted: Optional[Dict[str, Any]] = None,
    mcp_execution_success: Optional[bool] = None,
    pagination_recommended: bool = False,
) -> Dict[str, Any]:
    """Decide whether to auto-resolve or raise an exceptional replan trigger."""
    del task_spec  # reserved
    err = str(error or "")
    # Adapter / required-output gaps are never mcp_execution_failed.
    if err.startswith("required_outputs_missing") or err == "required_output_missing":
        if pagination_recommended:
            return {
                "action": "paginate",
                "trigger": None,
                "reason": "truncated_payload_missing_required_outputs",
            }
        return {
            "action": "exception",
            "trigger": "required_output_missing",
            "reason": err or "required_output_missing",
        }
    if not execution_ok and not mcp_execution_success:
        # Empty binder / unknown capability are contract failures, not MCP failures.
        if err in {
            "capability_contract_no_candidate",
            "unknown_capability",
            "missing_required_capability",
            "retrieve_missing_required_capability",
            "capability_catalog_injection_empty",
            "canonical_output_alias_or_provides_gap",
            "binding_expand_search_exhausted",
        } or err.startswith("unknown_capability"):
            return {
                "action": "exception",
                "trigger": (
                    "capability_contract_no_candidate"
                    if "no_candidate" in err or "alias" in err or "catalog" in err
                    else "capability_unsupported"
                ),
                "reason": err,
            }
        return {
            "action": "exception",
            "trigger": "mcp_execution_failed",
            "reason": err or "execution_failed",
        }

    adapted_view = adapted or (projection or {}).get("adapted") or {}
    if not adapted_view and projection:
        # Fallback: treat projected rows as adapted for offline unit checks.
        proj = projection.get("projected") or projection
        adapted_view = dict(proj) if isinstance(proj, dict) else {}
        rows = adapted_view.get("rows") or []
        if rows and isinstance(rows[0], dict):
            for k, v in rows[0].items():
                adapted_view.setdefault(k, v)

    if step.step_type == "retrieve" and (
        str(step.required_capability or "") == "disclosure.list"
        or list_only_should_not_resolve_narrative(
            operation_id=str(
                (projection or {}).get("operation_id")
                or step.selected_operation
                or step.invoked_operation
                or ""
            ),
            objective=step.objective,
            query_hint=step.retrieval_query_hint,
            required_capability=step.required_capability,
        )
    ):
        # List success with local_key → expand; list without identity still expand attempt.
        if adapted_view.get("local_key") or adapted_view.get("document_identified"):
            return {
                "action": "expand_dependency",
                "trigger": None,
                "reason": "list_identifies_document_not_final_narrative_evidence",
            }

    if step.step_type == "retrieve":
        required = list(step.required_outputs or [])
        check = required_outputs_present(adapted_view, required)
        if required and not check["ok"]:
            if pagination_recommended or adapted_view.get("pagination_needed"):
                return {
                    "action": "paginate",
                    "trigger": None,
                    "reason": "truncated_payload_missing_required_outputs",
                    "missing_outputs": check["missing"],
                }
            # Try fallback capability path via exceptional replan if configured.
            if step.fallback_capabilities:
                return {
                    "action": "exception",
                    "trigger": "required_output_missing",
                    "reason": f"missing_outputs:{check['missing']};try_fallback",
                    "missing_outputs": check["missing"],
                }
            return {
                "action": "exception",
                "trigger": "required_output_missing",
                "reason": f"missing_outputs:{check['missing']}",
                "missing_outputs": check["missing"],
            }
        return {"action": "resolve", "trigger": None, "reason": "required_outputs_met"}

    if step.step_type == "compute":
        if adapted_view.get("ok") or adapted_view.get("value") is not None:
            return {"action": "resolve", "trigger": None, "reason": "compute_ok"}
        return {
            "action": "exception",
            "trigger": "required_output_missing",
            "reason": "compute_result_missing",
        }

    return {"action": "resolve", "trigger": None, "reason": "success"}


def apply_deterministic_progress(
    *,
    plan_state: PlanState,
    step: PlanStep,
    classification: Dict[str, Any],
    observation_ids: Optional[Sequence[str]] = None,
    projection: Optional[Dict[str, Any]] = None,
    contracts: Optional[Dict[str, Dict[str, Any]]] = None,
    available_deps: Optional[Dict[str, Any]] = None,
    already_invoked: Optional[Sequence[str]] = None,
    metadata_by_id: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Mutate plan_state for normal progress / capability workflow expansion."""
    del metadata_by_id  # replaced by contracts in v1.2
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

    if action == "paginate":
        # Scheduler owns the next-page invoke; keep step ready/running for retry.
        if step.status == "failed":
            step.status = "ready"
            step.failure_reason = None
        out["action"] = "paginate"
        out["pagination"] = True
        return out

    if action == "expand_dependency":
        contracts = contracts or {}
        available = dict(available_deps or {})
        extracted = extract_provided_values(
            projection or {},
            provides=["local_key", "news_id", "release_date", "page_number"],
        )
        available.update(extracted)
        # Prefer target from notes tag disclosure_prefix_for:<cap>.
        target_cap = step.required_capability or "financial.official_narrative_evidence"
        notes = str(step.notes or "")
        if "disclosure_prefix_for:" in notes:
            target_cap = notes.split("disclosure_prefix_for:", 1)[1].split(";")[0].strip() or target_cap
        if str(step.required_capability or "") == "disclosure.list":
            target_cap = target_cap or "financial.official_narrative_evidence"
        nxt_cap = next_disclosure_capability(
            current_capability=str(step.required_capability or "disclosure.list"),
            target_capability=target_cap,
        )
        if not nxt_cap:
            out["action"] = "exception"
            out["exception_trigger"] = "capability_unsupported"
            plan_state.mark_failed(step.step_id, "disclosure_workflow_exhausted")
            return out
        suggestions = next_capability_operations(
            contracts=contracts,
            next_capability=nxt_cap,
            available=available,
            already_invoked=already_invoked or [],
        )
        if not suggestions:
            out["action"] = "exception"
            out["exception_trigger"] = "capability_unsupported"
            plan_state.mark_failed(
                step.step_id, "disclosure_list_without_followup_capability"
            )
            return out
        # List step resolves as document identified; follow-up carries narrative goal.
        plan_state.mark_resolved(step.step_id, observation_ids=observation_ids)
        out["resolved"] = True
        follow = build_followup_retrieve_step(
            parent_step_id=step.step_id,
            next_capability=nxt_cap,
            operation_hint=suggestions[0],
            available=available,
            index=1,
            required_outputs=["snippet", "local_key"]
            if nxt_cap != "disclosure.download"
            else ["local_key"],
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

    plan_state.mark_failed(
        step.step_id, classification.get("reason") or "exceptional_state"
    )
    return out


def needs_exceptional_replan(
    *,
    plan_state: PlanState,
    last_classification: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    # Independent ready branches continue even if another branch just failed.
    if plan_state.ready_steps():
        return None
    if last_classification and last_classification.get("action") == "exception":
        return last_classification.get("trigger") or "exception"
    if last_classification and last_classification.get("action") == "paginate":
        # Pagination exhausted without ready work → exceptional.
        return "required_output_missing"
    cov = plan_state.mandatory_coverage()
    if cov["mandatory_unresolved"]:
        return "no_ready_mandatory_unresolved"
    return None
