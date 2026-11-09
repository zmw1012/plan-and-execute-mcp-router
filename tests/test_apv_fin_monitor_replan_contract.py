"""Monitor replan contract — reject incomplete retrieve patches before PlanState."""

from __future__ import annotations

import json

import pytest

from hk_equity_mcp.agent.llm_client import ScriptedLLMClient
from hk_equity_mcp.apv_fin.monitor import run_monitor
from hk_equity_mcp.apv_fin.plan_state import PlanState, PlanStateError
from hk_equity_mcp.apv_fin.schemas import (
    MONITOR_ALLOWED_REPLAN_ACTIONS,
    SchemaValidationError,
    assert_monitor_patch_retrieve_contract,
    parse_and_validate,
)


def _base_plan_state() -> PlanState:
    return PlanState.from_initial_plan(
        {
            "steps": [
                {
                    "step_id": "s1_retrieve",
                    "step_type": "retrieve",
                    "objective": "Retrieve metric",
                    "depends_on": [],
                    "mandatory": True,
                    "success_criteria": "rows",
                    "required_capability": "financial.reported_metric_bundle",
                    "required_outputs": ["revenue", "attributable_profit"],
                    "argument_slots": {"symbol": "00700"},
                    "source_preference": ["structured_vendor"],
                    "fallback_capabilities": ["financial.official_narrative_evidence"],
                    "compute_spec": None,
                    "notes": None,
                    "retrieval_query_hint": "hk equity revenue",
                },
                {
                    "step_id": "s2_answer",
                    "step_type": "answer",
                    "objective": "Answer",
                    "depends_on": ["s1_retrieve"],
                    "mandatory": True,
                    "success_criteria": "final",
                    "compute_spec": None,
                    "required_capability": None,
                    "required_outputs": None,
                    "argument_slots": None,
                    "source_preference": None,
                    "fallback_capabilities": None,
                },
            ]
        }
    )


def test_monitor_allowed_replan_actions_frozen():
    assert set(MONITOR_ALLOWED_REPLAN_ACTIONS) == {
        "retry_current_capability",
        "switch_to_declared_fallback_capability",
        "reformulate_query",
        "clarify",
        "refuse",
        "fail",
    }


def test_monitor_missing_capability_new_step_rejected_by_schema():
    payload = {
        "schema_version": "apv_fin_monitor_v1",
        "decision": "replan",
        "replan_action": "retry_current_capability",
        "decision_reason": "try another retrieve",
        "ready_step_ids": [],
        "patch": {
            "completed_plan_step_ids": [],
            "updated_plan_steps": [],
            "new_plan_steps": [
                {
                    "step_id": "s1a_revenue_retrieve",
                    "step_type": "retrieve",
                    "objective": "Retrieve revenue",
                    "depends_on": [],
                    "mandatory": True,
                    "success_criteria": "revenue",
                }
            ],
            "removed_plan_step_ids": [],
        },
    }
    with pytest.raises(SchemaValidationError, match="missing contract fields"):
        assert_monitor_patch_retrieve_contract(payload)


def test_monitor_parse_rejects_incomplete_new_retrieve():
    raw = {
        "schema_version": "apv_fin_monitor_v1",
        "decision": "replan",
        "decision_reason": "missing capability patch",
        "ready_step_ids": [],
        "patch": {
            "completed_plan_step_ids": [],
            "updated_plan_steps": [],
            "new_plan_steps": [
                {
                    "step_id": "s1b_retrieve_fallback",
                    "step_type": "retrieve",
                    "objective": "fallback",
                    "depends_on": ["s1_retrieve"],
                    "mandatory": True,
                    "success_criteria": "close",
                }
            ],
            "removed_plan_step_ids": [],
        },
    }
    _, _, err = parse_and_validate(json.dumps(raw), "apv_fin_monitor_v1")
    assert err is not None
    assert "required_capability" in err or "missing contract" in err or "missing" in err


def test_monitor_reformulate_query_allowed():
    raw = {
        "schema_version": "apv_fin_monitor_v1",
        "decision": "replan",
        "replan_action": "reformulate_query",
        "decision_reason": "Tighten query hint",
        "ready_step_ids": [],
        "patch": {
            "completed_plan_step_ids": [],
            "updated_plan_steps": [
                {
                    "step_id": "s1_retrieve",
                    "retrieval_query_hint": "hong kong equity revenue attributable profit",
                }
            ],
            "new_plan_steps": [],
            "removed_plan_step_ids": [],
        },
    }
    payload, _, err = parse_and_validate(json.dumps(raw), "apv_fin_monitor_v1")
    assert err is None
    state = _base_plan_state()
    summary = state.apply_monitor_patch(payload["patch"])
    assert "s1_retrieve" in summary["updated_plan_steps"]
    assert "hong kong equity revenue" in (
        state.steps["s1_retrieve"].retrieval_query_hint or ""
    )


def test_monitor_llm_path_fail_closed_on_incomplete_patch():
    incomplete = {
        "schema_version": "apv_fin_monitor_v1",
        "decision": "replan",
        "decision_reason": "add retrieve",
        "ready_step_ids": [],
        "patch": {
            "completed_plan_step_ids": [],
            "updated_plan_steps": [],
            "new_plan_steps": [
                {
                    "step_id": "s1a_retrieve_eps",
                    "step_type": "retrieve",
                    "objective": "eps",
                    "depends_on": [],
                    "mandatory": True,
                    "success_criteria": "eps",
                }
            ],
            "removed_plan_step_ids": [],
        },
    }
    # First parse fails schema → repair attempt; both incomplete → fail closed.
    llm = ScriptedLLMClient([json.dumps(incomplete), json.dumps(incomplete)])
    out = run_monitor(
        llm,
        question="What is basic EPS?",
        plan_state=_base_plan_state(),
        observations=[],
        evidence_summary={},
        budget={},
        exception_trigger="required_output_missing",
    )
    assert out["ok"] is False
    assert out["fail_closed"] is True
    assert out["patch_summary"] is None
    # Must not have mutated plan with incomplete retrieve.
    assert "s1a_retrieve_eps" not in _base_plan_state().steps


def test_plan_state_rejects_retrieve_without_formal_capability():
    with pytest.raises(PlanStateError, match="required_capability"):
        PlanState.from_initial_plan(
            {
                "steps": [
                    {
                        "step_id": "s1",
                        "step_type": "retrieve",
                        "objective": "x",
                        "depends_on": [],
                        "mandatory": True,
                        "success_criteria": "y",
                        "compute_spec": None,
                    }
                ]
            }
        )
    with pytest.raises(PlanStateError, match="formal CAPABILITY_IDS"):
        PlanState.from_initial_plan(
            {
                "steps": [
                    {
                        "step_id": "s1",
                        "step_type": "retrieve",
                        "objective": "x",
                        "depends_on": [],
                        "mandatory": True,
                        "success_criteria": "y",
                        "required_capability": "market.equity_eod_price",
                        "required_outputs": ["close"],
                        "argument_slots": {},
                        "compute_spec": None,
                    }
                ]
            }
        )
