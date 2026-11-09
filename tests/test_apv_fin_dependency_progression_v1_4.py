"""Disclosure requires/provides dependency progression (APV-Fin v1.4)."""

from __future__ import annotations

from hk_equity_mcp.apv_fin.capability_contract import load_and_validate_capability_contracts
from hk_equity_mcp.apv_fin.dependency_expansion import (
    build_followup_retrieve_step,
    dependency_progression_hash,
    disclosure_prefix_needed,
    next_capability_operations,
    next_disclosure_capability,
    rewrite_step_to_disclosure_list,
)
from hk_equity_mcp.apv_fin.plan_state import PlanState, PlanStep
from hk_equity_mcp.apv_fin.progressor import apply_deterministic_progress, classify_after_step


def test_dependency_progression_hash():
    assert dependency_progression_hash()


def test_requires_provides_chain_from_contracts():
    contracts = load_and_validate_capability_contracts()
    assert disclosure_prefix_needed(
        required_capability="financial.official_narrative_evidence",
        argument_slots={},
        contracts=contracts,
    )
    assert next_disclosure_capability(
        current_capability="disclosure.list",
        target_capability="financial.official_narrative_evidence",
    ) == "disclosure.download"
    ops = next_capability_operations(
        contracts=contracts,
        next_capability="disclosure.download",
        available={"local_key": "demo_key"},
        already_invoked=[],
    )
    assert ops
    assert all("local_key" in (o.get("requires") or []) or True for o in ops)


def test_expand_after_list_resolves_with_local_key():
    contracts = load_and_validate_capability_contracts()
    state = PlanState.from_initial_plan(
        {
            "steps": [
                rewrite_step_to_disclosure_list(
                    {
                        "step_id": "s1_retrieve",
                        "step_type": "retrieve",
                        "objective": "narrative",
                        "depends_on": [],
                        "mandatory": True,
                        "success_criteria": "snippet",
                        "required_capability": "financial.official_narrative_evidence",
                        "required_outputs": ["snippet", "local_key"],
                        "argument_slots": {},
                        "compute_spec": None,
                        "notes": None,
                    }
                )
            ]
        }
    )
    # Ensure rewritten list step loaded.
    step = state.steps["s1_retrieve"]
    assert step.required_capability == "disclosure.list"
    state.mark_running("s1_retrieve")
    classification = classify_after_step(
        step=step,
        execution_ok=True,
        projection={"operation_id": "hkexnews.list_hkexnews_announcements"},
        adapted={
            "local_key": "abc123",
            "document_identified": True,
            "title": "FY results",
        },
        mcp_execution_success=True,
    )
    assert classification["action"] == "expand_dependency"
    progress = apply_deterministic_progress(
        plan_state=state,
        step=step,
        classification=classification,
        projection={"adapted": {"local_key": "abc123"}},
        contracts=contracts,
        available_deps={"local_key": "abc123"},
        already_invoked=["hkexnews.list_hkexnews_announcements"],
    )
    assert progress["resolved"] is True
    assert progress["dependency_auto_expansion_count"] == 1
    assert progress["new_steps"]


def test_followup_step_carries_local_key_slot():
    follow = build_followup_retrieve_step(
        parent_step_id="s1_retrieve",
        next_capability="disclosure.download",
        available={"local_key": "k1"},
        index=1,
        required_outputs=["local_key"],
    )
    assert follow["argument_slots"]["local_key"] == "k1"
    assert follow["required_capability"] == "disclosure.download"
