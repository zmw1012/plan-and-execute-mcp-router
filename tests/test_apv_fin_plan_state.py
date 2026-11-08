"""APV-Fin plan DAG / patch / multi-resolve tests."""

from __future__ import annotations

import pytest

from hk_equity_mcp.apv_fin.plan_state import PlanState, PlanStateError


def _steps():
    return [
        {
            "step_id": "a",
            "step_type": "retrieve",
            "objective": "A",
            "depends_on": [],
            "mandatory": True,
            "success_criteria": "ok",
        },
        {
            "step_id": "b",
            "step_type": "compute",
            "objective": "B",
            "depends_on": ["a"],
            "mandatory": True,
            "success_criteria": "ok",
        },
        {
            "step_id": "c",
            "step_type": "reason",
            "objective": "C",
            "depends_on": ["a"],
            "mandatory": False,
            "success_criteria": "ok",
        },
    ]


def test_duplicate_step_id():
    steps = _steps()
    steps.append(dict(steps[0]))
    with pytest.raises(PlanStateError, match="duplicate"):
        PlanState.from_initial_plan({"steps": steps})


def test_missing_dependency():
    steps = _steps()
    steps[1]["depends_on"] = ["missing"]
    with pytest.raises(PlanStateError, match="missing dependency"):
        PlanState.from_initial_plan({"steps": steps})


def test_cyclic_plan():
    steps = [
        {
            "step_id": "a",
            "step_type": "retrieve",
            "objective": "A",
            "depends_on": ["b"],
            "mandatory": True,
            "success_criteria": "ok",
        },
        {
            "step_id": "b",
            "step_type": "compute",
            "objective": "B",
            "depends_on": ["a"],
            "mandatory": True,
            "success_criteria": "ok",
        },
    ]
    with pytest.raises(PlanStateError, match="cyclic"):
        PlanState.from_initial_plan({"steps": steps})


def test_observation_resolves_multiple_steps():
    state = PlanState.from_initial_plan({"steps": _steps()})
    assert {s.step_id for s in state.ready_steps()} == {"a"}
    state.mark_running("a")
    resolved = state.resolve_steps_from_observation(["a", "c"], "obs_1")
    assert resolved == ["a", "c"]
    assert state.steps["a"].status == "resolved"
    assert state.steps["c"].status == "resolved"
    assert "obs_1" in state.steps["a"].observation_ids
    assert {s.step_id for s in state.ready_steps()} == {"b"}


def test_replan_patch():
    state = PlanState.from_initial_plan({"steps": _steps()})
    state.mark_resolved("a")
    summary = state.apply_monitor_patch(
        {
            "completed_steps": ["c"],
            "updated_steps": [
                {
                    "step_id": "b",
                    "objective": "B updated",
                    "depends_on": ["a"],
                }
            ],
            "new_steps": [
                {
                    "step_id": "d",
                    "step_type": "verify",
                    "objective": "verify",
                    "depends_on": ["b"],
                    "mandatory": True,
                    "success_criteria": "ok",
                    "retrieval_query_hint": None,
                    "notes": None,
                }
            ],
            "removed_steps": [],
        }
    )
    assert summary["revision"] == 1
    assert state.steps["b"].objective == "B updated"
    assert "d" in state.steps
    assert state.steps["c"].status == "resolved"
