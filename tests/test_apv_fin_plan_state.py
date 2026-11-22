"""APV-Fin plan DAG / patch / multi-resolve / field-ownership tests."""

from __future__ import annotations

import pytest

from hk_equity_mcp.apv_fin.plan_state import (
    MODEL_OWNED_STEP_FIELDS,
    RUNTIME_ONLY_STEP_FIELDS,
    PlanState,
    PlanStateError,
    PlanStep,
)


def _compute_spec():
    return {
        "expression": "(a / b) * 100",
        "operand_refs": {"a": "s1.a", "b": "s1.b"},
        "output_name": "ratio",
        "output_unit": "percent",
        "rounding": 2,
    }


def _steps():
    return [
        {
            "step_id": "a",
            "step_type": "retrieve",
            "objective": "A",
            "depends_on": [],
            "mandatory": True,
            "success_criteria": "ok",
            "required_capability": "financial.reported_metric",
            "required_outputs": ["value"],
            "argument_slots": {"ticker": "00700"},
            "compute_spec": None,
        },
        {
            "step_id": "b",
            "step_type": "compute",
            "objective": "B",
            "depends_on": ["a"],
            "mandatory": True,
            "success_criteria": "ok",
            "compute_spec": _compute_spec(),
        },
        {
            "step_id": "c",
            "step_type": "reason",
            "objective": "C",
            "depends_on": ["a"],
            "mandatory": False,
            "success_criteria": "ok",
            "compute_spec": None,
        },
    ]


def test_model_vs_runtime_field_sets():
    assert "status" in RUNTIME_ONLY_STEP_FIELDS
    assert "observation_ids" in RUNTIME_ONLY_STEP_FIELDS
    assert "compute_spec" in MODEL_OWNED_STEP_FIELDS
    assert "status" not in MODEL_OWNED_STEP_FIELDS


def test_model_plan_rejects_runtime_fields():
    with pytest.raises(PlanStateError, match="runtime"):
        PlanStep.from_dict(
            {
                "step_id": "a",
                "step_type": "retrieve",
                "objective": "A",
                "depends_on": [],
                "mandatory": True,
                "success_criteria": "ok",
                "status": "resolved",
            },
            allow_runtime=False,
        )


def test_compute_requires_compute_spec():
    with pytest.raises(PlanStateError, match="compute_spec"):
        PlanState.from_initial_plan(
            {
                "steps": [
                    {
                        "step_id": "b",
                        "step_type": "compute",
                        "objective": "B",
                        "depends_on": [],
                        "mandatory": True,
                        "success_criteria": "ok",
                    }
                ]
            }
        )


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
            "required_capability": "market.daily_ohlcv",
            "required_outputs": ["close"],
            "argument_slots": {},
            "compute_spec": None,
        },
        {
            "step_id": "b",
            "step_type": "compute",
            "objective": "B",
            "depends_on": ["a"],
            "mandatory": True,
            "success_criteria": "ok",
            "compute_spec": _compute_spec(),
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


def test_replan_patch_model_owned_only():
    state = PlanState.from_initial_plan({"steps": _steps()})
    state.mark_resolved("a")
    summary = state.apply_monitor_patch(
        {
            "completed_plan_step_ids": ["c"],
            "updated_plan_steps": [
                {
                    "step_id": "b",
                    "objective": "B updated",
                    "depends_on": ["a"],
                }
            ],
            "new_plan_steps": [
                {
                    "step_id": "d",
                    "step_type": "verify",
                    "objective": "verify",
                    "depends_on": ["b"],
                    "mandatory": True,
                    "success_criteria": "ok",
                    "retrieval_query_hint": None,
                    "compute_spec": None,
                    "notes": None,
                }
            ],
            "removed_plan_step_ids": [],
        }
    )
    assert summary["revision"] == 1
    assert state.steps["b"].objective == "B updated"
    assert "d" in state.steps
    # completed_plan_step_ids are suggestions only — runtime does not auto-resolve.
    assert state.steps["c"].status != "resolved"


def test_monitor_patch_rejects_runtime_fields():
    state = PlanState.from_initial_plan({"steps": _steps()})
    with pytest.raises(PlanStateError, match="runtime"):
        state.apply_monitor_patch(
            {
                "completed_plan_step_ids": [],
                "updated_plan_steps": [
                    {"step_id": "a", "status": "resolved", "observation_ids": ["x"]}
                ],
                "new_plan_steps": [],
                "removed_plan_step_ids": [],
            }
        )
