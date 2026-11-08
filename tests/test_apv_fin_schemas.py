"""APV-Fin schema parse / repair / fail-closed tests (offline)."""

from __future__ import annotations

import json

import pytest

from hk_equity_mcp.apv_fin.schemas import (
    APV_FIN_PLAN_V1,
    SchemaValidationError,
    assert_no_operation_id_in_plan,
    parse_and_validate,
    validate_against_schema,
)


def _valid_plan() -> dict:
    return {
        "schema_version": "apv_fin_plan_v1",
        "task_spec": {
            "goal": "get a metric",
            "entities": ["hk_listed_issuer"],
            "periods": ["FY2025"],
            "metrics": ["eps"],
            "constraints": [],
            "ambiguities": [],
            "answerability_hypothesis": "answerable",
            "answerability_reason": "Required evidence is available from supported sources.",
            "cutoff_date": "2026-08-12",
            "currency_unit_hints": ["HKD"],
            "metric_basis_hints": ["reported_ifrs"],
        },
        "initial_plan": {
            "steps": [
                {
                    "step_id": "r1",
                    "step_type": "retrieve",
                    "objective": "retrieve eps",
                    "depends_on": [],
                    "mandatory": True,
                    "success_criteria": "value found",
                    "retrieval_query_hint": "hk eps",
                    "notes": None,
                }
            ],
            "planning_notes": None,
        },
    }


def test_valid_plan_schema():
    payload = _valid_plan()
    validate_against_schema(payload, APV_FIN_PLAN_V1)
    parsed, repair, err = parse_and_validate(json.dumps(payload), "apv_fin_plan_v1")
    assert err is None
    assert repair == 0
    assert parsed["schema_version"] == "apv_fin_plan_v1"


def test_invalid_json_then_one_repair_then_fail_closed():
    bad = "not-json"
    repaired = json.dumps(_valid_plan())
    payload, repair, err = parse_and_validate(
        bad, "apv_fin_plan_v1", repair_content=repaired
    )
    assert err is None
    assert repair == 1
    assert payload["task_spec"]["goal"] == "get a metric"

    payload2, repair2, err2 = parse_and_validate(
        bad, "apv_fin_plan_v1", repair_content="still-not-json"
    )
    assert payload2 == {}
    assert repair2 == 1
    assert err2 is not None


def test_invalid_enum_fail_closed():
    payload = _valid_plan()
    payload["initial_plan"]["steps"][0]["step_type"] = "search"
    with pytest.raises(SchemaValidationError):
        validate_against_schema(payload, APV_FIN_PLAN_V1)


def test_additional_properties_rejected():
    payload = _valid_plan()
    payload["extra_field"] = 1
    with pytest.raises(SchemaValidationError, match="additional property"):
        validate_against_schema(payload, APV_FIN_PLAN_V1)


def test_plan_forbids_operation_id():
    payload = _valid_plan()
    payload["initial_plan"]["steps"][0]["operation_id"] = "aktools.stock_hk_hist"
    with pytest.raises(SchemaValidationError, match="operation_id"):
        assert_no_operation_id_in_plan(payload)
    # Notes mentioning the phrase are fine; only structured keys are banned.
    payload = _valid_plan()
    payload["initial_plan"]["steps"][0]["notes"] = "do not hard-code tools"
    assert_no_operation_id_in_plan(payload)
