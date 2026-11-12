"""Regression: PlanStep canonicalizer + grounding v2 (Smoke-5 v2 shapes, no gold)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict

import pytest

from hk_equity_mcp.pe_fin_lite.schemas import SchemaValidationError
from hk_equity_mcp.pe_fin_replan import (
    PE_FIN_REPLAN_ARCHITECTURE,
    PE_FIN_REPLAN_CONDITION,
    PE_FIN_REPLAN_METHOD_VERSION,
    PE_FIN_REPLAN_PROTOCOL_VERSION,
)
from hk_equity_mcp.pe_fin_replan.grounding import (
    GROUNDING_CONTRACT_VERSION,
    _tokenize_answer,
    grounding_contract_hash,
    validate_replanner_response,
)
from hk_equity_mcp.pe_fin_replan.identity import DEFAULT_BUDGETS, replan_protocol_identity
from hk_equity_mcp.pe_fin_replan.prompts import (
    REPLANNER_SYSTEM_PROMPT,
    render_replanner_user,
)
from hk_equity_mcp.pe_fin_lite.schemas import PLAN_STEP_SCHEMA, validate_against_schema
from hk_equity_mcp.pe_fin_replan.schemas import (
    COMPUTE_STEP_TEMPLATE,
    LOOP_POLICY,
    REASON_STEP_TEMPLATE,
    REPLAN_SCHEMA_VERSION,
    RETRIEVE_STEP_TEMPLATE,
    format_replanner_repair_message,
    parse_replanner_output,
    validate_replanner_payload,
)
from hk_equity_mcp.pe_fin_replan.step_canonicalizer import (
    CANONICALIZER_VERSION,
    canonicalize_replanner_payload,
    step_canonicalizer_hash,
)

ROOT = Path(__file__).resolve().parents[1]
SMOKE5_V2 = ROOT / "outputs" / "pe_fin_replan_smoke5_live_v2"
FORMAL = ROOT / "outputs" / "pe_fin_lite_r5_hk155_live_v1"
DIAG20 = ROOT / "benchmarks" / "manifests" / "hk_pe_fin_lite_eea_diagnostic20_v1.json"

SMOKE5_V2_SHA = {
    "predictions.jsonl": "44edcc205083a4a5c0dd801a59b2ce7a3f69906426ee333601d18e6607443aa2",
    "run_metadata.json": "6445b56aaf853ef51697bc3f3311b1a77820792bcd5e333afcf2567d9744f483",
}
FORMAL_SHA = {
    "predictions.jsonl": "4c5c4df3ababca99f9a91203605154ad8fc4a05cdfbc2ba2acb7cc11bc2a7d04",
    "run_metadata.json": "2a3dff284afa86db8ec93433965f2b451e7aef05c65a78edd53fce3f84b8d602",
}
DIAG20_SHA = "7076a39ffc6b896ddc2bd2153c8c8d4f9a6dca268e805e3bf239867077cad1a9"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_smoke5_v2_frozen_and_invalidated():
    for name, expected in SMOKE5_V2_SHA.items():
        assert _sha(SMOKE5_V2 / name) == expected
    marker = (SMOKE5_V2 / "INVALIDATED.md").read_text(encoding="utf-8")
    assert (
        "invalidated_engineering_smoke_due_to_replanner_step_contract_and_numeric_grounding_validator"
        in marker
    )
    assert "pe_fin_replan_smoke5_live_v3" in marker
    assert "not usable" in marker.lower()


def test_formal_and_diagnostic20_unchanged():
    for name, expected in FORMAL_SHA.items():
        assert _sha(FORMAL / name) == expected
    assert _sha(DIAG20) == DIAG20_SHA


def test_identity_v1_3_unchanged_architecture_budgets_r5():
    from hk_equity_mcp.pe_fin_lite.retrieval_formula import (
        DESCRIBE_N,
        RETRIEVAL_FORMULA_NAME,
        RETRIEVAL_TOP_K,
        retrieval_formula_hash,
    )

    assert PE_FIN_REPLAN_CONDITION == "pe_fin_replan"
    assert PE_FIN_REPLAN_METHOD_VERSION == "pe_fin_replan_v1_4"
    assert PE_FIN_REPLAN_PROTOCOL_VERSION == "planner_executor_replanner_contract_v4"
    assert PE_FIN_REPLAN_ARCHITECTURE == "plan_execute_replan"
    assert REPLAN_SCHEMA_VERSION == "pe_fin_replan_v1_4"
    assert CANONICALIZER_VERSION == "pe_fin_replan_step_canonicalizer_v2"
    assert GROUNDING_CONTRACT_VERSION == "pe_fin_replan_grounding_v3"
    assert DEFAULT_BUDGETS["max_total_llm_calls_per_task"] == 16
    assert DEFAULT_BUDGETS["max_execution_rounds"] == 7
    assert DEFAULT_BUDGETS["max_total_mcp_invocations"] == 8
    assert LOOP_POLICY["round7_response_only"] is True
    assert RETRIEVAL_FORMULA_NAME == "R5_global_lexical@0.12"
    assert RETRIEVAL_TOP_K == 5 and DESCRIBE_N == 3
    ident = replan_protocol_identity()
    assert ident["step_canonicalizer_hash"] == step_canonicalizer_hash()
    assert ident["grounding_contract_hash"] == grounding_contract_hash()
    assert ident["retrieval_formula_hash"] == retrieval_formula_hash()


def test_planstep_templates_match_schema_keys_only():
    required = set(PLAN_STEP_SCHEMA["required"])
    props = set(PLAN_STEP_SCHEMA["properties"])
    for tmpl in (RETRIEVE_STEP_TEMPLATE, COMPUTE_STEP_TEMPLATE, REASON_STEP_TEMPLATE):
        assert required.issubset(tmpl.keys())
        assert set(tmpl.keys()).issubset(props)
        validate_against_schema(tmpl, PLAN_STEP_SCHEMA, "$")
    assert '"id"' in REPLANNER_SYSTEM_PROMPT
    assert '"type"' in REPLANNER_SYSTEM_PROMPT
    assert "step_id" not in json.dumps(RETRIEVE_STEP_TEMPLATE)
    assert "retrieve step" in REPLANNER_SYSTEM_PROMPT.lower() or '"type": "retrieve"' in REPLANNER_SYSTEM_PROMPT
    user = render_replanner_user(
        compact_state={"q": 1},
        benchmark_cutoff="2026-08-31",
        response_only=False,
        execution_round=1,
        max_execution_rounds=7,
    )
    assert "retrieve_step" in user
    assert "compute_step" in user
    assert "reason_step" in user


def test_canonicalizer_accepts_live_shaped_aliases():
    # Smoke-5 v2 shape: step_id / step_type / missing root depends_on
    raw = {
        "schema_version": REPLAN_SCHEMA_VERSION,
        "action": "plan",
        "remaining_steps": [
            {
                "step_id": "s2",
                "step_type": "retrieve",
                "domain": "market",
                "objective": "retrieve unadjusted close",
                "required_outputs": ["close", "trade_date"],
            }
        ],
    }
    canon, rules = canonicalize_replanner_payload(raw)
    assert "step_id_to_id" in rules
    assert "step_type_to_type" in rules
    assert "first_or_root_step_missing_depends_on_to_empty_array" in rules
    step = canon["remaining_steps"][0]
    assert step["id"] == "s2" and step["type"] == "retrieve"
    assert step["depends_on"] == []
    obj, _, err = parse_replanner_output(json.dumps(raw))
    assert err is None and obj is not None
    assert obj["remaining_steps"][0]["id"] == "s2"

    # type aliases
    raw2 = {
        "schema_version": REPLAN_SCHEMA_VERSION,
        "action": "plan",
        "remaining_steps": [
            {
                "id": "r1",
                "type": "analysis",
                "domain": "general",
                "objective": "summarize",
                "required_outputs": [],
            }
        ],
    }
    obj2, _, err2 = parse_replanner_output(json.dumps(raw2))
    assert err2 is None
    assert obj2["remaining_steps"][0]["type"] == "reason"


def test_canonicalizer_fail_closed_unknown_and_missing_required_outputs():
    bad_unknown = {
        "schema_version": REPLAN_SCHEMA_VERSION,
        "action": "plan",
        "remaining_steps": [
            {
                "id": "s1",
                "type": "retrieve",
                "domain": "market",
                "objective": "x",
                "required_outputs": ["close"],
                "depends_on": [],
                "invented_field": "nope",
            }
        ],
    }
    with pytest.raises(SchemaValidationError):
        validate_replanner_payload(
            canonicalize_replanner_payload(bad_unknown)[0]
        )

    missing_ro = {
        "schema_version": REPLAN_SCHEMA_VERSION,
        "action": "plan",
        "remaining_steps": [
            {
                "id": "s1",
                "type": "retrieve",
                "domain": "market",
                "objective": "x",
                "depends_on": [],
            }
        ],
    }
    _, _, err = parse_replanner_output(json.dumps(missing_ro))
    assert err is not None
    assert "required_outputs" in err


def test_repair_prompt_includes_planstep_templates_and_path():
    msg = format_replanner_repair_message(
        "$.anyOf[0].remaining_steps[0]: missing required property 'type'",
        response_only=False,
        json_path="$.anyOf[0].remaining_steps[0]",
    )
    assert "JSON path: $.anyOf[0].remaining_steps[0]" in msg
    assert "Validation error:" in msg
    assert "response_only=False" in msg
    assert '"type": "retrieve"' in msg
    assert '"type": "compute"' in msg
    assert '"type": "reason"' in msg
    assert "Regenerate ONE complete JSON object" in msg
    assert "depends_on" in msg


def test_iso_date_not_split_into_negative_fragments():
    tokens, _ = _tokenize_answer("Revenue for period ended 2025-12-31 was 751.77 billion.")
    kinds = [t["kind"] for t in tokens]
    assert "iso_date" in kinds
    plains = [t for t in tokens if t["kind"] == "plain_number"]
    assert not any(float(t["value"]) < 0 for t in plains)
    assert not any(t["raw"] in {"-12", "-31", "12", "31"} for t in plains)


def test_mkt_dev_003_shaped_grounding_allows_date_and_close():
    question = (
        "For Alibaba (9988.HK) on 2026-05-15, report unadjusted open, high, low, "
        "and close in HKD (adjust=\"\", half-up 2dp, tolerance ±0.05 each)."
    )
    evidence = {
        "trade_date": "2026-05-15T00:00:00.000",
        "close": 132.3,
        "currency": "HKD",
        "issuer_code": "09988",
    }
    provenance = {
        "close": {
            "field_provenance": {
                "close": {
                    "value": 132.3,
                    "provenance_class": "observed",
                    "source": "mcp",
                },
                "trade_date": {
                    "value": "2026-05-15T00:00:00.000",
                    "provenance_class": "observed",
                    "source": "mcp",
                },
            }
        },
        "trade_date": {
            "field_provenance": {
                "trade_date": {
                    "value": "2026-05-15T00:00:00.000",
                    "provenance_class": "observed",
                    "source": "mcp",
                }
            }
        },
    }
    resp = {
        "completion": "answered",
        "final_answer": "On 2026-05-15, Alibaba (09988) unadjusted close was 132.3 HKD.",
        "reasoning_brief": "from verified market retrieve",
    }
    out = validate_replanner_response(
        response=resp,
        question=question,
        evidence_values=evidence,
        provenance_by_key=provenance,
    )
    assert out["ok"] is True, out
    assert "context_date_ok" in out["classifications"]
    assert "exact_evidence_numeric" in out["classifications"] or (
        "deterministic_rounding" in out["classifications"]
    )


def test_xd_dev_004_shaped_unit_conversion_and_rounding():
    question = (
        "For Tencent (00700): (a) FY2025 reported revenue (currency+unit); "
        "(b) exact HKEXnews title released 2026-08-06."
    )
    evidence = {
        "revenue": 751766000000,
        "currency": "人民币",
        "report_date": "2025-12-31",
        "fiscal_period": "FY2025",
        "issuer_code": "00700",
        "release_date": "2026-08-06",
        "title": "Monthly Return of Equity Issuer on Movements in Securities for the month ended 31 July 2026",
    }
    provenance = {
        "revenue": {
            "field_provenance": {
                "revenue": {
                    "value": 751766000000,
                    "provenance_class": "observed",
                    "source": "mcp",
                }
            }
        }
    }
    ok_exact = validate_replanner_response(
        response={
            "completion": "answered",
            "final_answer": (
                "FY2025 revenue was 751.766 billion 人民币; "
                "title released 2026-08-06 as in evidence."
            ),
            "reasoning_brief": "observed revenue",
        },
        question=question,
        evidence_values=evidence,
        provenance_by_key=provenance,
    )
    assert ok_exact["ok"] is True, ok_exact
    assert "derived_unit_conversion" in ok_exact["classifications"] or (
        "exact_evidence_numeric" in ok_exact["classifications"]
    )

    ok_round = validate_replanner_response(
        response={
            "completion": "answered",
            "final_answer": (
                "FY2025 revenue was 751.77 billion 人民币; "
                "Monthly Return released 2026-08-06."
            ),
            "reasoning_brief": "observed revenue rounded",
        },
        question=question,
        evidence_values=evidence,
        provenance_by_key=provenance,
    )
    assert ok_round["ok"] is True, ok_round
    assert (
        "deterministic_rounding" in ok_round["classifications"]
        or "derived_unit_conversion" in ok_round["classifications"]
    )

    # Unit not explicit → must not freely accept scaled display
    bad_unit = validate_replanner_response(
        response={
            "completion": "answered",
            "final_answer": "FY2025 revenue was 751.77.",
            "reasoning_brief": "no unit",
        },
        question=question,
        evidence_values=evidence,
        provenance_by_key=provenance,
    )
    assert bad_unit["ok"] is False
    assert any(
        c in {"unit_mismatch", "ungrounded_answer_value"}
        for c in bad_unit["classifications"]
    )


def test_question_only_target_number_rejected():
    question = "What is the close on 2026-05-15 if someone claims 999.99?"
    evidence = {
        "trade_date": "2026-05-15",
        "close": 132.3,
        "currency": "HKD",
    }
    provenance = {
        "close": {
            "field_provenance": {
                "close": {"value": 132.3, "provenance_class": "observed"}
            }
        }
    }
    out = validate_replanner_response(
        response={
            "completion": "answered",
            "final_answer": "The close is 999.99 HKD on 2026-05-15.",
            "reasoning_brief": "invented",
        },
        question=question,
        evidence_values=evidence,
        provenance_by_key=provenance,
    )
    assert out["ok"] is False
    assert any("999.99" in r or "ungrounded" in r for r in out["reasons"])


def test_fin_cf_shaped_profit_unit_scale():
    question = (
        "What was Alibaba's reported net income attributable to ordinary "
        "shareholders for FY2026 ended 31 March 2026? State the currency and unit."
    )
    evidence = {
        "attributable_profit": 103592000000,
        "currency": "人民币",
        "report_date": "2026-03-31",
        "fiscal_period": "FY2026",
        "issuer_code": "09988",
    }
    provenance = {
        "attributable_profit": {
            "field_provenance": {
                "attributable_profit": {
                    "value": 103592000000,
                    "provenance_class": "observed",
                }
            }
        }
    }
    out = validate_replanner_response(
        response={
            "completion": "answered",
            "final_answer": (
                "FY2026 attributable profit was 103.59 billion 人民币 "
                "(report date 2026-03-31)."
            ),
            "reasoning_brief": "observed",
        },
        question=question,
        evidence_values=evidence,
        provenance_by_key=provenance,
    )
    assert out["ok"] is True, out


def test_loop_invariants_documented_in_policy():
    assert LOOP_POLICY["planner_exactly_once"] is True
    assert LOOP_POLICY["executor_one_step_per_round"] is True
    assert LOOP_POLICY["no_executor_to_executor_edge"] is True
    assert LOOP_POLICY["final_response_from_replanner_only"] is True
    assert LOOP_POLICY["max_execution_rounds"] == 7
    assert LOOP_POLICY["round7_response_only"] is True
    assert LOOP_POLICY["one_replanner_repair"] is True
    assert DEFAULT_BUDGETS["max_total_llm_calls_per_task"] == 16
    assert DEFAULT_BUDGETS["max_total_mcp_invocations"] == 8
