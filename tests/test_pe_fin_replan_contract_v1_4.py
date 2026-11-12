"""Regression: canonicalizer v2, OHLC vocab, grounding v3, failed signatures."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict

import pytest

from hk_equity_mcp.pe_fin_lite.envelope_projection import REQUIRED_OUTPUT_SYNONYMS
from hk_equity_mcp.pe_fin_lite.evidence_acceptance_eea import ANSWER_BEARING_FIELDS as EEA_AB
from hk_equity_mcp.pe_fin_lite.evidence_provenance import ANSWER_BEARING_FIELDS as PROV_AB
from hk_equity_mcp.pe_fin_lite.output_vocabulary import (
    CANONICAL_OUTPUT_FIELDS,
    CANONICAL_OUTPUT_VOCABULARY_VERSION,
)
from hk_equity_mcp.pe_fin_lite.schemas import PLAN_STEP_SCHEMA, validate_against_schema
from hk_equity_mcp.pe_fin_lite.step_result import (
    finalize_retrieve_status,
    make_step_result,
    retrieve_step_succeeded,
)
from hk_equity_mcp.pe_fin_replan import (
    PE_FIN_REPLAN_ARCHITECTURE,
    PE_FIN_REPLAN_CONDITION,
    PE_FIN_REPLAN_METHOD_VERSION,
    PE_FIN_REPLAN_PROTOCOL_VERSION,
)
from hk_equity_mcp.pe_fin_replan.failed_invocation import (
    FailedInvocationGuard,
    failed_invocation_policy_hash,
    invocation_key,
)
from hk_equity_mcp.pe_fin_replan.grounding import (
    GROUNDING_CONTRACT_VERSION,
    _tokenize_answer,
    validate_replanner_response,
)
from hk_equity_mcp.pe_fin_replan.identity import DEFAULT_BUDGETS, replan_protocol_identity
from hk_equity_mcp.pe_fin_replan.prompts import REPLANNER_SYSTEM_PROMPT
from hk_equity_mcp.pe_fin_replan.schemas import (
    LOOP_POLICY,
    REPLAN_SCHEMA_VERSION,
    parse_replanner_output,
)
from hk_equity_mcp.pe_fin_replan.step_canonicalizer import CANONICALIZER_VERSION

ROOT = Path(__file__).resolve().parents[1]
SMOKE5_V3 = ROOT / "outputs" / "pe_fin_replan_smoke5_live_v3"
FORMAL = ROOT / "outputs" / "pe_fin_lite_r5_hk155_live_v1"
DIAG20 = ROOT / "benchmarks" / "manifests" / "hk_pe_fin_lite_eea_diagnostic20_v1.json"

SMOKE5_V3_SHA = {
    "predictions.jsonl": "bcbeecc5a6b0fe0d07c30703c67c704a726cc53261e3f92cf4ad2bea58bb9df9",
    "run_metadata.json": "eefbd4c0ce4cfdb4f98c793878b4ac0c7a2f3f28e5735a7d54da48d9642f1389",
}
FORMAL_SHA = {
    "predictions.jsonl": "4c5c4df3ababca99f9a91203605154ad8fc4a05cdfbc2ba2acb7cc11bc2a7d04",
    "run_metadata.json": "2a3dff284afa86db8ec93433965f2b451e7aef05c65a78edd53fce3f84b8d602",
}
DIAG20_SHA = "7076a39ffc6b896ddc2bd2153c8c8d4f9a6dca268e805e3bf239867077cad1a9"
R5_HASH = "7414751814f6858a375d95da918921e3ee365dfc5b69cdaba0cee1bf9e1ecb3e"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _plan(step: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "schema_version": REPLAN_SCHEMA_VERSION,
        "action": "plan",
        "remaining_steps": [step],
    }


def test_smoke5_v3_frozen_invalidated():
    for name, expected in SMOKE5_V3_SHA.items():
        assert _sha(SMOKE5_V3 / name) == expected
    marker = (SMOKE5_V3 / "INVALIDATED.md").read_text(encoding="utf-8")
    assert (
        "invalidated_engineering_smoke_due_to_replanner_scope_contract_ohlc_vocabulary_and_natural_date_grounding"
        in marker
    )
    assert "pe_fin_replan_smoke5_live_v4" in marker


def test_formal_diag20_r5_unchanged():
    for name, expected in FORMAL_SHA.items():
        assert _sha(FORMAL / name) == expected
    assert _sha(DIAG20) == DIAG20_SHA
    ident = replan_protocol_identity()
    assert ident["retrieval_formula_hash"] == R5_HASH
    assert PE_FIN_REPLAN_CONDITION == "pe_fin_replan"
    assert PE_FIN_REPLAN_ARCHITECTURE == "plan_execute_replan"
    assert PE_FIN_REPLAN_METHOD_VERSION == "pe_fin_replan_v1_4"
    assert PE_FIN_REPLAN_PROTOCOL_VERSION == "planner_executor_replanner_contract_v4"
    assert CANONICALIZER_VERSION == "pe_fin_replan_step_canonicalizer_v2"
    assert GROUNDING_CONTRACT_VERSION == "pe_fin_replan_grounding_v3"
    assert CANONICAL_OUTPUT_VOCABULARY_VERSION == "hk155_canonical_outputs_v2"
    assert DEFAULT_BUDGETS["max_total_llm_calls_per_task"] == 16
    assert DEFAULT_BUDGETS["max_execution_rounds"] == 7
    assert DEFAULT_BUDGETS["max_total_mcp_invocations"] == 8
    assert LOOP_POLICY["round7_response_only"] is True
    assert ident["failed_invocation_policy_hash"] == failed_invocation_policy_hash()


def test_filing_to_disclosure():
    obj, _, err = parse_replanner_output(
        json.dumps(
            _plan(
                {
                    "id": "s1",
                    "type": "retrieve",
                    "domain": "filing",
                    "objective": "retrieve disclosure",
                    "required_outputs": ["title"],
                }
            )
        )
    )
    assert err is None
    assert obj["remaining_steps"][0]["domain"] == "disclosure"


def test_ticker_entity_name_adjust_aliases():
    obj, _, err = parse_replanner_output(
        json.dumps(
            _plan(
                {
                    "id": "s1",
                    "type": "retrieve",
                    "domain": "market",
                    "objective": "ohlc",
                    "required_outputs": ["open", "high", "low", "close"],
                    "scope_constraints": {
                        "ticker": "09988",
                        "entity_name": "Alibaba",
                        "adjust": "",
                    },
                }
            )
        )
    )
    assert err is None
    scope = obj["remaining_steps"][0]["scope_constraints"]
    assert scope["entity"]["code"] == "09988"
    assert scope["entity"]["name"] == "Alibaba"
    assert scope["adjustment"] == ""
    assert "ticker" not in scope and "entity_name" not in scope and "adjust" not in scope


def test_alias_conflict_fail_closed():
    _, _, err = parse_replanner_output(
        json.dumps(
            _plan(
                {
                    "id": "s1",
                    "type": "retrieve",
                    "domain": "market",
                    "objective": "x",
                    "required_outputs": ["close"],
                    "scope_constraints": {
                        "ticker": "09988",
                        "entity": {"code": "00700"},
                    },
                }
            )
        )
    )
    assert err is not None and "canonicalizer_conflict" in err


def test_top_level_entity_dict_moves():
    obj, _, err = parse_replanner_output(
        json.dumps(
            _plan(
                {
                    "id": "s1",
                    "type": "retrieve",
                    "domain": "news",
                    "objective": "x",
                    "required_outputs": ["title"],
                    "depends_on": [],
                    "entity": {"name": "Tencent", "code": "00700"},
                }
            )
        )
    )
    assert err is None
    step = obj["remaining_steps"][0]
    assert step.get("entity") is None
    assert step["scope_constraints"]["entity"]["code"] == "00700"


def test_prompt_lists_domains_scope_and_ohlc():
    assert "Allowed domains:" in REPLANNER_SYSTEM_PROMPT
    assert "disclosure" in REPLANNER_SYSTEM_PROMPT
    assert "filing" in REPLANNER_SYSTEM_PROMPT  # negation guidance
    assert "scope_constraints.entity.code" in REPLANNER_SYSTEM_PROMPT
    assert "open" in CANONICAL_OUTPUT_FIELDS
    assert "high" in CANONICAL_OUTPUT_FIELDS
    assert "low" in CANONICAL_OUTPUT_FIELDS
    assert '"open"' in REPLANNER_SYSTEM_PROMPT


def test_ohlc_completeness_close_only_incomplete():
    assert {"open", "high", "low", "close"} <= set(CANONICAL_OUTPUT_FIELDS)
    assert "open" in EEA_AB and "high" in EEA_AB and "low" in EEA_AB
    assert "open" in PROV_AB
    assert REQUIRED_OUTPUT_SYNONYMS["open"] == ("open", "Open", "开盘")
    assert REQUIRED_OUTPUT_SYNONYMS["high"][2] == "最高"
    assert REQUIRED_OUTPUT_SYNONYMS["low"][2] == "最低"

    required = ["open", "high", "low", "close"]
    close_only = finalize_retrieve_status(
        make_step_result(
            execution_ok=True,
            values={"close": 132.3, "trade_date": "2026-05-15"},
            missing_outputs=["open", "high", "low"],
            scope_match=True,
            mcp_transport_ok=True,
            tool_result_valid=True,
            status="unavailable",
        )
    )
    assert not retrieve_step_succeeded(close_only)
    assert set(close_only["missing_outputs"]) >= {"open", "high", "low"}

    full = finalize_retrieve_status(
        make_step_result(
            execution_ok=True,
            values={"open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5},
            missing_outputs=[],
            scope_match=True,
            mcp_transport_ok=True,
            tool_result_valid=True,
            status="completed",
        )
    )
    assert retrieve_step_succeeded(full)
    # required_outputs enum accepts OHLC
    validate_against_schema(
        {
            "id": "s1",
            "type": "retrieve",
            "domain": "market",
            "objective": "ohlc",
            "required_outputs": required,
            "depends_on": [],
        },
        PLAN_STEP_SCHEMA,
        "$",
    )


def test_natural_language_dates_four_formats_and_invalid():
    for text, iso in [
        ("31 March 2026", "2026-03-31"),
        ("March 31, 2026", "2026-03-31"),
        ("31 Mar 2026", "2026-03-31"),
        ("Mar 31, 2026", "2026-03-31"),
    ]:
        toks, _ = _tokenize_answer(f"report date {text} ok")
        dates = [t for t in toks if t["kind"] == "iso_date"]
        assert dates and dates[0]["value"] == iso
        plains = [t for t in toks if t["kind"] == "plain_number"]
        assert not any(str(int(t["value"])) in {"31", "2026"} for t in plains)

    bad_toks, _ = _tokenize_answer("as of 31 February 2026")
    assert any(t["kind"] == "invalid_date" for t in bad_toks)


def test_fin_cf_019_natural_date_grounding_passes():
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
    ok = validate_replanner_response(
        response={
            "completion": "answered",
            "final_answer": (
                "For FY2026 ended 31 March 2026, attributable profit was "
                "103.59 billion 人民币."
            ),
            "reasoning_brief": "observed",
        },
        question=question,
        evidence_values=evidence,
        provenance_by_key=provenance,
    )
    assert ok["ok"] is True, ok
    assert "context_date_ok" in ok["classifications"]

    bad = validate_replanner_response(
        response={
            "completion": "answered",
            "final_answer": (
                "For FY2026 ended 31 March 2026, attributable profit was "
                "999.99 billion 人民币."
            ),
            "reasoning_brief": "invented",
        },
        question=question,
        evidence_values=evidence,
        provenance_by_key=provenance,
    )
    assert bad["ok"] is False


def test_failed_signature_blocks_identical_but_allows_fallback():
    guard = FailedInvocationGuard()
    args = {"symbol": "00700", "start_date": "20260807", "end_date": "20260807"}
    guard.record_failure(
        operation_id="aktools.stock_hk_hist",
        arguments=args,
        failure_class="upstream_rate_limit",
        error="IP限流",
    )
    blocked, reason = guard.is_blocked("aktools.stock_hk_hist", args)
    assert blocked and reason == "upstream_rate_limit"
    # Changing step_id is irrelevant — same op+args still blocked
    assert invocation_key("aktools.stock_hk_hist", args) in guard.blocked_keys
    # Different operation allowed
    blocked2, _ = guard.is_blocked("aktools.stock_hk_hist_min", args)
    assert blocked2 is False
    # Semantically different args allowed
    blocked3, _ = guard.is_blocked(
        "aktools.stock_hk_hist", {**args, "start_date": "20260801"}
    )
    assert blocked3 is False

    # Blocked invoke does not require MCP; count increments
    before = guard.signature_blocked_count
    guard.mark_blocked_attempt(
        operation_id="aktools.stock_hk_hist", arguments=args, reason="upstream_rate_limit"
    )
    assert guard.signature_blocked_count == before + 1


def test_loop_invariants():
    assert LOOP_POLICY["planner_exactly_once"] is True
    assert LOOP_POLICY["executor_one_step_per_round"] is True
    assert LOOP_POLICY["no_executor_to_executor_edge"] is True
    assert LOOP_POLICY["max_execution_rounds"] == 7
    assert DEFAULT_BUDGETS["max_total_llm_calls_per_task"] == 16
    assert DEFAULT_BUDGETS["max_total_mcp_invocations"] == 8
