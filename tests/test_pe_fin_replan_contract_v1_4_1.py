"""PE-Fin RePlan v1.4.1: identifier grounding, constraints, semantic repair, no-result sig."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from hk_equity_mcp.pe_fin_lite.budget import BudgetTracker, BudgetedLLMClient
from hk_equity_mcp.pe_fin_lite.schemas import PlanValidationError
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
    grounding_contract_hash,
    validate_replanner_response,
)
from hk_equity_mcp.pe_fin_replan.identity import DEFAULT_BUDGETS, replan_protocol_identity
from hk_equity_mcp.pe_fin_replan.plan_ops import validate_updated_remaining_plan
from hk_equity_mcp.pe_fin_replan.replanner import (
    run_replanner,
    run_replanner_semantic_repair,
)
from hk_equity_mcp.pe_fin_replan.schemas import (
    LOOP_POLICY,
    REPLAN_SCHEMA_VERSION,
    semantic_plan_repair_policy_hash,
)

ROOT = Path(__file__).resolve().parents[1]
SMOKE5_V4 = ROOT / "outputs" / "pe_fin_replan_smoke5_live_v4"
FORMAL = ROOT / "outputs" / "pe_fin_lite_r5_hk155_live_v1"

SMOKE5_V4_SHA = {
    "predictions.jsonl": (
        "04762ce17e5b1d8d57de9b067b11f886c38acae92256e8d6e515e6bf1be72882"
    ),
    "run_metadata.json": (
        "4a43cc9ad0dccd2acfdd0bc83e1e596d789e18d91e74eddadde2f5f7ff7eff2b"
    ),
}
FORMAL_PRED_SHA = (
    "4c5c4df3ababca99f9a91203605154ad8fc4a05cdfbc2ba2acb7cc11bc2a7d04"
)

R5_HASH = "7414751814f6858a375d95da918921e3ee365dfc5b69cdaba0cee1bf9e1ecb3e"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _step(sid: str, objective: str = "x") -> Dict[str, Any]:
    return {
        "id": sid,
        "type": "retrieve",
        "domain": "disclosure",
        "objective": objective,
        "required_outputs": ["title"],
        "depends_on": [],
    }


# ---------------------------------------------------------------------------
# Fixtures shaped from Smoke-5 v4 raw evidence (no gold edits)
# ---------------------------------------------------------------------------

XD008_QUESTION = (
    'For Alibaba (09988): (a) confirm the HKEXnews title "DATE OF BOARD MEETING" '
    "on 2026-08-07 in window 2026-07-01..2026-08-12; (b) report unadjusted close "
    "on 2026-08-07 (adjust=\"\", HKD, ±0.05). Dates are YYYY-MM-DD."
)

XD008_EVIDENCE = {
    "issuer_code": "09988",
    "news_id": "12277073",
    "title": "DATE OF BOARD MEETING",
    "release_date": "2026-08-07",
    "local_key": "09988_13150_2026-08-07_df274199",
    "file_link": "/listedco/listconews/sehk/2026/0807/2026080700604.pdf",
    "source_url": "/listedco/listconews/sehk/2026/0807/2026080700604.pdf",
    "close": 123.8,
    "trade_date": "2026-08-07",
    "currency": "HKD",
    "adjustment": "",
}

XD004_QUESTION = (
    "For Tencent (00700): (a) FY2025 reported revenue (currency+unit); "
    "(b) exact HKEXnews title of the Monthly Return released 2026-08-06 "
    "in window 2026-07-01..2026-08-12."
)

XD004_EVIDENCE = {
    "issuer_code": "00700",
    "revenue": 751766000000,
    "currency": "人民币",
    "fiscal_period": "FY2025",
    "report_date": "2025-12-31",
    "news_id": "12275186",
    "title": (
        "Monthly Return of Equity Issuer on Movements in Securities "
        "for the month ended 31 July 2026"
    ),
    "release_date": "2026-08-06",
    "local_key": "00700_GEN_2026-08-06_f90abb86",
    "file_link": "/listedco/listconews/sehk/2026/0806/2026080600836.pdf",
    "source_url": "/listedco/listconews/sehk/2026/0806/2026080600836.pdf",
}


def test_smoke5_v4_frozen_invalidated():
    for name, expected in SMOKE5_V4_SHA.items():
        assert _sha(SMOKE5_V4 / name) == expected
    marker = (SMOKE5_V4 / "INVALIDATED.md").read_text(encoding="utf-8")
    assert (
        "invalidated_engineering_smoke_due_to_identifier_grounding_and_unrepaired_plan_validation"
        in marker
    )
    assert "pe_fin_replan_smoke5_live_v5" in marker
    assert "Sina market path" in marker or "Sina" in marker


def test_identity_v1_4_1_and_r5_unchanged():
    assert PE_FIN_REPLAN_CONDITION == "pe_fin_replan"
    assert PE_FIN_REPLAN_ARCHITECTURE == "plan_execute_replan"
    assert PE_FIN_REPLAN_METHOD_VERSION == "pe_fin_replan_v1_4_1"
    assert PE_FIN_REPLAN_PROTOCOL_VERSION == "planner_executor_replanner_contract_v4_1"
    assert GROUNDING_CONTRACT_VERSION == "pe_fin_replan_grounding_v3_1"
    assert REPLAN_SCHEMA_VERSION == "pe_fin_replan_v1_4"
    assert DEFAULT_BUDGETS["max_new_steps_per_replan"] == 2
    assert DEFAULT_BUDGETS["max_total_llm_calls_per_task"] == 16
    assert DEFAULT_BUDGETS["max_execution_rounds"] == 7
    assert DEFAULT_BUDGETS["max_total_mcp_invocations"] == 8
    ident = replan_protocol_identity()
    assert ident["retrieval_formula_hash"] == R5_HASH
    assert ident["grounding_contract_hash"] == grounding_contract_hash()
    assert ident["failed_invocation_policy_hash"] == failed_invocation_policy_hash()
    assert ident["semantic_plan_repair_policy_hash"] == semantic_plan_repair_policy_hash()
    assert LOOP_POLICY["max_new_steps_per_replan"] == 2
    assert LOOP_POLICY["one_semantic_or_schema_repair"] is True
    assert _sha(FORMAL / "predictions.jsonl") == FORMAL_PRED_SHA


def test_verified_news_id_not_split():
    answer = (
        "Confirmed DATE OF BOARD MEETING (news_id=12277073) on 2026-08-07; "
        "close 123.8 HKD."
    )
    toks, _ = _tokenize_answer(
        answer,
        verified_identifiers=["12277073", XD008_EVIDENCE["local_key"]],
    )
    plains = [t for t in toks if t["kind"] == "plain_number"]
    assert not any(abs(float(t["value"]) - 12277073) < 1e-9 for t in plains)
    assert any(t["kind"] == "verified_evidence_identifier" for t in toks)
    out = validate_replanner_response(
        response={
            "completion": "answered",
            "final_answer": answer,
            "reasoning_brief": "verified",
        },
        question=XD008_QUESTION,
        evidence_values=XD008_EVIDENCE,
    )
    assert out["ok"] is True, out
    assert "verified_evidence_identifier" in out["classifications"]


def test_verified_local_key_not_split():
    lk = XD008_EVIDENCE["local_key"]
    answer = (
        f"Disclosure local_key={lk}; title DATE OF BOARD MEETING; "
        "unadjusted close 123.8 HKD within tolerance ±0.05."
    )
    toks, _ = _tokenize_answer(answer, verified_identifiers=[lk, "12277073"])
    plains = [t for t in toks if t["kind"] in {"plain_number", "hk_stock_code"}]
    assert not any(str(t.get("raw")) in {"13150", "274199"} for t in plains)
    assert not any(
        t.get("value") in {"13150", "013150"} for t in toks if t["kind"] == "hk_stock_code"
    )
    out = validate_replanner_response(
        response={
            "completion": "answered",
            "final_answer": answer,
            "reasoning_brief": "verified",
        },
        question=XD008_QUESTION,
        evidence_values=XD008_EVIDENCE,
    )
    assert out["ok"] is True, out


def test_verified_file_link_source_url_not_split():
    fl = XD004_EVIDENCE["file_link"]
    answer = (
        f"FY2025 revenue 751.766 billion 人民币; Monthly Return title confirmed; "
        f"file_link={fl}; news_id=12275186."
    )
    toks, _ = _tokenize_answer(
        answer, verified_identifiers=[fl, "12275186", XD004_EVIDENCE["local_key"]]
    )
    plains = [t for t in toks if t["kind"] in {"plain_number", "hk_stock_code"}]
    assert not any(str(t.get("raw")) in {"0806", "12275186", "2026080600836"} for t in plains)
    out = validate_replanner_response(
        response={
            "completion": "answered",
            "final_answer": answer,
            "reasoning_brief": "verified",
        },
        question=XD004_QUESTION,
        evidence_values=XD004_EVIDENCE,
    )
    assert out["ok"] is True, out


def test_unverified_id_still_rejected():
    answer = (
        "Confirmed title; invented news_id=99999999 and path "
        "/listedco/listconews/sehk/2026/0806/9999999999999.pdf; close 123.8."
    )
    out = validate_replanner_response(
        response={
            "completion": "answered",
            "final_answer": answer,
            "reasoning_brief": "invented ids",
        },
        question=XD008_QUESTION,
        evidence_values=XD008_EVIDENCE,
    )
    assert out["ok"] is False
    joined = ";".join(out["reasons"])
    assert "99999999" in joined or "ungrounded_identifier" in joined


def test_tolerance_constraint_ok_in_context():
    answer = (
        "Title DATE OF BOARD MEETING on 2026-08-07 confirmed; "
        "unadjusted close 123.8 HKD (tolerance ±0.05)."
    )
    out = validate_replanner_response(
        response={
            "completion": "answered",
            "final_answer": answer,
            "reasoning_brief": "verified",
        },
        question=XD008_QUESTION,
        evidence_values=XD008_EVIDENCE,
    )
    assert out["ok"] is True, out
    assert "question_constraint_ok" in out["classifications"]


def test_bare_0_05_without_evidence_still_rejected():
    answer = "The close was 0.05 HKD."
    out = validate_replanner_response(
        response={
            "completion": "answered",
            "final_answer": answer,
            "reasoning_brief": "no evidence for 0.05 as close",
        },
        question=XD008_QUESTION,
        evidence_values=XD008_EVIDENCE,
    )
    assert out["ok"] is False
    assert any("0.05" in r for r in out["reasons"])


def test_fin_cf_019_no_gold_specific_change():
    """Regression only: natural date grounding still works; no gold edits."""
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
    )
    assert ok["ok"] is True, ok


class _ScriptedLLM:
    def __init__(self, contents: List[str]):
        self.contents = list(contents)
        self.calls: List[str] = []

    def complete(
        self,
        messages,
        *,
        temperature: float = 0.0,
        max_tokens: int = 256,
        stage: str | None = None,
    ):
        self.calls.append(stage or "unknown")
        content = self.contents.pop(0) if self.contents else "{}"
        return {
            "content": content,
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            "finish_reason": "stop",
            "response_message_keys": ["role", "content"],
            "response_observability": {
                "content_present": True,
                "content_length": len(content),
                "finish_reason": "stop",
                "response_message_keys": ["role", "content"],
                "max_tokens_requested": max_tokens,
            },
        }


def _plan_json(steps: List[Dict[str, Any]]) -> str:
    return json.dumps(
        {
            "schema_version": REPLAN_SCHEMA_VERSION,
            "action": "plan",
            "remaining_steps": steps,
        }
    )


def test_three_new_steps_triggers_one_semantic_repair_then_pass():
    previous = [_step("s1")]
    bad = _plan_json([_step("n1"), _step("n2"), _step("n3")])
    good = _plan_json([_step("n1"), _step("n2")])
    raw = _ScriptedLLM([bad])
    budget = BudgetTracker(dict(DEFAULT_BUDGETS))
    llm = BudgetedLLMClient(raw, budget)
    out = run_replanner(
        llm=llm,
        compact_state={"question": "q", "remaining_steps": previous},
        execution_round=2,
        max_execution_rounds=7,
    )
    assert out["ok"] is True
    assert out["repair_count"] == 0
    assert budget.llm_calls == 1
    with pytest.raises(PlanValidationError, match="too_many_new_steps"):
        validate_updated_remaining_plan(
            proposed=list(out["payload"]["remaining_steps"]),
            previous_remaining=previous,
            completed_ids=set(),
            completed_objectives=[],
            max_new_steps=2,
        )
    # Semantic repair (counts toward LLM budget)
    raw2 = _ScriptedLLM([good])
    llm2 = BudgetedLLMClient(raw2, budget)
    repaired = run_replanner_semantic_repair(
        llm=llm2,
        messages=list(out["messages"]),
        previous_content=str(out["assistant_content"]),
        semantic_error="too_many_new_steps:3>max=2",
        max_new_steps=2,
        execution_round=2,
    )
    assert repaired["ok"] is True
    assert repaired["repair_count"] == 1
    assert budget.llm_calls == 2
    assert "replanner_semantic_repair" in raw2.calls
    new_remaining, _ = validate_updated_remaining_plan(
        proposed=list(repaired["payload"]["remaining_steps"]),
        previous_remaining=previous,
        completed_ids=set(),
        completed_objectives=[],
        max_new_steps=2,
    )
    assert len(new_remaining) == 2


def test_semantic_repair_still_gt2_controlled_failure():
    previous = [_step("s1")]
    still_bad = _plan_json([_step("n1"), _step("n2"), _step("n3")])
    budget = BudgetTracker(dict(DEFAULT_BUDGETS))
    raw = _ScriptedLLM([still_bad])
    llm = BudgetedLLMClient(raw, budget)
    repaired = run_replanner_semantic_repair(
        llm=llm,
        messages=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "user"},
        ],
        previous_content=_plan_json([_step("a"), _step("b"), _step("c")]),
        semantic_error="too_many_new_steps:3>max=2",
        max_new_steps=2,
        execution_round=3,
    )
    assert repaired["ok"] is True  # schema ok
    with pytest.raises(PlanValidationError, match="too_many_new_steps"):
        validate_updated_remaining_plan(
            proposed=list(repaired["payload"]["remaining_steps"]),
            previous_remaining=previous,
            completed_ids=set(),
            completed_objectives=[],
            max_new_steps=2,
        )
    assert budget.llm_calls == 1


def test_identical_hkex_no_result_signature_blocked():
    guard = FailedInvocationGuard()
    args = {
        "stock_code": "00700",
        "from_date": "2026-07-01",
        "to_date": "2026-08-12",
        "title": "ANNOUNCEMENT OF THE RESULTS",
    }
    guard.record_failure(
        operation_id="hkex.list_filings",
        arguments=args,
        failure_class="no_matching_result",
        error="no matching filings",
    )
    blocked, reason = guard.is_blocked("hkex.list_filings", args)
    assert blocked and reason == "no_matching_result"
    # step_id change cannot bypass
    assert invocation_key("hkex.list_filings", args) in guard.blocked_keys
    before = guard.signature_blocked_count
    resp = guard.blocked_response(
        operation_id="hkex.list_filings", reason="no_matching_result"
    )
    guard.mark_blocked_attempt(
        operation_id="hkex.list_filings",
        arguments=args,
        reason="no_matching_result",
    )
    assert guard.signature_blocked_count == before + 1
    assert resp.get("_signature_blocked") is True


def test_broadened_args_still_allowed():
    guard = FailedInvocationGuard()
    args = {
        "stock_code": "00700",
        "from_date": "2026-07-01",
        "to_date": "2026-08-12",
        "title": "ANNOUNCEMENT OF THE RESULTS",
    }
    guard.record_failure(
        operation_id="hkex.list_filings",
        arguments=args,
        failure_class="empty_result",
        error="empty",
    )
    broadened = {
        **args,
        "from_date": "2026-06-01",
        "title": "RESULTS",
    }
    blocked, _ = guard.is_blocked("hkex.list_filings", broadened)
    assert blocked is False
    # Different fallback operation allowed
    blocked2, _ = guard.is_blocked("hkex.search_filings", args)
    assert blocked2 is False


def test_max_new_steps_unchanged_at_2():
    assert DEFAULT_BUDGETS["max_new_steps_per_replan"] == 2
    assert LOOP_POLICY["max_new_steps_per_replan"] == 2
