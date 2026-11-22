"""Offline engineering-audit regressions for PE-Fin Lite protocol fixes.

No real LLM / Embedding / MCP / paid API.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from hk_equity_mcp.agent.llm_client import ScriptedLLMClient
from hk_equity_mcp.apv_fin.runtime_safety import summarize_prediction_outcomes
from hk_equity_mcp.experiment.unified_grading import (
    _looks_like_fabricated_numeric,
    grade_refusal,
)
from hk_equity_mcp.pe_fin_lite.agent import PeFinLiteAgent
from hk_equity_mcp.pe_fin_lite.argument_grounding import (
    argument_grounding_hash,
    collect_step_grounding,
    extract_explicit_hk_codes,
)
from hk_equity_mcp.pe_fin_lite.budget import BudgetTracker
from hk_equity_mcp.pe_fin_lite.capability_profile import load_capability_profile
from hk_equity_mcp.pe_fin_lite.disclosure_workflow import (
    build_stage_arguments,
    run_disclosure_composite,
)
from hk_equity_mcp.pe_fin_lite.envelope_projection import project_invoke_to_required_outputs
from hk_equity_mcp.pe_fin_lite.identity import (
    FORMAL_OUTPUT_DIRS,
    METRICS_VERSION,
    protocol_identity_hashes,
    worst_case_budget_estimate,
)
from hk_equity_mcp.pe_fin_lite.output_vocabulary import (
    CANONICAL_OUTPUT_FIELDS,
    canonical_output_vocabulary_hash,
    scope_contract_hash,
    validate_required_outputs,
)
from hk_equity_mcp.pe_fin_lite.planner import run_planner
from hk_equity_mcp.pe_fin_lite.prompts import pe_fin_lite_prompt_hash, render_planner_user
from hk_equity_mcp.pe_fin_lite.schemas import (
    OBJECTIVE_MAX_LENGTH,
    PE_FIN_LITE_PLAN_V1,
    PlanValidationError,
    SchemaValidationError,
    format_repair_user_message,
    parse_and_validate,
    validate_against_schema,
    validate_plan_semantics,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "apv_fin_v7_envelopes"


def _resolver_shape(outer: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "success": bool(outer.get("success")),
        "operation_id": outer.get("operation_id"),
        "arguments": outer.get("arguments") or {},
        "result": outer,
        "error": outer.get("error"),
    }


def test_objective_max_length_contract_and_repair_message():
    assert OBJECTIVE_MAX_LENGTH == 480
    over = "x" * (OBJECTIVE_MAX_LENGTH + 20)
    plan = {
        "steps": [
            {
                "id": "s1",
                "type": "retrieve",
                "domain": "financial",
                "objective": over,
                "required_outputs": ["basic_eps"],
                "depends_on": [],
            }
        ]
    }
    with pytest.raises(SchemaValidationError, match="maxLength"):
        validate_against_schema(plan, PE_FIN_LITE_PLAN_V1)
    msg = format_repair_user_message(
        "plan",
        "$.steps[0].objective: string length exceeds maxLength",
        previous_content=json.dumps(plan),
    )
    assert "actual_length=" in msg
    assert "allowed_length=" in msg
    assert "COMPLETE new plan" in msg or "regenerate the FULL" in msg or "COMPLETE" in msg
    assert "Do NOT splice" in msg or "splice" in msg
    prompt = render_planner_user(
        question="q",
        capability_profile={"profile_id": "x"},
        limits={},
        max_plan_steps=6,
    )
    assert str(OBJECTIVE_MAX_LENGTH) in prompt


def test_compute_spec_contract_requires_expression_and_typed_operands():
    missing_expr = {
        "steps": [
            {
                "id": "s1",
                "type": "retrieve",
                "domain": "financial",
                "objective": "rev",
                "required_outputs": ["revenue"],
                "depends_on": [],
            },
            {
                "id": "s2",
                "type": "compute",
                "domain": "compute",
                "objective": "ratio",
                "required_outputs": [],
                "depends_on": ["s1"],
                "compute_spec": {
                    "expression": "",
                    "operand_refs": {"a": "s1.revenue"},
                    "output_name": "out",
                },
            },
        ]
    }
    with pytest.raises((SchemaValidationError, PlanValidationError)):
        validate_against_schema(missing_expr, PE_FIN_LITE_PLAN_V1)
        validate_plan_semantics(missing_expr)

    missing_refs = {
        "steps": [
            {
                "id": "s1",
                "type": "retrieve",
                "domain": "financial",
                "objective": "rev",
                "required_outputs": ["revenue"],
                "depends_on": [],
            },
            {
                "id": "s2",
                "type": "compute",
                "domain": "compute",
                "objective": "ratio",
                "required_outputs": [],
                "depends_on": ["s1"],
                "compute_spec": {
                    "expression": "a/b",
                    "operand_refs": {},
                    "output_name": "out",
                },
            },
        ]
    }
    with pytest.raises(PlanValidationError, match="operand_refs"):
        validate_plan_semantics(missing_refs)

    prompt = render_planner_user(
        question="q",
        capability_profile={"profile_id": "x"},
        limits={},
        max_plan_steps=6,
    )
    assert "compute_spec" in prompt
    assert "operand_refs" in prompt
    assert "expression" in prompt


def test_canonical_required_outputs_reject_free_text():
    assert validate_required_outputs(["basic EPS value", "source evidence availability"])
    plan = {
        "steps": [
            {
                "id": "s1",
                "type": "retrieve",
                "domain": "financial",
                "objective": "eps",
                "required_outputs": ["basic EPS value"],
                "depends_on": [],
            }
        ]
    }
    with pytest.raises(SchemaValidationError):
        validate_against_schema(plan, PE_FIN_LITE_PLAN_V1)
    ok = {
        "steps": [
            {
                "id": "s1",
                "type": "retrieve",
                "domain": "financial",
                "objective": "eps",
                "required_outputs": ["basic_eps", "currency", "report_date"],
                "depends_on": [],
                "scope_constraints": {
                    "entity": {"code": "00005"},
                    "period": "FY2025",
                    "metric_basis": "reported",
                },
            }
        ]
    }
    validate_against_schema(ok, PE_FIN_LITE_PLAN_V1)
    validate_plan_semantics(ok)
    assert canonical_output_vocabulary_hash()
    assert scope_contract_hash()
    assert set(CANONICAL_OUTPUT_FIELDS)


def test_financial_market_disclosure_fixture_replay_canonical():
    fin = json.loads((FIXTURES / "financial_analysis_indicator.json").read_text())["invoke"]
    p = project_invoke_to_required_outputs(
        invoke_result=_resolver_shape(fin),
        required_outputs=["basic_eps", "revenue", "currency", "report_date", "issuer_code"],
        domain="financial",
    )
    assert p["values"]["basic_eps"] is not None
    assert p["values"]["revenue"] is not None
    assert p["values"]["currency"]
    assert p["values"]["report_date"]
    assert p["values"]["issuer_code"] == "00005"
    assert "issuer_code" not in p["missing_outputs"]

    long_path = FIXTURES / "financial_report_long_table.json"
    if long_path.exists():
        long = json.loads(long_path.read_text())
        inv = long.get("invoke") or long
        p2 = project_invoke_to_required_outputs(
            invoke_result=_resolver_shape(inv),
            required_outputs=["basic_eps", "revenue", "currency", "report_date"],
            domain="financial",
        )
        assert len(p2["missing_outputs"]) < 4

    mkt = json.loads((FIXTURES / "market_ohlcv_chinese.json").read_text())["invoke"]
    pm = project_invoke_to_required_outputs(
        invoke_result=_resolver_shape(mkt),
        required_outputs=["close", "volume", "trade_date"],
        domain="market",
    )
    assert pm["missing_outputs"] == []

    disc = json.loads((FIXTURES / "disclosure_list_hkex.json").read_text())["invoke"]
    pd = project_invoke_to_required_outputs(
        invoke_result=_resolver_shape(disc),
        required_outputs=["news_id", "title", "release_date", "issuer_code", "local_key"],
        domain="disclosure",
    )
    assert pd["missing_outputs"] == []
    assert pd["values"]["news_id"]
    assert pd["values"]["local_key"]


def test_entity_code_date_grounding_generic():
    assert extract_explicit_hk_codes("Tencent (00700)") == ["00700"]
    assert extract_explicit_hk_codes("0700.HK") == ["00700"]
    assert extract_explicit_hk_codes("HSBC 00005") == ["00005"]
    g = collect_step_grounding(
        question="For Tencent (00700) in 2026-07-01..2026-08-12, news titled 'Next Day Disclosure Return'",
        step={
            "domain": "disclosure",
            "objective": "list announcements",
            "required_outputs": ["news_id"],
            "depends_on": [],
        },
    )
    assert g["stock_code"] == "00700"
    assert g["from_date"] == "2026-07-01"
    assert g["to_date"] == "2026-08-12"
    assert g["title_keyword"] == "Next Day Disclosure Return"
    name_only = collect_step_grounding(
        question="What is the EPS for Some Unknown Co?",
        step={
            "scope_constraints": {"entity": {"name": "Some Unknown Co"}},
            "objective": "eps",
            "required_outputs": ["basic_eps"],
            "depends_on": [],
        },
    )
    assert name_only["stock_code"] is None
    assert name_only["needs_lookup"] is True
    assert argument_grounding_hash()


def test_disclosure_composite_gets_stock_code_from_question():
    meta = {
        "hkexnews.list_hkexnews_announcements": {
            "domain": "news",
            "capabilities": ["disclosure_search", "list"],
            "provides": ["local_key", "news_id", "title", "release_date"],
        }
    }
    contracts = {
        "hkexnews.list_hkexnews_announcements": {
            "json_schema": {
                "properties": {
                    "stock_code": {"type": "string"},
                    "from_date": {"type": "string"},
                    "to_date": {"type": "string"},
                    "keyword": {"type": "string"},
                },
                "required": ["stock_code", "from_date", "to_date"],
            }
        }
    }
    captured: List[Dict[str, Any]] = []

    def invoke(oid: str, args: Dict[str, Any]) -> Dict[str, Any]:
        captured.append({"oid": oid, "args": args})
        return {
            "success": True,
            "result": {
                "rows": [
                    {
                        "news_id": "1",
                        "title": "Next Day Disclosure Return",
                        "release_date": "2026-07-09",
                        "local_key": "k1",
                        "issuer_code": "00700",
                    }
                ]
            },
        }

    def describe(oid: str) -> Dict[str, Any]:
        return {"contract": contracts[oid]}

    out = run_disclosure_composite(
        step={
            "domain": "disclosure",
            "workflow": "retrieve_official_disclosure_evidence",
            "objective": "Find Next Day Disclosure Return",
            "required_outputs": ["news_id", "title", "release_date", "issuer_code", "local_key"],
            "depends_on": [],
        },
        metadata_by_id=meta,
        runtime_operation_ids=list(meta),
        invoke_fn=invoke,
        describe_fn=describe,
        question="For Tencent (00700) in 2026-07-01..2026-08-12, title 'Next Day Disclosure Return'",
        max_stages=1,
    )
    assert captured
    assert captured[0]["args"]["stock_code"] == "00700"
    assert captured[0]["args"]["from_date"] == "2026-07-01"
    assert captured[0]["args"]["to_date"] == "2026-08-12"
    assert out.get("error") is None or "missing_stage_args" not in str(out.get("error"))


def test_disclosure_missing_args_structured_not_crash():
    meta = {
        "hkexnews.list_hkexnews_announcements": {
            "domain": "news",
            "capabilities": ["list"],
            "provides": ["news_id"],
        }
    }

    def invoke(oid: str, args: Dict[str, Any]) -> Dict[str, Any]:
        raise AssertionError("must not invoke when args missing")

    def describe(oid: str) -> Dict[str, Any]:
        return {
            "contract": {
                "json_schema": {
                    "properties": {"stock_code": {"type": "string"}},
                    "required": ["stock_code"],
                }
            }
        }

    out = run_disclosure_composite(
        step={
            "domain": "disclosure",
            "workflow": "retrieve_official_disclosure_evidence",
            "objective": "list",
            "required_outputs": ["news_id"],
            "depends_on": [],
        },
        metadata_by_id=meta,
        runtime_operation_ids=list(meta),
        invoke_fn=invoke,
        describe_fn=describe,
        question="What filings exist for an unnamed issuer?",
        max_stages=1,
    )
    assert out["status"] != "completed"
    assert "missing_stage_args" in str(out.get("error") or "")


class _NullContentLLM:
    """Provider-style null/empty content without using ScriptedLLMClient's len()."""

    provider = "mock"
    model = "mock-llm"

    def __init__(self, contents: List[Any]):
        self._queue = list(contents)

    def complete(self, messages, *, temperature=0.0, max_tokens=1024, stage=None):
        content = self._queue.pop(0) if self._queue else ""
        return {
            "content": content,
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            "latency_ms": 0.0,
            "provider": self.provider,
            "model": self.model,
        }


def test_planner_provider_style_failures_are_structured():
    profile = load_capability_profile(ROOT)
    limits = {"max_total_llm_calls_per_task": 8, "max_total_mcp_invocations": 8}

    # Truncated / empty / null content → structured failure after repair.
    for content in (None, "", "{", '{"steps":['):
        llm = _NullContentLLM([content, content])
        out = run_planner(
            llm=llm,  # type: ignore[arg-type]
            question="q",
            capability_profile=profile,
            limits=limits,
        )
        assert out["ok"] is False
        assert out["error_type"] == "planner_schema_or_semantics"
        assert out["stopping_reason"]

    # Overlong objective survives into repair then still invalid.
    over = "y" * (OBJECTIVE_MAX_LENGTH + 50)
    bad_plan = {
        "steps": [
            {
                "id": "s1",
                "type": "retrieve",
                "domain": "financial",
                "objective": over,
                "required_outputs": ["basic_eps"],
                "depends_on": [],
            }
        ]
    }
    payload = json.dumps(bad_plan)
    llm = ScriptedLLMClient([payload, payload])
    out = run_planner(
        llm=llm, question="q", capability_profile=profile, limits=limits
    )
    assert out["ok"] is False
    assert "maxLength" in str(out["error"]) or "length" in str(out["error"]).lower()


def test_agent_planner_failure_does_not_raise_runner_exception():
    class _FakeResolver:
        def list_underlying_operation_ids(self):
            return []

        @property
        def registry(self):
            return {}

    profile = load_capability_profile(ROOT)
    llm = _NullContentLLM([None, ""])
    agent = PeFinLiteAgent(
        llm=llm,  # type: ignore[arg-type]
        resolver=_FakeResolver(),  # type: ignore[arg-type]
        capability_profile=profile,
        config={
            "limits": {
                "max_total_llm_calls_per_task": 8,
                "max_total_mcp_invocations": 8,
            },
            "use_mock_binder": True,
        },
        live_mode=False,
    )
    result = agent.run("dummy question")
    assert result["status"] == "failed"
    assert result["error_type"] == "planner_schema_or_semantics"
    assert result["stopping_reason"]
    assert result.get("monitor_decisions") == []
    assert result.get("plan_revisions") == []


def test_metrics_consistency_and_all_failed_artifacts():
    expected = ["a", "b", "c"]
    preds = {
        "a": {"status": "failed"},
        "b": {"status": "partial"},
        "c": {"status": "failed"},
    }
    out = summarize_prediction_outcomes(
        expected_ids=expected, predictions_by_id=preds
    )
    assert out["processed_task_count"] == 3
    assert out["full_count"] == 0
    assert out["partial_count"] == 1
    assert out["failed_count"] == 2
    assert out["missing_ids"] == []
    assert METRICS_VERSION == "pe_fin_lite_metrics_v3"
    assert FORMAL_OUTPUT_DIRS["smoke5_live"].endswith("v3")


def test_no_monitor_no_global_replan_budgets_unchanged():
    limits = {
        "max_plan_steps": 6,
        "max_total_llm_calls_per_task": 8,
        "max_total_mcp_invocations": 8,
        "max_local_fallback_per_retrieve": 1,
        "retrieval_top_k": 5,
        "describe_top_n": 3,
    }
    est = worst_case_budget_estimate(limits)
    assert est["monitor_calls"] == 0
    assert est["global_replan_rounds"] == 0
    assert est["per_task_max_mcp_invocations"] == 8
    assert est["max_total_llm_calls_per_task"] == 8
    assert est["retrieval_top_k"] == 5
    assert est["max_local_fallback_per_retrieve"] == 1
    b = BudgetTracker(limits)
    assert b.limits["max_total_mcp_invocations"] == 8
    assert b.limits["max_total_llm_calls_per_task"] == 8


def test_grader_ignores_question_grounded_stock_code():
    text = (
        "Unavailable: no verified closing-price data was provided for Tencent "
        "(e.g., Tencent Holdings, 0700.HK), so the latest/today's close, currency, "
        "and period cannot be confirmed."
    )
    assert _looks_like_fabricated_numeric(
        text, question="What is Tencent's latest / current / today's closing price?"
    ) is False
    row = {
        "final_answer": text,
        "predicted_status": "unavailable",
        "gold_status": "unanswerable",
        "question": "What is Tencent's latest / current / today's closing price?",
    }
    grade = grade_refusal(row, {"expected_behavior": "unanswerable", "question": row["question"]})
    assert grade["answer_correct"] is True
    assert grade["reason"] == "refusal_status_ok"
    # Still flag a real fabricated price.
    bad = "Unavailable, but the close is 512.3 HKD."
    assert _looks_like_fabricated_numeric(bad, question=row["question"]) is True


def test_protocol_identity_hashes_present():
    ids = protocol_identity_hashes()
    assert ids["prompt_hash"] == pe_fin_lite_prompt_hash()
    assert ids["canonical_output_vocabulary_hash"]
    assert ids["scope_contract_hash"]
    assert ids["argument_grounding_hash"]
    assert ids["metrics_version"] == METRICS_VERSION
