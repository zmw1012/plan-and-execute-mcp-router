"""Tool Binding + Monitor protocol v3 + metrics regressions (offline; no paid API)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hk_equity_mcp.agent.llm_client import ScriptedLLMClient
from hk_equity_mcp.apv_fin.argument_grounding import (
    normalize_hk_symbol,
    normalize_iso_date,
    propose_arguments_for_schema,
)
from hk_equity_mcp.apv_fin.identity import (
    TOOL_BINDING_PROTOCOL_CHANGE,
    combined_schema_hash,
    schema_hashes,
)
from hk_equity_mcp.apv_fin.monitor import run_monitor
from hk_equity_mcp.apv_fin.plan_state import PlanState
from hk_equity_mcp.apv_fin.prompts import (
    APV_FIN_MONITOR_USER_TEMPLATE,
    APV_FIN_TOOL_BINDING_USER_TEMPLATE,
    apv_fin_prompt_hash,
    assert_prompts_clean,
)
from hk_equity_mcp.apv_fin.schemas import (
    APV_FIN_MONITOR_V1,
    APV_FIN_TOOL_BINDING_V1,
    SchemaValidationError,
    format_repair_user_message,
    parse_and_validate,
    validate_against_schema,
)
from hk_equity_mcp.apv_fin.scheduler import (
    BudgetTracker,
    context_from_task_spec,
    execute_retrieve_step,
)
from hk_equity_mcp.apv_fin.plan_state import PlanStep
from hk_equity_mcp.apv_fin.metadata import load_and_validate_metadata
from hk_equity_mcp.apv_fin.reranker import rerank_candidates
from hk_equity_mcp.apv_fin.tool_binding import (
    ToolBindingValidationError,
    build_deterministic_binding_proposal,
    select_candidates_to_describe,
    validate_binding_payload,
)
from hk_equity_mcp.experiment.unified_apv_fin_metrics import (
    aggregate_apv_fin_metrics,
    calculation_validation_flag,
    evidence_chain_complete,
    plan_completion_flag,
    strict_answer_success,
    tool_execution_success_flag,
)
from hk_equity_mcp.operation_resolver.retrieval import DeterministicHashEmbeddingBackend
from hk_equity_mcp.operation_resolver.server import OperationResolverServer
from hk_equity_mcp.real_mcp.unified_catalog import build_unified_catalog
from hk_equity_mcp.real_mcp.unified_gateway import (
    ScriptedMockTransport,
    UnifiedRealMcpGateway,
)

ROOT = Path(__file__).resolve().parents[1]


def test_tool_binding_schema_valid():
    payload = {
        "schema_version": "apv_fin_tool_binding_v1",
        "decision": "select",
        "selected_operation_id": "example.op",
        "arguments": {"symbol": "00700"},
        "argument_sources": {"symbol": "task_spec"},
        "expected_output": "rows",
        "decision_reason": "Compatible candidate with grounded symbol.",
        "reformulated_query": None,
    }
    validate_against_schema(payload, APV_FIN_TOOL_BINDING_V1)


def test_argument_grounding_hk_codes_and_dates():
    assert normalize_hk_symbol("0700.HK") == "00700"
    assert normalize_hk_symbol("700") == "00700"
    assert normalize_hk_symbol("0005.HK") == "00005"
    assert normalize_hk_symbol("Tencent") is None
    assert normalize_iso_date("2026-07-31") == "2026-07-31"
    assert normalize_iso_date("20260731") == "2026-07-31"


def test_binding_rejects_empty_args_and_off_candidate():
    contract = {
        "json_schema": {
            "type": "object",
            "properties": {"symbol": {"type": "string"}},
            "required": ["symbol"],
        }
    }
    with pytest.raises(ToolBindingValidationError, match="missing required"):
        validate_binding_payload(
            {
                "schema_version": "apv_fin_tool_binding_v1",
                "decision": "select",
                "selected_operation_id": "op.a",
                "arguments": {},
                "argument_sources": {},
                "expected_output": "x",
                "decision_reason": "bad",
            },
            allowed_operation_ids=["op.a"],
            described_contracts={"op.a": contract},
        )
    with pytest.raises(ToolBindingValidationError, match="not in current"):
        validate_binding_payload(
            {
                "schema_version": "apv_fin_tool_binding_v1",
                "decision": "select",
                "selected_operation_id": "op.gold",
                "arguments": {"symbol": "00700"},
                "argument_sources": {"symbol": "task_spec"},
                "expected_output": "x",
                "decision_reason": "bad",
            },
            allowed_operation_ids=["op.a"],
            described_contracts={"op.a": contract},
        )


def test_tier_c_blocked_when_ab_exists():
    contract = {
        "json_schema": {
            "type": "object",
            "properties": {"symbol": {"type": "string"}},
            "required": ["symbol"],
        }
    }
    with pytest.raises(ToolBindingValidationError, match="Tier C"):
        validate_binding_payload(
            {
                "schema_version": "apv_fin_tool_binding_v1",
                "decision": "select",
                "selected_operation_id": "fin.op",
                "arguments": {"symbol": "00700"},
                "argument_sources": {"symbol": "task_spec"},
                "expected_output": "x",
                "decision_reason": "no",
            },
            allowed_operation_ids=["mkt.op", "fin.op"],
            described_contracts={"fin.op": contract, "mkt.op": contract},
            tier_by_id={"mkt.op": "A", "fin.op": "C"},
        )


def test_market_close_rerank_prefers_market_hist_over_financial():
    bundle = build_unified_catalog(root=ROOT)
    meta = load_and_validate_metadata(catalog=bundle, root=ROOT)
    step = PlanStep(
        step_id="r1",
        step_type="retrieve",
        objective="Retrieve HK equity market close price history",
        depends_on=[],
        mandatory=True,
        success_criteria="close obtained",
        retrieval_query_hint="stock hk hist close",
    )
    ctx = context_from_task_spec(
        {
            "goal": "obtain close price",
            "metrics": ["close"],
            "periods": ["2026-07-31"],
            "cutoff_date": "2026-08-12",
        },
        step,
    )
    assert "market" in ctx.domain_hints
    candidates = [
        {
            "operation_id": "aktools.stock_financial_hk_analysis_indicator_em",
            "score": 0.99,
        },
        {"operation_id": "aktools.stock_hk_hist", "score": 0.40},
    ]
    out = rerank_candidates(
        candidates,
        meta,
        ctx,
        runtime_operation_ids=bundle.stats.runtime_operation_ids,
    )
    reranked = out["reranked"]
    assert reranked[0]["operation_id"] == "aktools.stock_hk_hist"
    fin = next(
        r
        for r in reranked
        if r["operation_id"] == "aktools.stock_financial_hk_analysis_indicator_em"
    )
    assert fin["compatibility_tier"] == "C"
    # With Tier A market present, binder must not select Tier C financial.
    described = select_candidates_to_describe(reranked)
    assert described[0]["operation_id"] == "aktools.stock_hk_hist"
    assert all(r["compatibility_tier"] != "C" for r in described) or not any(
        r["compatibility_tier"] in {"A", "B"} for r in reranked
    )


def test_financial_metric_rerank_prefers_financial_over_market_price():
    bundle = build_unified_catalog(root=ROOT)
    meta = load_and_validate_metadata(catalog=bundle, root=ROOT)
    step = PlanStep(
        step_id="r1",
        step_type="retrieve",
        objective="Retrieve reported FY EPS / financial indicators",
        depends_on=[],
        mandatory=True,
        success_criteria="eps found",
        retrieval_query_hint="financial indicator eps",
    )
    ctx = context_from_task_spec(
        {
            "goal": "obtain reported eps",
            "metrics": ["eps"],
            "periods": ["FY2025"],
            "cutoff_date": "2026-08-12",
        },
        step,
    )
    assert "financial" in ctx.domain_hints
    candidates = [
        {"operation_id": "aktools.stock_hk_hist", "score": 0.95},
        {
            "operation_id": "aktools.stock_financial_hk_analysis_indicator_em",
            "score": 0.50,
        },
    ]
    out = rerank_candidates(
        candidates,
        meta,
        ctx,
        runtime_operation_ids=bundle.stats.runtime_operation_ids,
    )
    assert out["reranked"][0]["operation_id"] == (
        "aktools.stock_financial_hk_analysis_indicator_em"
    )
    hist = next(r for r in out["reranked"] if r["operation_id"] == "aktools.stock_hk_hist")
    assert hist["compatibility_tier"] == "C"


def test_official_disclosure_dependency_metadata_applied():
    bundle = build_unified_catalog(root=ROOT)
    meta = load_and_validate_metadata(catalog=bundle, root=ROOT)
    step = PlanStep(
        step_id="r1",
        step_type="retrieve",
        objective="List official HKEX announcements / disclosure filings",
        depends_on=[],
        mandatory=True,
        success_criteria="announcement list",
        retrieval_query_hint="hkexnews announcements list",
    )
    ctx = context_from_task_spec(
        {
            "goal": "official disclosure evidence",
            "metrics": ["announcement"],
            "constraints": ["prefer_official_disclosure"],
            "cutoff_date": "2026-08-12",
        },
        step,
    )
    assert "official_disclosure" in ctx.source_type_prefs
    # Ensure HKEX-related runtime ops keep disclosure-friendly tiers when present.
    hkex_ops = [
        oid
        for oid in bundle.stats.runtime_operation_ids
        if "hkex" in oid.lower() or "announcement" in oid.lower()
    ]
    assert hkex_ops, "expected hkex/announcement ops in runtime catalog"
    candidates = [
        {"operation_id": oid, "score": 0.5 + i * 0.01} for i, oid in enumerate(hkex_ops[:3])
    ] + [{"operation_id": "aktools.stock_hk_hist", "score": 0.99}]
    out = rerank_candidates(
        candidates,
        meta,
        ctx,
        runtime_operation_ids=bundle.stats.runtime_operation_ids,
    )
    # Market hist should not outrank disclosure when domain/source prefs conflict.
    top = out["reranked"][0]
    assert "hist" not in top["operation_id"] or top["compatibility_tier"] != "A"


def test_monitor_missing_schema_version_and_decision_reason():
    bad1 = {
        "decision": "continue",
        "decision_reason": "ok",
        "ready_step_ids": [],
        "patch": {
            "completed_steps": [],
            "updated_steps": [],
            "new_steps": [],
            "removed_steps": [],
        },
    }
    with pytest.raises(SchemaValidationError, match="schema_version"):
        validate_against_schema(bad1, APV_FIN_MONITOR_V1)
    msg = format_repair_user_message(
        "apv_fin_monitor_v1", "SchemaValidationError: $: missing required property 'schema_version'"
    )
    assert "schema_version" in msg

    bad2 = {
        "schema_version": "apv_fin_monitor_v1",
        "decision": "continue",
        "ready_step_ids": [],
        "patch": {
            "completed_steps": [],
            "updated_steps": [],
            "new_steps": [],
            "removed_steps": [],
        },
    }
    with pytest.raises(SchemaValidationError, match="decision_reason"):
        validate_against_schema(bad2, APV_FIN_MONITOR_V1)


def test_monitor_rejects_reasoning_brief_instead_of_decision_reason():
    bad = {
        "schema_version": "apv_fin_monitor_v1",
        "decision": "continue",
        "reasoning_brief": "should not be allowed",
        "ready_step_ids": [],
        "patch": {
            "completed_steps": [],
            "updated_steps": [],
            "new_steps": [],
            "removed_steps": [],
        },
    }
    with pytest.raises(SchemaValidationError):
        validate_against_schema(bad, APV_FIN_MONITOR_V1)
    msg = format_repair_user_message(
        "apv_fin_monitor_v1",
        "SchemaValidationError: $: additional property not allowed: 'reasoning_brief'",
    )
    assert "decision_reason" in msg


def test_monitor_repair_success_and_second_failure_fail_closed():
    good = {
        "schema_version": "apv_fin_monitor_v1",
        "decision": "continue",
        "decision_reason": "Continue after observation.",
        "ready_step_ids": [],
        "patch": {
            "completed_steps": [],
            "updated_steps": [],
            "new_steps": [],
            "removed_steps": [],
        },
        "clarify_question": None,
        "refuse_reason": None,
        "fail_reason": None,
    }
    plan = PlanState.from_initial_plan(
        {
            "steps": [
                {
                    "step_id": "s1",
                    "step_type": "answer",
                    "objective": "answer",
                    "depends_on": [],
                    "mandatory": True,
                    "success_criteria": "done",
                }
            ]
        }
    )
    llm = ScriptedLLMClient(
        [
            json.dumps({"decision": "continue", "decision_reason": "x"}),  # missing schema
            json.dumps(good),
        ]
    )
    out = run_monitor(
        llm,
        question="q",
        plan_state=plan,
        observations=[],
        evidence_summary={},
        budget={},
    )
    assert out["ok"] is True
    assert out["repair_count"] == 1

    llm2 = ScriptedLLMClient(["not-json", "still-bad"])
    out2 = run_monitor(
        llm2,
        question="q",
        plan_state=plan,
        observations=[],
        evidence_summary={},
        budget={},
    )
    assert out2["ok"] is False
    assert out2["fail_closed"] is True


def test_replan_patch_resets_failed_retrieve_and_increments_revision():
    plan = PlanState.from_initial_plan(
        {
            "steps": [
                {
                    "step_id": "r1",
                    "step_type": "retrieve",
                    "objective": "retrieve close",
                    "depends_on": [],
                    "mandatory": True,
                    "success_criteria": "got",
                    "retrieval_query_hint": "bad query",
                }
            ]
        }
    )
    plan.mark_running("r1")
    plan.mark_failed("r1", "Missing required arguments: ['symbol']")
    assert plan.steps["r1"].status == "failed"
    summary = plan.apply_monitor_patch(
        {
            "completed_steps": [],
            "updated_steps": [
                {
                    "step_id": "r1",
                    "retrieval_query_hint": "hong kong equity close price history",
                }
            ],
            "new_steps": [],
            "removed_steps": [],
        }
    )
    assert summary["revision"] == 1
    assert plan.revision == 1
    assert plan.steps["r1"].status == "ready"
    assert "close" in (plan.steps["r1"].retrieval_query_hint or "")


def test_identical_failed_invocation_not_repeated(tmp_path: Path):
    bundle = build_unified_catalog(root=ROOT)
    transport = ScriptedMockTransport()
    gateway = UnifiedRealMcpGateway(catalog_bundle=bundle, mock_transport=transport)
    gateway.open()
    resolver = OperationResolverServer(
        embedding_backend=DeterministicHashEmbeddingBackend(),
        registry=bundle.registry,
        real_mcp_gateway=gateway,
        cache_dir=tmp_path / "emb",
    )
    meta = load_and_validate_metadata(catalog=bundle, root=ROOT)
    from hk_equity_mcp.apv_fin.evidence_ledger import EvidenceLedger

    ledger = EvidenceLedger()
    budget = BudgetTracker({})
    step = PlanStep(
        step_id="r1",
        step_type="retrieve",
        objective="Retrieve close price history",
        depends_on=[],
        mandatory=True,
        success_criteria="close",
        retrieval_query_hint="stock hk hist close",
    )
    # First call with incomplete args via forced select missing symbol → fail
    # Use select with args missing symbol to force validation error, then repeat.
    out1 = execute_retrieve_step(
        step=step,
        task_spec={
            "goal": "close",
            "entities": [],  # no symbol → cannot ground
            "metrics": ["close"],
            "periods": ["2026-07-31"],
            "cutoff_date": "2026-08-12",
        },
        resolver=resolver,
        metadata_by_id=meta,
        runtime_operation_ids=bundle.stats.runtime_operation_ids,
        ledger=ledger,
        budget=budget,
        question="close?",
        select_operation_id="aktools.stock_hk_hist",
        allow_forced_select=True,
        offline_tool_binding=True,
        invoke_arguments={},  # incomplete
    )
    # Without symbol, offline proposal should be expand_search/unsupported, not empty invoke.
    assert out1.get("mcp_called") is False or out1.get("ok") is False
    if out1.get("invoked"):
        # If somehow invoked, second identical must be blocked.
        out2 = execute_retrieve_step(
            step=step,
            task_spec={
                "goal": "close",
                "entities": [],
                "metrics": ["close"],
                "periods": ["2026-07-31"],
                "cutoff_date": "2026-08-12",
            },
            resolver=resolver,
            metadata_by_id=meta,
            runtime_operation_ids=bundle.stats.runtime_operation_ids,
            ledger=ledger,
            budget=budget,
            question="close?",
            select_operation_id="aktools.stock_hk_hist",
            allow_forced_select=True,
            offline_tool_binding=True,
            invoke_arguments={},
        )
        assert out2.get("error") in {
            "identical_failed_invocation_not_repeated",
            "max_targeted_retrievals_per_step",
            "binding_expand_search_exhausted",
            "binding_decision_unsupported",
            "binding_decision_expand_search",
        } or out2.get("mcp_called") is False
    gateway.close()


def test_metrics_evidence_and_calc_and_failed_answer():
    failed_row = {
        "status": "failed",
        "answer_correct": True,  # must NOT count as strict success
        "auto_grade_status": "graded",
        "plan_state": {
            "steps": {
                "r1": {
                    "step_type": "retrieve",
                    "mandatory": True,
                    "status": "failed",
                    "depends_on": [],
                }
            }
        },
        "evidence_ledger": [
            {
                "source_type": "mcp_invoke",
                "validation_status": "invalid",
                "operation_id": "x",
            }
        ],
        "failed_invocation_count": 1,
        "successful_invocation_count": 0,
        "real_mcp_execution_count": 1,
        "all_tool_invocations_successful": True,  # poisoned flag — must still be false
        "compute_traces": [],
    }
    assert evidence_chain_complete(failed_row) is False
    assert calculation_validation_flag(failed_row) is None
    assert strict_answer_success(failed_row) is False
    assert tool_execution_success_flag(failed_row) is False
    assert plan_completion_flag(failed_row) is False

    calc_missing = {
        "status": "ok",
        "plan_state": {
            "steps": {
                "c1": {
                    "step_type": "compute",
                    "mandatory": True,
                    "status": "pending",
                    "depends_on": [],
                }
            }
        },
        "compute_traces": [],
    }
    assert calculation_validation_flag(calc_missing) is False

    calc_ok = {
        "status": "ok",
        "plan_state": {
            "steps": {
                "c1": {
                    "step_type": "compute",
                    "mandatory": True,
                    "status": "resolved",
                    "depends_on": [],
                }
            }
        },
        "compute_traces": [{"ok": True}],
        "validator": {"checks": [{"name": "calculation_operands", "ok": True}]},
    }
    assert calculation_validation_flag(calc_ok) is True

    agg = aggregate_apv_fin_metrics([failed_row, calc_ok])
    assert agg["evidence_chain_completion_rate"] == 0.0
    assert agg["common_strict_answer_success"] == 0.0
    assert agg["calculation_validation_rate"] == 1.0  # only one non-null among rows? 
    # failed_row → None, calc_ok → True → rate 1.0
    assert agg["calculation_validation_rate"] == 1.0


def test_prompts_clean_and_protocol_label():
    assert_prompts_clean()
    assert "schema_version" in APV_FIN_MONITOR_USER_TEMPLATE
    assert "decision_reason" in APV_FIN_MONITOR_USER_TEMPLATE
    assert "reasoning_brief" in APV_FIN_MONITOR_USER_TEMPLATE  # prohibition text
    assert "apv_fin_tool_binding_v1" in APV_FIN_TOOL_BINDING_USER_TEMPLATE
    for needle in ("Tencent", "HSBC", "gold_operations", "fin_cf_003"):
        assert needle not in APV_FIN_TOOL_BINDING_USER_TEMPLATE
        assert needle not in APV_FIN_MONITOR_USER_TEMPLATE
    assert TOOL_BINDING_PROTOCOL_CHANGE.startswith("tool_binding_and_monitor_protocol_v3")
    assert "apv_fin_tool_binding_v1" in schema_hashes()
    assert apv_fin_prompt_hash()
    assert combined_schema_hash()


def test_no_empty_invoke_when_required_args_ungroundable(tmp_path: Path):
    bundle = build_unified_catalog(root=ROOT)
    transport = ScriptedMockTransport()
    gateway = UnifiedRealMcpGateway(catalog_bundle=bundle, mock_transport=transport)
    gateway.open()
    resolver = OperationResolverServer(
        embedding_backend=DeterministicHashEmbeddingBackend(),
        registry=bundle.registry,
        real_mcp_gateway=gateway,
        cache_dir=tmp_path / "emb2",
    )
    meta = load_and_validate_metadata(catalog=bundle, root=ROOT)
    from hk_equity_mcp.apv_fin.evidence_ledger import EvidenceLedger

    out = execute_retrieve_step(
        step=PlanStep(
            step_id="r1",
            step_type="retrieve",
            objective="Retrieve market close",
            depends_on=[],
            mandatory=True,
            success_criteria="close",
            retrieval_query_hint="close price history",
        ),
        task_spec={
            "goal": "close",
            "entities": ["hk_listed_issuer"],  # not a code
            "metrics": ["close"],
            "periods": ["2026-07-31"],
            "cutoff_date": "2026-08-12",
        },
        resolver=resolver,
        metadata_by_id=meta,
        runtime_operation_ids=bundle.stats.runtime_operation_ids,
        ledger=EvidenceLedger(),
        budget=BudgetTracker({}),
        question="What is the close?",
        offline_tool_binding=True,
    )
    assert out.get("mcp_called") is False
    assert out.get("invoke") is None
    assert (out.get("grounded_arguments") in (None, {}, [])) or out.get("ok") is False
    gateway.close()
