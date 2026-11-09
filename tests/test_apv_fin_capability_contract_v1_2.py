"""APV-Fin v1.2 capability-contract controller — offline unit + v5 replay fixtures."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hk_equity_mcp.apv_fin import APV_FIN_METHOD_VERSION
from hk_equity_mcp.apv_fin.argument_slots import can_deterministic_bind, map_slots_to_arguments
from hk_equity_mcp.apv_fin.capability_contract import (
    load_and_validate_capability_contracts,
    operations_for_capability,
    validate_capability_contracts,
)
from hk_equity_mcp.apv_fin.capability_filter import filter_by_capability_contract
from hk_equity_mcp.apv_fin.capability_ontology import CAPABILITY_IDS
from hk_equity_mcp.apv_fin.dependency_expansion import (
    build_followup_retrieve_step,
    extract_provided_values,
    next_capability_operations,
    next_disclosure_capability,
)
from hk_equity_mcp.apv_fin.evidence_ledger import EvidenceLedger
from hk_equity_mcp.apv_fin.output_adapters import (
    adapt_disclosure_list,
    adapt_financial_reported_metric_bundle,
    adapt_market_daily_ohlcv,
    adapt_mcp_result,
    required_outputs_present,
)
from hk_equity_mcp.apv_fin.plan_state import PlanState, PlanStateError
from hk_equity_mcp.apv_fin.progressor import apply_deterministic_progress, classify_after_step
from hk_equity_mcp.apv_fin.scheduler import BudgetTracker, execute_compute_step
from hk_equity_mcp.experiment.unified_apv_fin_metrics import (
    aggregate_apv_fin_metrics,
    tool_execution_success_flag,
)
from hk_equity_mcp.experiment.unified_apv_fin_runner import load_apv_fin_config

ROOT = Path(__file__).resolve().parents[1]
V5_PRED = ROOT / "outputs" / "apv_fin_real_smoke4_v5" / "predictions.jsonl"


def test_method_version_v1_2():
    # Suite still covers v1.2 contracts; live method version advanced to v1.4.
    assert APV_FIN_METHOD_VERSION == "apv_fin_v1_4_gateway_adapter_and_progression"
    cfg = load_apv_fin_config()
    assert cfg["method_version"] == APV_FIN_METHOD_VERSION


def test_capability_ontology_coverage():
    report = validate_capability_contracts()
    assert report["ok"] is True
    contracts = load_and_validate_capability_contracts()
    assert len(contracts) >= 39
    for cap in CAPABILITY_IDS:
        assert operations_for_capability(contracts, cap), f"no provider for {cap}"


def test_planner_retrieve_requires_capability():
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


def test_capability_filter_excludes_financial_for_market_ohlcv():
    contracts = load_and_validate_capability_contracts()
    candidates = [
        {
            "operation_id": "aktools.stock_financial_hk_analysis_indicator_em",
            "base_score": 0.9,
            "compatibility_tier": "C",
            "final_rank": 1,
        },
        {
            "operation_id": "aktools.stock_hk_hist",
            "base_score": 0.4,
            "compatibility_tier": "C",
            "final_rank": 5,
        },
        {
            "operation_id": "aktools.stock_hk_daily",
            "base_score": 0.39,
            "compatibility_tier": "C",
            "final_rank": 7,
        },
    ]
    out = filter_by_capability_contract(
        candidates=candidates,
        contracts=contracts,
        required_capability="market.daily_ohlcv",
        required_outputs=["trade_date", "close"],
        runtime_operation_ids=[c["operation_id"] for c in candidates],
    )
    ids = {r["operation_id"] for r in out["binder_candidates"]}
    assert "aktools.stock_hk_hist" in ids or "aktools.stock_hk_daily" in ids
    assert "aktools.stock_financial_hk_analysis_indicator_em" not in ids
    assert all(r.get("compatibility_tier") in {"A", "B"} for r in out["binder_candidates"])


def test_capability_candidate_injection_when_embedding_misses():
    contracts = load_and_validate_capability_contracts()
    # Embedding returned only financial ops.
    candidates = [
        {
            "operation_id": "aktools.stock_financial_hk_analysis_indicator_em",
            "base_score": 0.9,
            "compatibility_tier": "C",
            "final_rank": 1,
        }
    ]
    runtime = list(contracts.keys())
    out = filter_by_capability_contract(
        candidates=candidates,
        contracts=contracts,
        required_capability="market.daily_ohlcv",
        required_outputs=["close"],
        runtime_operation_ids=runtime,
    )
    assert out["capability_injected_candidate_count"] >= 1
    injected = [
        r for r in out["compatible_candidates"] if r.get("capability_injected_candidate")
    ]
    assert any(
        "hist" in r["operation_id"] or "daily" in r["operation_id"] for r in injected
    )


def test_deterministic_argument_binding():
    schema = {
        "type": "object",
        "required": ["symbol", "start_date", "end_date"],
        "properties": {
            "symbol": {"type": "string"},
            "start_date": {"type": "string"},
            "end_date": {"type": "string"},
            "adjust": {"type": "string", "enum": ["", "qfq", "hfq"]},
        },
    }
    slots = {
        "ticker": "00700",
        "start_date": "2026-08-12",
        "end_date": "2026-08-12",
        "adjust": "",
    }
    ok, mapped = can_deterministic_bind(
        argument_slots=slots, json_schema=schema, candidate_count=1
    )
    assert ok is True
    assert mapped["arguments"]["symbol"] == "00700"
    assert mapped["arguments"]["adjust"] == ""
    multi_ok, _ = can_deterministic_bind(
        argument_slots=slots, json_schema=schema, candidate_count=2
    )
    assert multi_ok is False


def test_output_adapters_financial_market_disclosure():
    fin = adapt_financial_reported_metric_bundle(
        [
            {
                "OPERATE_INCOME": 100.0,
                "HOLDER_PROFIT": 10.0,
                "BASIC_EPS": 1.2,
                "CURRENCY": "CNY",
                "REPORT_DATE": "2025-03-31",
            }
        ]
    )
    assert fin["metrics"]["revenue"] == 100.0
    assert fin["metrics"]["attributable_profit"] == 10.0
    assert fin["raw_field_map"]["revenue"] == "OPERATE_INCOME"
    assert required_outputs_present(fin, ["revenue", "attributable_profit"])["ok"]

    mkt = adapt_market_daily_ohlcv(
        [{"日期": "2026-08-12", "收盘": 320.5, "开盘": 318.0}],
        ticker="00700",
    )
    assert mkt["close"] == 320.5
    assert mkt["trade_date"] == "2026-08-12"

    disc = adapt_disclosure_list(
        [
            {
                "title": "Annual Results",
                "release_date": "2026-01-01",
                "news_id": "n1",
                "local_key": "lk1",
                "file_link": "http://x",
            }
        ]
    )
    assert disc["local_key"] == "lk1"
    assert disc["documents"][0]["title"] == "Annual Results"


def test_required_output_completion_blocks_partial():
    adapted = {"metrics": {"revenue": 1.0}, "capability": "financial.reported_metric_bundle"}
    check = required_outputs_present(adapted, ["revenue", "attributable_profit"])
    assert check["ok"] is False
    assert "attributable_profit" in check["missing"]


def test_disclosure_dependency_chain_synthetic():
    contracts = load_and_validate_capability_contracts()
    state = PlanState.from_initial_plan(
        {
            "steps": [
                {
                    "step_id": "s1",
                    "step_type": "retrieve",
                    "objective": "official narrative evidence",
                    "depends_on": [],
                    "mandatory": True,
                    "success_criteria": "snippet",
                    "required_capability": "financial.official_narrative_evidence",
                    "required_outputs": ["snippet", "local_key"],
                    "argument_slots": {"ticker": "00700"},
                    "compute_spec": None,
                },
                {
                    "step_id": "s2",
                    "step_type": "answer",
                    "objective": "answer",
                    "depends_on": ["s1"],
                    "mandatory": True,
                    "success_criteria": "done",
                    "compute_spec": None,
                },
            ]
        }
    )
    step = state.steps["s1"]
    state.mark_running("s1")
    adapted = adapt_disclosure_list(
        [{"title": "t", "local_key": "doc_abc", "news_id": "1", "release_date": "2026-01-01"}]
    )
    cls = classify_after_step(
        step=step,
        execution_ok=True,
        projection={"operation_id": "hkexnews.list_hkexnews_announcements", "adapted": adapted},
        adapted=adapted,
    )
    assert cls["action"] == "expand_dependency"
    progress = apply_deterministic_progress(
        plan_state=state,
        step=step,
        classification=cls,
        observation_ids=["e1"],
        projection={"adapted": adapted, "operation_id": "hkexnews.list_hkexnews_announcements"},
        contracts=contracts,
        already_invoked=["hkexnews.list_hkexnews_announcements"],
    )
    assert progress["dependency_auto_expansion_count"] == 1
    assert progress["new_steps"]
    follow_id = progress["new_steps"][0]
    follow = state.steps[follow_id]
    assert follow.required_capability in {
        "disclosure.download",
        "disclosure.parse",
        "disclosure.keyword_search",
        "disclosure.page_read",
    }
    assert follow.argument_slots.get("local_key") == "doc_abc"

    # Continue chain until keyword/page evidence.
    nxt = next_disclosure_capability(
        current_capability=follow.required_capability,
        target_capability="financial.official_narrative_evidence",
    )
    assert nxt is not None


def test_compute_operand_binding_generic_margin_fixture():
    ledger = EvidenceLedger()
    adapted = adapt_financial_reported_metric_bundle(
        [{"OPERATE_INCOME": 200.0, "HOLDER_PROFIT": 25.0, "CURRENCY": "CNY"}]
    )
    ledger.add(
        step_id="s1",
        claim="bundle",
        source_type="mcp_invoke",
        raw_value={},
        normalized_value=adapted,
        validation_status="valid",
        adapted=adapted,
    )
    state = PlanState.from_initial_plan(
        {
            "steps": [
                {
                    "step_id": "s1",
                    "step_type": "retrieve",
                    "objective": "financials",
                    "depends_on": [],
                    "mandatory": True,
                    "success_criteria": "ok",
                    "required_capability": "financial.reported_metric_bundle",
                    "required_outputs": ["revenue", "attributable_profit"],
                    "argument_slots": {"ticker": "09988"},
                    "compute_spec": None,
                },
                {
                    "step_id": "s2",
                    "step_type": "compute",
                    "objective": "margin",
                    "depends_on": ["s1"],
                    "mandatory": True,
                    "success_criteria": "ok",
                    "compute_spec": {
                        "expression": "(net_income / revenue) * 100",
                        "operand_refs": {
                            "net_income": "s1.metrics.attributable_profit",
                            "revenue": "s1.metrics.revenue",
                        },
                        "output_name": "margin",
                        "output_unit": "percent",
                        "rounding": 2,
                    },
                },
            ]
        }
    )
    state.mark_resolved("s1")
    state.mark_running("s2")
    out = execute_compute_step(step=state.steps["s2"], ledger=ledger)
    assert out["ok"] is True
    assert out["value"] == 12.5


def test_cross_domain_parallel_readiness():
    state = PlanState.from_initial_plan(
        {
            "steps": [
                {
                    "step_id": "s1_eps",
                    "step_type": "retrieve",
                    "objective": "EPS",
                    "depends_on": [],
                    "mandatory": True,
                    "success_criteria": "eps",
                    "required_capability": "financial.reported_metric",
                    "required_outputs": ["basic_eps"],
                    "argument_slots": {"ticker": "00005"},
                    "compute_spec": None,
                },
                {
                    "step_id": "s2_vol",
                    "step_type": "retrieve",
                    "objective": "volume",
                    "depends_on": [],
                    "mandatory": True,
                    "success_criteria": "volume",
                    "required_capability": "market.daily_ohlcv",
                    "required_outputs": ["trade_date", "volume"],
                    "argument_slots": {
                        "ticker": "00005",
                        "start_date": "2026-06-30",
                        "end_date": "2026-06-30",
                    },
                    "compute_spec": None,
                },
                {
                    "step_id": "s3_answer",
                    "step_type": "answer",
                    "objective": "answer",
                    "depends_on": ["s1_eps", "s2_vol"],
                    "mandatory": True,
                    "success_criteria": "done",
                    "compute_spec": None,
                },
            ]
        }
    )
    ready = {s.step_id for s in state.ready_steps()}
    assert ready == {"s1_eps", "s2_vol"}
    # Resolve only one branch — answer must not become ready.
    state.mark_running("s1_eps")
    state.mark_resolved("s1_eps")
    ready2 = {s.step_id for s in state.ready_steps()}
    assert "s2_vol" in ready2
    assert "s3_answer" not in ready2


def test_offline_v5_market_routing_replay():
    """Category fixture A: market capability must select hist/daily, not financial."""
    if not V5_PRED.exists():
        pytest.skip("v5 predictions not present")
    contracts = load_and_validate_capability_contracts()
    # Use stored described candidates pattern from mkt_dev_014 failure mode.
    candidates = [
        {"operation_id": "hkexnews.list_hkexnews_announcements", "base_score": 0.5, "compatibility_tier": "C", "final_rank": 1},
        {"operation_id": "aktools.stock_financial_hk_analysis_indicator_em", "base_score": 0.49, "compatibility_tier": "C", "final_rank": 2},
        {"operation_id": "aktools.stock_hk_fhpx_detail_ths", "base_score": 0.48, "compatibility_tier": "C", "final_rank": 3},
    ]
    out = filter_by_capability_contract(
        candidates=candidates,
        contracts=contracts,
        required_capability="market.daily_ohlcv",
        required_outputs=["trade_date", "close"],
        runtime_operation_ids=list(contracts.keys()),
    )
    binder_ids = {r["operation_id"] for r in out["binder_candidates"]}
    assert binder_ids & {"aktools.stock_hk_hist", "aktools.stock_hk_daily"}
    assert "aktools.stock_financial_hk_analysis_indicator_em" not in binder_ids
    # Args ground from slots.
    schema = {
        "required": ["symbol", "start_date", "end_date"],
        "properties": {
            "symbol": {"type": "string"},
            "start_date": {"type": "string"},
            "end_date": {"type": "string"},
            "adjust": {"type": "string"},
        },
    }
    mapped = map_slots_to_arguments(
        argument_slots={
            "ticker": "00700",
            "start_date": "2026-08-12",
            "end_date": "2026-08-12",
            "adjust": "",
        },
        json_schema=schema,
    )
    assert mapped["ok"] is True


def test_offline_v5_financial_margin_replay_synthetic_raw():
    """Category fixture B: structured financial → compute margin (no gold)."""
    raw = [
        {
            "OPERATE_INCOME": 941168000000.0,
            "HOLDER_PROFIT": 123000000000.0,
            "CURRENCY": "CNY",
            "REPORT_DATE": "2025-03-31",
        }
    ]
    adapted = adapt_mcp_result(
        capability="financial.reported_metric_bundle",
        raw_result=raw,
        contract=load_and_validate_capability_contracts()[
            "aktools.stock_financial_hk_analysis_indicator_em"
        ],
    )
    assert required_outputs_present(adapted, ["revenue", "attributable_profit"])["ok"]
    ledger = EvidenceLedger()
    ledger.add(
        step_id="s1",
        claim="fin",
        source_type="mcp_invoke",
        normalized_value=adapted,
        validation_status="valid",
        adapted=adapted,
    )
    step = PlanState.from_initial_plan(
        {
            "steps": [
                {
                    "step_id": "s2",
                    "step_type": "compute",
                    "objective": "margin",
                    "depends_on": [],
                    "mandatory": True,
                    "success_criteria": "ok",
                    "compute_spec": {
                        "expression": "(ni / rev) * 100",
                        "operand_refs": {
                            "ni": "s1.metrics.attributable_profit",
                            "rev": "s1.metrics.revenue",
                        },
                        "output_name": "margin",
                        "output_unit": "percent",
                        "rounding": 2,
                    },
                }
            ]
        }
    ).steps["s2"]
    # Fix: operand step refs need s1 evidence — use ledger step_id s1
    out = execute_compute_step(step=step, ledger=ledger)
    assert out["ok"] is True
    assert out["value"] == pytest.approx(13.07, abs=0.01)


def test_metrics_consistency_counters_not_null():
    rows = [
        {
            "status": "full",
            "deterministic_step_transitions": 0,
            "exception_replan_calls": 0,
            "raw_observation_chars": 0,
            "projected_observation_chars": 0,
            "all_tool_invocations_successful": None,
            "successful_invocation_count": 0,
            "failed_invocation_count": 0,
            "real_mcp_execution_count": 0,
            "plan_state": {"steps": {}},
            "compute_traces": [],
            "monitor_decisions": [],
        },
        {
            "status": "failed",
            "error_type": "resource_limit_failure",
            "stopping_reason": "resource_limit_failure",
            "deterministic_step_transitions": 1,
            "exception_replan_calls": 2,
            "raw_observation_chars": 100,
            "projected_observation_chars": 40,
            "compression_ratio": 0.4,
            "all_tool_invocations_successful": True,
            "successful_invocation_count": 1,
            "failed_invocation_count": 0,
            "real_mcp_execution_count": 1,
            "plan_state": {"steps": {}},
            "compute_traces": [],
            "monitor_decisions": [{"decision": "replan"}],
        },
    ]
    assert tool_execution_success_flag(rows[0]) is None
    agg = aggregate_apv_fin_metrics(rows)
    assert agg["deterministic_step_transitions_mean"] is not None
    assert agg["raw_observation_chars_mean"] is not None
    assert agg["cost_summary"]["successful_tasks"] == 1
    assert agg["resource_limit_failures"] == 1
    assert agg["monitor_revision_count_mean"] == 0.5
    assert agg["exception_replan_calls_sum"] == 2
