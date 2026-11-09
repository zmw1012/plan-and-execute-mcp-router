"""APV-Fin v1.3 capability requirement compiler — offline unit tests."""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from hk_equity_mcp.apv_fin import APV_FIN_METHOD_VERSION
from hk_equity_mcp.apv_fin.capability_filter import filter_by_capability_contract
from hk_equity_mcp.apv_fin.capability_contract import load_and_validate_capability_contracts
from hk_equity_mcp.apv_fin.capability_ontology import CAPABILITY_IDS
from hk_equity_mcp.apv_fin.capability_requirement_compiler import (
    COMPILER_VERSION,
    CapabilityCompilerError,
    SEMANTIC_OUTPUT_ALIAS_MAP,
    capability_compiler_hash,
    compile_capability_requirement,
    compile_retrieve_step,
    normalize_capability_id,
    output_alias_map_hash,
)
from hk_equity_mcp.apv_fin.identity import CAPABILITY_COMPILER_PROTOCOL_CHANGE
from hk_equity_mcp.apv_fin.schemas import FORMAL_CAPABILITY_ENUM, PLAN_STEP_SCHEMA
from hk_equity_mcp.experiment.unified_apv_fin_runner import load_apv_fin_config

ROOT = Path(__file__).resolve().parents[1]


def test_method_version_and_protocol_v1_3():
    # Compiler suite remains; live method advanced to v1.4 adapter/progression.
    assert APV_FIN_METHOD_VERSION == "apv_fin_v1_4_1_crash_safe_runtime"
    cfg = load_apv_fin_config()
    assert cfg["method_version"] == APV_FIN_METHOD_VERSION
    assert (
        CAPABILITY_COMPILER_PROTOCOL_CHANGE
        == "capability_requirement_compiler_before_valid_method_evaluation"
    )
    assert COMPILER_VERSION
    assert capability_compiler_hash()
    assert output_alias_map_hash()


def test_planner_schema_required_capability_is_formal_enum():
    prop = PLAN_STEP_SCHEMA["properties"]["required_capability"]
    assert "enum" in prop
    for cap in CAPABILITY_IDS:
        assert cap in prop["enum"]
    assert None in prop["enum"]
    assert set(FORMAL_CAPABILITY_ENUM) == set(prop["enum"])


def test_invalid_capability_alias_normalization():
    assert (
        normalize_capability_id("market.equity_eod_price") == "market.latest_close"
    )
    assert (
        normalize_capability_id("market.daily_trading_volume") == "market.daily_ohlcv"
    )
    assert (
        normalize_capability_id("financial.standard_metric_bundle")
        == "financial.reported_metric_bundle"
    )
    assert (
        normalize_capability_id("financial.official_filing_narrative_metric")
        == "financial.official_narrative_evidence"
    )


def test_unknown_capability_fail_closed():
    with pytest.raises(CapabilityCompilerError) as ei:
        normalize_capability_id("market.totally_made_up")
    assert ei.value.error_type == "unknown_capability"


def test_semantic_to_canonical_output_mapping():
    out = compile_capability_requirement(
        capability_hint="market.equity_eod_price",
        required_output_semantics=["closing_price", "trade_date", "currency"],
        step_objective="Get latest close",
    )
    assert out["canonical_required_capability"] == "market.latest_close"
    assert "close" in out["canonical_required_outputs"]
    assert "trade_date" in out["canonical_required_outputs"]
    assert (
        out["semantic_to_canonical_output_map"]["closing_price"] == "close"
    )
    assert "closing_price" in out["semantic_output_requirements"]


def test_latest_close_routing_binder_not_financial_only():
    contracts = load_and_validate_capability_contracts()
    compiled = compile_capability_requirement(
        capability_hint="market.equity_eod_price",
        required_output_semantics=["closing_price", "trade_date"],
    )
    candidates = [
        {
            "operation_id": "aktools.stock_financial_hk_analysis_indicator_em",
            "base_score": 0.99,
            "compatibility_tier": "C",
            "final_rank": 1,
        }
    ]
    out = filter_by_capability_contract(
        candidates=candidates,
        contracts=contracts,
        required_capability=compiled["canonical_required_capability"],
        required_outputs=compiled["canonical_required_outputs"],
        runtime_operation_ids=list(contracts.keys()),
    )
    ids = {r["operation_id"] for r in out["binder_candidates"]}
    assert "aktools.stock_hk_hist" in ids or "aktools.stock_hk_daily" in ids
    assert "aktools.stock_financial_hk_analysis_indicator_em" not in ids
    assert out["capability_injected_candidate_count"] >= 1


def test_daily_volume_routing():
    out = compile_capability_requirement(
        capability_hint="market.daily_trading_volume",
        required_output_semantics=[
            "trading_volume",
            "volume_unit",
            "trade_date",
            "adjustment",
        ],
    )
    assert out["canonical_required_capability"] == "market.daily_ohlcv"
    assert "volume" in out["canonical_required_outputs"]
    assert "trade_date" in out["canonical_required_outputs"]
    assert out["semantic_to_canonical_output_map"]["trading_volume"] == "volume"


def test_financial_margin_operand_binding():
    out = compile_capability_requirement(
        capability_hint="financial.reported_metric_bundle",
        required_output_semantics=[
            "revenue",
            "net_income_attributable_to_ordinary_shareholders",
        ],
        metric_semantics=["revenue", "net_income_attributable_to_ordinary_shareholders"],
        basis_adjustment_constraints=["reported"],
    )
    assert out["canonical_required_capability"] == "financial.reported_metric_bundle"
    assert "attributable_profit" in out["canonical_required_outputs"]
    assert "revenue" in out["canonical_required_outputs"]
    assert (
        out["semantic_to_canonical_output_map"][
            "net_income_attributable_to_ordinary_shareholders"
        ]
        == "attributable_profit"
    )


def test_non_ifrs_narrative_routing():
    out = compile_capability_requirement(
        capability_hint="financial.reported_metric_bundle",
        required_output_semantics=[
            "non_ifrs_profit_attributable_to_equity_holders",
            "currency",
            "unit",
            "period",
        ],
        metric_semantics=["non_IFRS_profit_attributable_to_equity_holders"],
        basis_adjustment_constraints=["non-IFRS"],
        step_objective="Retrieve non-IFRS profit attributable to equity holders",
    )
    assert out["canonical_required_capability"] == "financial.official_narrative_evidence"
    assert out["narrative_routing"] is True
    # Must not force structured bundle exact semantic field.
    assert "non_ifrs_profit_attributable_to_equity_holders" not in out[
        "canonical_required_outputs"
    ]
    assert "snippet" in out["canonical_required_outputs"]
    assert "local_key" in out["canonical_required_outputs"]


def test_capability_injection_not_gold_based():
    src = inspect.getsource(filter_by_capability_contract)
    assert "gold" not in src.lower()
    contracts = load_and_validate_capability_contracts()
    out = filter_by_capability_contract(
        candidates=[],
        contracts=contracts,
        required_capability="market.latest_close",
        required_outputs=["trade_date", "close"],
        runtime_operation_ids=list(contracts.keys()),
    )
    assert out["capability_injected_candidate_count"] >= 1


def test_no_task_id_or_company_branching_in_compiler():
    src = Path(
        ROOT
        / "src/hk_equity_mcp/apv_fin/capability_requirement_compiler.py"
    ).read_text(encoding="utf-8")
    for banned in (
        "fin_cf_",
        "mkt_dev_",
        "xd_dev_",
        "Tencent",
        "HSBC",
        "gold_operations",
    ):
        assert banned not in src
    # No behavioral branching keys.
    assert "if task_id" not in src
    assert "elif task_id" not in src
    assert "company_whitelist" not in src


def test_compile_retrieve_step_preserves_semantic_provenance():
    result = compile_retrieve_step(
        {
            "step_id": "s1",
            "step_type": "retrieve",
            "objective": "latest close",
            "depends_on": [],
            "mandatory": True,
            "success_criteria": "close",
            "required_capability": "market.equity_eod_price",
            "required_outputs": ["closing_price", "trade_date"],
            "argument_slots": {"symbol": "00700"},
            "source_preference": [],
            "fallback_capabilities": [],
            "compute_spec": None,
        }
    )
    assert result["compilation"]["ok"] is True
    step = result["step"]
    assert step["required_capability"] == "market.latest_close"
    assert step["required_outputs"] == ["close", "trade_date"] or set(
        step["required_outputs"]
    ) >= {"close", "trade_date"}
    assert "closing_price" in step["semantic_output_requirements"]
    assert step["semantic_to_canonical_output_map"]["closing_price"] == "close"


def test_alias_table_covers_required_mappings():
    required = {
        "closing_price": "close",
        "latest_close": "close",
        "trading_volume": "volume",
        "net_income_attributable_to_ordinary_shareholders": "attributable_profit",
        "profit_attributable_to_equity_holders": "attributable_profit",
        "reporting_period": "report_date",
        "period": "report_date",
        "eps_currency": "currency",
        "reporting_currency": "currency",
    }
    for k, v in required.items():
        assert SEMANTIC_OUTPUT_ALIAS_MAP[k] == v
