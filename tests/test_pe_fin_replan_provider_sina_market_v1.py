"""Offline tests: RePlan Sina market provider availability overlay."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from hk_equity_mcp.apv_fin.metadata import load_and_validate_metadata
from hk_equity_mcp.experiment.unified_pe_fin_lite_runner import (
    UnifiedHkPeFinLiteRunner,
    assert_output_dir_safe,
)
from hk_equity_mcp.pe_fin_lite.envelope_projection import project_invoke_to_required_outputs
from hk_equity_mcp.pe_fin_lite.retrieval_formula import retrieval_formula_hash
from hk_equity_mcp.pe_fin_replan import PE_FIN_REPLAN_METHOD_VERSION
from hk_equity_mcp.pe_fin_replan.provider_availability import (
    ProviderAvailabilityError,
    apply_provider_availability_filter,
    assert_tool_readiness_for_live,
    load_provider_availability_profile,
    profile_hash,
    validate_bound_operation,
)
from hk_equity_mcp.pe_fin_replan.tool_readiness import (
    EXPECTED_RUNTIME_OPERATION_COUNT,
    run_contract_audit,
)
from hk_equity_mcp.real_mcp.unified_catalog import build_unified_catalog

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "configs/operations/hk_runtime_provider_availability_sina_market_v1.yaml"
EXPECTED_EM_MARKET = {
    "aktools.stock_hk_famous_spot_em",
    "aktools.stock_hk_hist",
    "aktools.stock_hk_hot_rank_em",
    "aktools.stock_hk_index_daily_em",
}
EXPECTED_FIN_EM = {
    "aktools.stock_financial_hk_analysis_indicator_em",
    "aktools.stock_financial_hk_report_em",
    "aktools.stock_hk_company_profile_em",
    "aktools.stock_hk_dividend_payout_em",
    "aktools.stock_hk_financial_indicator_em",
    "aktools.stock_hk_growth_comparison_em",
    "aktools.stock_hk_scale_comparison_em",
    "aktools.stock_hk_security_profile_em",
    "aktools.stock_hk_valuation_comparison_em",
}
R5_HASH_PREFIX = "74147518"


@pytest.fixture(scope="module")
def catalog_bundle():
    return build_unified_catalog(root=ROOT)


@pytest.fixture(scope="module")
def metadata(catalog_bundle):
    return load_and_validate_metadata(
        ROOT / "configs/operations/hk_unified_operation_metadata_v1.yaml",
        catalog=catalog_bundle,
        root=ROOT,
    )


@pytest.fixture(scope="module")
def profile():
    return load_provider_availability_profile(PROFILE, condition="pe_fin_replan")


@pytest.fixture(scope="module")
def filter_trace(catalog_bundle, metadata, profile):
    return apply_provider_availability_filter(
        operation_ids=list(catalog_bundle.stats.runtime_operation_ids),
        metadata_by_id=metadata,
        profile=profile,
        raw_ops=dict(catalog_bundle.raw_ops),
        records=dict(catalog_bundle.records),
    )


def test_profile_stable_sha256(profile):
    assert profile["profile_id"] == "hk_runtime_sina_market_v1"
    h = profile_hash(profile)
    assert len(h) == 64
    assert h == hashlib.sha256(PROFILE.read_bytes()).hexdigest()


def test_eastmoney_market_disabled_financial_kept(filter_trace):
    disabled = set(filter_trace["disabled_operation_ids"])
    assert disabled == EXPECTED_EM_MARKET
    for oid in EXPECTED_FIN_EM:
        assert oid not in disabled
    after = set(filter_trace["operations_after_provider_filter"])
    assert "aktools.stock_hk_daily" in after
    assert "aktools.stock_hk_index_daily_sina" in after
    assert "aktools.stock_hk_hist" not in after


def test_em_suffix_not_used_for_disable(catalog_bundle, metadata, profile):
    # Financial _em ops must remain even though name contains _em.
    filt = apply_provider_availability_filter(
        operation_ids=["aktools.stock_financial_hk_report_em", "aktools.stock_hk_hist"],
        metadata_by_id=metadata,
        profile=profile,
        raw_ops=dict(catalog_bundle.raw_ops),
        records=dict(catalog_bundle.records),
    )
    assert "aktools.stock_hk_hist" in filt["disabled_operation_ids"]
    assert "aktools.stock_financial_hk_report_em" not in filt["disabled_operation_ids"]


def test_single_equity_cannot_bind_index_tool(profile, filter_trace):
    step = {
        "objective": "Get Tencent 00700 unadjusted daily OHLCV",
        "required_outputs": ["open", "high", "low", "close"],
        "scope_constraints": {"entity": {"code": "00700"}, "adjustment": ""},
    }
    with pytest.raises(ProviderAvailabilityError, match="index_tool_for_single_equity"):
        validate_bound_operation(
            operation_id="aktools.stock_hk_index_daily_sina",
            arguments={"symbol": "HSI"},
            step=step,
            profile=profile,
            filter_trace=filter_trace,
            question="腾讯控股 00700 无复权收盘价",
        )


def test_index_task_can_bind_sina_index(profile, filter_trace, catalog_bundle):
    step = {
        "objective": "Get Hang Seng Index daily OHLC",
        "required_outputs": ["open", "high", "low", "close"],
        "scope_constraints": {"entity": "HSI"},
    }
    obs = validate_bound_operation(
        operation_id="aktools.stock_hk_index_daily_sina",
        arguments={"symbol": "HSI"},
        step=step,
        profile=profile,
        filter_trace=filter_trace,
        question="恒生指数历史行情",
        raw_ops=dict(catalog_bundle.raw_ops),
    )
    assert obs["provider_capability_match"] is True
    assert obs["provider_substitution"] is True


def test_unadjusted_allows_stock_hk_daily(profile, filter_trace, catalog_bundle):
    step = {
        "objective": "unadjusted daily OHLCV",
        "scope_constraints": {"adjustment": "", "entity": {"code": "09988"}},
    }
    obs = validate_bound_operation(
        operation_id="aktools.stock_hk_daily",
        arguments={"symbol": "09988.HK", "adjust": ""},
        step=step,
        profile=profile,
        filter_trace=filter_trace,
        question="阿里巴巴 09988 无复权",
        raw_ops=dict(catalog_bundle.raw_ops),
    )
    assert obs["requested_adjustment"] == ""
    assert obs["provider_capability_match"] is True


def test_qfq_cannot_use_stock_hk_daily(profile, filter_trace):
    step = {
        "objective": "qfq adjusted close",
        "scope_constraints": {"adjustment": "qfq", "entity": {"code": "00700"}},
    }
    with pytest.raises(ProviderAvailabilityError, match="unsupported_provider_capability"):
        validate_bound_operation(
            operation_id="aktools.stock_hk_daily",
            arguments={"symbol": "00700.HK", "adjust": "qfq"},
            step=step,
            profile=profile,
            filter_trace=filter_trace,
            question="前复权收盘价",
        )


def test_binder_cannot_select_disabled(profile, filter_trace):
    step = {"objective": "price", "scope_constraints": {"adjustment": ""}}
    with pytest.raises(ProviderAvailabilityError, match="disabled_operation_selected"):
        validate_bound_operation(
            operation_id="aktools.stock_hk_hist",
            arguments={"symbol": "00700", "adjust": ""},
            step=step,
            profile=profile,
            filter_trace=filter_trace,
            question="00700 close",
        )


def test_ohlc_missing_field_not_completed():
    invoke = {
        "success": True,
        "normalized": {
            "success": True,
            "data": {
                "trade_date": "2026-05-15",
                "open": 1.0,
                "high": 2.0,
                "low": 0.5,
                # close missing
            },
        },
    }
    out = project_invoke_to_required_outputs(
        invoke_result=invoke,
        required_outputs=["open", "high", "low", "close"],
        domain="market",
    )
    assert "close" in out["missing_outputs"]
    assert out["missing_outputs"]  # incomplete → cannot mark completed


def test_provider_profile_in_run_resume_identity(tmp_path, profile, filter_trace):
    out = tmp_path / "dry_with_profile"
    runner = UnifiedHkPeFinLiteRunner(
        output_dir=out,
        split="exploratory_all",
        replan_smoke5=True,
        replan_enabled=True,
        dry_run=True,
        allow_dirty=True,
        root=ROOT,
        market_provider_profile=str(PROFILE),
    )
    meta = runner.run()
    assert meta["market_provider_profile_id"] == "hk_runtime_sina_market_v1"
    assert meta["market_provider_profile_hash"] == profile_hash(profile)
    assert set(meta["disabled_operation_ids"]) == EXPECTED_EM_MARKET
    assert meta["preferred_single_equity_operation"] == "aktools.stock_hk_daily"
    assert meta["preferred_index_operation"] == "aktools.stock_hk_index_daily_sina"
    assert meta["method_version"] == PE_FIN_REPLAN_METHOD_VERSION


def test_old_output_dir_resume_refused_without_profile(tmp_path, profile):
    out = tmp_path / "old"
    out.mkdir()
    existing = {
        "condition": "pe_fin_replan",
        "method_version": PE_FIN_REPLAN_METHOD_VERSION,
        "protocol_version": "planner_executor_replanner_contract_v4",
        "architecture": "plan_execute_replan",
        "subset_kind": "replan_smoke5",
        "mock": False,
        "dry_run": True,
        "market_provider_profile_id": None,
        "market_provider_profile_hash": None,
    }
    (out / "run_metadata.json").write_text(json.dumps(existing), encoding="utf-8")
    (out / "predictions.jsonl").write_text("", encoding="utf-8")
    current = dict(existing)
    current["market_provider_profile_id"] = "hk_runtime_sina_market_v1"
    current["market_provider_profile_hash"] = profile_hash(profile)
    with pytest.raises(RuntimeError, match="identity mismatch"):
        assert_output_dir_safe(
            output_dir=out,
            dry_run=True,
            mock=False,
            resume=True,
            run_meta=current,
        )


def test_r5_formula_hash_unchanged():
    h = retrieval_formula_hash()
    assert h.startswith(R5_HASH_PREFIX)
    assert len(h) == 64


def test_contract_audit_39(tmp_path):
    summary = run_contract_audit(
        root=ROOT,
        output_dir=tmp_path / "contract",
        profile_path=PROFILE,
    )
    assert summary["runtime_operation_count"] == EXPECTED_RUNTIME_OPERATION_COUNT
    assert summary["status"] == "pass"
    assert summary["network_called"] is False
    assert summary["live_pass_forged"] is False
    assert set(summary["disabled_operation_ids"]) == EXPECTED_EM_MARKET


def test_mock_dry_run_cannot_forge_live_readiness(tmp_path, profile):
    skipped = assert_tool_readiness_for_live(
        artifact_path=tmp_path / "missing.json",
        expected_profile_hash=profile_hash(profile),
        mock=True,
        dry_run=False,
    )
    assert skipped["status"] == "skipped_offline"
    assert skipped["live_pass_forged"] is False

    with pytest.raises(ProviderAvailabilityError, match="readiness artifact missing"):
        assert_tool_readiness_for_live(
            artifact_path=tmp_path / "missing.json",
            expected_profile_hash=profile_hash(profile),
            mock=False,
            dry_run=False,
        )


def test_readiness_hash_mismatch_fail_closed(tmp_path, profile):
    from hk_equity_mcp.pe_fin_lite.envelope_projection import (
        MARKET_GATEWAY_PROJECTION_VERSION,
        market_gateway_projection_hash,
    )
    from hk_equity_mcp.pe_fin_replan.sina_symbol import (
        SINA_HK_SYMBOL_NORMALIZER_VERSION,
        sina_symbol_normalizer_hash,
    )

    art = tmp_path / "readiness.json"
    art.write_text(
        json.dumps(
            {
                "mode": "market_sina_probe",
                "status": "pass",
                "gateway_success": True,
                "normalization_success": True,
                "projection_success": True,
                "scope_match": True,
                "missing_outputs": [],
                "provider": "sina_via_akshare",
                "network_called": True,
                "live_probe_pass": True,
                "normalized_symbol": "09988",
                "effective_session_date": "2026-05-15",
                "market_provider_profile_hash": "0" * 64,
                "sina_symbol_normalizer_version": SINA_HK_SYMBOL_NORMALIZER_VERSION,
                "sina_symbol_normalizer_hash": sina_symbol_normalizer_hash(),
                "market_gateway_projection_version": MARKET_GATEWAY_PROJECTION_VERSION,
                "market_gateway_projection_hash": market_gateway_projection_hash(),
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ProviderAvailabilityError, match="hash mismatch"):
        assert_tool_readiness_for_live(
            artifact_path=art,
            expected_profile_hash=profile_hash(profile),
            mock=False,
            dry_run=False,
        )


def test_require_tool_readiness_live_without_artifact_fails(tmp_path, profile):
    out = tmp_path / "live_need_ready"
    with pytest.raises(RuntimeError, match="readiness artifact missing"):
        UnifiedHkPeFinLiteRunner(
            output_dir=out,
            split="exploratory_all",
            replan_smoke5=True,
            replan_enabled=True,
            dry_run=False,
            mock=False,
            allow_dirty=True,
            root=ROOT,
            market_provider_profile=str(PROFILE),
            require_tool_readiness=True,
            tool_readiness_artifact=str(tmp_path / "nope.json"),
        ).run()


def test_docs_index_identity_stable_under_overlay(catalog_bundle):
    from hk_equity_mcp.pe_fin_lite.operation_embedding_index import documents_hash

    h1 = documents_hash(catalog_bundle.registry)
    profile = load_provider_availability_profile(PROFILE, condition="pe_fin_replan")
    apply_provider_availability_filter(
        operation_ids=list(catalog_bundle.stats.runtime_operation_ids),
        metadata_by_id={},
        profile=profile,
        raw_ops=dict(catalog_bundle.raw_ops),
        records=dict(catalog_bundle.records),
    )
    h2 = documents_hash(catalog_bundle.registry)
    assert h1 == h2
    assert len(catalog_bundle.stats.runtime_operation_ids) == 39
