"""Offline tests for Sina HK symbol normalizer and readiness v2 identity."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from hk_equity_mcp.pe_fin_replan.provider_availability import (
    ProviderAvailabilityError,
    assert_tool_readiness_for_live,
    load_provider_availability_profile,
    profile_hash,
)
from hk_equity_mcp.pe_fin_replan.sina_symbol import (
    SINA_HK_SYMBOL_NORMALIZER_VERSION,
    SinaHkSymbolError,
    normalize_sina_hk_symbol_v1,
    sina_symbol_normalizer_hash,
)
from hk_equity_mcp.real_mcp.market_gateway import MarketRealMcpGateway
from hk_equity_mcp.real_mcp.types import RealMcpResult

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "configs/operations/hk_runtime_provider_availability_sina_market_v1.yaml"
V1_PROBE = ROOT / "outputs/pe_fin_replan_tool_readiness_v1/market_sina_probe"
V1_READINESS_SHA = "7a81d9536ea60898f771a17569beda9617545b4a8cd424c2a0b4bce5b14877fb"
V1_RAW_SHA = "e8e0cd5c9d7237082b54f59a4d6a5f59c4a062303d4f99a0d0a03438b8f17720"
DOCS_SHA = "021b46626e99da6efa65602b559571f75dfcad17e05702c4eabbd7c8e46315ad"
R5_SHA = "7414751814f6858a375d95da918921e3ee365dfc5b69cdaba0cee1bf9e1ecb3e"
PROFILE_HASH_BEFORE = "ae4c22409e6778a5b06ff15d0cbfe6f2896dd8d60eefc86334865cc4b91b63c4"


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("9988.HK", "09988"),
        ("09988.HK", "09988"),
        ("hk09988", "09988"),
        ("HK09988", "09988"),
        ("9988", "09988"),
        ("09988", "09988"),
    ],
)
def test_normalize_sina_hk_symbol_aliases(raw, expected):
    out = normalize_sina_hk_symbol_v1(raw)
    assert out["normalized_symbol"] == expected
    assert out["original_symbol"] == raw
    assert out["argument_transform_applied"] is True
    assert out["symbol_normalizer_version"] == SINA_HK_SYMBOL_NORMALIZER_VERSION
    assert out["symbol_normalizer_hash"] == sina_symbol_normalizer_hash()


@pytest.mark.parametrize("raw", ["", None, "ABC", "99.88", "hk", "09988.SZ", "123456"])
def test_normalize_sina_hk_symbol_fail_closed(raw):
    with pytest.raises(SinaHkSymbolError, match="invalid_symbol_format"):
        normalize_sina_hk_symbol_v1(raw)


def test_normalizer_only_stock_hk_daily():
    gw = MarketRealMcpGateway.__new__(MarketRealMcpGateway)
    gw.catalog_path = Path("x")
    gw.cutoff = __import__("datetime").date(2026, 8, 12)
    calls = {}

    def fake_get_data(operation_id, raw, execution, arguments):
        calls[operation_id] = dict(arguments)
        return RealMcpResult(
            success=True,
            server_id="aktools",
            operation_id=operation_id,
            arguments=arguments,
            normalized={
                "success": True,
                "data": {"date": "2026-05-15"},
                "provenance": {},
            },
            protocol_success=True,
            execution_success=True,
            real_mcp=True,
        )

    gw._invoke_get_data = fake_get_data  # type: ignore[method-assign]
    gw.registry = {
        "aktools.stock_hk_daily": MagicMock(optional_params=lambda: []),
        "aktools.stock_hk_hist": MagicMock(optional_params=lambda: []),
        "aktools.stock_hk_index_daily_sina": MagicMock(optional_params=lambda: []),
    }
    gw._raw_ops = {
        "aktools.stock_hk_daily": {
            "server_id": "aktools",
            "provider": "sina_via_akshare",
            "execution": {"kind": "aktools_get_data", "interface": "stock_hk_daily"},
        },
        "aktools.stock_hk_hist": {
            "server_id": "aktools",
            "provider": "eastmoney_via_akshare",
            "execution": {
                "kind": "aktools_get_data",
                "interface": "stock_hk_hist",
                "param_keys": ["symbol", "adjust"],
            },
        },
        "aktools.stock_hk_index_daily_sina": {
            "server_id": "aktools",
            "provider": "sina_via_akshare",
            "execution": {
                "kind": "aktools_get_data",
                "interface": "stock_hk_index_daily_sina",
                "param_keys": ["symbol"],
            },
        },
    }

    daily = gw.invoke("aktools.stock_hk_daily", {"symbol": "09988.HK"})
    assert calls["aktools.stock_hk_daily"]["symbol"] == "09988"
    assert daily.normalized["symbol_format_normalized"] == "09988"
    assert daily.normalized["argument_transform_applied"] is True

    hist = gw.invoke(
        "aktools.stock_hk_hist",
        {"symbol": "09988.HK", "adjust": ""},
    )
    assert calls["aktools.stock_hk_hist"]["symbol"] == "09988.HK"
    assert "symbol_format_normalized" not in (hist.normalized or {})

    idx = gw.invoke(
        "aktools.stock_hk_index_daily_sina",
        {"symbol": "HSI"},
    )
    assert calls["aktools.stock_hk_index_daily_sina"]["symbol"] == "HSI"


def test_v1_failed_readiness_not_accepted():
    profile = load_provider_availability_profile(PROFILE, condition="pe_fin_replan")
    art = V1_PROBE / "readiness.json"
    assert art.is_file()
    with pytest.raises(ProviderAvailabilityError, match="invalidated|status"):
        assert_tool_readiness_for_live(
            artifact_path=art,
            expected_profile_hash=profile_hash(profile),
            mock=False,
            dry_run=False,
        )


def test_v1_probe_artifacts_unchanged():
    readiness = V1_PROBE / "readiness.json"
    raw = V1_PROBE / "raw_stock_hk_daily_09988.json"
    assert hashlib.sha256(readiness.read_bytes()).hexdigest() == V1_READINESS_SHA
    assert hashlib.sha256(raw.read_bytes()).hexdigest() == V1_RAW_SHA
    marker = json.loads((V1_PROBE / "INVALIDATED.json").read_text(encoding="utf-8"))
    assert (
        marker["marker"]
        == "invalidated_market_sina_probe_due_to_incorrect_hk_symbol_format"
    )
    data = json.loads(readiness.read_text(encoding="utf-8"))
    assert data["status"] == "fail"
    assert data["arguments_redacted"]["symbol"] == "09988.HK"
    assert data["failure_class"] == "market_provider_unavailable"


def test_readiness_v2_identity_requires_normalizer(tmp_path):
    from hk_equity_mcp.pe_fin_lite.envelope_projection import (
        MARKET_GATEWAY_PROJECTION_VERSION,
        market_gateway_projection_hash,
    )

    profile = load_provider_availability_profile(PROFILE, condition="pe_fin_replan")
    # Profile hash must have changed vs pre-symbol-format revision.
    assert profile_hash(profile) != PROFILE_HASH_BEFORE
    art = tmp_path / "readiness.json"
    payload = {
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
        "market_provider_profile_hash": profile_hash(profile),
        "sina_symbol_normalizer_version": SINA_HK_SYMBOL_NORMALIZER_VERSION,
        "sina_symbol_normalizer_hash": sina_symbol_normalizer_hash(),
        "market_gateway_projection_version": MARKET_GATEWAY_PROJECTION_VERSION,
        "market_gateway_projection_hash": market_gateway_projection_hash(),
    }
    art.write_text(json.dumps(payload), encoding="utf-8")
    meta = assert_tool_readiness_for_live(
        artifact_path=art,
        expected_profile_hash=profile_hash(profile),
        mock=False,
        dry_run=False,
    )
    assert meta["status"] == "pass"
    assert meta["sina_symbol_normalizer_hash"] == sina_symbol_normalizer_hash()

    bad = dict(payload)
    bad["sina_symbol_normalizer_hash"] = "0" * 64
    art.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ProviderAvailabilityError, match="sina_symbol_normalizer_hash"):
        assert_tool_readiness_for_live(
            artifact_path=art,
            expected_profile_hash=profile_hash(profile),
            mock=False,
            dry_run=False,
        )


def test_docs_r5_unchanged():
    from hk_equity_mcp.pe_fin_lite.operation_embedding_index import documents_hash
    from hk_equity_mcp.pe_fin_lite.retrieval_formula import retrieval_formula_hash
    from hk_equity_mcp.real_mcp.unified_catalog import build_unified_catalog

    bundle = build_unified_catalog(root=ROOT)
    assert documents_hash(bundle.registry) == DOCS_SHA
    assert retrieval_formula_hash() == R5_SHA


def test_mock_dry_cannot_forge_live_pass(tmp_path):
    profile = load_provider_availability_profile(PROFILE, condition="pe_fin_replan")
    skipped = assert_tool_readiness_for_live(
        artifact_path=tmp_path / "x.json",
        expected_profile_hash=profile_hash(profile),
        mock=True,
        dry_run=False,
    )
    assert skipped["live_pass_forged"] is False
    assert skipped["status"] == "skipped_offline"


def test_profile_contains_single_equity_symbol_format():
    profile = load_provider_availability_profile(PROFILE, condition="pe_fin_replan")
    pref = profile["preferred_operations"]["single_equity_unadjusted_daily_ohlcv"]
    fmt = pref["single_equity_symbol_format"]
    assert fmt["invoke_format"] == "five_digit_numeric"
    assert fmt["example"] == "09988"
    assert "09988.HK" in fmt["input_aliases"]
