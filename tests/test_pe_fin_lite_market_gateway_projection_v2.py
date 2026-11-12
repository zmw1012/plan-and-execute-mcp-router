"""Offline regression: market gateway normalized.data → required_outputs."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Dict

import pytest

from hk_equity_mcp.pe_fin_lite.envelope_projection import (
    MARKET_GATEWAY_PROJECTION_VERSION,
    market_gateway_projection_hash,
    project_invoke_to_required_outputs,
)
from hk_equity_mcp.pe_fin_lite.retrieval_formula import retrieval_formula_hash
from hk_equity_mcp.pe_fin_lite.operation_embedding_index import documents_hash
from hk_equity_mcp.pe_fin_replan.provider_availability import (
    ProviderAvailabilityError,
    assert_tool_readiness_for_live,
    load_provider_availability_profile,
    profile_hash,
)
from hk_equity_mcp.pe_fin_replan.tool_readiness import (
    classify_readiness_failure,
    evaluate_market_probe_invoke,
)
from hk_equity_mcp.real_mcp.unified_catalog import build_unified_catalog

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = (
    ROOT
    / "tests/fixtures/pe_fin_replan/market_sina_probe_v2_raw_stock_hk_daily_09988.json"
)
V2_RAW = (
    ROOT
    / "outputs/pe_fin_replan_tool_readiness_v1/market_sina_probe_v2/raw_stock_hk_daily_09988.json"
)
V2_READINESS = (
    ROOT
    / "outputs/pe_fin_replan_tool_readiness_v1/market_sina_probe_v2/readiness.json"
)
V2_RAW_SHA = "38f6cba44cd4c0af6992a9f98f7aab4440667216b8496156a63fa623d112033c"
V2_READINESS_SHA = "9598260d0c4b8d9305b78459291df049ee5a74190424fa3ebdf32008ae73a5b3"
DOCS_SHA = "021b46626e99da6efa65602b559571f75dfcad17e05702c4eabbd7c8e46315ad"
R5_SHA = "7414751814f6858a375d95da918921e3ee365dfc5b69cdaba0cee1bf9e1ecb3e"
PROFILE = ROOT / "configs/operations/hk_runtime_provider_availability_sina_market_v1.yaml"

REQUIRED = [
    "trade_date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "adjustment",
    "currency",
    "issuer_code",
]


def _load_fixture() -> Dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _invoke_from_raw(raw: dict, *, shape: str) -> dict:
    norm = copy.deepcopy(raw["normalized"])
    if shape == "outer.normalized.data":
        return {
            "success": True,
            "operation_id": "aktools.stock_hk_daily",
            "normalized": norm,
        }
    if shape == "result.normalized.data":
        return {
            "success": True,
            "operation_id": "aktools.stock_hk_daily",
            "result": {
                "success": True,
                "operation_id": "aktools.stock_hk_daily",
                "normalized": norm,
            },
        }
    if shape == "normalized.data":
        # Payload is already the normalized layer (success wrapper still present).
        return {"success": True, "normalized": norm}
    if shape == "data":
        return {
            "success": True,
            "data": copy.deepcopy(norm["data"]),
            "provider": "sina_via_akshare",
            "source_operation": "aktools.stock_hk_daily",
        }
    raise AssertionError(shape)


def test_frozen_v2_raw_hashes_unchanged():
    assert hashlib.sha256(V2_RAW.read_bytes()).hexdigest() == V2_RAW_SHA
    assert hashlib.sha256(V2_READINESS.read_bytes()).hexdigest() == V2_READINESS_SHA
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == V2_RAW_SHA
    marker = json.loads(
        (
            ROOT
            / "outputs/pe_fin_replan_tool_readiness_v1/market_sina_probe_v2/INVALIDATED.json"
        ).read_text(encoding="utf-8")
    )
    assert (
        marker["marker"]
        == "invalidated_market_sina_probe_v2_due_to_gateway_envelope_projection_contract"
    )


@pytest.mark.parametrize(
    "shape",
    [
        "outer.normalized.data",
        "result.normalized.data",
        "normalized.data",
        "data",
    ],
)
def test_market_gateway_shapes_project(shape):
    raw = _load_fixture()
    invoke = _invoke_from_raw(raw, shape=shape)
    out = project_invoke_to_required_outputs(
        invoke_result=invoke,
        required_outputs=REQUIRED,
        domain="market",
    )
    assert out["missing_outputs"] == []
    assert out["values"]["trade_date"] == "2026-05-15"
    assert out["values"]["open"] == 138.0
    assert out["values"]["high"] == 138.0
    assert out["values"]["low"] == 131.1
    assert out["values"]["close"] == 132.3
    assert out["values"]["volume"] == 90144879.0
    assert out["values"]["adjustment"] == ""
    assert out["values"]["currency"] == "HKD"
    assert out["values"]["issuer_code"] == "09988"
    assert out.get("projection_ok") is True
    assert out.get("normalization_error") in (None, "")


def test_aliases_trade_date_currency_issuer():
    raw = _load_fixture()
    norm = copy.deepcopy(raw["normalized"])
    data = norm["data"]
    # Force alias-only paths
    data.pop("trade_date", None)
    assert "date" in data
    data.pop("currency", None)
    data.pop("issuer_code", None)
    data.pop("adjustment", None)
    out = project_invoke_to_required_outputs(
        invoke_result={"success": True, "normalized": norm},
        required_outputs=REQUIRED,
        domain="market",
    )
    assert out["missing_outputs"] == []
    assert out["values"]["trade_date"] == "2026-05-15"
    assert out["values"]["currency"] == "HKD"  # from currency_or_unit
    assert out["values"]["issuer_code"] == "09988"  # from symbol
    assert out["values"]["adjustment"] == ""  # from adjustment_mode


def test_adjustment_empty_legal_other_empty_missing_none_adjustment_missing():
    raw = _load_fixture()
    # empty open must be missing; empty adjustment legal
    bad = copy.deepcopy(raw["normalized"])
    bad["data"]["open"] = ""
    out = project_invoke_to_required_outputs(
        invoke_result={"success": True, "normalized": bad},
        required_outputs=REQUIRED,
        domain="market",
    )
    assert "open" in out["missing_outputs"]
    assert "adjustment" not in out["missing_outputs"]
    assert out["values"].get("adjustment") == ""

    none_adj = copy.deepcopy(raw["normalized"])
    none_adj["data"]["adjustment"] = None
    none_adj["adjustment_mode"] = None
    out2 = project_invoke_to_required_outputs(
        invoke_result={"success": True, "normalized": none_adj},
        required_outputs=REQUIRED,
        domain="market",
    )
    assert "adjustment" in out2["missing_outputs"]


def test_ohlc_missing_any_field_not_complete():
    raw = _load_fixture()
    for field in ("open", "high", "low", "close"):
        broken = copy.deepcopy(raw["normalized"])
        broken["data"].pop(field, None)
        out = project_invoke_to_required_outputs(
            invoke_result={"success": True, "normalized": broken},
            required_outputs=REQUIRED,
            domain="market",
        )
        assert field in out["missing_outputs"]
        assert out.get("projection_ok") is False


def test_v2_fixture_offline_replay_and_agent_readiness_parity():
    raw = _load_fixture()
    invoke = {
        "success": True,
        "operation_id": "aktools.stock_hk_daily",
        "normalized": raw["normalized"],
        "result": {
            "success": True,
            "operation_id": "aktools.stock_hk_daily",
            "normalized": raw["normalized"],
        },
    }
    agent = project_invoke_to_required_outputs(
        invoke_result=invoke,
        required_outputs=REQUIRED,
        domain="market",
    )
    ready = evaluate_market_probe_invoke(
        invoke_result=invoke,
        required_outputs=REQUIRED,
        target_date="2026-05-15",
    )
    assert agent["missing_outputs"] == []
    assert ready["missing_outputs"] == []
    assert agent["values"] == ready["values"]
    assert ready["projection_success"] is True
    assert ready["scope_match"] is True
    assert ready["effective_session_date"] == "2026-05-15"
    assert ready["failure_class"] is None


def test_wrong_requested_date_scope_mismatch():
    raw = _load_fixture()
    invoke = {
        "success": True,
        "normalized": raw["normalized"],
    }
    ready = evaluate_market_probe_invoke(
        invoke_result=invoke,
        required_outputs=REQUIRED,
        target_date="2026-05-16",
    )
    assert ready["projection_success"] is True
    assert ready["scope_match"] is False
    assert ready["failure_class"] == "scope_mismatch"


def test_gateway_success_malformed_shape_projection_failure():
    invoke = {
        "success": True,
        "normalized": {"success": True, "provider": "sina_via_akshare", "data": "oops"},
    }
    ready = evaluate_market_probe_invoke(
        invoke_result=invoke,
        required_outputs=REQUIRED,
        target_date="2026-05-15",
    )
    assert ready["gateway_success"] is True
    assert ready["projection_success"] is False
    assert ready["failure_class"] == "gateway_envelope_projection_failure"
    assert (
        classify_readiness_failure(
            gateway_success=True,
            projection_success=False,
            missing_outputs=["open"],
            scope_match=False,
        )
        == "gateway_envelope_projection_failure"
    )


def test_v2_readiness_rejected_for_live():
    profile = load_provider_availability_profile(PROFILE, condition="pe_fin_replan")
    with pytest.raises(ProviderAvailabilityError, match="invalidated|projection"):
        assert_tool_readiness_for_live(
            artifact_path=V2_READINESS,
            expected_profile_hash=profile_hash(profile),
            mock=False,
            dry_run=False,
        )


def test_docs_r5_unchanged_and_projection_identity():
    bundle = build_unified_catalog(root=ROOT)
    assert documents_hash(bundle.registry) == DOCS_SHA
    assert retrieval_formula_hash() == R5_SHA
    assert MARKET_GATEWAY_PROJECTION_VERSION == "pe_fin_lite_market_gateway_projection_v2"
    assert len(market_gateway_projection_hash()) == 64


def test_mock_dry_cannot_forge_live_pass(tmp_path):
    profile = load_provider_availability_profile(PROFILE, condition="pe_fin_replan")
    skipped = assert_tool_readiness_for_live(
        artifact_path=tmp_path / "x.json",
        expected_profile_hash=profile_hash(profile),
        mock=True,
        dry_run=False,
    )
    assert skipped["live_pass_forged"] is False
