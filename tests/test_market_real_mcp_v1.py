"""Frozen Market Real-MCP catalog + adapter unit tests (no paid LLM)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
import yaml

from hk_equity_mcp.real_mcp.catalog import load_real_mcp_catalog, load_real_mcp_registry
from hk_equity_mcp.real_mcp.market_gateway import (
    BLOCKED_MARKET_OPS,
    DEFAULT_MARKET_CATALOG,
    MarketRealMcpGateway,
)
from hk_equity_mcp.real_mcp.market_normalize import (
    MARKET_CUTOFF,
    filter_cutoff,
    previous_valid_session,
    row_date,
)
from hk_equity_mcp.real_mcp.types import RealMcpResult

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "configs" / "operations" / "market_real_mcp_v1_1.yaml"
CATALOG_V1 = ROOT / "configs" / "operations" / "market_real_mcp_v1.yaml"


def test_default_catalog_path():
    assert DEFAULT_MARKET_CATALOG == CATALOG
    assert CATALOG.exists()
    assert CATALOG_V1.exists()


def test_v1_superseded_and_v1_1_package_blocked_ids():
    v1 = load_real_mcp_catalog(CATALOG_V1)
    assert v1.get("status") == "superseded_before_any_agent_run"
    catalog = load_real_mcp_catalog(CATALOG)
    assert catalog["catalog_id"] == "market_real_mcp_v1_1"
    servers = catalog["servers"]
    ak = servers[0] if isinstance(servers, list) else servers["aktools"]
    assert ak["package"] == "aktools"
    assert str(ak["package_version"]) == "0.1.4"
    blocked = {b["operation_id"] for b in catalog["blocked_operations"]}
    assert blocked == {
        "ftshare.ft_v2_hk_hk_candlesticks",
        "ftshare.ft_v1_hk_hsi_daily_weight",
    }


def test_active_role_counts_and_no_blocked():
    catalog = load_real_mcp_catalog(CATALOG)
    ops = catalog["operations"]
    assert len(ops) == 13
    roles = {}
    for op in ops:
        roles[op["role"]] = roles.get(op["role"], 0) + 1
        assert op["operation_id"] not in BLOCKED_MARKET_OPS
        assert op["operation_id"] not in {
            "ftshare.ft_v2_hk_hk_candlesticks",
            "ftshare.ft_v1_hk_hsi_daily_weight",
            "aktools.stock_hk_hist_min_em",
        }
    assert roles == {"primary": 2, "supporting": 6, "distractor": 5}
    assert len(catalog.get("blocked_operations") or []) == 2
    assert len(catalog.get("deferred_operations") or []) == 1
    registry = load_real_mcp_registry(CATALOG)
    assert set(registry) == {o["operation_id"] for o in ops}
    assert "aktools.tool_trade_date_hist_sina" in registry
    assert catalog.get("tool_trade_date_hist_sina_is_hkex_official_calendar") is False


def test_distractor_families_present():
    ops = load_real_mcp_catalog(CATALOG)["operations"]
    families = {
        o.get("distractor_family")
        for o in ops
        if o.get("role") == "distractor"
    }
    assert families == {
        "spot",
        "hot_rank",
        "hk_yield",
        "valuation_indicator",
        "non_hk_trading_calendar",
    }


def test_gateway_rejects_blocked_ops_in_constructor(tmp_path):
    bad = {
        "operations": [
            {
                "operation_id": "ftshare.hk_hist",
                "server_id": "ftshare",
                "name": "hk_hist",
                "summary": "bad",
                "description": "bad",
                "parameters": [],
                "execution": {"kind": "mcp_tool", "tool": "x"},
            }
        ]
    }
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="Blocked"):
        MarketRealMcpGateway(catalog_path=path, aktools=object())  # type: ignore[arg-type]


def test_post_filter_drops_post_cutoff():
    rows = [
        {"date": "2026-08-11", "close": 1},
        {"date": "2026-08-12", "close": 2},
        {"date": "2026-08-13", "close": 3},
    ]
    kept = filter_cutoff(rows, cutoff=MARKET_CUTOFF)
    assert [row_date(r).isoformat() for r in kept] == ["2026-08-11", "2026-08-12"]


def test_previous_valid_session_no_forward_fill():
    rows = [
        {"date": "2026-08-07", "close": 10},
        {"date": "2026-08-10", "close": 11},
        {"date": "2026-08-11", "close": 12},
        {"date": "2026-08-12", "close": 13},
        {"date": "2026-08-13", "close": 99},
    ]
    # Sunday 2026-08-09 → Friday 2026-08-07
    hit = previous_valid_session(rows, date(2026, 8, 9), cutoff=MARKET_CUTOFF)
    assert row_date(hit) == date(2026, 8, 7)
    # Exact session
    hit2 = previous_valid_session(rows, date(2026, 8, 12), cutoff=MARKET_CUTOFF)
    assert row_date(hit2) == date(2026, 8, 12)


class _FakeAktools:
    def __init__(self, payload):
        self.payload = payload
        self.opened = False

    def open(self):
        self.opened = True

    def close(self):
        self.opened = False

    def get_data(self, **kwargs):
        return RealMcpResult(
            success=True,
            server_id="aktools",
            operation_id="tmp",
            arguments=kwargs,
            normalized=self.payload,
            real_mcp=True,
            protocol_success=True,
            execution_success=True,
            server_version="fake-test",
        )

    def call_mcp_tool(self, tool_name, arguments=None):
        return RealMcpResult(
            success=True,
            server_id="aktools",
            operation_id=f"aktools.{tool_name}",
            arguments=arguments or {},
            normalized={"ok": True},
            real_mcp=True,
            protocol_success=True,
            execution_success=True,
        )


def test_gateway_hist_default_adjust_and_schema():
    fake = _FakeAktools(
        [
            {"日期": "2026-08-11", "收盘": 100.0, "开盘": 99, "最高": 101, "最低": 98, "成交量": 1},
            {"日期": "2026-08-12", "收盘": 105.0, "开盘": 100, "最高": 106, "最低": 99, "成交量": 2},
            {"日期": "2026-08-13", "收盘": 999.0, "开盘": 1, "最高": 1, "最低": 1, "成交量": 9},
        ]
    )
    gw = MarketRealMcpGateway(catalog_path=CATALOG, aktools=fake)
    res = gw.invoke(
        "aktools.stock_hk_hist",
        {
            "symbol": "00700",
            "start_date": "20260811",
            "end_date": "20260812",
            "target_date": "2026-08-12",
        },
    )
    assert res.success
    norm = res.normalized
    assert norm["success"] is True
    assert norm["adjustment_mode"] == ""
    assert norm["currency_or_unit"] == "HKD"
    assert norm["currency_provenance"] == "exchange_convention"
    assert norm["timezone"] == "Asia/Hong_Kong"
    assert norm["effective_session_date"] == "2026-08-12"
    assert norm["data"]["close"] == 105.0
    assert norm["provenance"]["mock_fallback"] is False
    # post-cutoff row must not appear in range data path either
    res2 = gw.invoke(
        "aktools.stock_hk_hist",
        {"symbol": "00700", "start_date": "20260811", "end_date": "20260812"},
    )
    data = res2.normalized["data"]
    assert isinstance(data, list)
    assert all(r["date"] <= "2026-08-12" for r in data)


def test_gateway_rejects_end_after_cutoff():
    gw = MarketRealMcpGateway(catalog_path=CATALOG, aktools=_FakeAktools([]))
    res = gw.invoke(
        "aktools.stock_hk_hist",
        {
            "symbol": "00700",
            "start_date": "20260801",
            "end_date": "20260820",
            "adjust": "",
        },
    )
    assert not res.success
    assert res.error_type == "cutoff_violation"


def test_gateway_no_mock_financial_server_attribute():
    gw = MarketRealMcpGateway(catalog_path=CATALOG, aktools=_FakeAktools([]))
    assert not hasattr(gw, "financial")
