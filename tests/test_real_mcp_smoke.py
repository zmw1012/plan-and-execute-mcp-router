"""Live Real-MCP smoke tests (no paid LLM). Skips if server binaries missing."""

from __future__ import annotations

from pathlib import Path

import pytest

from hk_equity_mcp.operation_resolver.retrieval import DeterministicHashEmbeddingBackend
from hk_equity_mcp.operation_resolver.server import OperationResolverServer
from hk_equity_mcp.real_mcp.adapters.aktools import AktoolsAdapter, default_aktools_command
from hk_equity_mcp.real_mcp.adapters.hkexnews import (
    HkexnewsAdapter,
    default_hkexnews_command,
)
from hk_equity_mcp.real_mcp.catalog import load_real_mcp_registry
from hk_equity_mcp.real_mcp.client import resolve_command
from hk_equity_mcp.real_mcp.gateway import RealMcpGateway
from hk_equity_mcp.real_mcp.reconcile import reconcile_tencent_fy2025_revenue


def _has_cmd(command: str) -> bool:
    try:
        resolve_command(command)
        return True
    except Exception:
        return False


def _has_mcp_sdk() -> bool:
    try:
        import mcp  # noqa: F401
        from mcp import ClientSession  # noqa: F401
        from mcp.client.stdio import stdio_client  # noqa: F401

        return True
    except Exception:
        return False


requires_aktools = pytest.mark.skipif(
    not (
        _has_mcp_sdk()
        and bool(default_aktools_command(require=False))
    ),
    reason="mcp SDK or aktools-mcp binary not available in this Python env",
)
requires_hkexnews = pytest.mark.skipif(
    not (
        _has_mcp_sdk()
        and bool(default_hkexnews_command(require=False))
    ),
    reason="mcp SDK or agentladle-mcp-hkexnews binary not available in this Python env",
)


@requires_aktools
def test_aktools_structured_smoke_tencent_fy2025_revenue():
    adapter = AktoolsAdapter(timeout_sec=120)
    # Protocol: tools/list
    listed = adapter.list_tools()
    assert listed["success"] is True
    names = {t["name"] for t in listed["tools"]}
    assert {"search_interfaces", "describe_interface", "get_data"} <= names

    result = adapter.get_data(
        interface="stock_financial_hk_analysis_indicator_em",
        params={"symbol": "00700", "indicator": "年度"},
        offset=0,
        max_rows=5,
        operation_id="aktools.stock_financial_hk_analysis_indicator_em",
    )
    assert result.protocol_success is True
    assert result.execution_success is True
    assert result.real_mcp is True
    assert result.success is True
    ok, note = reconcile_tencent_fy2025_revenue(result.normalized)
    result.reconciliation_success = ok
    result.reconciliation_note = note
    assert ok, note


@requires_hkexnews
def test_hkexnews_evidence_smoke_tencent_fy2025_results():
    cache = Path(__file__).resolve().parents[1] / "outputs" / "hkexnews_cache"
    cache.mkdir(parents=True, exist_ok=True)
    adapter = HkexnewsAdapter(allow_download=True, cache_dir=cache, timeout_sec=180)

    listed = adapter.list_tools()
    assert listed["success"] is True
    assert len(listed["tools"]) == 6

    listed_anns = adapter.call_mcp_tool(
        "list_hkexnews_announcements",
        {
            "stock_code": "00700",
            "title_keyword": "ANNUAL RESULTS",
            "start_date": "2026-03-01",
            "end_date": "2026-03-31",
            "limit": 5,
        },
        operation_id="hkexnews.list_hkexnews_announcements",
    )
    assert listed_anns.protocol_success is True
    assert listed_anns.execution_success is True

    # Single audited sample download (same filing as gap discovery).
    dl = adapter.call_mcp_tool(
        "download_hkexnews_announcement",
        {
            "stock_code": "00700",
            "release_date": "2026-03-18",
            "title_keyword": "ANNUAL RESULTS",
            "news_id": "12056832",
        },
        operation_id="hkexnews.download_hkexnews_announcement",
    )
    assert dl.protocol_success is True
    assert dl.execution_success is True, dl.error
    local_key = (dl.normalized or {}).get("data", {}).get("local_key")
    assert local_key

    parsed = adapter.call_mcp_tool(
        "parse_hkexnews_announcement",
        {"local_key": local_key},
        operation_id="hkexnews.parse_hkexnews_announcement",
    )
    assert parsed.execution_success is True

    kw = adapter.call_mcp_tool(
        "keyword_search",
        {
            "local_key": local_key,
            "keywords": ["non-IFRS", "Non-IFRS", "Revenues", "RMB"],
            "match_mode": "ANY",
            "max_results": 8,
        },
        operation_id="hkexnews.keyword_search",
    )
    assert kw.execution_success is True
    data = (kw.normalized or {}).get("data") or {}
    matches = data.get("matches") or []
    assert matches, "expected keyword hits for revenue/Non-IFRS"
    blob = str(matches).lower()
    assert "revenue" in blob or "non-ifrs" in blob or "rmb" in blob


@requires_aktools
def test_l5_fail_closed_unavailable_metric():
    gateway = RealMcpGateway(allow_hkex_download=False)
    # Unknown operation
    missing = gateway.invoke("aktools.does_not_exist", {})
    assert missing.success is False
    assert missing.error_type in {"unknown_operation", "unavailable"}
    assert missing.normalized is None

    # Unsupported / unreliable bank gross-margin style request via bogus interface name
    bad = gateway.aktools.get_data(
        interface="stock_hk_bank_gross_margin_DOES_NOT_EXIST",
        params={"symbol": "00005"},
        operation_id="aktools.stock_hk_bank_gross_margin_DOES_NOT_EXIST",
    )
    assert bad.protocol_success is True  # MCP session worked
    assert bad.execution_success is False
    assert bad.success is False
    assert bad.error
    # Must not fabricate rows.
    assert not (
        isinstance(bad.normalized, dict)
        and bad.normalized.get("rows")
        and not bad.error
    )


@requires_aktools
def test_operation_visible_resolver_real_mcp_invoke(tmp_path: Path):
    gateway = RealMcpGateway(allow_hkex_download=False)
    resolver = OperationResolverServer(
        embedding_backend=DeterministicHashEmbeddingBackend(),
        registry=load_real_mcp_registry(),
        cache_dir=tmp_path / "c",
        real_mcp_gateway=gateway,
    )
    resp = resolver.invoke_api(
        "aktools.stock_financial_hk_analysis_indicator_em",
        {"symbol": "00700", "indicator": "年度", "max_rows": 3},
    )
    assert resp["success"] is True
    assert resp["result"]["real_mcp"] is True
    assert resp["result"]["transport"] == "stdio"
    ok, note = reconcile_tencent_fy2025_revenue(resp["result"].get("normalized"))
    assert ok, note
