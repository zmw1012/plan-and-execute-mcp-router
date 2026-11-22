"""Unit tests for MCP-Zero hierarchical retrieval (offline, no APIs)."""

from __future__ import annotations

from typing import Dict, List

import pytest

from hk_equity_mcp.mcp_zero.active_request import (
    format_active_tool_request,
    parse_active_tool_request,
)
from hk_equity_mcp.mcp_zero.embedding import CountingEmbedder, EmbeddingTable
from hk_equity_mcp.mcp_zero.extension import (
    TERMINATION_DUPLICATE_REQUEST,
    TERMINATION_MAX_EXTENSION_ROUNDS,
    TERMINATION_NO_NEW_TOOLS,
    ExtensionController,
)
from hk_equity_mcp.mcp_zero.metadata import McpZeroCatalog, ServerMeta, ToolMeta
from hk_equity_mcp.mcp_zero.router import HierarchicalRouter
from hk_equity_mcp.mcp_zero.schemas import ActiveToolRequest
from hk_equity_mcp.mcp_zero.scoring import (
    SCORE_OFFICIAL_COMBINED,
    official_combined_score,
    sort_key,
)


class _FixedEmbedder:
    model_name = "mock-fixed"

    def __init__(self, table: Dict[str, List[float]]):
        self.table = table

    def embed(self, texts):
        out = []
        for text in texts:
            if text not in self.table:
                raise KeyError(f"missing embedding for {text!r}")
            out.append(list(self.table[text]))
        return out


def _toy_catalog() -> McpZeroCatalog:
    return McpZeroCatalog(
        [
            ServerMeta(
                server_id="alpha",
                description="alpha server finance",
                summary="alpha summary finance indicators",
                tools=[
                    ToolMeta(
                        server_id="alpha",
                        tool_name="fin_indicator",
                        operation_id="alpha.fin_indicator",
                        description="financial indicator tool",
                        text="fin_indicator financial indicator tool",
                    ),
                    ToolMeta(
                        server_id="alpha",
                        tool_name="fin_other",
                        operation_id="alpha.fin_other",
                        description="other finance helper",
                        text="fin_other other finance helper",
                    ),
                ],
            ),
            ServerMeta(
                server_id="beta",
                description="beta server news",
                summary="beta summary news filings",
                tools=[
                    ToolMeta(
                        server_id="beta",
                        tool_name="news_search",
                        operation_id="beta.news_search",
                        description="news search tool",
                        text="news_search news search tool",
                    ),
                ],
            ),
            ServerMeta(
                server_id="gamma",
                description="gamma server market",
                summary=None,
                tools=[
                    ToolMeta(
                        server_id="gamma",
                        tool_name="blocked_looking",
                        operation_id="gamma.should_not_appear_if_excluded",
                        description="market quote",
                        text="blocked_looking market quote",
                    ),
                ],
            ),
        ],
        source="toy",
    )


def _unit(v: float, dim: int = 4, idx: int = 0) -> List[float]:
    vec = [0.0] * dim
    vec[idx] = v
    return vec


def _build_router() -> HierarchicalRouter:
    catalog = _toy_catalog()
    # Place alpha near finance query; beta near news; tools aligned similarly.
    table = EmbeddingTable(
        model_name="mock-fixed",
        fingerprint="toy",
        server_desc={
            "alpha": _unit(1.0, idx=0),
            "beta": _unit(1.0, idx=1),
            "gamma": _unit(1.0, idx=2),
        },
        server_summary={
            "alpha": _unit(0.9, idx=0),
            "beta": _unit(0.9, idx=1),
        },
        tool={
            "alpha.fin_indicator": _unit(1.0, idx=0),
            "alpha.fin_other": _unit(0.5, idx=0),
            "beta.news_search": _unit(1.0, idx=1),
            "gamma.should_not_appear_if_excluded": _unit(1.0, idx=2),
        },
    )
    # Query embeddings keyed by the Active Tool Request strings used in tests.
    q = {
        "finance server": _unit(1.0, idx=0),
        "financial indicator": _unit(1.0, idx=0),
        "news server": _unit(1.0, idx=1),
        "news search": _unit(1.0, idx=1),
        "market server": _unit(1.0, idx=2),
        "market quote": _unit(1.0, idx=2),
        "finance server again": _unit(1.0, idx=0),
        "financial indicator refined": _unit(1.0, idx=0),
    }
    embedder = CountingEmbedder(_FixedEmbedder(q), model_name="mock-fixed")
    return HierarchicalRouter(
        catalog,
        table,
        embedder,
        server_top_k=2,
        tool_top_k=2,
        score_mode=SCORE_OFFICIAL_COMBINED,
        use_summary=True,
    )


def test_active_tool_request_parse():
    text = format_active_tool_request("HK finance MCP", "indicator lookup")
    req = parse_active_tool_request(text)
    assert req is not None
    assert req.server == "HK finance MCP"
    assert req.tool == "indicator lookup"
    assert parse_active_tool_request("no tags") is None


def test_server_first_then_tools_only_in_selected_servers():
    router = _build_router()
    req = ActiveToolRequest(server="finance server", tool="financial indicator")
    round_ = router.retrieve(req)
    assert [s.server_id for s in round_.servers][0] == "alpha"
    assert "beta" not in {t.server_id for t in round_.tools} or all(
        t.server_id in {s.server_id for s in round_.servers} for t in round_.tools
    )
    selected = {s.server_id for s in round_.servers}
    for tool in round_.tools:
        assert tool.server_id in selected
    # gamma not selected for finance query → its tools never scored into top.
    assert all(t.operation_id != "gamma.should_not_appear_if_excluded" for t in round_.tools)
    assert round_.tools[0].operation_id == "alpha.fin_indicator"


def test_deterministic_tie_breaking():
    # Equal scores → lexicographic server/tool/op id.
    a = sort_key(0.5, "b_server", "tool", "op_b")
    b = sort_key(0.5, "a_server", "tool", "op_a")
    assert b < a
    assert official_combined_score(0.8, 0.5) == pytest.approx((0.8 * 0.5) * 0.8)


def test_iterative_extension_and_termination():
    router = _build_router()
    ctrl = ExtensionController(max_extension_rounds=1)
    r0 = ActiveToolRequest(server="finance server", tool="financial indicator", round_index=0)
    d0 = ctrl.decide(r0)
    assert d0.allow and not d0.is_extension
    round0 = ctrl.commit_round(router.retrieve(r0, is_extension=False))
    assert round0.new_operation_ids

    # Duplicate rejected.
    d_dup = ctrl.decide(r0)
    assert not d_dup.allow
    assert d_dup.reason == TERMINATION_DUPLICATE_REQUEST

    # Extension allowed once.
    r1 = ActiveToolRequest(
        server="finance server again",
        tool="financial indicator refined",
        round_index=1,
        source="extension",
    )
    d1 = ctrl.decide(r1)
    assert d1.allow and d1.is_extension
    round1 = ctrl.commit_round(
        router.retrieve(
            r1,
            exclude_operation_ids=ctrl.already_seen_ops(),
            is_extension=True,
        )
    )
    # After excluding already-seen ops, new tools may be empty → termination signal.
    if not round1.new_operation_ids:
        assert round1.stop_reason == TERMINATION_NO_NEW_TOOLS

    # Second extension blocked by max rounds.
    r2 = ActiveToolRequest(
        server="news server", tool="news search", round_index=2, source="extension"
    )
    d2 = ctrl.decide(r2)
    assert not d2.allow
    assert d2.reason == TERMINATION_MAX_EXTENSION_ROUNDS


def test_tool_retrieval_never_leaves_selected_servers():
    router = _build_router()
    req = ActiveToolRequest(server="news server", tool="news search")
    round_ = router.retrieve(req)
    assert round_.servers
    selected = {s.server_id for s in round_.servers}
    assert all(t.server_id in selected for t in round_.tools)
    # Ensure we did not silently pull alpha tools for a news query top hit.
    if round_.tools:
        assert round_.tools[0].server_id == "beta"
