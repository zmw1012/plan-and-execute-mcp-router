"""Hierarchical Semantic Routing: server-first then tool-within-selected-servers.

Paper formula (official matcher):
  server_score = max(cos(q_server, desc), cos(q_server, summary))
  final_score  = (s_server * s_tool) * max(s_server, s_tool)

HK adapted defaults (documented): ``server_top_k=3``, ``tool_top_k=5``,
``score_mode=official_combined``, embedding model pinned to the shared
experiment backbone ``text-embedding-3-small`` (not the paper's *-large*).
"""

from __future__ import annotations

import time
from typing import Iterable, List, Optional, Sequence

from hk_equity_mcp.mcp_zero.embedding import CountingEmbedder, EmbeddingTable
from hk_equity_mcp.mcp_zero.metadata import McpZeroCatalog
from hk_equity_mcp.mcp_zero.schemas import ActiveToolRequest, ExtensionRound, ToolHit
from hk_equity_mcp.mcp_zero.scoring import SCORE_OFFICIAL_COMBINED
from hk_equity_mcp.mcp_zero.server_retriever import ServerRetriever
from hk_equity_mcp.mcp_zero.tool_retriever import ToolRetriever
from hk_equity_mcp.mcp_zero import (
    DEFAULT_SERVER_TOP_K,
    DEFAULT_SIMILARITY_THRESHOLD,
    DEFAULT_TOOL_TOP_K,
)


class HierarchicalRouter:
    """Two-stage retrieval with a visible ``ExtensionRound`` trace."""

    def __init__(
        self,
        catalog: McpZeroCatalog,
        table: EmbeddingTable,
        embedder: CountingEmbedder,
        *,
        server_top_k: int = DEFAULT_SERVER_TOP_K,
        tool_top_k: int = DEFAULT_TOOL_TOP_K,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        score_mode: str = SCORE_OFFICIAL_COMBINED,
        use_summary: bool = True,
    ):
        self.catalog = catalog
        self.table = table
        self.embedder = embedder
        self.server_top_k = int(server_top_k)
        self.tool_top_k = int(tool_top_k)
        self.similarity_threshold = float(similarity_threshold)
        self.score_mode = score_mode
        self.server_retriever = ServerRetriever(
            catalog,
            table,
            embedder,
            server_top_k=server_top_k,
            similarity_threshold=similarity_threshold,
            use_summary=use_summary,
        )
        self.tool_retriever = ToolRetriever(
            catalog,
            table,
            embedder,
            tool_top_k=tool_top_k,
            similarity_threshold=similarity_threshold,
            score_mode=score_mode,
        )

    def retrieve(
        self,
        request: ActiveToolRequest,
        *,
        exclude_operation_ids: Iterable[str] = (),
        is_extension: bool = False,
    ) -> ExtensionRound:
        started = time.perf_counter()
        calls_before = self.embedder.calls
        tokens_before = self.embedder.tokens
        servers = self.server_retriever.match(request.server)
        tools = self.tool_retriever.retrieve(
            servers,
            request.tool,
            exclude_operation_ids=exclude_operation_ids,
        )
        latency_ms = (time.perf_counter() - started) * 1000.0
        return ExtensionRound(
            round_index=int(request.round_index),
            request=request,
            servers=servers,
            tools=tools,
            new_operation_ids=[t.operation_id for t in tools],
            is_extension=bool(is_extension),
            latency_ms=latency_ms,
            embedding_calls=self.embedder.calls - calls_before,
            embedding_tokens=self.embedder.tokens - tokens_before,
        )

    def format_tools_for_prompt(self, tools: Sequence[ToolHit]) -> str:
        lines: List[str] = []
        for hit in tools:
            meta = self.catalog.tool(hit.operation_id)
            schema = (meta.schema if meta else {}) or {}
            lines.append(
                f"- operation_id: {hit.operation_id}\n"
                f"  server_id: {hit.server_id}\n"
                f"  tool_name: {hit.tool_name}\n"
                f"  description: {hit.description}\n"
                f"  server_score: {hit.server_score:.6f}\n"
                f"  tool_score: {hit.tool_score:.6f}\n"
                f"  final_score: {hit.final_score:.6f}\n"
                f"  rank: {hit.rank}\n"
                f"  schema: {schema}"
            )
        return "\n".join(lines) if lines else "(no tools retrieved)"
