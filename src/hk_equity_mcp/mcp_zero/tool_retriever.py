"""Stage 2 of hierarchical retrieval: rank tools *only inside selected servers*.

Mirrors ``ToolMatcher.match_tools`` (official) / ``_retrieve_tools`` (ETOM MCP0):
tools of servers that were not selected in stage 1 are never scored.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

from hk_equity_mcp.mcp_zero.embedding import CountingEmbedder, EmbeddingTable
from hk_equity_mcp.mcp_zero.metadata import McpZeroCatalog
from hk_equity_mcp.mcp_zero.schemas import ServerHit, ToolHit
from hk_equity_mcp.mcp_zero.scoring import (
    SCORE_MODES,
    SCORE_OFFICIAL_COMBINED,
    combine_scores,
    cosine,
    passes_threshold,
    rank_items,
    sort_key,
)


class ToolRetriever:
    def __init__(
        self,
        catalog: McpZeroCatalog,
        table: EmbeddingTable,
        embedder: CountingEmbedder,
        *,
        tool_top_k: int = 5,
        similarity_threshold: float = 0.0,
        score_mode: str = SCORE_OFFICIAL_COMBINED,
    ):
        if int(tool_top_k) < 1:
            raise ValueError("tool_top_k must be >= 1")
        if score_mode not in SCORE_MODES:
            raise ValueError(f"Unknown score_mode {score_mode!r}")
        self.catalog = catalog
        self.table = table
        self.embedder = embedder
        self.tool_top_k = int(tool_top_k)
        self.similarity_threshold = float(similarity_threshold)
        self.score_mode = score_mode

    def embed_query(self, text: str) -> List[float]:
        return self.embedder.embed([text])[0]

    def retrieve(
        self,
        server_hits: Sequence[ServerHit],
        tool_desc: str,
        *,
        exclude_operation_ids: Iterable[str] = (),
        query_vec: Optional[Sequence[float]] = None,
    ) -> List[ToolHit]:
        if not server_hits or not tool_desc or not tool_desc.strip():
            return []
        excluded = set(exclude_operation_ids)
        qvec = list(query_vec) if query_vec is not None else self.embed_query(tool_desc)
        candidates: List[ToolHit] = []
        for server_hit in server_hits:
            # Scope: ONLY tools belonging to selected servers are ever scored.
            for tool in self.catalog.tools_for(server_hit.server_id):
                if tool.operation_id in excluded:
                    continue
                vec = self.table.tool.get(tool.operation_id)
                if vec is None:
                    continue  # official: tools without embedding skipped
                tool_sim = cosine(qvec, vec)
                if not passes_threshold(tool_sim, self.similarity_threshold):
                    continue
                final = combine_scores(self.score_mode, server_hit.score, tool_sim)
                candidates.append(
                    ToolHit(
                        server_id=tool.server_id,
                        tool_name=tool.tool_name,
                        operation_id=tool.operation_id,
                        server_score=server_hit.score,
                        tool_score=tool_sim,
                        final_score=final,
                        description=tool.description,
                    )
                )
        top = rank_items(
            candidates,
            lambda h: sort_key(h.final_score, h.server_id, h.tool_name, h.operation_id),
            self.tool_top_k,
        )
        for index, hit in enumerate(top, start=1):
            hit.rank = index
        return top
