"""Stage 1 of hierarchical retrieval: rank *servers* by the request's server text.

Mirrors ``ToolMatcher.match_servers`` in the official matcher: the query
embedding of the ``server:`` description is compared with both the server
description embedding and (if present) the server summary embedding; the score
is the max of the two.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from hk_equity_mcp.mcp_zero.embedding import CountingEmbedder, EmbeddingTable
from hk_equity_mcp.mcp_zero.metadata import McpZeroCatalog
from hk_equity_mcp.mcp_zero.schemas import ServerHit
from hk_equity_mcp.mcp_zero.scoring import (
    cosine,
    passes_threshold,
    rank_items,
    server_score,
    sort_key,
)


class ServerRetriever:
    def __init__(
        self,
        catalog: McpZeroCatalog,
        table: EmbeddingTable,
        embedder: CountingEmbedder,
        *,
        server_top_k: int = 3,
        similarity_threshold: float = 0.0,
        use_summary: bool = True,
    ):
        if int(server_top_k) < 1:
            raise ValueError("server_top_k must be >= 1")
        self.catalog = catalog
        self.table = table
        self.embedder = embedder
        self.server_top_k = int(server_top_k)
        self.similarity_threshold = float(similarity_threshold)
        self.use_summary = bool(use_summary)

    def embed_query(self, text: str) -> List[float]:
        return self.embedder.embed([text])[0]

    def match(
        self,
        server_desc: str,
        query_vec: Optional[Sequence[float]] = None,
    ) -> List[ServerHit]:
        if not server_desc or not server_desc.strip():
            return []
        qvec = list(query_vec) if query_vec is not None else self.embed_query(server_desc)
        hits: List[ServerHit] = []
        for server in self.catalog.servers():
            desc_vec = self.table.server_desc.get(server.server_id)
            if desc_vec is None:
                continue  # official: servers without description embedding skipped
            desc_sim = cosine(qvec, desc_vec)
            summary_sim: Optional[float] = None
            if self.use_summary:
                sum_vec = self.table.server_summary.get(server.server_id)
                if sum_vec is not None:
                    summary_sim = cosine(qvec, sum_vec)
            score = server_score(desc_sim, summary_sim)
            if not passes_threshold(score, self.similarity_threshold):
                continue
            hits.append(
                ServerHit(
                    server_id=server.server_id,
                    score=score,
                    description_score=desc_sim,
                    summary_score=summary_sim,
                )
            )
        top = rank_items(
            hits, lambda h: sort_key(h.score, h.server_id), self.server_top_k
        )
        for index, hit in enumerate(top, start=1):
            hit.rank = index
        return top
