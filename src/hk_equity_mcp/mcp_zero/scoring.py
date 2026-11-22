"""Scoring helpers for MCP-Zero hierarchical retrieval (pure Python, no numpy).

Official matcher (``data/etom/raw_mcp0_pin/official_matcher.py``):
  server score = max(cos(q_server, server.description), cos(q_server, server.summary))
  tool score   = (s_server * s_tool) * max(s_server, s_tool)

ETOM pinned MCP0 orchestrator (``orchestrator_mcp0.py``) *overrides* the final
score with ``tool_similarity`` only; both modes are available here.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple, TypeVar

SCORE_OFFICIAL_COMBINED = "official_combined"
SCORE_TOOL_ONLY = "tool_similarity_only"
SCORE_MODES = (SCORE_OFFICIAL_COMBINED, SCORE_TOOL_ONLY)

# Round scores before tie-breaking so equal-in-practice floats order
# identically across platforms / summation orders.
SCORE_SORT_DECIMALS = 12

T = TypeVar("T")


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity; zero vectors yield 0.0 (as in the official matcher)."""
    if len(a) != len(b):
        raise ValueError(
            f"Embedding dimension mismatch: {len(a)} vs {len(b)}. "
            "Do not mix embedding models."
        )
    if not a:
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def server_score(
    description_sim: float, summary_sim: Optional[float]
) -> float:
    """Dual match: max(description, summary) like the official matcher.

    When a server has no summary embedding the description similarity is used
    directly (the official matcher would clamp at ``max(desc, 0)``; the ETOM
    pinned orchestrator has a single server embedding and no clamp).
    """
    if summary_sim is None:
        return float(description_sim)
    return float(max(description_sim, summary_sim))


def official_combined_score(s_server: float, s_tool: float) -> float:
    """Paper / official matcher formula."""
    return (s_server * s_tool) * max(s_server, s_tool)


def tool_only_score(_s_server: float, s_tool: float) -> float:
    """ETOM pinned MCP0: ``final_score = tool_similarity``."""
    return float(s_tool)


def combine_scores(mode: str, s_server: float, s_tool: float) -> float:
    if mode == SCORE_OFFICIAL_COMBINED:
        return official_combined_score(s_server, s_tool)
    if mode == SCORE_TOOL_ONLY:
        return tool_only_score(s_server, s_tool)
    raise ValueError(f"Unknown score mode {mode!r}; expected one of {SCORE_MODES}")


def sort_key(score: float, server_id: str, tool_name: str = "", operation_id: str = "") -> Tuple:
    """Deterministic ordering: (-score, server_id, tool_name, operation_id)."""
    return (-round(float(score), SCORE_SORT_DECIMALS), server_id, tool_name, operation_id)


def rank_items(
    items: Iterable[T],
    key_fn: Callable[[T], Tuple],
    top_k: int,
) -> List[T]:
    ordered = sorted(items, key=key_fn)
    return ordered[: max(0, int(top_k))]


def passes_threshold(sim: float, threshold: float) -> bool:
    """ETOM pinned MCP0 keeps ``sim >= threshold`` (threshold 0 drops negatives)."""
    return float(sim) >= float(threshold)


def any_nan(values: Iterable[Any]) -> bool:
    return any(isinstance(v, float) and math.isnan(v) for v in values)
