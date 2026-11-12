"""Step-level tool retrieval for PE-Fin RePlan-ETOM: R5 + safety-only filter.

R5: score = cosine_similarity + 0.12 * lexical_score  (shared with HK-155 via
``pe_fin_lite.retrieval_formula``). Top-K = 5, Describe-N = 3. The only filter is
a safety filter (runtime-exposed and not blocked); there is NO domain hard filter.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Set

from hk_equity_mcp.operation_resolver.schemas import OperationSpec
from hk_equity_mcp.pe_fin_lite.retrieval_formula import (
    DESCRIBE_N,
    DOMAIN_HARD_FILTER,
    RETRIEVAL_FORMULA_NAME,
    RETRIEVAL_TOP_K,
    r5_score_from_vectors,
)
from hk_equity_mcp.pe_fin_lite.tool_retrieval import safety_filter_operations

ETOM_RETRIEVAL_TOP_K = RETRIEVAL_TOP_K  # 5
ETOM_DESCRIBE_N = DESCRIBE_N  # 3
assert ETOM_RETRIEVAL_TOP_K == 5 and ETOM_DESCRIBE_N == 3 and DOMAIN_HARD_FILTER is False


def build_step_search_query(step: Dict[str, Any]) -> str:
    """Query from step fields only (never from gold, never the task id)."""
    parts: List[str] = []
    objective = step.get("objective")
    if objective:
        parts.append(str(objective))
    domain = str(step.get("domain") or "")
    if domain and domain not in ("general", "reason", "compute"):
        parts.append(f"area: {domain}")
    req = step.get("required_outputs") or []
    if req:
        parts.append("outputs: " + ", ".join(str(x) for x in req))
    scope = step.get("scope_constraints")
    if isinstance(scope, dict):
        vals = [f"{k}={v}" for k, v in scope.items() if v not in (None, "")]
        if vals:
            parts.append("scope: " + ", ".join(vals))
    return " | ".join(parts).strip()


def rank_r5(
    *,
    query: str,
    query_vector: Sequence[float],
    registry: Dict[str, OperationSpec],
    vectors: Dict[str, Sequence[float]],
    allowed_ids: Optional[Set[str]] = None,
    top_k: int = ETOM_RETRIEVAL_TOP_K,
) -> List[Dict[str, Any]]:
    """Rank allowed operations by R5; ties break on operation_id (deterministic)."""
    scored = []
    for oid in sorted(registry):
        if allowed_ids is not None and oid not in allowed_ids:
            continue
        parts = r5_score_from_vectors(
            query=query,
            query_vector=query_vector,
            doc_vector=vectors[oid],
            operation=registry[oid],
        )
        scored.append((parts["score"], oid, parts))
    scored.sort(key=lambda row: (-row[0], row[1]))
    rows: List[Dict[str, Any]] = []
    for rank, (score, oid, parts) in enumerate(scored[: max(1, int(top_k))], start=1):
        spec = registry[oid]
        rows.append(
            {
                "rank": rank,
                "operation_id": oid,
                "score": float(score),
                "cosine_similarity": float(parts["cosine_similarity"]),
                "lexical_score": float(parts["lexical_score"]),
                "lexical_weight": float(parts["lexical_weight"]),
                "server_name": spec.service_id,
                "name": spec.name,
                "summary": (spec.summary or "")[:320],
            }
        )
    return rows


def select_describe_candidates(
    topk: Sequence[Dict[str, Any]], *, describe_n: int = ETOM_DESCRIBE_N
) -> List[Dict[str, Any]]:
    n = max(1, int(describe_n))
    return [dict(row) for row in list(topk)[:n]]


def retrieve_candidates_for_step(
    *,
    step: Dict[str, Any],
    registry: Dict[str, OperationSpec],
    vectors: Dict[str, Sequence[float]],
    embedder: Any,
    budget: Any = None,
    metadata_by_id: Optional[Dict[str, Dict[str, Any]]] = None,
    runtime_operation_ids: Optional[Sequence[str]] = None,
    top_k: int = ETOM_RETRIEVAL_TOP_K,
    describe_n: int = ETOM_DESCRIBE_N,
) -> Dict[str, Any]:
    """One embedding query -> safety filter -> R5 Top-K -> describe top N."""
    query = build_step_search_query(step)
    if not query:
        return {
            "ok": False,
            "error": "empty_step_query",
            "search_query": query,
            "topk": [],
            "described": [],
            "embedding_query_calls": 0,
        }
    registry_ids = sorted(registry)
    runtime_ids = list(runtime_operation_ids) if runtime_operation_ids is not None else registry_ids
    safety = safety_filter_operations(
        registry_ids=registry_ids,
        metadata_by_id=metadata_by_id or {},
        runtime_operation_ids=runtime_ids,
    )
    allowed = set(safety["kept"])
    query_vectors, _usage = embedder.embed([query])
    if budget is not None:
        budget.embedding_query_calls += 1
    topk = rank_r5(
        query=query,
        query_vector=query_vectors[0],
        registry=registry,
        vectors=vectors,
        allowed_ids=allowed,
        top_k=top_k,
    )
    described = select_describe_candidates(topk, describe_n=describe_n)
    return {
        "ok": bool(topk),
        "error": None if topk else "no_candidates_after_safety_filter",
        "search_query": query,
        "retrieval_formula": RETRIEVAL_FORMULA_NAME,
        "top_k": int(top_k),
        "describe_n": int(describe_n),
        "domain_hard_filter_applied": False,
        "safety_filter": {
            k: safety[k]
            for k in ("kept_count", "rejected_count", "domain_hard_filter_applied", "filter_kind")
        },
        "topk": topk,
        "described": described,
        "embedding_query_calls": 1,
    }
