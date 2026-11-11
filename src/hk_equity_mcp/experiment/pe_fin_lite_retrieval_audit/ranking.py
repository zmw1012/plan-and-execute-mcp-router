"""Offline ranking variants R0–R3 and R_runtime (zero-network from caches)."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from hk_equity_mcp.experiment.pe_fin_lite_retrieval_audit import (
    DESCRIBE_N,
    DOMAIN_FILTER_VERSION,
    HYBRID_ALPHAS,
    OUTPUT_COVERAGE_VERSION,
    RUNTIME_LEXICAL_WEIGHT,
    TOP_K,
)
from hk_equity_mcp.experiment.pe_fin_lite_retrieval_audit.embedding_identity_gate import (
    EmbeddingDimensionMismatch,
    assert_vector_dimensions_match,
)
from hk_equity_mcp.operation_resolver.retrieval import (
    LEXICAL_WEIGHT,
    DeterministicHashEmbeddingBackend,
    lexical_score,
)
from hk_equity_mcp.operation_resolver.schemas import OperationSpec
from hk_equity_mcp.pe_fin_lite.tool_retrieval import (
    build_step_search_query,
    hard_filter_operations,
    select_embedding_topk,
)

# Empty required_outputs → coverage=1.0 (unit-neutral). Hybrid then becomes
# alpha*cos + (1-alpha)*1.0, an affine transform of cosine that preserves order.
EMPTY_REQUIRED_OUTPUT_COVERAGE = 1.0


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity; fail-closed on length mismatch (never zip/truncate)."""
    assert_vector_dimensions_match(a, b, context="cosine")
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return float(dot / (na * nb))


def required_output_coverage(
    *,
    required_outputs: Sequence[str],
    provides: Sequence[str],
) -> float:
    """Deterministic required-output coverage in [0, 1].

    If ``required_outputs`` is empty, returns ``EMPTY_REQUIRED_OUTPUT_COVERAGE``
    (=1.0) so R3 hybrid never lets a missing coverage term dominate ordering
    (affine in cosine when coverage is constant).
    """
    req = [str(x).lower() for x in (required_outputs or []) if str(x).strip()]
    if not req:
        return float(EMPTY_REQUIRED_OUTPUT_COVERAGE)
    prov = {str(x).lower() for x in (provides or [])}
    hit = sum(1 for r in req if r in prov or any(r in p or p in r for p in prov))
    return hit / float(len(req))


class OfflineEmbeddingIndex:
    """In-memory operation embedding index.

    - mock mode: DeterministicHashEmbeddingBackend
    - offline ranking: construct via ``from_preloaded_vectors`` (zero network)
    - never silently falls back from real → hash
    """

    def __init__(
        self,
        *,
        documents: Dict[str, Dict[str, Any]],
        metadata_by_id: Dict[str, Dict[str, Any]],
        runtime_operation_ids: Sequence[str],
        embedding_backend: Optional[Any] = None,
        preloaded_vectors: Optional[Dict[str, Sequence[float]]] = None,
        embedding_model: Optional[str] = None,
        allow_query_embed: bool = True,
        refuse_network: bool = True,
        registry: Optional[Dict[str, OperationSpec]] = None,
    ):
        self.documents = documents
        self.metadata_by_id = metadata_by_id
        self.runtime_operation_ids = list(runtime_operation_ids)
        self.registry = registry or {}
        self.ordered_ids = sorted(
            oid
            for oid in documents.keys()
            if oid in set(runtime_operation_ids) or oid in (preloaded_vectors or {})
        )
        if not self.ordered_ids:
            self.ordered_ids = sorted(documents.keys())
        self.allow_query_embed = bool(allow_query_embed)
        self.backend = None
        if preloaded_vectors is not None:
            self.ordered_ids = [oid for oid in sorted(preloaded_vectors.keys())]
            self.vectors = {
                oid: [float(x) for x in preloaded_vectors[oid]] for oid in self.ordered_ids
            }
            self.embedding_model = embedding_model or "preloaded"
            self.embedding_dim = (
                len(next(iter(self.vectors.values()))) if self.vectors else None
            )
            self.backend = None
        else:
            if refuse_network and embedding_backend is not None:
                name = getattr(embedding_backend, "model_name", "")
                if "mock-hash" not in str(name) and not isinstance(
                    embedding_backend, DeterministicHashEmbeddingBackend
                ):
                    raise RuntimeError(
                        "OfflineEmbeddingIndex refuse non-hash backend without "
                        "preloaded vectors (use query/catalog caches for live ranking)"
                    )
            self.backend = embedding_backend or DeterministicHashEmbeddingBackend()
            texts = [
                (documents.get(oid) or {}).get("document_text") or oid
                for oid in self.ordered_ids
            ]
            vectors = self.backend.embed(texts)
            self.vectors = {oid: vec for oid, vec in zip(self.ordered_ids, vectors)}
            self.embedding_model = embedding_model or getattr(
                self.backend, "model_name", "mock-hash-embedding-v1"
            )
            self.embedding_dim = len(vectors[0]) if vectors else None

    @classmethod
    def from_preloaded_vectors(
        cls,
        *,
        documents: Dict[str, Dict[str, Any]],
        metadata_by_id: Dict[str, Dict[str, Any]],
        runtime_operation_ids: Sequence[str],
        vectors: Dict[str, Sequence[float]],
        embedding_model: str,
        registry: Optional[Dict[str, OperationSpec]] = None,
    ) -> "OfflineEmbeddingIndex":
        return cls(
            documents=documents,
            metadata_by_id=metadata_by_id,
            runtime_operation_ids=runtime_operation_ids,
            preloaded_vectors=vectors,
            embedding_model=embedding_model,
            allow_query_embed=False,
            refuse_network=True,
            registry=registry,
        )

    def embed_query(self, text: str) -> List[float]:
        if not self.allow_query_embed or self.backend is None:
            raise RuntimeError(
                "Query embedding disabled for this index; pass precomputed query_vector"
            )
        return self.backend.embed([text])[0]


def _hard_exclusions(
    metadata_by_id: Dict[str, Dict[str, Any]],
    runtime_ids: Set[str],
) -> Tuple[Set[str], List[Dict[str, str]]]:
    """blocked / rejected / not runtime-exposed."""
    excluded: Set[str] = set()
    reasons: List[Dict[str, str]] = []
    for oid, meta in metadata_by_id.items():
        avail = str(meta.get("availability") or "").lower()
        role = str(meta.get("runtime_role") or meta.get("role") or "").lower()
        if oid not in runtime_ids:
            excluded.add(oid)
            reasons.append({"operation_id": oid, "reason": "not_runtime_exposed"})
            continue
        if avail in {"blocked", "rejected", "unavailable"} or role in {
            "blocked",
            "rejected",
        }:
            excluded.add(oid)
            reasons.append({"operation_id": oid, "reason": "blocked_or_rejected"})
            continue
        if meta.get("blocked") is True or meta.get("rejected") is True:
            excluded.add(oid)
            reasons.append({"operation_id": oid, "reason": "blocked_or_rejected"})
    return excluded, reasons


def assert_lexical_weight_frozen() -> None:
    if abs(float(LEXICAL_WEIGHT) - float(RUNTIME_LEXICAL_WEIGHT)) > 1e-12:
        raise RuntimeError(
            f"LEXICAL_WEIGHT={LEXICAL_WEIGHT} drifted from frozen "
            f"RUNTIME_LEXICAL_WEIGHT={RUNTIME_LEXICAL_WEIGHT}"
        )


def runtime_score_all_operations(
    *,
    query: str,
    query_vector: Sequence[float],
    vectors: Dict[str, Sequence[float]],
    registry: Dict[str, OperationSpec],
    operation_ids: Sequence[str],
) -> List[Dict[str, Any]]:
    """Mirror ``OperationRetriever.rank`` scoring (pre top-k cut).

    score = cosine + LEXICAL_WEIGHT * lexical_score
    sort: score desc, operation_id asc
    """
    assert_lexical_weight_frozen()
    scored: List[Dict[str, Any]] = []
    for oid in operation_ids:
        if oid not in vectors or oid not in registry:
            continue
        cos = cosine(query_vector, vectors[oid])
        lex = lexical_score(query, registry[oid])
        score = float(cos) + float(LEXICAL_WEIGHT) * float(lex)
        scored.append(
            {
                "operation_id": oid,
                "score": score,
                "cosine_similarity": cos,
                "lexical_score": lex,
                "lexical_weight": float(LEXICAL_WEIGHT),
            }
        )
    scored.sort(key=lambda r: (-float(r["score"]), str(r["operation_id"])))
    return scored


def apply_runtime_topk_then_hard_filter(
    *,
    scored_all: Sequence[Dict[str, Any]],
    allowed_ids: Set[str],
    top_k: int,
) -> List[Dict[str, Any]]:
    """Mirror executor: resolver returns Top-K from full registry, then
    ``select_embedding_topk`` keeps order among domain/runtime-allowed ids.
    """
    pre = []
    for row in list(scored_all)[: max(1, int(top_k))]:
        item = dict(row)
        # Match OperationRetriever.rank output rounding for emitted score.
        item["score"] = round(float(item["score"]), 6)
        pre.append(item)
    return select_embedding_topk(pre, allowed_ids=allowed_ids, top_k=top_k)


def rank_operations(
    *,
    index: OfflineEmbeddingIndex,
    query: str,
    variant: str,
    alpha: float = 1.0,
    step_domain: str = "general",
    required_outputs: Optional[Sequence[str]] = None,
    top_k: int = TOP_K,
    gold_ops_for_filter_diag: Optional[Sequence[str]] = None,
    query_vector: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """Rank catalog ops for one query.

    Variants:
      R0/R1: global cosine (no domain filter)
      R2: domain-filtered cosine
      R3: domain-filtered hybrid (alpha*cos + (1-alpha)*coverage)
      R4: global hybrid coverage (no domain filter; ablates coverage vs R1)
      R5: global cosine + fixed 0.12 lexical (no domain filter; ablates lexical vs R1)
      R_runtime: exact runtime path (cosine + 0.12*lexical, Top-K then hard filter)
    """
    if variant not in {"R0", "R1", "R2", "R3", "R4", "R5", "R_runtime"}:
        raise ValueError(f"Unknown ranking variant: {variant}")
    if variant in {"R3", "R4"} and alpha not in HYBRID_ALPHAS:
        raise ValueError(f"Unsupported alpha {alpha}; allowed={HYBRID_ALPHAS}")

    runtime = set(index.runtime_operation_ids)
    excluded, exclusion_reasons = _hard_exclusions(index.metadata_by_id, runtime)
    if query_vector is not None:
        qvec = [float(x) for x in query_vector]
    else:
        qvec = index.embed_query(query)

    if index.vectors:
        sample_oid = next(iter(index.vectors))
        try:
            assert_vector_dimensions_match(
                qvec, index.vectors[sample_oid], context=f"rank_operations:{variant}"
            )
        except EmbeddingDimensionMismatch:
            raise
        if getattr(index, "embedding_dim", None) not in (None, len(qvec)):
            raise EmbeddingDimensionMismatch(
                f"index.embedding_dim={index.embedding_dim} != query_dim={len(qvec)}"
            )

    use_domain = variant in {"R2", "R3"}
    use_hybrid = variant in {"R3", "R4"}
    use_global_lexical = variant == "R5"
    is_runtime = variant == "R_runtime"

    filter_info: Dict[str, Any] = {
        "domain_filter_applied": use_domain or is_runtime,
        "domain_filter_version": DOMAIN_FILTER_VERSION if (use_domain or is_runtime) else None,
        "domain_filter_timing": (
            "after_global_topk" if is_runtime else ("before_scoring" if use_domain else None)
        ),
        "step_domain": step_domain,
        "kept": [],
        "rejected": [],
        "candidates_before_filter": None,
        "candidates_after_filter": None,
        "gold_present_before_filter": None,
        "gold_present_after_filter": None,
        "filter_miss": False,
        "embedding_miss": False,
        "output_coverage_version": OUTPUT_COVERAGE_VERSION if use_hybrid else None,
    }

    if is_runtime:
        registry = index.registry or {}
        if not registry:
            raise RuntimeError(
                "R_runtime requires OfflineEmbeddingIndex.registry (OperationSpec map)"
            )
        # Hard filter set (same as executor) — applied AFTER global Top-K.
        filt = hard_filter_operations(
            registry_ids=list(registry.keys()),
            metadata_by_id=index.metadata_by_id,
            runtime_operation_ids=list(runtime),
            step_domain=step_domain,
        )
        allowed = set(filt["kept"])
        filter_info["kept"] = list(filt["kept"])
        filter_info["rejected"] = list(filt["rejected"])
        filter_info["candidates_before_filter"] = len(registry)
        filter_info["candidates_after_filter"] = len(allowed)

        score_ids = [oid for oid in registry.keys() if oid in index.vectors]
        scored_all = runtime_score_all_operations(
            query=query,
            query_vector=qvec,
            vectors=index.vectors,
            registry=registry,
            operation_ids=sorted(score_ids),
        )
        top = apply_runtime_topk_then_hard_filter(
            scored_all=scored_all,
            allowed_ids=allowed,
            top_k=top_k,
        )
        # Enrich domain for diagnostics.
        for row in top:
            meta = index.metadata_by_id.get(str(row["operation_id"])) or {}
            row["domain"] = meta.get("domain")
            row["required_output_coverage"] = None

        gold = [str(x) for x in (gold_ops_for_filter_diag or [])]
        before = any(g in score_ids for g in gold) if gold else None
        after = any(g in allowed for g in gold) if gold else None
        filter_info["gold_present_before_filter"] = before
        filter_info["gold_present_after_filter"] = after
        filter_info["filter_miss"] = bool(gold) and before is True and after is False

        ranked_ids = [r["operation_id"] for r in top]
        if gold and not any(g in ranked_ids for g in gold):
            survived = not (
                filter_info.get("filter_miss")
                or filter_info.get("gold_present_after_filter") is False
            )
            filter_info["embedding_miss"] = bool(survived and gold)

        blocked_leakage = [r for r in ranked_ids if r in excluded]
        return {
            "variant": variant,
            "alpha": None,
            "lexical_weight": float(LEXICAL_WEIGHT),
            "query": query,
            "top_k": top_k,
            "describe_n": DESCRIBE_N,
            "ranked": top,
            "ranked_ids": ranked_ids,
            "filter": filter_info,
            "hard_exclusions": exclusion_reasons[:20],
            "blocked_leakage": blocked_leakage,
            "n_candidates_scored": len(scored_all),
            "runtime_parity": True,
        }

    candidate_ids = [oid for oid in index.ordered_ids if oid not in excluded]
    if use_domain:
        filt = hard_filter_operations(
            registry_ids=candidate_ids,
            metadata_by_id=index.metadata_by_id,
            runtime_operation_ids=list(runtime),
            step_domain=step_domain,
        )
        kept = list(filt["kept"])
        filter_info["kept"] = kept
        filter_info["rejected"] = list(filt["rejected"])
        filter_info["candidates_before_filter"] = len(candidate_ids)
        filter_info["candidates_after_filter"] = len(kept)
        gold = [str(x) for x in (gold_ops_for_filter_diag or [])]
        before = any(g in candidate_ids for g in gold) if gold else None
        after = any(g in kept for g in gold) if gold else None
        filter_info["gold_present_before_filter"] = before
        filter_info["gold_present_after_filter"] = after
        filter_info["filter_miss"] = bool(gold) and before is True and after is False
        pool = kept
    else:
        pool = candidate_ids
        filter_info["kept"] = pool
        filter_info["candidates_before_filter"] = len(candidate_ids)
        filter_info["candidates_after_filter"] = len(pool)

    if use_global_lexical:
        assert_lexical_weight_frozen()
        registry = index.registry or {}
        if not registry:
            raise RuntimeError(
                "R5 requires OfflineEmbeddingIndex.registry for lexical_score"
            )

    scored: List[Dict[str, Any]] = []
    for oid in pool:
        meta = index.metadata_by_id.get(oid) or {}
        cos = cosine(qvec, index.vectors[oid])
        provides = list(meta.get("provides") or [])
        cov = required_output_coverage(
            required_outputs=required_outputs or [],
            provides=provides,
        )
        lex = None
        if use_global_lexical:
            if oid not in registry:
                raise RuntimeError(f"R5 missing OperationSpec for {oid}")
            lex = float(lexical_score(query, registry[oid]))
            score = float(cos) + float(LEXICAL_WEIGHT) * lex
            cov_out = None
        elif use_hybrid:
            score = float(alpha) * cos + (1.0 - float(alpha)) * cov
            cov_out = cov
        else:
            score = cos
            cov_out = None if variant in {"R0", "R1", "R2"} else cov
        row = {
            "operation_id": oid,
            "score": score,
            "cosine_similarity": cos,
            "required_output_coverage": cov_out,
            "lexical_score": lex,
            "lexical_weight": float(LEXICAL_WEIGHT) if use_global_lexical else None,
            "domain": meta.get("domain"),
            "provides_empty": (not provides),
        }
        scored.append(row)
    scored.sort(key=lambda r: (-float(r["score"]), str(r["operation_id"])))
    for i, row in enumerate(scored):
        row["final_rank"] = i + 1
        row["final_score"] = float(row["score"])
    top = scored[: max(1, int(top_k))]
    ranked_ids = [r["operation_id"] for r in top]
    gold = [str(x) for x in (gold_ops_for_filter_diag or [])]
    if gold and not any(g in ranked_ids for g in gold):
        survived = True
        if use_domain and filter_info.get("filter_miss"):
            survived = False
        elif use_domain and gold and filter_info.get("gold_present_after_filter") is False:
            survived = False
        filter_info["embedding_miss"] = bool(survived and gold)

    blocked_leakage = [r for r in ranked_ids if r in excluded]

    return {
        "variant": variant,
        "alpha": float(alpha) if use_hybrid else None,
        "lexical_weight": float(LEXICAL_WEIGHT) if use_global_lexical else None,
        "query": query,
        "top_k": top_k,
        "describe_n": DESCRIBE_N,
        "ranked": top,
        "ranked_ids": ranked_ids,
        "ranked_full": scored,
        "filter": filter_info,
        "hard_exclusions": exclusion_reasons[:20],
        "blocked_leakage": blocked_leakage,
        "n_candidates_scored": len(scored),
        "runtime_parity": False,
    }


def build_question_query(question: str) -> str:
    return str(question or "").strip()


def build_step_query(step: Dict[str, Any]) -> str:
    return build_step_search_query(step)
