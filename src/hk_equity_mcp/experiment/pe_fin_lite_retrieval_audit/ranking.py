"""Offline ranking variants R0–R3 (no network in mock/hash mode)."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from hk_equity_mcp.experiment.pe_fin_lite_retrieval_audit import (
    DESCRIBE_N,
    DOMAIN_FILTER_VERSION,
    HYBRID_ALPHAS,
    TOP_K,
)
from hk_equity_mcp.operation_resolver.retrieval import DeterministicHashEmbeddingBackend
from hk_equity_mcp.pe_fin_lite.tool_retrieval import (
    build_step_search_query,
    domain_compatible,
    hard_filter_operations,
)


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return float(dot / (na * nb))


def required_output_coverage(
    *,
    required_outputs: Sequence[str],
    provides: Sequence[str],
) -> float:
    req = [str(x).lower() for x in (required_outputs or []) if str(x).strip()]
    if not req:
        return 1.0
    prov = {str(x).lower() for x in (provides or [])}
    hit = sum(1 for r in req if r in prov or any(r in p or p in r for p in prov))
    return hit / float(len(req))


class OfflineEmbeddingIndex:
    """In-memory operation embedding index (hash backend by default; offline)."""

    def __init__(
        self,
        *,
        documents: Dict[str, Dict[str, Any]],
        metadata_by_id: Dict[str, Dict[str, Any]],
        runtime_operation_ids: Sequence[str],
        embedding_backend: Optional[Any] = None,
        allow_network: bool = False,
    ):
        if allow_network:
            raise RuntimeError(
                "OfflineEmbeddingIndex refuse network this round; use hash backend"
            )
        self.backend = embedding_backend or DeterministicHashEmbeddingBackend()
        self.documents = documents
        self.metadata_by_id = metadata_by_id
        self.runtime_operation_ids = list(runtime_operation_ids)
        self.ordered_ids = sorted(documents.keys())
        texts = [documents[oid]["document_text"] for oid in self.ordered_ids]
        vectors = self.backend.embed(texts)
        self.vectors = {oid: vec for oid, vec in zip(self.ordered_ids, vectors)}
        self.embedding_model = getattr(self.backend, "model_name", "mock-hash-embedding-v1")
        self.embedding_dim = len(vectors[0]) if vectors else None

    def embed_query(self, text: str) -> List[float]:
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
) -> Dict[str, Any]:
    """Rank catalog ops for one query.

    Variants:
      R0/R1: global cosine (no domain filter)
      R2: domain-filtered cosine
      R3: domain-filtered hybrid
    """
    if variant not in {"R0", "R1", "R2", "R3"}:
        raise ValueError(f"Unknown ranking variant: {variant}")
    if variant == "R3" and alpha not in HYBRID_ALPHAS:
        raise ValueError(f"Unsupported alpha {alpha}; allowed={HYBRID_ALPHAS}")

    runtime = set(index.runtime_operation_ids)
    excluded, exclusion_reasons = _hard_exclusions(index.metadata_by_id, runtime)
    qvec = index.embed_query(query)

    use_domain = variant in {"R2", "R3"}
    use_hybrid = variant == "R3"

    filter_info: Dict[str, Any] = {
        "domain_filter_applied": use_domain,
        "domain_filter_version": DOMAIN_FILTER_VERSION if use_domain else None,
        "step_domain": step_domain,
        "kept": [],
        "rejected": [],
        "gold_present_before_filter": None,
        "gold_present_after_filter": None,
        "filter_miss": False,
        "embedding_miss": False,
    }

    candidate_ids = [oid for oid in index.ordered_ids if oid not in excluded]
    if use_domain:
        filt = hard_filter_operations(
            registry_ids=candidate_ids,
            metadata_by_id=index.metadata_by_id,
            runtime_operation_ids=list(runtime),
            step_domain=step_domain,
        )
        # hard_filter also drops non-runtime; already excluded.
        kept = list(filt["kept"])
        filter_info["kept"] = kept
        filter_info["rejected"] = list(filt["rejected"])
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

    scored: List[Dict[str, Any]] = []
    for oid in pool:
        meta = index.metadata_by_id.get(oid) or {}
        cos = cosine(qvec, index.vectors[oid])
        cov = required_output_coverage(
            required_outputs=required_outputs or [],
            provides=list(meta.get("provides") or []),
        )
        if use_hybrid:
            score = float(alpha) * cos + (1.0 - float(alpha)) * cov
        else:
            score = cos
            cov = None if variant in {"R0", "R1", "R2"} else cov
        scored.append(
            {
                "operation_id": oid,
                "score": score,
                "cosine_similarity": cos,
                "required_output_coverage": cov,
                "domain": meta.get("domain"),
            }
        )
    scored.sort(key=lambda r: (-float(r["score"]), str(r["operation_id"])))
    top = scored[: max(1, int(top_k))]
    ranked_ids = [r["operation_id"] for r in top]
    gold = [str(x) for x in (gold_ops_for_filter_diag or [])]
    if gold and not any(g in ranked_ids for g in gold):
        # embedding miss if gold survived filter (or no filter) but not in Top-K
        survived = True
        if use_domain and filter_info.get("filter_miss"):
            survived = False
        elif use_domain and gold and filter_info.get("gold_present_after_filter") is False:
            survived = False
        filter_info["embedding_miss"] = bool(survived and gold)

    # Blocked leakage: excluded ids must never appear.
    blocked_leakage = [r for r in ranked_ids if r in excluded]

    return {
        "variant": variant,
        "alpha": alpha if use_hybrid else None,
        "query": query,
        "top_k": top_k,
        "describe_n": DESCRIBE_N,
        "ranked": top,
        "ranked_ids": ranked_ids,
        "filter": filter_info,
        "hard_exclusions": exclusion_reasons[:20],
        "blocked_leakage": blocked_leakage,
        "n_candidates_scored": len(scored),
    }


def build_question_query(question: str) -> str:
    return str(question or "").strip()


def build_step_query(step: Dict[str, Any]) -> str:
    return build_step_search_query(step)
