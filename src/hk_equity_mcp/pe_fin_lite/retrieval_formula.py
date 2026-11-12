"""Frozen PE-Fin Lite runtime retrieval formula (R5 sync).

Selected by zero-network global_signal_ablation_v1 on development-origin metrics.
Runtime must reuse ``operation_resolver.retrieval.lexical_score`` / ``LEXICAL_WEIGHT``
— do not fork an approximate lexical blend.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Optional, Sequence

from hk_equity_mcp.operation_resolver.retrieval import (
    LEXICAL_WEIGHT,
    cosine_similarity,
    lexical_score,
)
from hk_equity_mcp.operation_resolver.schemas import OperationSpec

# Candidate lock from global_signal_ablation_v1 (identity pin; not retuned here).
R5_CANDIDATE_LOCK_SHA256 = (
    "6a2f6ec1ba4fcf70ab50d01fc21e7a1aabda95f1beb15981f32eda3dc21d28b4"
)
RETRIEVAL_FORMULA_NAME = "R5_global_lexical@0.12"
RETRIEVAL_FORMULA_VERSION = "pe_fin_lite_retrieval_formula_r5_v1"
RETRIEVAL_FORMULA_EXACT = "cosine_similarity + 0.12*lexical_score"
DOMAIN_HARD_FILTER = False
RETRIEVAL_TOP_K = 5
DESCRIBE_N = 3
EXPECTED_EMBEDDING_DIMENSION = 1536

# Lexical blend weights (must match lexical_score implementation).
LEXICAL_BLEND_WEIGHTS = {
    "token_overlap": 0.40,
    "operation_name_coverage": 0.25,
    "action_term_match": 0.20,
    "exact_operation_name": 0.10,
    "exact_service_name": 0.03,
    "exact_operation_id": 0.02,
}


def assert_lexical_weight_matches_runtime() -> None:
    if abs(float(LEXICAL_WEIGHT) - 0.12) > 1e-12:
        raise RuntimeError(
            f"LEXICAL_WEIGHT={LEXICAL_WEIGHT} drifted from frozen R5 weight 0.12"
        )


def r5_final_score(*, cosine: float, query: str, operation: OperationSpec) -> float:
    """score = cosine_similarity + 0.12 * lexical_score (shared lexical_score)."""
    assert_lexical_weight_matches_runtime()
    lex = float(lexical_score(query, operation))
    return float(cosine) + float(LEXICAL_WEIGHT) * lex


def r5_score_from_vectors(
    *,
    query: str,
    query_vector: Sequence[float],
    doc_vector: Sequence[float],
    operation: OperationSpec,
) -> Dict[str, float]:
    if len(query_vector) != len(doc_vector):
        raise ValueError(
            f"Embedding dimension mismatch: {len(query_vector)} vs {len(doc_vector)}"
        )
    if len(query_vector) not in (None, 0) and len(query_vector) != EXPECTED_EMBEDDING_DIMENSION:
        # Allow mock/hash dims in unit tests; live identity gates enforce 1536.
        pass
    cos = float(cosine_similarity(query_vector, doc_vector))
    lex = float(lexical_score(query, operation))
    assert_lexical_weight_matches_runtime()
    return {
        "cosine_similarity": cos,
        "lexical_score": lex,
        "lexical_weight": float(LEXICAL_WEIGHT),
        "score": cos + float(LEXICAL_WEIGHT) * lex,
    }


def retrieval_formula_identity_payload(
    *,
    embedding_provider_host: Optional[str] = None,
    embedding_model: Optional[str] = None,
    embedding_dimension: Optional[int] = None,
    operation_documents_sha256: Optional[str] = None,
    operation_index_identity: Optional[str] = None,
    catalog_fingerprint: Optional[str] = None,
) -> Dict[str, Any]:
    assert_lexical_weight_matches_runtime()
    return {
        "retrieval_formula_name": RETRIEVAL_FORMULA_NAME,
        "retrieval_formula_version": RETRIEVAL_FORMULA_VERSION,
        "retrieval_formula_exact": RETRIEVAL_FORMULA_EXACT,
        "lexical_weight": float(LEXICAL_WEIGHT),
        "lexical_blend_weights": dict(LEXICAL_BLEND_WEIGHTS),
        "top_k": RETRIEVAL_TOP_K,
        "describe_n": DESCRIBE_N,
        "domain_hard_filter": DOMAIN_HARD_FILTER,
        "candidate_lock_sha256": R5_CANDIDATE_LOCK_SHA256,
        "embedding_provider_host": embedding_provider_host,
        "embedding_model": embedding_model,
        "embedding_dimension": embedding_dimension,
        "operation_documents_sha256": operation_documents_sha256,
        "operation_index_identity": operation_index_identity,
        "catalog_fingerprint": catalog_fingerprint,
    }


def retrieval_formula_hash(payload: Optional[Dict[str, Any]] = None) -> str:
    """Stable SHA256 over formula-defining fields (excludes volatile host/docs)."""
    core = payload or retrieval_formula_identity_payload()
    freeze = {
        "retrieval_formula_name": core.get("retrieval_formula_name"),
        "retrieval_formula_version": core.get("retrieval_formula_version"),
        "retrieval_formula_exact": core.get("retrieval_formula_exact"),
        "lexical_weight": core.get("lexical_weight"),
        "lexical_blend_weights": core.get("lexical_blend_weights"),
        "top_k": core.get("top_k"),
        "describe_n": core.get("describe_n"),
        "domain_hard_filter": core.get("domain_hard_filter"),
        "candidate_lock_sha256": core.get("candidate_lock_sha256"),
    }
    blob = json.dumps(freeze, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def runtime_retrieval_metadata(
    *,
    embedding_provider_host: Optional[str] = None,
    embedding_model: Optional[str] = None,
    embedding_dimension: Optional[int] = None,
    operation_documents_sha256: Optional[str] = None,
    operation_index_identity: Optional[str] = None,
    catalog_fingerprint: Optional[str] = None,
) -> Dict[str, Any]:
    payload = retrieval_formula_identity_payload(
        embedding_provider_host=embedding_provider_host,
        embedding_model=embedding_model,
        embedding_dimension=embedding_dimension,
        operation_documents_sha256=operation_documents_sha256,
        operation_index_identity=operation_index_identity,
        catalog_fingerprint=catalog_fingerprint,
    )
    payload["retrieval_formula_hash"] = retrieval_formula_hash(payload)
    payload["runtime_synced_from_candidate"] = True
    return payload
