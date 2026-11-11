"""Fail-closed embedding identity / dimension gates (offline; no API)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import List

import pytest

from hk_equity_mcp.experiment.pe_fin_lite_retrieval_audit.embedding_identity_gate import (
    EmbeddingDimensionMismatch,
    MOCK_HASH_DIM,
    REAL_TEXT_EMBEDDING_DIM,
    assert_query_catalog_alignment,
    assert_real_backend_identity,
    assert_vector_dimensions_match,
    classify_catalog_source,
    list_catalog_cache_inventory,
    resolve_embedding_meta_for_offline,
    select_matching_catalog_cache,
)
from hk_equity_mcp.experiment.pe_fin_lite_retrieval_audit.live_path import LivePathError
from hk_equity_mcp.experiment.pe_fin_lite_retrieval_audit.ranking import (
    OfflineEmbeddingIndex,
    cosine,
    rank_operations,
)

ROOT = Path(__file__).resolve().parents[1]
PLANNER_V3 = ROOT / "outputs" / "pe_fin_lite_retrieval_all155_planner_v3"
QQ_VECTORS = (
    PLANNER_V3 / "question_query_cache" / "question_query_vectors.json"
)
EXPECTED_QQ_SHA = (
    "4082f51fa357c8bdb805c77fc3018dcc170f1f7a6b4c3fbeef3a944bca1a0c5d"
)
EXPECTED_REAL_CATALOG = (
    "ac4c109bed0edb21fb2b837dfdbd745680b24f33efa62e13ad6bca53f174b0df"
)
WRONG_MOCK_CATALOG = (
    "1caff45e2b73b0a929172cd2e9a5727d0cb20c79c5b262aff50d2f303aa92843"
)


def _fake_vec(dim: int, seed: float = 1.0) -> List[float]:
    return [float((i + 1) * seed) for i in range(dim)]


def test_cosine_rejects_dimension_mismatch_no_truncate():
    a = _fake_vec(1536)
    b = _fake_vec(64)
    with pytest.raises(EmbeddingDimensionMismatch, match="fail-closed"):
        cosine(a, b)
    with pytest.raises(EmbeddingDimensionMismatch):
        assert_vector_dimensions_match(a, b, context="unit")


def test_cosine_allows_matching_dims():
    a = _fake_vec(8, 1.0)
    b = _fake_vec(8, 2.0)
    score = cosine(a, b)
    assert isinstance(score, float)
    assert -1.0 <= score <= 1.0


def test_matching_1536_ranking_allowed():
    docs = {f"op{i}": {"document_text": f"doc{i}"} for i in range(3)}
    meta = {f"op{i}": {"availability": "available"} for i in range(3)}
    vectors = {f"op{i}": _fake_vec(REAL_TEXT_EMBEDDING_DIM, float(i + 1)) for i in range(3)}
    index = OfflineEmbeddingIndex.from_preloaded_vectors(
        documents=docs,
        metadata_by_id=meta,
        runtime_operation_ids=list(docs),
        vectors=vectors,
        embedding_model="text-embedding-3-small",
    )
    index.embedding_provider_host = "api.zhizengzeng.com"
    index.catalog_source = "real"
    assert_real_backend_identity(
        embedding_backend="real_text_embedding_3_small",
        embedding_provider_host="api.zhizengzeng.com",
        vector_dimension=REAL_TEXT_EMBEDDING_DIM,
        source="real",
    )
    out = rank_operations(
        index=index,
        query="ignored",
        variant="R0",
        query_vector=_fake_vec(REAL_TEXT_EMBEDDING_DIM, 9.0),
    )
    assert len(out["ranked_ids"]) >= 1


def test_mock_64_ranking_allowed_only_with_matching_dims():
    docs = {"op0": {"document_text": "a"}, "op1": {"document_text": "b"}}
    meta = {"op0": {"availability": "available"}, "op1": {"availability": "available"}}
    vectors = {
        "op0": _fake_vec(MOCK_HASH_DIM, 1.0),
        "op1": _fake_vec(MOCK_HASH_DIM, 2.0),
    }
    index = OfflineEmbeddingIndex.from_preloaded_vectors(
        documents=docs,
        metadata_by_id=meta,
        runtime_operation_ids=list(docs),
        vectors=vectors,
        embedding_model="mock-hash-embedding-v1",
    )
    out = rank_operations(
        index=index,
        query="ignored",
        variant="R0",
        query_vector=_fake_vec(MOCK_HASH_DIM, 3.0),
    )
    assert out["ranked_ids"]
    with pytest.raises(EmbeddingDimensionMismatch):
        rank_operations(
            index=index,
            query="ignored",
            variant="R0",
            query_vector=_fake_vec(REAL_TEXT_EMBEDDING_DIM, 3.0),
        )


def test_real_backend_rejects_host_null_and_dim64():
    with pytest.raises(LivePathError, match="host=null"):
        assert_real_backend_identity(
            embedding_backend="real_text_embedding_3_small",
            embedding_provider_host=None,
            vector_dimension=64,
        )
    with pytest.raises(LivePathError, match="mock/hash"):
        assert_real_backend_identity(
            embedding_backend="real_text_embedding_3_small",
            embedding_provider_host="api.zhizengzeng.com",
            vector_dimension=MOCK_HASH_DIM,
            source="mock_hash",
        )
    with pytest.raises(LivePathError, match="64"):
        assert_real_backend_identity(
            embedding_backend="real_text_embedding_3_small",
            embedding_provider_host="api.zhizengzeng.com",
            vector_dimension=MOCK_HASH_DIM,
            source="unknown",
        )


def test_classify_and_select_exact_identity(tmp_path: Path):
    inv = [
        {
            "identity_digest": "aaa",
            "embedding_provider_host": None,
            "embedding_model": "text-embedding-3-small",
            "vector_dimension": 64,
            "source": "mock_hash",
            "documents_hash": "d1",
            "registry_fingerprint": "r1",
            "catalog_fingerprint": "c1",
        },
        {
            "identity_digest": "bbb",
            "embedding_provider_host": "api.zhizengzeng.com",
            "embedding_model": "text-embedding-3-small",
            "vector_dimension": 1536,
            "source": "real",
            "documents_hash": "d1",
            "registry_fingerprint": "r1",
            "catalog_fingerprint": "c1",
        },
    ]
    assert classify_catalog_source(
        embedding_provider_host=None, vector_dimension=64
    ) == "mock_hash"
    chosen = select_matching_catalog_cache(
        inventory=inv,
        expected_identity={
            "embedding_provider_host": "api.zhizengzeng.com",
            "embedding_model": "text-embedding-3-small",
            "documents_hash": "d1",
            "registry_fingerprint": "r1",
            "catalog_fingerprint": "c1",
        },
        expected_digest="bbb",
        require_real=True,
        require_dim=1536,
    )
    assert chosen["identity_digest"] == "bbb"
    with pytest.raises(LivePathError):
        select_matching_catalog_cache(
            inventory=inv,
            expected_identity={"embedding_provider_host": None},
            expected_digest="aaa",
            require_real=True,
            require_dim=64,
        )


def test_multiple_catalog_matches_fail_closed():
    inv = [
        {"identity_digest": "x", "source": "real", "vector_dimension": 1536,
         "embedding_provider_host": "h", "embedding_model": "m"},
        {"identity_digest": "x", "source": "real", "vector_dimension": 1536,
         "embedding_provider_host": "h", "embedding_model": "m"},
    ]
    with pytest.raises(LivePathError, match="Multiple"):
        select_matching_catalog_cache(
            inventory=inv,
            expected_identity={},
            expected_digest="x",
            require_real=True,
        )


def test_offline_meta_requires_env_or_query_identity(tmp_path: Path):
    with pytest.raises(LivePathError, match="embedding-env-file"):
        resolve_embedding_meta_for_offline(
            root=tmp_path, embedding_env_file=None, query_cache_identity=None
        )
    meta = resolve_embedding_meta_for_offline(
        root=tmp_path,
        embedding_env_file=None,
        query_cache_identity={
            "embedding_provider_host": "api.zhizengzeng.com",
            "embedding_model": "text-embedding-3-small",
            "embedding_dim": 1536,
        },
    )
    assert meta["host"] == "api.zhizengzeng.com"
    assert meta["identity_source"] == "query_cache_identity"


def test_query_catalog_alignment_gates():
    assert_query_catalog_alignment(
        query_dim=1536,
        catalog_dim=1536,
        query_host="api.zhizengzeng.com",
        catalog_host="api.zhizengzeng.com",
        query_model="text-embedding-3-small",
        catalog_model="text-embedding-3-small",
    )
    with pytest.raises(EmbeddingDimensionMismatch):
        assert_query_catalog_alignment(
            query_dim=1536,
            catalog_dim=64,
            query_host="api.zhizengzeng.com",
            catalog_host="api.zhizengzeng.com",
            query_model="text-embedding-3-small",
            catalog_model="text-embedding-3-small",
        )
    with pytest.raises(LivePathError, match="host"):
        assert_query_catalog_alignment(
            query_dim=1536,
            catalog_dim=1536,
            query_host="api.zhizengzeng.com",
            catalog_host=None,
            query_model="text-embedding-3-small",
            catalog_model="text-embedding-3-small",
        )


def test_mock_mode_64_classified_as_mock_hash():
    assert (
        classify_catalog_source(
            embedding_provider_host="local_hash",
            vector_dimension=64,
            embedding_model="mock-hash-embedding-v1",
        )
        in {"mock_hash", "unknown"}
    )
    # host present but dim 64 still mock_hash via dim rule
    assert (
        classify_catalog_source(
            embedding_provider_host=None, vector_dimension=MOCK_HASH_DIM
        )
        == "mock_hash"
    )


@pytest.mark.skipif(not PLANNER_V3.exists(), reason="planner_v3 outputs missing")
def test_invalid_r0_marker_present_and_raw_unchanged():
    marker = PLANNER_V3 / "question_level" / "INVALIDATED_R0_MIXED_EMBEDDING_DIMENSIONS.json"
    assert marker.exists()
    payload = json.loads(marker.read_text(encoding="utf-8"))
    assert (
        payload["marker"]
        == "invalidated_retrieval_r0_due_to_mixed_embedding_dimensions_and_mock_catalog_selection"
    )
    assert payload["usable_for_report"] is False
    assert payload["usable_for_formula_selection"] is False
    assert payload["query"]["embedding_dim"] == 1536
    assert payload["selected_wrong_catalog"]["vector_dimension"] == 64
    assert payload["selected_wrong_catalog"]["identity_sha256"].startswith("1caff45e")
    assert payload["expected_real_catalog"]["identity_sha256"].startswith("ac4c109b")

    # Raw ranking artifacts must keep pre-invalidation hashes.
    frozen = payload["raw_hashes_at_invalidation"]
    for rel, info in frozen.items():
        if not str(rel).startswith("question_level/"):
            continue
        path = PLANNER_V3 / rel
        assert path.exists()
        blob = path.read_bytes()
        assert hashlib.sha256(blob).hexdigest() == info["sha256"]
        assert len(blob) == info["bytes"]


@pytest.mark.skipif(not QQ_VECTORS.exists(), reason="QQ cache missing")
def test_valid_question_query_cache_sha_unchanged():
    digest = hashlib.sha256(QQ_VECTORS.read_bytes()).hexdigest()
    assert digest == EXPECTED_QQ_SHA
    # Marker must also record the same reusable QQ sha.
    marker = json.loads(
        (
            PLANNER_V3
            / "question_level"
            / "INVALIDATED_R0_MIXED_EMBEDDING_DIMENSIONS.json"
        ).read_text(encoding="utf-8")
    )
    assert marker["query"]["vectors_sha256"] == EXPECTED_QQ_SHA
    assert marker["query"]["reusable"] is True


@pytest.mark.skipif(
    not (ROOT / ".cache" / "pe_fin_lite_operation_embeddings_v1").exists(),
    reason="embedding cache dir missing",
)
def test_inventory_lists_real_and_mock_caches():
    cache = ROOT / ".cache" / "pe_fin_lite_operation_embeddings_v1"
    inv = list_catalog_cache_inventory(cache)
    digests = {row["identity_digest"] for row in inv}
    assert EXPECTED_REAL_CATALOG in digests
    assert WRONG_MOCK_CATALOG in digests
    by = {row["identity_digest"]: row for row in inv}
    assert by[EXPECTED_REAL_CATALOG]["source"] == "real"
    assert by[EXPECTED_REAL_CATALOG]["vector_dimension"] == REAL_TEXT_EMBEDDING_DIM
    assert by[WRONG_MOCK_CATALOG]["source"] == "mock_hash"
    assert by[WRONG_MOCK_CATALOG]["vector_dimension"] == MOCK_HASH_DIM
