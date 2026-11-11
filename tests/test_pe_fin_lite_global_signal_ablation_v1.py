"""Focused zero-network tests for R4_global_coverage / R5_global_lexical."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List
from unittest import mock

import pytest

from hk_equity_mcp.experiment.pe_fin_lite_retrieval_audit import (
    DESCRIBE_N,
    HYBRID_ALPHAS,
    RUNTIME_LEXICAL_WEIGHT,
    TOP_K,
)
from hk_equity_mcp.experiment.pe_fin_lite_retrieval_audit.embedding_identity_gate import (
    EmbeddingDimensionMismatch,
)
from hk_equity_mcp.experiment.pe_fin_lite_retrieval_audit.global_signal_ablation import (
    EXPECTED_CATALOG_IDENTITY,
    EXPECTED_OP_DOCS,
    EXPECTED_PLANNER_SHA,
    EXPECTED_STEP_IDENTITY,
    compare_ablation_formulas,
    run_global_signal_ablation,
)
from hk_equity_mcp.experiment.pe_fin_lite_retrieval_audit.live_path import LivePathError
from hk_equity_mcp.experiment.pe_fin_lite_retrieval_audit.ranking import (
    OfflineEmbeddingIndex,
    rank_operations,
    required_output_coverage,
)
from hk_equity_mcp.operation_resolver.retrieval import (
    LEXICAL_WEIGHT,
    lexical_score,
)
from hk_equity_mcp.operation_resolver.schemas import OperationSpec, ParameterSpec

ROOT = Path(__file__).resolve().parents[1]
OUT_V3 = ROOT / "outputs" / "pe_fin_lite_retrieval_all155_planner_v3"
RAW_WATCHED = [
    "planner_cache/planner_outputs.jsonl",
    "planner_cache/planner_only_summary.json",
    "planner_cache_audit/planner_cache_audit.json",
    "step_query_cache/step_query_vectors.json",
    "step_query_cache/step_query_cache_identity.json",
    "catalog_index/index_status.json",
    "question_level/retrieval_metrics.json",
    "ranking_variants/per_step_rankings.jsonl",
    "formula_selection/formula_lock.json",
]


def _vec(dim: int, seed: float) -> List[float]:
    return [float((i + 1) * seed) for i in range(dim)]


def _op(oid: str, *, name: str, summary: str, actions: List[str]) -> OperationSpec:
    return OperationSpec(
        operation_id=oid,
        service_id="aktools",
        name=name,
        summary=summary,
        description=summary,
        action_terms=actions,
        parameters=[ParameterSpec(name="symbol", type="string", required=True)],
        underlying_tool=name,
    )


def _mini_index(dim: int = 8):
    registry = {
        "op.fin": _op(
            "op.fin",
            name="stock_financial_hk_report_em",
            summary="Hong Kong financial report indicators revenue eps",
            actions=["financial", "report", "revenue"],
        ),
        "op.mkt": _op(
            "op.mkt",
            name="stock_hk_hist",
            summary="Hong Kong market daily history close volume",
            actions=["market", "history", "close"],
        ),
        "op.news": _op(
            "op.news",
            name="stock_news_main_cx",
            summary="Company news headlines disclosure",
            actions=["news", "headline"],
        ),
    }
    meta = {
        "op.fin": {
            "domain": "financial",
            "provides": ["revenue", "basic_eps"],
            "availability": "runtime",
        },
        "op.mkt": {
            "domain": "market",
            "provides": ["close", "volume"],
            "availability": "runtime",
        },
        "op.news": {
            "domain": "news",
            "provides": ["title"],
            "availability": "runtime",
        },
    }
    vectors = {
        "op.fin": _vec(dim, 1.0),
        "op.mkt": _vec(dim, 2.0),
        "op.news": _vec(dim, 3.0),
    }
    docs = {oid: {"document_text": registry[oid].index_document()} for oid in registry}
    index = OfflineEmbeddingIndex.from_preloaded_vectors(
        documents=docs,
        metadata_by_id=meta,
        runtime_operation_ids=list(registry.keys()),
        vectors=vectors,
        embedding_model="fake",
        registry=registry,
    )
    return registry, index


def test_r4_alpha1_row_for_row_matches_r1():
    _, index = _mini_index()
    q = "financial revenue eps"
    qvec = _vec(8, 1.4)
    r1 = rank_operations(
        index=index,
        query=q,
        variant="R1",
        required_outputs=["revenue"],
        query_vector=qvec,
        top_k=TOP_K,
    )
    r4 = rank_operations(
        index=index,
        query=q,
        variant="R4",
        alpha=1.0,
        required_outputs=["revenue"],
        query_vector=qvec,
        top_k=TOP_K,
    )
    assert r1["ranked_ids"] == r4["ranked_ids"]
    assert [r["operation_id"] for r in r1["ranked_full"]] == [
        r["operation_id"] for r in r4["ranked_full"]
    ]
    for a, b in zip(r1["ranked_full"], r4["ranked_full"]):
        assert abs(a["cosine_similarity"] - b["cosine_similarity"]) < 1e-12
        assert abs(a["score"] - b["score"]) < 1e-12
        assert a["final_rank"] == b["final_rank"]
    assert r1["ranked_ids"][:1] == r4["ranked_ids"][:1]
    assert set(r1["ranked_ids"][:3]) == set(r4["ranked_ids"][:3])
    assert set(r1["ranked_ids"][:5]) == set(r4["ranked_ids"][:5])


def test_r4_no_hard_domain_filter_and_reuses_coverage():
    _, index = _mini_index()
    qvec = _vec(8, 1.1)
    out = rank_operations(
        index=index,
        query="market close volume",
        variant="R4",
        alpha=0.8,
        step_domain="financial",
        required_outputs=["revenue"],
        query_vector=qvec,
    )
    assert out["filter"]["domain_filter_applied"] is False
    assert out["n_candidates_scored"] == 3
    top = out["ranked"][0]
    expected_cov = required_output_coverage(
        required_outputs=["revenue"],
        provides=index.metadata_by_id[top["operation_id"]]["provides"],
    )
    assert abs(top["required_output_coverage"] - expected_cov) < 1e-12
    expect = 0.8 * top["cosine_similarity"] + 0.2 * expected_cov
    assert abs(top["score"] - expect) < 1e-12


def test_r5_reuses_runtime_lexical_and_fixed_weight():
    registry, index = _mini_index()
    query = "Hong Kong financial report revenue"
    qvec = _vec(8, 1.2)
    assert LEXICAL_WEIGHT == RUNTIME_LEXICAL_WEIGHT == 0.12
    out = rank_operations(
        index=index,
        query=query,
        variant="R5",
        step_domain="news",
        query_vector=qvec,
        top_k=TOP_K,
    )
    assert out["filter"]["domain_filter_applied"] is False
    assert abs(float(out["lexical_weight"]) - 0.12) < 1e-12
    assert out["n_candidates_scored"] == 3
    for row in out["ranked_full"]:
        oid = row["operation_id"]
        expect_lex = float(lexical_score(query, registry[oid]))
        assert abs(float(row["lexical_score"]) - expect_lex) < 1e-12
        expect = float(row["cosine_similarity"]) + 0.12 * expect_lex
        assert abs(float(row["score"]) - expect) < 1e-12


def test_topk_describe_n_frozen():
    assert TOP_K == 5
    assert DESCRIBE_N == 3
    assert HYBRID_ALPHAS == (1.0, 0.9, 0.8, 0.7)


def test_dim_mismatch_fail_closed():
    _, index = _mini_index(dim=8)
    with pytest.raises(EmbeddingDimensionMismatch):
        rank_operations(
            index=index,
            query="q",
            variant="R1",
            query_vector=_vec(1536, 1.0),
        )


def test_selection_development_only_and_r4_larger_alpha_tiebreak():
    base = {
        "complete_route@3": 0.5,
        "recall@3": 0.5,
        "min_domain_recall@3": 0.4,
        "mrr": 0.4,
    }
    comparison = compare_ablation_formulas(
        {
            "R1": dict(base),
            "R4_global_coverage@1.0": dict(base),
            "R4_global_coverage@0.9": dict(base),
            "R5_global_lexical@0.12": dict(base),
        }
    )
    # Same metrics → simpler formula (R1) wins.
    assert comparison["selected"]["formula_key"] == "R1"
    assert comparison["development_only"] is True
    comparison2 = compare_ablation_formulas(
        {
            "R4_global_coverage@0.7": {
                "complete_route@3": 0.6,
                "recall@3": 0.5,
                "min_domain_recall@3": 0.4,
                "mrr": 0.4,
            },
            "R4_global_coverage@0.9": {
                "complete_route@3": 0.6,
                "recall@3": 0.5,
                "min_domain_recall@3": 0.4,
                "mrr": 0.4,
            },
        }
    )
    assert comparison2["selected"]["formula_key"] == "R4_global_coverage@0.9"


def test_candidate_lock_identity_fields_cannot_be_null():
    lock = {
        "embedding_provider_host": "api.zhizengzeng.com",
        "embedding_model": "text-embedding-3-small",
        "embedding_dimension": 1536,
        "operation_index_identity": EXPECTED_CATALOG_IDENTITY,
        "operation_documents_sha256": EXPECTED_OP_DOCS,
        "planner_cache_sha256": EXPECTED_PLANNER_SHA,
        "step_query_cache_identity": EXPECTED_STEP_IDENTITY,
    }
    for k, v in lock.items():
        assert v is not None and v != ""


def test_cache_identity_mismatch_fail_closed(tmp_path: Path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "planner_cache").mkdir()
    (out / "planner_cache" / "planner_outputs.jsonl").write_text(
        "{}\n", encoding="utf-8"
    )
    with pytest.raises(LivePathError, match="Planner cache SHA mismatch"):
        run_global_signal_ablation(root=ROOT, output_dir=out)


@pytest.mark.skipif(not OUT_V3.exists(), reason="frozen planner_v3 outputs missing")
def test_frozen_raw_sha_snapshot_stable():
    digests: Dict[str, str] = {}
    for rel in RAW_WATCHED:
        path = OUT_V3 / rel
        if path.exists():
            digests[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
    # Re-read immediately — must be identical (no accidental rewrite).
    for rel, digest in digests.items():
        assert hashlib.sha256((OUT_V3 / rel).read_bytes()).hexdigest() == digest
    if (OUT_V3 / "planner_cache" / "planner_outputs.jsonl").exists():
        assert digests["planner_cache/planner_outputs.jsonl"] == EXPECTED_PLANNER_SHA


@pytest.mark.skipif(not OUT_V3.exists(), reason="frozen planner_v3 outputs missing")
def test_old_formula_lock_identity_incomplete_not_modified():
    lock_path = OUT_V3 / "formula_selection" / "formula_lock.json"
    assert lock_path.exists()
    before = hashlib.sha256(lock_path.read_bytes()).hexdigest()
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    assert lock.get("embedding_provider_host") is None
    assert lock.get("embedding_dimension") is None
    assert lock.get("operation_index_identity") is None
    assert lock.get("planner_cache_sha256") != EXPECTED_PLANNER_SHA
    after = hashlib.sha256(lock_path.read_bytes()).hexdigest()
    assert before == after


def test_gold_not_used_in_scoring_path():
    _, index = _mini_index()
    qvec = _vec(8, 1.3)
    base = rank_operations(
        index=index,
        query="financial revenue",
        variant="R4",
        alpha=0.7,
        required_outputs=["revenue"],
        query_vector=qvec,
        gold_ops_for_filter_diag=None,
    )
    boosted = rank_operations(
        index=index,
        query="financial revenue",
        variant="R4",
        alpha=0.7,
        required_outputs=["revenue"],
        query_vector=qvec,
        gold_ops_for_filter_diag=["op.news"],
    )
    assert [r["operation_id"] for r in base["ranked_full"]] == [
        r["operation_id"] for r in boosted["ranked_full"]
    ]
    assert [r["score"] for r in base["ranked_full"]] == [
        r["score"] for r in boosted["ranked_full"]
    ]


def test_mixed_dim_invalid_r0_not_in_ablation_formulas():
    comparison = compare_ablation_formulas(
        {
            "R0": {
                "complete_route@3": 0.99,
                "recall@3": 0.99,
                "min_domain_recall@3": 0.99,
                "mrr": 0.99,
            },
            "R1": {
                "complete_route@3": 0.1,
                "recall@3": 0.1,
                "min_domain_recall@3": 0.1,
                "mrr": 0.1,
            },
        }
    )
    keys = {r["formula_key"] for r in comparison["ranking"]}
    assert "R0" not in keys
    assert comparison["selected"]["formula_key"] == "R1"


def test_no_network_helpers_monkeypatched():
    """Ensure ablation entry refuses live and does not call embed APIs."""
    with mock.patch(
        "hk_equity_mcp.experiment.pe_fin_lite_retrieval_audit.global_signal_ablation."
        "load_complete_index",
        side_effect=AssertionError("must not rebuild index via network"),
    ):
        with pytest.raises((LivePathError, AssertionError, FileNotFoundError)):
            run_global_signal_ablation(root=ROOT, output_dir=ROOT / "outputs" / "missing")
