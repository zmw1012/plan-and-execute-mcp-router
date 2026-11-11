"""Formula experiment readiness tests (fake vectors; zero network)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List

import pytest

from hk_equity_mcp.experiment.pe_fin_lite_retrieval_audit import (
    HYBRID_ALPHAS,
    RUNTIME_LEXICAL_WEIGHT,
)
from hk_equity_mcp.experiment.pe_fin_lite_retrieval_audit.embedding_identity_gate import (
    EmbeddingDimensionMismatch,
)
from hk_equity_mcp.experiment.pe_fin_lite_retrieval_audit.formula_selection import (
    assert_formal_formula_lock_allowed,
    assert_frozen_evaluation_allowed,
    build_formula_lock,
    compare_formulas,
    filter_plan_first_selection_inputs,
)
from hk_equity_mcp.experiment.pe_fin_lite_retrieval_audit.planner_cache_gate import (
    IncompletePlannerCacheError,
    assert_planner_cache_for_step_pipeline,
    audit_planner_cache_zero_network,
)
from hk_equity_mcp.experiment.pe_fin_lite_retrieval_audit.ranking import (
    EMPTY_REQUIRED_OUTPUT_COVERAGE,
    OfflineEmbeddingIndex,
    apply_runtime_topk_then_hard_filter,
    cosine,
    rank_operations,
    required_output_coverage,
    runtime_score_all_operations,
)
from hk_equity_mcp.experiment.pe_fin_lite_retrieval_audit.runner import (
    RetrievalAuditRunner,
)
from hk_equity_mcp.operation_resolver.retrieval import (
    LEXICAL_WEIGHT,
    OperationRetriever,
    lexical_score,
)
from hk_equity_mcp.operation_resolver.schemas import OperationSpec, ParameterSpec
from hk_equity_mcp.pe_fin_lite.tool_retrieval import (
    hard_filter_operations,
    select_embedding_topk,
)

ROOT = Path(__file__).resolve().parents[1]
OUT_V3 = ROOT / "outputs" / "pe_fin_lite_retrieval_all155_planner_v3"
INVALID_DIR = OUT_V3 / "question_level_invalid_mixed_dim64_v1"
QQ_VECTORS = OUT_V3 / "question_query_cache" / "question_query_vectors.json"
EXPECTED_QQ_SHA = (
    "4082f51fa357c8bdb805c77fc3018dcc170f1f7a6b4c3fbeef3a944bca1a0c5d"
)


def _vec(dim: int, seed: float) -> List[float]:
    return [float((i + 1) * seed) for i in range(dim)]


def _op(
    oid: str,
    *,
    name: str,
    summary: str,
    actions: List[str],
) -> OperationSpec:
    return OperationSpec(
        operation_id=oid,
        service_id="aktools",
        name=name,
        summary=summary,
        description=summary,
        action_terms=actions,
        parameters=[
            ParameterSpec(name="symbol", type="string", required=True),
        ],
        underlying_tool=name,
    )


def _mini_runtime_fixture(dim: int = 8):
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
        "op.blocked": _op(
            "op.blocked",
            name="blocked_tool",
            summary="should never rank",
            actions=["blocked"],
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
        "op.blocked": {
            "domain": "financial",
            "provides": [],
            "availability": "blocked",
            "blocked": True,
            "runtime_role": "blocked",
        },
    }
    vectors = {
        "op.fin": _vec(dim, 1.0),
        "op.mkt": _vec(dim, 2.0),
        "op.news": _vec(dim, 3.0),
        "op.blocked": _vec(dim, 4.0),
    }
    docs = {
        oid: {"document_text": registry[oid].index_document()} for oid in registry
    }
    index = OfflineEmbeddingIndex.from_preloaded_vectors(
        documents=docs,
        metadata_by_id=meta,
        runtime_operation_ids=["op.fin", "op.mkt", "op.news"],
        vectors=vectors,
        embedding_model="fake",
        registry=registry,
    )
    return registry, meta, vectors, index


def test_cosine_dim_mismatch_fail_closed():
    with pytest.raises(EmbeddingDimensionMismatch):
        cosine(_vec(1536, 1.0), _vec(64, 1.0))


def test_lexical_weight_frozen_at_0_12():
    assert LEXICAL_WEIGHT == RUNTIME_LEXICAL_WEIGHT == 0.12


def test_required_output_coverage_deterministic_and_empty_neutral():
    assert required_output_coverage(required_outputs=[], provides=["x"]) == (
        EMPTY_REQUIRED_OUTPUT_COVERAGE
    )
    a = required_output_coverage(
        required_outputs=["revenue", "eps"], provides=["revenue", "basic_eps"]
    )
    b = required_output_coverage(
        required_outputs=["revenue", "eps"], provides=["revenue", "basic_eps"]
    )
    assert a == b
    assert 0.0 < a <= 1.0


def test_r3_alpha_formula_exact():
    _, _, _, index = _mini_runtime_fixture()
    q = _vec(8, 1.5)
    out = rank_operations(
        index=index,
        query="financial revenue",
        variant="R3",
        alpha=0.8,
        step_domain="financial",
        required_outputs=["revenue"],
        query_vector=q,
    )
    assert out["alpha"] == 0.8
    top = out["ranked"][0]
    expected = 0.8 * top["cosine_similarity"] + 0.2 * top["required_output_coverage"]
    assert abs(top["score"] - expected) < 1e-12
    with pytest.raises(ValueError):
        rank_operations(
            index=index,
            query="q",
            variant="R3",
            alpha=0.55,
            step_domain="financial",
            query_vector=q,
        )


def test_domain_filter_diagnostics_and_blocked_never_leak():
    _, _, _, index = _mini_runtime_fixture()
    out = rank_operations(
        index=index,
        query="financial revenue",
        variant="R2",
        step_domain="market",
        query_vector=_vec(8, 1.0),
        gold_ops_for_filter_diag=["op.fin"],
    )
    assert out["filter"]["domain_filter_applied"] is True
    assert out["filter"]["candidates_before_filter"] is not None
    assert out["filter"]["candidates_after_filter"] is not None
    assert out["filter"]["filter_miss"] is True
    assert "op.blocked" not in out["ranked_ids"]
    assert out["blocked_leakage"] == []


def test_r_runtime_parity_with_operation_retriever_path():
    registry, meta, vectors, index = _mini_runtime_fixture()
    query = "Hong Kong financial report revenue eps"
    qvec = _vec(8, 1.1)

    class FixedBackend:
        model_name = "fake-fixed"

        def embed(self, texts):
            # Only query embed is invoked in rank() when docs preloaded.
            return [list(qvec) for _ in texts]

    retriever = OperationRetriever(
        registry=registry,
        embedding_backend=FixedBackend(),  # type: ignore[arg-type]
        preloaded_doc_vectors=vectors,
        ensure_docs=False,
    )
    raw = retriever.rank(query, top_k=5)
    filt = hard_filter_operations(
        registry_ids=list(registry.keys()),
        metadata_by_id=meta,
        runtime_operation_ids=["op.fin", "op.mkt", "op.news"],
        step_domain="financial",
    )
    runtime_ids = [
        r["operation_id"]
        for r in select_embedding_topk(
            raw, allowed_ids=set(filt["kept"]), top_k=5
        )
    ]

    offline = rank_operations(
        index=index,
        query=query,
        variant="R_runtime",
        step_domain="financial",
        query_vector=qvec,
        top_k=5,
    )
    assert offline["runtime_parity"] is True
    assert offline["lexical_weight"] == 0.12
    assert offline["filter"]["domain_filter_timing"] == "after_global_topk"
    assert offline["ranked_ids"] == runtime_ids
    # Also equal to shared scoring helper path.
    scored = runtime_score_all_operations(
        query=query,
        query_vector=qvec,
        vectors=vectors,
        registry=registry,
        operation_ids=sorted(registry),
    )
    helper_ids = [
        r["operation_id"]
        for r in apply_runtime_topk_then_hard_filter(
            scored_all=scored, allowed_ids=set(filt["kept"]), top_k=5
        )
    ]
    assert helper_ids == runtime_ids


def test_r0_excluded_from_plan_first_selection():
    comparison = compare_formulas(
        development_metrics_by_formula={
            "R0": {
                "complete_route@3": 0.99,
                "recall@3": 0.99,
                "min_domain_recall@3": 0.99,
                "mrr": 0.99,
            },
            "R1": {
                "complete_route@3": 0.40,
                "recall@3": 0.40,
                "min_domain_recall@3": 0.40,
                "mrr": 0.40,
            },
        }
    )
    assert comparison["selected"]["formula_key"] == "R1"
    assert "R0" in comparison["excluded_from_inputs"]
    assert filter_plan_first_selection_inputs({"R0": {}, "R2": {}}) == {"R2": {}}


def test_runtime_sync_flags_on_selection():
    rt = compare_formulas(
        development_metrics_by_formula={
            "R_runtime": {
                "complete_route@3": 0.9,
                "recall@3": 0.9,
                "min_domain_recall@3": 0.9,
                "mrr": 0.9,
            },
            "R1": {
                "complete_route@3": 0.1,
                "recall@3": 0.1,
                "min_domain_recall@3": 0.1,
                "mrr": 0.1,
            },
        }
    )
    assert rt["runtime_sync_required"] is False
    r1 = compare_formulas(
        development_metrics_by_formula={
            "R1": {
                "complete_route@3": 0.9,
                "recall@3": 0.9,
                "min_domain_recall@3": 0.9,
                "mrr": 0.9,
            },
            "R_runtime": {
                "complete_route@3": 0.1,
                "recall@3": 0.1,
                "min_domain_recall@3": 0.1,
                "mrr": 0.1,
            },
        }
    )
    assert r1["runtime_sync_required"] is True
    assert r1["runtime_sync_specification"]["minimal_sync"]["drop_lexical_term"] is True


def test_formal_lock_fields_and_mock_refused():
    with pytest.raises(RuntimeError, match="mock"):
        assert_formal_formula_lock_allowed(
            embedding_backend="real_text_embedding_3_small", mock=True
        )
    selected = {
        "variant": "R2",
        "alpha": None,
        "formula_key": "R2",
        "metrics": {"complete_route@3": 0.5},
    }
    lock = build_formula_lock(
        selected=selected,
        development_metrics=selected["metrics"],
        runtime_sync_required=True,
        runtime_sync_specification={"note": "sync later"},
        config={
            "catalog_fingerprint": "cat",
            "operation_documents_sha256": "docs",
            "embedding_model": "text-embedding-3-small",
            "embedding_provider_host": "api.zhizengzeng.com",
            "embedding_dimension": 1536,
            "operation_index_identity": "ac4c109b",
            "planner_cache_sha256": "p",
            "step_query_cache_sha256": "s",
            "embedding_backend": "real_text_embedding_3_small",
        },
    )
    assert lock["no_retune_on_reserved85"] is True
    assert lock["runtime_sync_required"] is True
    assert lock["embedding_dimension"] == 1536
    assert lock["lock_sha256"]
    assert_frozen_evaluation_allowed(
        lock=lock, requested_variant="R2", requested_alpha=None
    )
    with pytest.raises(RuntimeError, match="cannot retune"):
        assert_frozen_evaluation_allowed(
            lock=lock, requested_variant="R3", requested_alpha=0.8
        )


def test_incomplete_planner_and_step_cache_refused(tmp_path: Path):
    rows = [{"task_id": f"t{i}", "ok": True} for i in range(10)]
    with pytest.raises(IncompletePlannerCacheError):
        assert_planner_cache_for_step_pipeline(
            rows=rows, allow_incomplete=False, formal=False, stage="rank"
        )
    with pytest.raises(IncompletePlannerCacheError):
        assert_planner_cache_for_step_pipeline(
            rows=[{"task_id": f"t{i:03d}", "ok": i != 0} for i in range(155)],
            allow_incomplete=False,
            formal=True,
            stage="select",
        )


def test_alphas_are_discrete_four_only():
    assert HYBRID_ALPHAS == (1.0, 0.9, 0.8, 0.7)


def test_changing_alpha_uses_same_vectors_zero_network():
    _, _, _, index = _mini_runtime_fixture()
    q = _vec(8, 1.2)
    # Two alphas; no embed_query path (preloaded query vector).
    a = rank_operations(
        index=index,
        query="financial revenue",
        variant="R3",
        alpha=1.0,
        step_domain="financial",
        required_outputs=["revenue"],
        query_vector=q,
    )
    b = rank_operations(
        index=index,
        query="financial revenue",
        variant="R3",
        alpha=0.7,
        step_domain="financial",
        required_outputs=["revenue"],
        query_vector=q,
    )
    assert a["n_candidates_scored"] == b["n_candidates_scored"]
    # Alpha swap must not require re-embedding; scores follow closed-form hybrid.
    for row_a, row_b in zip(a["ranked"], b["ranked"]):
        assert row_a["cosine_similarity"] == row_b["cosine_similarity"]
        assert row_a["required_output_coverage"] == row_b["required_output_coverage"]
        expect_a = 1.0 * row_a["cosine_similarity"] + 0.0 * row_a["required_output_coverage"]
        expect_b = 0.7 * row_b["cosine_similarity"] + 0.3 * row_b["required_output_coverage"]
        assert abs(row_a["score"] - expect_a) < 1e-12
        assert abs(row_b["score"] - expect_b) < 1e-12


def test_mock_offline_rank_pipeline_includes_r_runtime(tmp_path: Path):
    out = tmp_path / "mock_out"
    runner = RetrievalAuditRunner(root=ROOT, output_dir=out, mock=True)
    runner.run_planner_only()
    # Build a minimal step cache by ranking from mock planner (embeds via hash).
    meta = runner.run_rank_from_planner_cache()
    assert meta["zero_network"] is True
    cmp_path = out / "ranking_variants" / "formula_comparison.json"
    payload = json.loads(cmp_path.read_text(encoding="utf-8"))
    keys = set(payload.get("formulas") or {})
    assert "R1" in keys and "R2" in keys and "R_runtime" in keys
    assert any(k.startswith("R3@") for k in keys)
    assert "R0" not in keys
    sel = runner.run_select_formula(formal=True)
    assert sel.get("formal_lock_written") is False


def test_dry_run_cli_helpers(tmp_path: Path):
    out = tmp_path / "dry"
    runner = RetrievalAuditRunner(root=ROOT, output_dir=out, dry_run=True)
    status = runner.get_index_status()
    assert status.get("zero_network") is True
    assert status.get("selection_mode") == "inventory_only"


@pytest.mark.skipif(not INVALID_DIR.exists(), reason="invalid R0 archive missing")
def test_invalid_mixed_dim_artifacts_unchanged():
    marker = INVALID_DIR / "INVALIDATED_R0_MIXED_EMBEDDING_DIMENSIONS.json"
    assert marker.exists()
    payload = json.loads(marker.read_text(encoding="utf-8"))
    frozen = payload.get("raw_hashes_at_invalidation") or {}
    # Prefer hashes relative to archive dir files themselves if listed with question_level/
    for name in [
        "run_metadata.json",
        "retrieval_metrics.json",
        "per_query_rankings.jsonl",
    ]:
        path = INVALID_DIR / name
        assert path.exists()
        # Marker may key as question_level/... from original tree.
        key = f"question_level/{name}"
        if key in frozen:
            blob = path.read_bytes()
            assert hashlib.sha256(blob).hexdigest() == frozen[key]["sha256"]
            assert len(blob) == frozen[key]["bytes"]


@pytest.mark.skipif(not QQ_VECTORS.exists(), reason="QQ cache missing")
def test_question_query_cache_sha_reusable_unchanged():
    assert hashlib.sha256(QQ_VECTORS.read_bytes()).hexdigest() == EXPECTED_QQ_SHA


def test_planner_cache_audit_zero_network(tmp_path: Path):
    out = tmp_path / "pa"
    pc = out / "planner_cache"
    pc.mkdir(parents=True)
    rows = []
    for i in range(155):
        rows.append(
            {
                "task_id": f"task_{i:03d}",
                "ok": True,
                "repair_count": 0,
                "latency_ms": 10 + i,
                "usage": {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3},
                "plan": {
                    "steps": [
                        {"type": "retrieve", "domain": "financial", "id": "s1"},
                        {"type": "reason", "id": "s2"},
                    ]
                },
            }
        )
    (pc / "planner_outputs.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )
    (pc / "planner_cache_identity.json").write_text(
        json.dumps(
            {
                "kind": "planner_only_v3",
                "protocol_version": "pe_fin_lite_planner_only_v3",
                "planner_prompt_hash": "abc",
                "planner_schema_hash": "def",
                "identity_sha256": "xyz",
            }
        ),
        encoding="utf-8",
    )
    expected = [f"task_{i:03d}" for i in range(155)]
    audit = audit_planner_cache_zero_network(
        output_dir=out, expected_task_ids=expected
    )
    assert audit["zero_network"] is True
    assert audit["complete_155_ok"] is True
    assert audit["retrieve_compute_reason_counts"]["retrieve"] == 155
