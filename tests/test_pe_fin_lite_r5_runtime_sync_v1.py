"""R5 retrieval formula runtime sync — offline parity and fail-closed tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pytest

from hk_equity_mcp.experiment.pe_fin_lite_retrieval_audit.ranking import (
    OfflineEmbeddingIndex,
    rank_operations,
)
from hk_equity_mcp.experiment.unified_pe_fin_lite_runner import (
    PE_RESUME_IDENTITY_KEYS,
    assert_output_dir_safe,
)
from hk_equity_mcp.operation_resolver.retrieval import (
    LEXICAL_WEIGHT,
    OperationRetriever,
    lexical_score,
)
from hk_equity_mcp.operation_resolver.schemas import OperationSpec, ParameterSpec
from hk_equity_mcp.pe_fin_lite.budget import BudgetTracker
from hk_equity_mcp.pe_fin_lite.evidence_store import EvidenceStore
from hk_equity_mcp.pe_fin_lite import executor as ex
from hk_equity_mcp.pe_fin_lite.retrieval_formula import (
    DESCRIBE_N,
    DOMAIN_HARD_FILTER,
    R5_CANDIDATE_LOCK_SHA256,
    RETRIEVAL_FORMULA_NAME,
    RETRIEVAL_TOP_K,
    r5_final_score,
    retrieval_formula_hash,
)
from hk_equity_mcp.pe_fin_lite.tool_retrieval import (
    safety_filter_operations,
    select_describe_candidates,
)
from hk_equity_mcp.pe_fin_lite.tool_binding import (
    ToolBindingError,
    validate_binding_payload,
)

ROOT = Path(__file__).resolve().parents[1]


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


def _fixture(dim: int = 8):
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
            "provides": ["revenue"],
            "availability": "runtime",
        },
        "op.mkt": {
            "domain": "market",
            "provides": ["close"],
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
    docs = {oid: {"document_text": registry[oid].index_document()} for oid in registry}
    index = OfflineEmbeddingIndex.from_preloaded_vectors(
        documents=docs,
        metadata_by_id=meta,
        runtime_operation_ids=["op.fin", "op.mkt", "op.news"],
        vectors={k: vectors[k] for k in ["op.fin", "op.mkt", "op.news"]},
        embedding_model="fake",
        registry={k: registry[k] for k in ["op.fin", "op.mkt", "op.news"]},
    )
    return registry, meta, vectors, index


def test_r5_formula_constants_frozen():
    assert RETRIEVAL_FORMULA_NAME == "R5_global_lexical@0.12"
    assert LEXICAL_WEIGHT == 0.12
    assert DOMAIN_HARD_FILTER is False
    assert RETRIEVAL_TOP_K == 5
    assert DESCRIBE_N == 3
    assert R5_CANDIDATE_LOCK_SHA256 == (
        "6a2f6ec1ba4fcf70ab50d01fc21e7a1aabda95f1beb15981f32eda3dc21d28b4"
    )
    assert retrieval_formula_hash()
    assert "retrieval_formula_hash" in PE_RESUME_IDENTITY_KEYS


def test_runtime_score_matches_cosine_plus_012_lexical():
    registry, _, vectors, _ = _fixture()
    query = "Hong Kong financial report revenue"
    qvec = _vec(8, 1.15)

    class FixedBackend:
        model_name = "fake-fixed"

        def embed(self, texts):
            return [list(qvec) for _ in texts]

    retriever = OperationRetriever(
        registry={k: registry[k] for k in ["op.fin", "op.mkt", "op.news"]},
        embedding_backend=FixedBackend(),  # type: ignore[arg-type]
        preloaded_doc_vectors={k: vectors[k] for k in ["op.fin", "op.mkt", "op.news"]},
        ensure_docs=False,
    )
    ranked = retriever.rank(query, top_k=5)
    assert len(ranked) <= 5
    for row in ranked:
        oid = str(row["operation_id"])
        expect = r5_final_score(
            cosine=float(row["cosine_similarity"]),
            query=query,
            operation=registry[oid],
        )
        assert abs(float(row["score"]) - round(expect, 6)) < 1e-9
        assert abs(float(row["lexical_score"]) - float(lexical_score(query, registry[oid]))) < 1e-12
        assert float(row["lexical_weight"]) == 0.12


def test_runtime_rank_parity_with_audit_r5():
    registry, meta, vectors, index = _fixture()
    query = "Hong Kong financial report revenue eps"
    qvec = _vec(8, 1.2)
    audit = rank_operations(
        index=index,
        query=query,
        variant="R5",
        step_domain="news",  # different domain — must not filter under R5
        query_vector=qvec,
        top_k=5,
    )
    assert audit["filter"]["domain_filter_applied"] is False

    class FixedBackend:
        model_name = "fake-fixed"

        def embed(self, texts):
            return [list(qvec) for _ in texts]

    runtime_ids = ["op.fin", "op.mkt", "op.news"]
    safe = safety_filter_operations(
        registry_ids=list(registry.keys()),
        metadata_by_id=meta,
        runtime_operation_ids=runtime_ids,
    )
    assert set(safe["kept"]) == set(runtime_ids)
    retriever = OperationRetriever(
        registry={k: registry[k] for k in runtime_ids},
        embedding_backend=FixedBackend(),  # type: ignore[arg-type]
        preloaded_doc_vectors={k: vectors[k] for k in runtime_ids},
        ensure_docs=False,
    )
    runtime = retriever.rank(query, top_k=5, allowed_ids=sorted(safe["kept"]))
    assert [r["operation_id"] for r in runtime] == audit["ranked_ids"]
    for a, b in zip(runtime, audit["ranked"]):
        assert a["operation_id"] == b["operation_id"]
        assert abs(float(a["score"]) - round(float(b["score"]), 6)) < 1e-9


def test_cross_domain_ops_not_removed_by_r5_safety_filter():
    _, meta, _, _ = _fixture()
    safe = safety_filter_operations(
        registry_ids=["op.fin", "op.mkt", "op.news", "op.blocked"],
        metadata_by_id=meta,
        runtime_operation_ids=["op.fin", "op.mkt", "op.news", "op.blocked"],
    )
    assert "op.mkt" in safe["kept"]
    assert "op.fin" in safe["kept"]
    assert "op.news" in safe["kept"]
    assert "op.blocked" not in safe["kept"]


def test_blocked_never_leaks_and_topk_describe_bounds():
    registry, meta, vectors, _ = _fixture()
    qvec = _vec(8, 1.0)

    class FixedBackend:
        model_name = "fake"

        def embed(self, texts):
            return [list(qvec) for _ in texts]

    # Include blocked in registry vectors — must still be excluded via allowed_ids.
    retriever = OperationRetriever(
        registry=registry,
        embedding_backend=FixedBackend(),  # type: ignore[arg-type]
        preloaded_doc_vectors=vectors,
        ensure_docs=False,
    )
    safe = safety_filter_operations(
        registry_ids=list(registry.keys()),
        metadata_by_id=meta,
        runtime_operation_ids=["op.fin", "op.mkt", "op.news", "op.blocked"],
    )
    ranked = retriever.rank("q", top_k=5, allowed_ids=safe["kept"])
    assert all(r["operation_id"] != "op.blocked" for r in ranked)
    assert len(ranked) <= 5
    described = select_describe_candidates(ranked, top_n=3)
    assert len(described) <= 3
    assert 2 <= len(described) <= 3 or len(ranked) < 2


def test_binder_only_described_candidates():
    contracts = {
        "op.fin": {
            "json_schema": {
                "properties": {"symbol": {"type": "string"}},
                "required": ["symbol"],
            }
        },
        "op.mkt": {
            "json_schema": {
                "properties": {"symbol": {"type": "string"}},
                "required": ["symbol"],
            }
        },
    }
    ok = validate_binding_payload(
        {
            "primary": {"operation_id": "op.fin", "arguments": {"symbol": "0700"}},
            "fallback": {"operation_id": "op.mkt", "arguments": {"symbol": "0700"}},
        },
        described_operation_ids=["op.fin", "op.mkt"],
        described_contracts=contracts,
    )
    assert ok["primary"]["operation_id"] == "op.fin"
    with pytest.raises(ToolBindingError):
        validate_binding_payload(
            {
                "primary": {
                    "operation_id": "op.news",
                    "arguments": {"symbol": "0700"},
                },
                "fallback": None,
            },
            described_operation_ids=["op.fin", "op.mkt"],
            described_contracts=contracts,
        )


def test_resume_refuses_old_formula_identity(tmp_path: Path):
    out = tmp_path / "old_formula"
    out.mkdir()
    old = {
        "condition": "pe_fin_lite",
        "method_version": "pe_fin_lite_v1.2",
        "architecture": "plan_first_step_level_retrieval_v1",
        "dry_run": False,
        "mock": True,
        "subset_kind": "smoke5",
        "retrieval_formula_name": "legacy_domain_filter_runtime",
        "retrieval_formula_hash": "0" * 64,
        "domain_hard_filter": True,
        "candidate_lock_sha256": "old",
        "lexical_weight": 0.12,
    }
    (out / "run_metadata.json").write_text(json.dumps(old), encoding="utf-8")
    (out / "predictions.jsonl").write_text("{}\n", encoding="utf-8")
    new = dict(old)
    new["retrieval_formula_name"] = RETRIEVAL_FORMULA_NAME
    new["retrieval_formula_hash"] = retrieval_formula_hash()
    new["domain_hard_filter"] = False
    new["candidate_lock_sha256"] = R5_CANDIDATE_LOCK_SHA256
    with pytest.raises(RuntimeError, match="identity mismatch"):
        assert_output_dir_safe(
            output_dir=out,
            dry_run=False,
            mock=True,
            resume=True,
            run_meta=new,
        )


def test_executor_r5_keeps_cross_domain_and_records_formula(monkeypatch):
    calls: List[str] = []

    class FakeResolver:
        registry = {"op.fin": object(), "op.mkt": object(), "op.news": object()}

        def find_operations(self, query, top_k=5, *, allowed_ids=None):
            # Return cross-domain market op first for a financial step query.
            assert allowed_ids is not None
            assert "op.mkt" in allowed_ids
            return {
                "success": True,
                "candidates": [
                    {"operation_id": "op.mkt", "score": 0.99},
                    {"operation_id": "op.fin", "score": 0.90},
                    {"operation_id": "op.news", "score": 0.80},
                ],
            }

        def describe_operation(self, oid):
            return {
                "success": True,
                "contract": {
                    "json_schema": {
                        "properties": {"symbol": {"type": "string"}},
                        "required": ["symbol"],
                    },
                    "summary": oid,
                },
            }

        def invoke_api(self, oid, arguments=None):
            calls.append(oid)
            return {
                "success": True,
                "operation_id": oid,
                "arguments": arguments or {},
                "result": {
                    "success": True,
                    "raw": {
                        "is_error": False,
                        "content": [{"json": {"close": 1.0}}],
                    },
                },
                "error_type": None,
                "error": None,
            }

    class FakeLLM:
        def complete(self, messages, *, temperature=0.0, max_tokens=1024, stage=None):
            return {
                "content": json.dumps(
                    {
                        "primary": {
                            "operation_id": "op.mkt",
                            "arguments": {"symbol": "0700"},
                        },
                        "fallback": {
                            "operation_id": "op.fin",
                            "arguments": {"symbol": "0700"},
                        },
                    }
                ),
                "usage": {},
            }

    store = EvidenceStore()
    budget = BudgetTracker(
        {
            "max_total_mcp_invocations": 8,
            "retrieval_top_k": 5,
            "describe_top_n": 3,
            "max_local_fallback_per_retrieve": 1,
        }
    )
    meta = {
        "op.fin": {"domain": "financial", "availability": "runtime"},
        "op.mkt": {"domain": "market", "availability": "runtime"},
        "op.news": {"domain": "news", "availability": "runtime"},
    }
    out = ex.execute_retrieve_step(
        step={
            "id": "s1",
            "type": "retrieve",
            "domain": "financial",
            "objective": "close price",
            "required_outputs": ["close"],
            "depends_on": [],
        },
        question="what is close?",
        resolver=FakeResolver(),
        metadata_by_id=meta,
        runtime_operation_ids=["op.fin", "op.mkt", "op.news"],
        store=store,
        budget=budget,
        llm=FakeLLM(),
        use_mock_binder=False,
    )
    funnel = out["funnel"]
    assert funnel["domain_hard_filter"] is False
    assert funnel["retrieval_formula"]["name"] == RETRIEVAL_FORMULA_NAME
    assert "op.mkt" in [r["operation_id"] for r in funnel["embedding_topk"]]
    assert len(funnel["described_candidates"]) <= 3
    assert funnel["binder_selection_validation"]["primary_in_described"] is True
    # market op survived financial step — domain filter removed
    assert calls[0] == "op.mkt"
    assert budget.embedding_query_calls == 1
    assert budget.binder_calls == 1
