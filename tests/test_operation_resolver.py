"""Tests for Operation-Resolver retrieval + meta-tools (mock embeddings)."""

from pathlib import Path

import pytest

from hk_equity_mcp.operation_resolver.registry import build_default_registry
from hk_equity_mcp.operation_resolver.retrieval import (
    MAX_TOP_K,
    DeterministicHashEmbeddingBackend,
    OpenAICompatibleEmbeddingBackend,
    OperationRetriever,
    cosine_similarity,
)
from hk_equity_mcp.operation_resolver.server import META_TOOLS, OperationResolverServer


@pytest.fixture
def resolver(tmp_path: Path) -> OperationResolverServer:
    backend = DeterministicHashEmbeddingBackend()
    return OperationResolverServer(
        embedding_backend=backend,
        cache_dir=tmp_path / "emb_cache",
    )


def test_meta_tools_exactly_three(resolver: OperationResolverServer):
    assert resolver.list_meta_tools() == list(META_TOOLS)
    assert META_TOOLS == (
        "find_operations",
        "describe_operation",
        "invoke_api",
    )


def test_top_k_cap_and_stable_ranking(resolver: OperationResolverServer):
    q = "retrieve revenue financial metric for HK ticker"
    a = resolver.find_operations(q, top_k=50)
    b = resolver.find_operations(q, top_k=20)
    assert a["success"] and b["success"]
    assert a["top_k"] == MAX_TOP_K == 20
    assert b["top_k"] == 20
    assert a["candidates"] == b["candidates"]
    # Identical inputs → identical ranked IDs/scores.
    c = resolver.find_operations(q, top_k=20)
    assert [row["operation_id"] for row in a["candidates"]] == [
        row["operation_id"] for row in c["candidates"]
    ]
    assert [row["score"] for row in a["candidates"]] == [
        row["score"] for row in c["candidates"]
    ]
    for row in a["candidates"]:
        assert set(row) >= {
            "operation_id",
            "score",
            "service_id",
            "name",
            "summary",
        }
        assert "result" not in row


def test_describe_operation_contract(resolver: OperationResolverServer):
    resp = resolver.describe_operation("financial.get_financial_metric")
    assert resp["success"] is True
    contract = resp["contract"]
    assert contract["operation_id"] == "financial.get_financial_metric"
    assert contract["service_id"] == "financial"
    assert "json_schema" in contract
    assert contract["required_parameters"]
    assert "invocation_guidance" in contract


def test_describe_unknown_operation(resolver: OperationResolverServer):
    resp = resolver.describe_operation("financial.does_not_exist")
    assert resp["success"] is False
    assert resp["contract"] is None
    assert "Unknown operation_id" in resp["error"]


def test_invoke_rejects_missing_unknown_mistyped(resolver: OperationResolverServer):
    op_id = "financial.get_financial_metric"
    missing = resolver.invoke_api(op_id, {"ticker": "0700.HK"})
    assert missing["success"] is False
    assert missing["error_type"] == "validation_error"
    assert "Missing required" in missing["error"]

    unknown = resolver.invoke_api(
        op_id, {"ticker": "0700.HK", "metric": "revenue", "bogus": 1}
    )
    assert unknown["success"] is False
    assert "Unknown arguments" in unknown["error"]

    mistyped = resolver.invoke_api(op_id, {"ticker": "0700.HK", "metric": 123})
    assert mistyped["success"] is False
    assert "must be string" in mistyped["error"]

    bad_enum = resolver.invoke_api(
        op_id, {"ticker": "0700.HK", "metric": "not_a_metric"}
    )
    assert bad_enum["success"] is False
    assert "must be one of" in bad_enum["error"]


def test_invoke_unknown_operation_and_underlying_success(
    resolver: OperationResolverServer,
):
    unknown = resolver.invoke_api("nope.op", {})
    assert unknown["success"] is False
    assert unknown["error_type"] == "unknown_operation"

    ok = resolver.invoke_api(
        "financial.get_financial_metric",
        {
            "ticker": "0700.HK",
            "metric": "revenue",
            "period": "FY2025",
            "report_type": "annual",
        },
    )
    assert ok["success"] is True
    assert ok["result"]["success"] is True
    assert ok["result"]["data"]["value"] == 751766000000


def test_refuse_direct_underlying_via_call_meta(resolver: OperationResolverServer):
    resp = resolver.call_meta("get_financial_metric", ticker="0700.HK", metric="revenue")
    assert resp["success"] is False
    assert "Unknown meta-tool" in resp["error"]


def test_embedding_backend_fails_without_config(monkeypatch):
    monkeypatch.delenv("EMBEDDING_BASE_URL", raising=False)
    monkeypatch.delenv("EMBEDDING_API_KEY", raising=False)
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    with pytest.raises(RuntimeError, match="EMBEDDING_BASE_URL"):
        OpenAICompatibleEmbeddingBackend()


def test_embedding_model_auto_rejected(monkeypatch):
    with pytest.raises(RuntimeError, match="auto"):
        OpenAICompatibleEmbeddingBackend(
            base_url="https://example.invalid",
            api_key="dummy",
            model="auto",
        )


def test_cosine_dimension_mismatch():
    with pytest.raises(ValueError, match="dimension"):
        cosine_similarity([1.0, 0.0], [1.0])


def test_embedding_cache_key_uses_model_and_fingerprint(tmp_path: Path):
    registry = build_default_registry()
    backend = DeterministicHashEmbeddingBackend(model_name="mock-a")
    r1 = OperationRetriever(registry, backend, cache_dir=tmp_path / "c")
    q = "filing summary annual report"
    rank1 = r1.rank(q, top_k=2)
    # Reuse cache with same model.
    r2 = OperationRetriever(registry, backend, cache_dir=tmp_path / "c")
    rank2 = r2.rank(q, top_k=2)
    assert rank1 == rank2
    # Cache files must not contain api_key-like fields.
    for path in (tmp_path / "c").glob("*.json"):
        text = path.read_text(encoding="utf-8")
        assert "api_key" not in text.lower()
        assert "Authorization" not in text
