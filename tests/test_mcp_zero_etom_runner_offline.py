"""Offline mock/dry-run/preflight/index/cache tests for MCP-Zero × ETOM."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hk_equity_mcp.mcp_zero.embedding import (
    CountingEmbedder,
    build_embedding_table,
    embedding_cache_identity,
    identity_digest,
    load_complete_table,
    table_status,
)
from hk_equity_mcp.mcp_zero.metadata import McpZeroCatalog, ServerMeta, ToolMeta
from hk_equity_mcp.mcp_zero.router import HierarchicalRouter
from hk_equity_mcp.mcp_zero.schemas import ActiveToolRequest
from hk_equity_mcp.mcp_zero_etom import EMBEDDING_NAMESPACE, EXPECTED_HK_METHOD_CARD_SHA256
from hk_equity_mcp.mcp_zero_etom.identity import assert_resume_compatible
from hk_equity_mcp.mcp_zero_etom.metadata import build_etom_catalog
from hk_equity_mcp.mcp_zero_etom.recorder import EtomRouteRecorder
from hk_equity_mcp.mcp_zero_etom.runner import EtomMcpZeroRunner
from hk_equity_mcp.operation_resolver.retrieval import DeterministicHashEmbeddingBackend
from hk_equity_mcp.mcp_zero import method_identity_sha256

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "etom_mini"


@pytest.fixture()
def fixture_catalog():
    payload = json.loads((FIXTURE / "data" / "mcp_registry.json").read_text())
    return build_etom_catalog(payload)


def test_hierarchical_server_then_tool_restriction(fixture_catalog):
    backend = DeterministicHashEmbeddingBackend(model_name="text-embedding-3-small")
    embedder = CountingEmbedder(backend, model_name="text-embedding-3-small")
    table = build_embedding_table(
        fixture_catalog,
        embedder,
        cache_dir=None,
        use_summary=True,
        namespace=EMBEDDING_NAMESPACE,
    )
    router = HierarchicalRouter(
        fixture_catalog, table, embedder, server_top_k=1, tool_top_k=5
    )
    req = ActiveToolRequest(server="notes markdown", tool="list notes", round_index=0)
    round_ = router.retrieve(req)
    assert round_.servers
    selected_server = round_.servers[0].server_id
    for hit in round_.tools:
        assert hit.server_id == selected_server


def test_incomplete_cache_fail_closed(tmp_path, fixture_catalog):
    status = table_status(
        fixture_catalog,
        model_name="text-embedding-3-small",
        cache_dir=tmp_path,
        use_summary=True,
        namespace=EMBEDDING_NAMESPACE,
    )
    assert status["complete"] is False
    with pytest.raises(Exception):
        load_complete_table(
            fixture_catalog,
            model_name="text-embedding-3-small",
            cache_dir=tmp_path,
            use_summary=True,
            namespace=EMBEDDING_NAMESPACE,
        )


def test_cache_checkpoint_resume(tmp_path, fixture_catalog):
    backend = DeterministicHashEmbeddingBackend(model_name="text-embedding-3-small")
    embedder = CountingEmbedder(backend, model_name="text-embedding-3-small")
    table = build_embedding_table(
        fixture_catalog,
        embedder,
        cache_dir=tmp_path,
        use_summary=True,
        namespace=EMBEDDING_NAMESPACE,
    )
    assert table.complete
    assert table.identity["namespace"] == EMBEDDING_NAMESPACE
    # reuse
    table2 = load_complete_table(
        fixture_catalog,
        model_name="text-embedding-3-small",
        cache_dir=tmp_path,
        use_summary=True,
        namespace=EMBEDDING_NAMESPACE,
    )
    assert table2.cache_reused is True
    # HK digest namespace default unchanged
    hk_ident = embedding_cache_identity(
        embedding_model="text-embedding-3-small",
        catalog_fingerprint="x",
        metadata_hash="y",
    )
    assert hk_ident["namespace"] == "mcp_zero_hk_v1"
    assert method_identity_sha256() == EXPECTED_HK_METHOD_CARD_SHA256


def test_recorder_no_real_execution(fixture_catalog):
    payload = json.loads((FIXTURE / "data" / "mcp_registry.json").read_text())
    from hk_equity_mcp.etom.convert import convert_registry
    _, official = convert_registry(payload)
    rec = EtomRouteRecorder(official)
    oid = next(iter(official))
    rec.set_retrieved({oid})
    out = rec.invoke(oid, {}, [])
    assert out["executed"] is False
    assert rec.real_tool_executions == 0


def test_resume_identity_prefix():
    base = {
        "experiment": "e",
        "condition": "mcp_zero",
        "method_identity_sha256": "a",
        "question_ids": ["L1-q1", "L2-q1"],
        "server_top_k": 3,
        "tool_top_k": 5,
    }
    ok = dict(base)
    ok["question_ids"] = ["L1-q1", "L2-q1", "L3-q1"]
    assert_resume_compatible(base, ok)
    bad = dict(base)
    bad["server_top_k"] = 5
    with pytest.raises(RuntimeError):
        assert_resume_compatible(base, bad)


def test_dry_run_preflight_mock(tmp_path, monkeypatch):
    # Fail closed if any network attempted
    import urllib.request

    def boom(*a, **k):
        raise RuntimeError("network forbidden in offline test")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    out = tmp_path / "dry"
    runner = EtomMcpZeroRunner(
        ROOT,
        out,
        split="formal100",
        mode="dry-run",
        fixture_dir=str(FIXTURE),
        embedding_cache_dir=tmp_path / "cache",
        allow_dirty=True,
    )
    report = runner.execute()
    assert report["action"] == "dry-run"
    assert (out / "preflight_report.json").exists()

    out2 = tmp_path / "pre"
    runner2 = EtomMcpZeroRunner(
        ROOT,
        out2,
        split="pilot10",
        mode="preflight",
        fixture_dir=str(FIXTURE),
        embedding_cache_dir=tmp_path / "cache2",
        allow_dirty=True,
    )
    r2 = runner2.execute()
    assert r2["action"] == "preflight"

    out3 = tmp_path / "mock"
    runner3 = EtomMcpZeroRunner(
        ROOT,
        out3,
        split="formal100",
        mode="mock",
        fixture_dir=str(FIXTURE),
        embedding_cache_dir=tmp_path / "cache3",
        allow_dirty=True,
        max_tasks=5,
    )
    r3 = runner3.execute()
    assert r3["action"] == "mock"
    preds = (out3 / "predictions.jsonl").read_text().strip().splitlines()
    assert len(preds) >= 1
    row = json.loads(preds[0])
    assert row["real_tool_executions"] == 0
    # raw_content preserved
    assert any(
        isinstance(e, dict) and e.get("raw_content") is not None
        for e in (row.get("trace") or [])
    )
    for name in [
        "run_metadata.json",
        "grouped_results.json",
        "cost_summary.json",
        "retrieval_analysis.json",
        "error_summary.json",
        "protocol_summary.json",
        "per_task_metrics.csv",
        "source_manifest.json",
    ]:
        assert (out3 / name).exists(), name
