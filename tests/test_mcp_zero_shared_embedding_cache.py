"""Shared MCP-Zero embedding cache lifecycle (offline; no paid APIs)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import pytest

from hk_equity_mcp.experiment.unified_mcp_zero_runner import (
    UnifiedHkMcpZeroRunner,
    catalog_fingerprint,
)
from hk_equity_mcp.mcp_zero import method_identity_sha256
from hk_equity_mcp.mcp_zero.embedding import (
    CACHE_NAMESPACE,
    CacheNamespaceError,
    CountingEmbedder,
    IncompleteEmbeddingTableError,
    build_embedding_table,
    default_shared_cache_dir,
    file_sha256,
    install_verified_index,
    load_complete_table,
    resolve_embedding_cache_dir,
    table_paths,
    table_status,
)
from hk_equity_mcp.mcp_zero.metadata import build_hk_metadata
from hk_equity_mcp.mcp_zero.prompt import mcp_zero_prompt_hash
from hk_equity_mcp.operation_resolver.retrieval import DeterministicHashEmbeddingBackend
from hk_equity_mcp.real_mcp.unified_catalog import build_unified_catalog

ROOT = Path(__file__).resolve().parents[1]
PROMPT_HASH = "0879f47079d8ba345cda167829fa3474375aefd647f92efc139014d7a2885ab8"
METHOD_HASH = "77cc3b07a4c2b9af00ad3f6bfb7cba59c82084d94b7592f2c820f46f1d3e8d3f"


def _catalog():
    bundle = build_unified_catalog(root=ROOT)
    catalog = build_hk_metadata(bundle)
    fp = catalog_fingerprint(bundle.stats, bundle.source_catalogs)
    return bundle, catalog, fp


def test_hashes_unchanged_by_cache_plumbing():
    assert mcp_zero_prompt_hash() == PROMPT_HASH
    assert method_identity_sha256() == METHOD_HASH


def test_default_shared_cache_independent_of_output_dir(tmp_path: Path):
    shared = default_shared_cache_dir(ROOT)
    assert shared.name == CACHE_NAMESPACE
    assert shared.parent.name == ".cache"
    a = resolve_embedding_cache_dir(ROOT, None)
    b = resolve_embedding_cache_dir(ROOT, tmp_path / "custom" / CACHE_NAMESPACE)
    assert a == shared
    assert b == tmp_path / "custom" / CACHE_NAMESPACE
    # Identity does not include output_dir; two outputs share one cache path.
    out1 = tmp_path / "out_a"
    out2 = tmp_path / "out_b"
    r1 = UnifiedHkMcpZeroRunner(
        output_dir=out1, mock=True, allow_dirty=True, root=ROOT, dry_run=True
    )
    r2 = UnifiedHkMcpZeroRunner(
        output_dir=out2, mock=True, allow_dirty=True, root=ROOT, dry_run=True
    )
    assert r1.embedding_cache_dir == r2.embedding_cache_dir == shared


def test_matching_final_index_produces_zero_embedding_calls(tmp_path: Path):
    _, catalog, fp = _catalog()
    cache = tmp_path / CACHE_NAMESPACE
    embedder = CountingEmbedder(
        DeterministicHashEmbeddingBackend(), model_name="text-embedding-3-small"
    )
    built = build_embedding_table(
        catalog, embedder, cache_dir=cache, catalog_fingerprint=fp, use_summary=True
    )
    assert built.complete and built.build_calls > 0
    calls_after_build = embedder.calls
    reused = build_embedding_table(
        catalog, embedder, cache_dir=cache, catalog_fingerprint=fp, use_summary=True
    )
    assert reused.cache_reused is True
    assert reused.build_calls == 0
    assert embedder.calls == calls_after_build
    loaded = load_complete_table(
        catalog,
        model_name="text-embedding-3-small",
        cache_dir=cache,
        catalog_fingerprint=fp,
        use_summary=True,
    )
    assert loaded.cache_reused is True
    assert loaded.build_calls == 0
    assert embedder.calls == calls_after_build


def test_partial_timeout_not_treated_as_complete(tmp_path: Path):
    _, catalog, fp = _catalog()
    cache = tmp_path / CACHE_NAMESPACE
    embedder = CountingEmbedder(
        DeterministicHashEmbeddingBackend(), model_name="text-embedding-3-small"
    )
    # Build once to learn identity paths, then simulate partial-only state.
    table = build_embedding_table(
        catalog, embedder, cache_dir=cache, catalog_fingerprint=fp, use_summary=True
    )
    final = Path(table.cache_path)
    partial = Path(str(final).replace(".json", ".partial.jsonl"))
    # Move complete aside and leave only one checkpointed unit as partial.
    payload = json.loads(final.read_text(encoding="utf-8"))
    key = next(iter(payload["vectors"]))
    vec = payload["vectors"][key]
    final.unlink()
    partial.write_text(json.dumps({"k": key, "v": vec}) + "\n", encoding="utf-8")
    status = table_status(
        catalog,
        model_name="text-embedding-3-small",
        cache_dir=cache,
        catalog_fingerprint=fp,
        use_summary=True,
    )
    assert status["complete"] is False
    assert status["final_complete"] is False
    assert status["partial_completed_count"] == 1
    with pytest.raises(IncompleteEmbeddingTableError):
        load_complete_table(
            catalog,
            model_name="text-embedding-3-small",
            cache_dir=cache,
            catalog_fingerprint=fp,
            use_summary=True,
        )


def test_partial_batches_are_resumable(tmp_path: Path):
    _, catalog, fp = _catalog()
    cache = tmp_path / "partial_resume"
    embedder = CountingEmbedder(
        DeterministicHashEmbeddingBackend(), model_name="text-embedding-3-small"
    )
    from hk_equity_mcp.mcp_zero.embedding import _units, embedding_cache_identity

    units = _units(catalog, True)
    identity = embedding_cache_identity(
        embedding_model="text-embedding-3-small",
        catalog_fingerprint=fp,
        metadata_hash=catalog.fingerprint(),
        use_summary=True,
    )
    final, partial, _ = table_paths(cache, identity)
    cache.mkdir(parents=True)
    kept = units[:3]
    lines: List[str] = []
    for key, text in kept:
        vec = embedder.embed([text])[0]
        lines.append(json.dumps({"k": key, "v": vec}))
    partial.write_text("\n".join(lines) + "\n", encoding="utf-8")
    embedder.calls = 0
    resumed = build_embedding_table(
        catalog, embedder, cache_dir=cache, catalog_fingerprint=fp, use_summary=True
    )
    assert resumed.index_action == "resume_partial"
    assert resumed.complete is True
    assert final.exists()
    assert not partial.exists()
    # Only remaining units were embedded.
    assert embedder.calls >= 1


def test_foreign_identity_and_damaged_json_rejected(tmp_path: Path):
    _, catalog, fp = _catalog()
    cache = tmp_path / CACHE_NAMESPACE
    cache.mkdir(parents=True)
    from hk_equity_mcp.mcp_zero.embedding import embedding_cache_identity

    identity = embedding_cache_identity(
        embedding_model="text-embedding-3-small",
        catalog_fingerprint=fp,
        metadata_hash=catalog.fingerprint(),
        use_summary=True,
    )
    final, _, _ = table_paths(cache, identity)
    final.write_text("{not-json", encoding="utf-8")
    with pytest.raises(IncompleteEmbeddingTableError):
        load_complete_table(
            catalog,
            model_name="text-embedding-3-small",
            cache_dir=cache,
            catalog_fingerprint=fp,
            use_summary=True,
        )
    final.write_text(
        json.dumps({"namespace": "etom_catalog", "complete": True, "vectors": {}}),
        encoding="utf-8",
    )
    with pytest.raises(CacheNamespaceError):
        load_complete_table(
            catalog,
            model_name="text-embedding-3-small",
            cache_dir=cache,
            catalog_fingerprint=fp,
            use_summary=True,
        )


def test_require_complete_index_missing_makes_no_network_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    calls = {"n": 0}

    class _Boom:
        def __init__(self, *a, **k):
            calls["n"] += 1
            raise AssertionError("embedding client must not be constructed")

    monkeypatch.setattr(
        "hk_equity_mcp.experiment.unified_mcp_zero_runner.OpenAICompatibleEmbeddingBackend",
        _Boom,
    )
    monkeypatch.setattr(
        "hk_equity_mcp.experiment.unified_mcp_zero_runner.OpenAICompatibleChatClient",
        _Boom,
    )
    runner = UnifiedHkMcpZeroRunner(
        output_dir=tmp_path / "formal",
        split="development",
        max_tasks=1,
        allow_dirty=True,
        root=ROOT,
        embedding_cache_dir=tmp_path / "empty_shared" / CACHE_NAMESPACE,
        require_complete_index=True,
        mock=False,
        dry_run=False,
    )
    # Skip env/git gates by calling build_agent directly after setting emb model.
    runner.emb_meta = {"model": "text-embedding-3-small"}
    with pytest.raises(IncompleteEmbeddingTableError):
        runner.build_agent()
    assert calls["n"] == 0


def test_smoke_and_formal_outputs_reuse_same_shared_index(tmp_path: Path):
    shared = tmp_path / "shared" / CACHE_NAMESPACE
    smoke_out = tmp_path / "smoke"
    formal_out = tmp_path / "formal"
    r_smoke = UnifiedHkMcpZeroRunner(
        output_dir=smoke_out,
        split="development",
        smoke=True,
        max_tasks=1,
        mock=True,
        allow_dirty=True,
        root=ROOT,
        embedding_cache_dir=shared,
    )
    meta1 = r_smoke.run()
    assert meta1["embedding_index"]["complete"] is True
    assert str(meta1["embedding_index"]["cache_path"]).startswith(str(shared))
    assert meta1["embedding_cache_dir"] == str(shared)
    # Second output reuses without rebuild cost in report.
    r_formal = UnifiedHkMcpZeroRunner(
        output_dir=formal_out,
        split="development",
        smoke=True,
        max_tasks=1,
        mock=True,
        allow_dirty=True,
        root=ROOT,
        embedding_cache_dir=shared,
        require_complete_index=False,
    )
    meta2 = r_formal.run()
    assert meta2["embedding_index"]["cache_reused"] is True
    assert meta2["embedding_index"]["build_calls"] == 0
    assert meta2["embedding_index"]["identity_sha256"] == meta1["embedding_index"][
        "identity_sha256"
    ]
    assert not (smoke_out / ".cache").exists()
    assert not (formal_out / ".cache").exists()


def test_install_verified_index_copies_without_api(tmp_path: Path):
    _, catalog, fp = _catalog()
    src_dir = tmp_path / "src" / CACHE_NAMESPACE
    embedder = CountingEmbedder(
        DeterministicHashEmbeddingBackend(), model_name="text-embedding-3-small"
    )
    table = build_embedding_table(
        catalog, embedder, cache_dir=src_dir, catalog_fingerprint=fp, use_summary=True
    )
    src = Path(table.cache_path)
    src_sha = file_sha256(src)
    dest = tmp_path / "shared" / CACHE_NAMESPACE
    report = install_verified_index(
        src,
        cache_dir=dest,
        catalog=catalog,
        model_name="text-embedding-3-small",
        catalog_fingerprint=fp,
        use_summary=True,
        expected_identity_sha256=table.identity_sha256,
    )
    assert report["complete"] is True
    assert report["cache_file_sha256"] == src_sha
    assert Path(report["cache_path"]).exists()
    assert src.exists()  # source preserved
    status = table_status(
        catalog,
        model_name="text-embedding-3-small",
        cache_dir=dest,
        catalog_fingerprint=fp,
        use_summary=True,
    )
    assert status["complete"] is True
    assert status["cache_reused"] is True
    assert status["identity_sha256"] == table.identity_sha256


def test_invalidated_output_dir_refuses_run(tmp_path: Path):
    out = tmp_path / "bad"
    out.mkdir()
    (out / "INVALIDATED.md").write_text("invalidated", encoding="utf-8")
    runner = UnifiedHkMcpZeroRunner(
        output_dir=out,
        split="development",
        mock=True,
        allow_dirty=True,
        root=ROOT,
        max_tasks=1,
    )
    with pytest.raises(RuntimeError, match="invalidated"):
        runner.run()
