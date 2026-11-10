"""Live-path hardening tests for Retrieval Audit (fake transports only)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Sequence

import pytest

from hk_equity_mcp.experiment.pe_fin_lite_retrieval_audit.formula_selection import (
    REAL_EMBEDDING_BACKEND,
    assert_formal_formula_lock_allowed,
)
from hk_equity_mcp.experiment.pe_fin_lite_retrieval_audit.live_path import (
    LivePathError,
    build_embedding_backend,
    load_and_assert_embedding,
    load_and_assert_generation,
    public_meta_without_secrets,
    resolve_mode,
)
from hk_equity_mcp.experiment.pe_fin_lite_retrieval_audit.planner_only import (
    run_planner_only,
)
from hk_equity_mcp.experiment.pe_fin_lite_retrieval_audit.query_cache import (
    QueryCacheError,
    embed_planner_step_queries,
    embed_question_queries,
    load_complete_question_query_cache,
    load_complete_step_query_cache,
    question_query_cache_identity,
    step_query_cache_identity,
)
from hk_equity_mcp.experiment.pe_fin_lite_retrieval_audit.runner import (
    RetrievalAuditRunner,
)
from hk_equity_mcp.operation_resolver.retrieval import DeterministicHashEmbeddingBackend
from hk_equity_mcp.pe_fin_lite.identity import EXPECTED_EMBEDDING_MODEL
from hk_equity_mcp.pe_fin_lite.operation_embedding_index import (
    CACHE_NAMESPACE,
    build_index_identity,
    build_or_resume_index,
    index_status,
)

ROOT = Path(__file__).resolve().parents[1]


class CountingEmbedder:
    def __init__(self, dim: int = 8):
        self.dim = dim
        self.calls = 0
        self.model_name = EXPECTED_EMBEDDING_MODEL
        self._backend = DeterministicHashEmbeddingBackend(
            model_name=EXPECTED_EMBEDDING_MODEL, dim=dim
        )

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        self.calls += 1
        return self._backend.embed(list(texts))


def test_no_mock_flag_refused():
    with pytest.raises(LivePathError, match="--no-mock"):
        resolve_mode(no_mock=True)


def test_live_without_env_fails(tmp_path: Path):
    runner = RetrievalAuditRunner(
        root=ROOT, output_dir=tmp_path / "live", live=True
    )
    with pytest.raises(LivePathError, match="embedding-env-file"):
        runner.build_index_only()
    with pytest.raises(LivePathError, match="env-file"):
        runner.run_planner_only()


def test_live_wrong_provider_model_host(tmp_path: Path, monkeypatch):
    gen = tmp_path / "bad_gen.env"
    gen.write_text(
        "\n".join(
            [
                "LLM_PROVIDER=openai",
                "LLM_BASE_URL=https://api.openai.com/v1",
                "LLM_API_KEY=sk-test",
                "LLM_MODEL=gpt-4o-mini",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(LivePathError, match="generation provider"):
        load_and_assert_generation(ROOT, str(gen))

    emb = tmp_path / "bad_emb.env"
    emb.write_text(
        "\n".join(
            [
                "EMBEDDING_BASE_URL=https://example.invalid/v1",
                "EMBEDDING_API_KEY=sk-emb",
                "EMBEDDING_MODEL=text-embedding-ada-002",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(LivePathError, match="embedding model"):
        load_and_assert_embedding(ROOT, str(emb))


def test_planner_live_llm_none_refuses_mock_fallback(tmp_path: Path):
    with pytest.raises(RuntimeError, match="llm=None"):
        run_planner_only(
            root=ROOT,
            tasks=[{"task_id": "t1", "question": "q", "domain": "financial"}],
            output_dir=tmp_path / "planner",
            cache_identity={"identity_sha256": "x", "kind": "test"},
            llm=None,
            live=True,
            mock=False,
            dry_run=False,
            resume=False,
            capability_profile={},
        )


def test_catalog_checkpoint_resume(tmp_path: Path):
    from hk_equity_mcp.real_mcp.unified_catalog import build_unified_catalog

    bundle = build_unified_catalog(root=ROOT)
    # Tiny subset registry for speed.
    subset_ids = sorted(bundle.registry.keys())[:5]
    registry = {oid: bundle.registry[oid] for oid in subset_ids}
    cache_dir = tmp_path / CACHE_NAMESPACE
    identity = build_index_identity(
        registry=registry,
        catalog_fingerprint="test-fp",
        embedding_model=EXPECTED_EMBEDDING_MODEL,
        embedding_provider_host="api.zhizengzeng.com",
    )
    emb = CountingEmbedder(dim=8)

    # Interrupt after first batch by wrapping embed.
    calls_before_interrupt = {"n": 0}
    original = emb.embed

    def flaky(texts):
        calls_before_interrupt["n"] += 1
        if calls_before_interrupt["n"] == 1:
            out = original(texts[:1])
            # Write only first then raise to simulate interrupt mid-run via
            # partial-only path: call build with batch_size=1 and stop early.
            return out
        return original(texts)

    # First pass: batch_size=1, only process by raising after first successful batch
    # using a custom embed that succeeds once then fails.
    class OnceThenFail:
        def __init__(self):
            self.n = 0
            self.model_name = EXPECTED_EMBEDDING_MODEL
            self._b = DeterministicHashEmbeddingBackend(
                model_name=EXPECTED_EMBEDDING_MODEL, dim=8
            )

        def embed(self, texts):
            self.n += 1
            if self.n > 2:
                raise RuntimeError("simulated interrupt")
            return self._b.embed(texts)

    once = OnceThenFail()
    with pytest.raises(RuntimeError, match="simulated interrupt"):
        build_or_resume_index(
            registry=registry,
            identity=identity,
            cache_dir=cache_dir,
            embed_fn=once.embed,
            batch_size=1,
        )
    st = index_status(registry=registry, identity=identity, cache_dir=cache_dir)
    assert st.complete is False
    assert st.completed >= 1
    partial = Path(st.partial_path) if st.partial_path else None
    assert partial is not None and partial.exists()

    # Resume completes without wiping partial.
    emb2 = CountingEmbedder(dim=8)
    result = build_or_resume_index(
        registry=registry,
        identity=identity,
        cache_dir=cache_dir,
        embed_fn=emb2.embed,
        batch_size=2,
    )
    assert result["complete"] is True
    assert emb2.calls >= 1


def test_question_and_step_query_checkpoint_resume(tmp_path: Path):
    tasks = [
        {"task_id": "t1", "question": "revenue of A?"},
        {"task_id": "t2", "question": "close price of B?"},
        {"task_id": "t3", "question": "announcement about C?"},
    ]
    identity = question_query_cache_identity(
        dataset_identities={
            "manifest_sha256": "m",
            "ordered_task_ids_hash": "o",
            "question_identity_hash": "q",
        },
        embedding_model=EXPECTED_EMBEDDING_MODEL,
        embedding_provider_host="api.zhizengzeng.com",
        embedding_dim=None,
    )
    emb = CountingEmbedder(dim=8)

    class FailAfterOne:
        def __init__(self):
            self.n = 0
            self._b = DeterministicHashEmbeddingBackend(dim=8)

        def embed(self, texts):
            self.n += 1
            if self.n > 1:
                raise RuntimeError("interrupt qq")
            return self._b.embed(texts)

    fail = FailAfterOne()
    with pytest.raises(RuntimeError, match="interrupt qq"):
        embed_question_queries(
            tasks=tasks,
            output_dir=tmp_path / "qq",
            identity=identity,
            embed_fn=fail.embed,
            batch_size=1,
            resume=True,
        )
    partial = tmp_path / "qq" / "question_query_vectors.partial.jsonl"
    assert partial.exists()
    lines = [ln for ln in partial.read_text().splitlines() if ln.strip()]
    assert len(lines) == 1

    summary = embed_question_queries(
        tasks=tasks,
        output_dir=tmp_path / "qq",
        identity=identity,
        embed_fn=emb.embed,
        batch_size=2,
        resume=True,
    )
    assert summary["complete"] is True
    assert summary["newly_written"] == 2
    payload = load_complete_question_query_cache(
        tmp_path / "qq" / "question_query_vectors.json"
    )
    assert len(payload["vectors_by_task_id"]) == 3

    # Step queries
    planner_rows = [
        {
            "task_id": "t1",
            "retrieve_steps": [
                {
                    "id": "step1",
                    "type": "retrieve",
                    "domain": "financial",
                    "search_query": "financial revenue",
                    "required_outputs": ["revenue"],
                }
            ],
        },
        {
            "task_id": "t2",
            "retrieve_steps": [
                {
                    "id": "step1",
                    "type": "retrieve",
                    "domain": "market",
                    "search_query": "market close",
                    "required_outputs": ["close"],
                }
            ],
        },
    ]
    s_identity = step_query_cache_identity(
        dataset_identities={
            "manifest_sha256": "m",
            "ordered_task_ids_hash": "o",
            "question_identity_hash": "q",
        },
        planner_cache_sha256="plannerhash",
        embedding_model=EXPECTED_EMBEDDING_MODEL,
        embedding_provider_host="api.zhizengzeng.com",
        embedding_dim=None,
    )
    class FailStep:
        def __init__(self):
            self.n = 0
            self._b = DeterministicHashEmbeddingBackend(dim=8)

        def embed(self, texts):
            self.n += 1
            if self.n > 1:
                raise RuntimeError("interrupt sq")
            return self._b.embed(texts)

    with pytest.raises(RuntimeError, match="interrupt sq"):
        embed_planner_step_queries(
            planner_rows=planner_rows,
            output_dir=tmp_path / "sq",
            identity=s_identity,
            embed_fn=FailStep().embed,
            batch_size=1,
            resume=True,
        )
    assert (tmp_path / "sq" / "step_query_vectors.partial.jsonl").exists()
    s2 = embed_planner_step_queries(
        planner_rows=planner_rows,
        output_dir=tmp_path / "sq",
        identity=s_identity,
        embed_fn=emb.embed,
        batch_size=2,
        resume=True,
    )
    assert s2["complete"] is True
    load_complete_step_query_cache(tmp_path / "sq" / "step_query_vectors.json")


def test_incomplete_caches_refuse_ranking(tmp_path: Path):
    out = tmp_path / "audit"
    # offline_rank mode without caches
    runner = RetrievalAuditRunner(
        root=ROOT, output_dir=out, offline_rank=True
    )
    with pytest.raises(LivePathError, match="Incomplete catalog|Incomplete question"):
        runner.run_question_level()
    # planner cache missing
    with pytest.raises(FileNotFoundError, match="planner cache"):
        runner.run_rank_from_planner_cache()


def test_ranking_zero_network_from_mock_is_ok(tmp_path: Path):
    out = tmp_path / "mock"
    runner = RetrievalAuditRunner(root=ROOT, output_dir=out, mock=True)
    runner.run_planner_only()
    meta = runner.run_rank_from_planner_cache()
    assert meta.get("zero_network") is True
    assert meta.get("live") is False


def test_mock_cannot_formal_lock():
    with pytest.raises(RuntimeError, match="mock results cannot"):
        assert_formal_formula_lock_allowed(
            embedding_backend=REAL_EMBEDDING_BACKEND, mock=True
        )
    with pytest.raises(RuntimeError, match="real_text_embedding_3_small"):
        assert_formal_formula_lock_allowed(
            embedding_backend="mock_hash", mock=False
        )


def test_live_metadata_strips_secrets():
    dirty = {
        "provider": "openai_compatible",
        "api_key": "sk-secret",
        "token": "tok",
        "model": EXPECTED_EMBEDDING_MODEL,
        "password": "x",
    }
    clean = public_meta_without_secrets(dirty)
    assert "api_key" not in clean
    assert "token" not in clean
    assert "password" not in clean
    assert clean["model"] == EXPECTED_EMBEDDING_MODEL


def test_identity_mismatch_fail_closed(tmp_path: Path):
    tasks = [{"task_id": "t1", "question": "q1"}]
    identity_a = question_query_cache_identity(
        dataset_identities={
            "manifest_sha256": "m1",
            "ordered_task_ids_hash": "o",
            "question_identity_hash": "q",
        },
        embedding_model=EXPECTED_EMBEDDING_MODEL,
        embedding_provider_host="host-a",
        embedding_dim=8,
    )
    emb = CountingEmbedder(dim=8)
    embed_question_queries(
        tasks=tasks,
        output_dir=tmp_path / "qq",
        identity=identity_a,
        embed_fn=emb.embed,
        batch_size=1,
        resume=True,
    )
    identity_b = question_query_cache_identity(
        dataset_identities={
            "manifest_sha256": "m2",
            "ordered_task_ids_hash": "o",
            "question_identity_hash": "q",
        },
        embedding_model=EXPECTED_EMBEDDING_MODEL,
        embedding_provider_host="host-a",
        embedding_dim=8,
    )
    with pytest.raises(QueryCacheError, match="identity mismatch"):
        embed_question_queries(
            tasks=tasks,
            output_dir=tmp_path / "qq",
            identity=identity_b,
            embed_fn=emb.embed,
            batch_size=1,
            resume=True,
        )


def test_cli_no_mock_and_live_without_env(tmp_path: Path):
    import importlib.util

    script = ROOT / "scripts" / "run_pe_fin_lite_retrieval_audit.py"
    spec = importlib.util.spec_from_file_location("retrieval_audit_cli", script)
    assert spec and spec.loader
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)

    rc = cli.main(
        [
            "--root",
            str(ROOT),
            "--output-dir",
            str(tmp_path / "cli"),
            "--planner-only",
            "--no-mock",
        ]
    )
    assert rc == 2

    rc2 = cli.main(
        [
            "--root",
            str(ROOT),
            "--output-dir",
            str(tmp_path / "cli2"),
            "--build-index-only",
            "--live",
        ]
    )
    assert rc2 == 2


def test_select_formula_refuses_mock_formal(tmp_path: Path):
    out = tmp_path / "sel"
    runner = RetrievalAuditRunner(root=ROOT, output_dir=out, mock=True)
    runner.run_planner_only()
    runner.run_rank_from_planner_cache()
    meta = runner.run_select_formula(formal=True)
    assert meta["formal_lock_written"] is False
    assert "formal_lock_refused" in meta or not (
        out / "formula_selection" / "formula_lock.json"
    ).exists()


def test_dry_run_index_status_zero_network(tmp_path: Path):
    runner = RetrievalAuditRunner(
        root=ROOT, output_dir=tmp_path / "st", dry_run=True
    )
    status = runner.get_index_status()
    assert "completed" in status
    assert "expected" in status or "total" in status
    assert status.get("zero_network") is True


def test_build_embedding_backend_refuses_offline_rank():
    mode = resolve_mode(offline_rank=True)
    with pytest.raises(LivePathError, match="dry-run/offline-rank"):
        build_embedding_backend(mode=mode, emb_meta={"model": "x"})
