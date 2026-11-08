"""APV-Fin operation embedding index transport + checkpoint tests (offline)."""

from __future__ import annotations

import http.client
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence
from unittest.mock import MagicMock

import pytest

from hk_equity_mcp.apv_fin.operation_embedding_index import (
    CACHE_NAMESPACE,
    IncompleteOperationIndexError,
    OperationIndexIdentityError,
    build_index_identity,
    build_or_resume_index,
    index_status,
    load_complete_index,
    try_import_ov_compatible_cache,
)
from hk_equity_mcp.experiment.unified_apv_fin_runner import (
    UnifiedHkApvFinRunner,
    load_apv_fin_config,
)
from hk_equity_mcp.operation_resolver.retrieval import (
    DeterministicHashEmbeddingBackend,
    OpenAICompatibleEmbeddingBackend,
)
from hk_equity_mcp.real_mcp.unified_catalog import build_unified_catalog

ROOT = Path(__file__).resolve().parents[1]


class FlakyEmbedder:
    """Scripted failures then success; records requested text batches."""

    def __init__(self, plan: List[Any], dim: int = 8):
        self.plan = list(plan)
        self.dim = dim
        self.calls: List[List[str]] = []
        self.model_name = "text-embedding-3-small"

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        texts = list(texts)
        self.calls.append(texts)
        if not self.plan:
            raise RuntimeError("FlakyEmbedder plan exhausted")
        event = self.plan.pop(0)
        if isinstance(event, Exception):
            raise event
        if event == "ok":
            backend = DeterministicHashEmbeddingBackend(dim=self.dim)
            return backend.embed(texts)
        raise RuntimeError(f"unknown plan event {event!r}")


def _identity(registry, catalog_fp: str = "fp-test") -> Dict[str, Any]:
    return build_index_identity(
        registry=registry,
        catalog_fingerprint=catalog_fp,
        embedding_model="text-embedding-3-small",
        embedding_provider_host="api.zhizengzeng.com",
    )


def test_config_embedding_transport_knobs():
    cfg = load_apv_fin_config(root=ROOT)
    assert cfg["embedding_batch_size"] == 16
    assert cfg["embedding_request_timeout_sec"] == 120
    assert cfg["embedding_max_retries"] == 3
    assert cfg["embedding_retry_backoff_seconds"] == [1, 2, 4]
    assert cfg["limits"]["max_total_llm_calls_per_task"] == 14


def test_timeout_then_success_via_backend(monkeypatch):
    attempts = {"n": 0}

    def fake_once(self, texts):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise TimeoutError("boom")
        return DeterministicHashEmbeddingBackend(dim=4).embed(texts)

    monkeypatch.setattr(
        OpenAICompatibleEmbeddingBackend, "_embed_once", fake_once
    )
    sleeps: List[float] = []
    monkeypatch.setattr(
        "hk_equity_mcp.operation_resolver.retrieval.time.sleep",
        lambda s: sleeps.append(s),
    )
    backend = OpenAICompatibleEmbeddingBackend(
        base_url="https://example.invalid/v1",
        api_key="dummy",
        model="text-embedding-3-small",
        timeout_sec=1,
        max_retries=3,
        retry_backoff_seconds=[1, 2, 4],
    )
    out = backend.embed(["hello"])
    assert len(out) == 1
    assert attempts["n"] == 2
    assert sleeps == [1]
    assert backend.retry_count == 1


def test_429_then_success(monkeypatch):
    attempts = {"n": 0}

    def fake_once(self, texts):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise RuntimeError(
                "Embedding request failed with HTTP 429 for model text-embedding-3-small."
            )
        return [[0.1, 0.2]]

    monkeypatch.setattr(OpenAICompatibleEmbeddingBackend, "_embed_once", fake_once)
    monkeypatch.setattr(
        "hk_equity_mcp.operation_resolver.retrieval.time.sleep", lambda s: None
    )
    backend = OpenAICompatibleEmbeddingBackend(
        base_url="https://example.invalid/v1",
        api_key="dummy",
        model="text-embedding-3-small",
        max_retries=2,
        retry_backoff_seconds=[1, 2],
    )
    assert backend.embed(["x"]) == [[0.1, 0.2]]
    assert attempts["n"] == 2


def test_remote_disconnected_then_success(monkeypatch):
    attempts = {"n": 0}

    def fake_once(self, texts):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise http.client.RemoteDisconnected("closed")
        return [[0.3]]

    monkeypatch.setattr(OpenAICompatibleEmbeddingBackend, "_embed_once", fake_once)
    monkeypatch.setattr(
        "hk_equity_mcp.operation_resolver.retrieval.time.sleep", lambda s: None
    )
    backend = OpenAICompatibleEmbeddingBackend(
        base_url="https://example.invalid/v1",
        api_key="dummy",
        model="text-embedding-3-small",
        max_retries=2,
        retry_backoff_seconds=[1],
    )
    assert backend.embed(["x"]) == [[0.3]]


def test_non_retryable_400(monkeypatch):
    def fake_once(self, texts):
        raise RuntimeError(
            "Embedding request failed with HTTP 400 for model text-embedding-3-small."
        )

    monkeypatch.setattr(OpenAICompatibleEmbeddingBackend, "_embed_once", fake_once)
    backend = OpenAICompatibleEmbeddingBackend(
        base_url="https://example.invalid/v1",
        api_key="dummy",
        model="text-embedding-3-small",
        max_retries=3,
        retry_backoff_seconds=[1, 2, 4],
    )
    with pytest.raises(RuntimeError, match="HTTP 400"):
        backend.embed(["x"])


def test_retry_exhaustion(monkeypatch):
    def fake_once(self, texts):
        raise TimeoutError("still down")

    monkeypatch.setattr(OpenAICompatibleEmbeddingBackend, "_embed_once", fake_once)
    monkeypatch.setattr(
        "hk_equity_mcp.operation_resolver.retrieval.time.sleep", lambda s: None
    )
    backend = OpenAICompatibleEmbeddingBackend(
        base_url="https://example.invalid/v1",
        api_key="dummy",
        model="text-embedding-3-small",
        max_retries=2,
        retry_backoff_seconds=[1, 2],
    )
    with pytest.raises(RuntimeError, match="TimeoutError"):
        backend.embed(["x"])
    assert backend.retry_count == 2


def test_batch_checkpoint_resume_and_no_reembed(tmp_path: Path):
    bundle = build_unified_catalog(root=ROOT)
    # Use a tiny synthetic registry subset for speed.
    ids = sorted(bundle.registry.keys())[:5]
    registry = {oid: bundle.registry[oid] for oid in ids}
    cache_dir = tmp_path / CACHE_NAMESPACE
    identity = _identity(registry)
    # Interrupt after first batch of size 2.
    call_log: List[int] = []

    class CountingBackend:
        model_name = "text-embedding-3-small"

        def __init__(self):
            self.inner = DeterministicHashEmbeddingBackend(dim=8)
            self.fail_after = 1
            self.n = 0

        def embed(self, texts):
            call_log.append(len(texts))
            self.n += 1
            if self.n == self.fail_after:
                # Checkpoint first batch then fail.
                vecs = self.inner.embed(texts)
                # Raise after returning? simulate mid-run by failing on 2nd call.
                return vecs
            if self.n == 2:
                raise TimeoutError("interrupted")
            return self.inner.embed(texts)

    backend = CountingBackend()
    with pytest.raises(TimeoutError):
        build_or_resume_index(
            registry=registry,
            identity=identity,
            cache_dir=cache_dir,
            embed_fn=backend.embed,
            batch_size=2,
        )
    st = index_status(registry=registry, identity=identity, cache_dir=cache_dir)
    assert st.complete is False
    assert st.completed == 2
    # Resume — remaining only.
    backend.fail_after = 99
    backend.n = 10
    before_calls = len(call_log)
    result = build_or_resume_index(
        registry=registry,
        identity=identity,
        cache_dir=cache_dir,
        embed_fn=backend.embed,
        batch_size=2,
    )
    assert result["complete"] is True
    # Only missing ops requested (3 remaining → batches 2+1).
    new_sizes = call_log[before_calls:]
    assert sum(new_sizes) == 3
    # Reload complete — no embed.
    load_complete_index(registry=registry, identity=identity, cache_dir=cache_dir)
    again = build_or_resume_index(
        registry=registry,
        identity=identity,
        cache_dir=cache_dir,
        embed_fn=backend.embed,
        batch_size=2,
    )
    assert again["action"] == "reused_final"


def test_identity_and_dimension_mismatch_fail_closed(tmp_path: Path):
    bundle = build_unified_catalog(root=ROOT)
    ids = sorted(bundle.registry.keys())[:3]
    registry = {oid: bundle.registry[oid] for oid in ids}
    cache_dir = tmp_path / CACHE_NAMESPACE
    identity = _identity(registry)
    backend = DeterministicHashEmbeddingBackend(dim=8)
    build_or_resume_index(
        registry=registry,
        identity=identity,
        cache_dir=cache_dir,
        embed_fn=backend.embed,
        batch_size=16,
    )
    from hk_equity_mcp.apv_fin.operation_embedding_index import _load_final, index_paths

    bad = dict(identity)
    bad["documents_hash"] = "0" * 64
    final, _, _, _ = index_paths(cache_dir, identity)
    with pytest.raises(OperationIndexIdentityError):
        _load_final(final, bad)

    # Dimension drift on partial append
    cache2 = tmp_path / "dim" / CACHE_NAMESPACE
    identity2 = _identity(registry, catalog_fp="fp2")
    identity2["expected_vector_dimension"] = 8

    def wrong_dim(texts):
        return DeterministicHashEmbeddingBackend(dim=4).embed(texts)

    with pytest.raises(OperationIndexIdentityError):
        build_or_resume_index(
            registry=registry,
            identity=identity2,
            cache_dir=cache2,
            embed_fn=wrong_dim,
            batch_size=16,
        )


def test_batch_size_sixteen(tmp_path: Path):
    bundle = build_unified_catalog(root=ROOT)
    # 39 ops with batch 16 → sizes 16,16,7
    cache_dir = tmp_path / CACHE_NAMESPACE
    identity = _identity(bundle.registry)
    sizes: List[int] = []

    def capture(texts):
        sizes.append(len(texts))
        return DeterministicHashEmbeddingBackend(dim=8).embed(texts)

    result = build_or_resume_index(
        registry=bundle.registry,
        identity=identity,
        cache_dir=cache_dir,
        embed_fn=capture,
        batch_size=16,
    )
    assert result["complete"] is True
    assert sizes == [16, 16, 7]


def test_require_complete_index_rejection(tmp_path: Path):
    cache_dir = tmp_path / CACHE_NAMESPACE
    cache_dir.mkdir(parents=True)
    with pytest.raises(RuntimeError, match="require-complete-index|incomplete"):
        UnifiedHkApvFinRunner(
            output_dir=tmp_path / "out",
            split="development",
            smoke=True,
            max_tasks=1,
            mock=False,
            dry_run=False,
            allow_dirty=True,
            root=ROOT,
            require_complete_index=True,
            embedding_cache_dir=cache_dir,
        ).build_agent()


def test_live_run_cannot_auto_build_index(tmp_path: Path, monkeypatch):
    cache_dir = tmp_path / CACHE_NAMESPACE
    cache_dir.mkdir(parents=True)
    # Pretend env is present so we get past profile loads if needed.
    monkeypatch.setenv("EMBEDDING_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("EMBEDDING_API_KEY", "dummy")
    monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-3-small")
    monkeypatch.setenv("LLM_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("LLM_API_KEY", "dummy")
    monkeypatch.setenv("LLM_MODEL", "deepseek-v4.1-flash")
    monkeypatch.setenv("LLM_PROVIDER", "openai_compatible")
    runner = UnifiedHkApvFinRunner(
        output_dir=tmp_path / "out",
        split="development",
        smoke=True,
        max_tasks=1,
        mock=False,
        allow_dirty=True,
        root=ROOT,
        require_complete_index=False,
        allow_auto_build_index=False,
        embedding_cache_dir=cache_dir,
    )
    with pytest.raises(RuntimeError, match="auto-build disabled|incomplete"):
        runner.build_agent()


def test_build_index_only_does_not_instantiate_generation_llm(tmp_path: Path, monkeypatch):
    cache_dir = tmp_path / CACHE_NAMESPACE
    created = {"llm": False}

    class BoomLLM:
        def __init__(self, *a, **k):
            created["llm"] = True
            raise AssertionError("LLM must not be constructed")

    monkeypatch.setattr(
        "hk_equity_mcp.experiment.unified_apv_fin_runner.OpenAICompatibleChatClient",
        BoomLLM,
    )
    runner = UnifiedHkApvFinRunner(
        output_dir=tmp_path / "out",
        mock=True,
        allow_dirty=True,
        root=ROOT,
        embedding_cache_dir=cache_dir,
    )
    result = runner.build_index_only()
    assert result["complete"] is True
    assert created["llm"] is False


def test_index_status_zero_network(tmp_path: Path):
    cache_dir = tmp_path / CACHE_NAMESPACE
    runner = UnifiedHkApvFinRunner(
        output_dir=tmp_path / "out",
        mock=True,
        allow_dirty=True,
        root=ROOT,
        embedding_cache_dir=cache_dir,
    )
    # Spy: if embed called during status, fail.
    backend = MagicMock()
    runner._make_embedding_backend = MagicMock(return_value=backend)  # type: ignore
    status = runner.get_index_status()
    assert status["complete"] is False
    assert status["total"] == 39
    backend.embed.assert_not_called()


def test_ov_import_requires_full_identity_match(tmp_path: Path):
    bundle = build_unified_catalog(root=ROOT)
    ids = sorted(bundle.registry.keys())[:2]
    registry = {oid: bundle.registry[oid] for oid in ids}
    identity = _identity(registry)
    ov = tmp_path / "operation_embeddings"
    ov.mkdir()
    # Wrong model → no import
    assert (
        try_import_ov_compatible_cache(
            registry=registry, identity=identity, ov_cache_dir=ov
        )
        is None
    )
