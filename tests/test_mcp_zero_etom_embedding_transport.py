"""Offline embedding transport / checkpoint reliability tests (no network)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pytest

from hk_equity_mcp.mcp_zero import method_identity_sha256
from hk_equity_mcp.mcp_zero.embedding import (
    CountingEmbedder,
    IncompleteEmbeddingTableError,
    build_embedding_table,
    table_status,
)
from hk_equity_mcp.mcp_zero_etom import (
    EMBEDDING_NAMESPACE,
    EXPECTED_FORMAL_SAMPLE_SHA256,
    EXPECTED_HK_METHOD_CARD_SHA256,
    EXPECTED_ORDERED_IDS_SHA256,
    FROZEN_EMBEDDING_BUILD,
    assert_embedding_build_config,
    etom_method_identity_sha256,
    frozen_embedding_build,
)
from hk_equity_mcp.mcp_zero_etom.metadata import build_etom_catalog
from hk_equity_mcp.mcp_zero_etom.prompt import etom_prompt_hash
from hk_equity_mcp.mcp_zero_etom.source import assert_formal_lock_identity
from hk_equity_mcp.operation_resolver.retrieval import (
    DeterministicHashEmbeddingBackend,
    OpenAICompatibleEmbeddingBackend,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "etom_mini"


@pytest.fixture()
def fixture_catalog():
    payload = json.loads((FIXTURE / "data" / "mcp_registry.json").read_text())
    return build_etom_catalog(payload)


class _ScriptedBackend:
    """Fake paid-like backend: scripted failures then success. No network."""

    def __init__(
        self,
        *,
        model_name: str = "text-embedding-3-small",
        dim: int = 8,
        failures: Optional[Sequence[BaseException]] = None,
        bad_count: bool = False,
    ):
        self.model_name = model_name
        self.dim = dim
        self.failures = list(failures or [])
        self.bad_count = bad_count
        self.calls = 0
        self.retry_count = 0
        self.on_attempt = None
        self.max_retries = 3
        self.retry_backoff_seconds = [0, 0, 0]  # no sleep in tests
        self._inner = DeterministicHashEmbeddingBackend(model_name=model_name, dim=dim)

    def embed(self, texts: List[str]) -> List[List[float]]:
        # Mimic OpenAICompatibleEmbeddingBackend retry loop.
        max_attempts = 1 + int(self.max_retries)
        last_exc: Optional[BaseException] = None
        for attempt in range(1, max_attempts + 1):
            if self.on_attempt:
                self.on_attempt(
                    {
                        "phase": "attempt",
                        "attempt": attempt,
                        "max_attempts": max_attempts,
                        "batch_size": len(texts),
                    }
                )
            self.calls += 1
            if self.failures:
                exc = self.failures.pop(0)
                last_exc = exc
                retryable = OpenAICompatibleEmbeddingBackend._is_retryable(exc)
                if (not retryable) or attempt >= max_attempts:
                    raise exc
                self.retry_count += 1
                if self.on_attempt:
                    self.on_attempt(
                        {
                            "phase": "retry",
                            "attempt": attempt,
                            "max_attempts": max_attempts,
                            "delay_sec": 0,
                            "error_type": type(exc).__name__,
                        }
                    )
                continue
            vecs = self._inner.embed(list(texts))
            if self.bad_count:
                return vecs[:-1]  # count mismatch
            return vecs
        assert last_exc is not None
        raise last_exc


def test_frozen_transport_and_hashes_unchanged():
    assert frozen_embedding_build() == {
        "batch_size": 32,
        "request_timeout_sec": 120,
        "max_retries": 3,
        "retry_backoff_seconds": [1, 2, 4],
    }
    assert FROZEN_EMBEDDING_BUILD["batch_size"] == 32
    assert method_identity_sha256() == EXPECTED_HK_METHOD_CARD_SHA256
    assert etom_prompt_hash() == (
        "8ebac6ce045af0176c1fb9e580bad483b7ca4564e3521adc433c09476db285d7"
    )
    ident = assert_formal_lock_identity(ROOT)
    assert ident["sample_sha256"] == EXPECTED_FORMAL_SAMPLE_SHA256
    assert ident["ordered_task_ids_sha256"] == EXPECTED_ORDERED_IDS_SHA256
    # method card must not include transport knobs
    card = json.dumps(
        __import__(
            "hk_equity_mcp.mcp_zero_etom", fromlist=["frozen_etom_method_identity"]
        ).frozen_etom_method_identity(),
        sort_keys=True,
    )
    assert "batch_size" not in card
    assert "request_timeout_sec" not in card
    # etom_method hash stable relative to transport (transport not in card)
    _ = etom_method_identity_sha256()


def test_timeout_then_success_retries(monkeypatch):
    sleeps: List[float] = []
    monkeypatch.setattr(
        "hk_equity_mcp.operation_resolver.retrieval.time.sleep",
        lambda s: sleeps.append(float(s)),
    )
    backend = OpenAICompatibleEmbeddingBackend(
        base_url="https://example.invalid/v1",
        api_key="test-key-not-real",
        model="text-embedding-3-small",
        timeout_sec=120,
        max_retries=3,
        retry_backoff_seconds=[1, 2, 4],
    )
    attempts = {"n": 0}

    def _once(texts):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise TimeoutError("boom")
        return [[0.1] * 4 for _ in texts]

    monkeypatch.setattr(backend, "_embed_once", _once)
    out = backend.embed(["a", "b"])
    assert len(out) == 2
    assert attempts["n"] == 2
    assert backend.retry_count == 1
    assert sleeps == [1.0]


def test_http_429_and_503_retry(monkeypatch):
    monkeypatch.setattr(
        "hk_equity_mcp.operation_resolver.retrieval.time.sleep", lambda s: None
    )
    backend = OpenAICompatibleEmbeddingBackend(
        base_url="https://example.invalid/v1",
        api_key="test-key-not-real",
        model="text-embedding-3-small",
        max_retries=3,
        retry_backoff_seconds=[0, 0, 0],
    )
    seq = [
        RuntimeError("Embedding request failed with HTTP 429 for model x."),
        RuntimeError("Embedding request failed with HTTP 503 for model x."),
    ]
    attempts = {"n": 0}

    def _once(texts):
        attempts["n"] += 1
        if seq:
            raise seq.pop(0)
        return [[0.2] * 3 for _ in texts]

    monkeypatch.setattr(backend, "_embed_once", _once)
    assert len(backend.embed(["q"])) == 1
    assert attempts["n"] == 3
    assert backend.retry_count == 2


def test_http_400_fail_closed_no_retry(monkeypatch):
    monkeypatch.setattr(
        "hk_equity_mcp.operation_resolver.retrieval.time.sleep", lambda s: None
    )
    backend = OpenAICompatibleEmbeddingBackend(
        base_url="https://example.invalid/v1",
        api_key="test-key-not-real",
        model="text-embedding-3-small",
        max_retries=3,
        retry_backoff_seconds=[1, 2, 4],
    )
    attempts = {"n": 0}

    def _once(texts):
        attempts["n"] += 1
        raise RuntimeError(
            "Embedding request failed with HTTP 400 for model text-embedding-3-small. "
            "No lexical-only fallback."
        )

    monkeypatch.setattr(backend, "_embed_once", _once)
    with pytest.raises(RuntimeError, match="HTTP 400"):
        backend.embed(["q"])
    assert attempts["n"] == 1
    assert backend.retry_count == 0


def test_batch_checkpoint_only_after_success(tmp_path, fixture_catalog):
    events: List[Dict[str, Any]] = []
    backend = _ScriptedBackend(
        failures=[TimeoutError("first")],
    )
    embedder = CountingEmbedder(backend, model_name="text-embedding-3-small")
    # Force tiny batches so we can observe checkpoints.
    table = build_embedding_table(
        fixture_catalog,
        embedder,
        cache_dir=tmp_path,
        use_summary=True,
        catalog_fingerprint=fixture_catalog.fingerprint(),
        namespace=EMBEDDING_NAMESPACE,
        batch_size=32,
        event_callback=events.append,
    )
    assert table.complete
    assert table.build_retries >= 1
    # CountingEmbedder only counts successful embed() returns (retries internal).
    assert embedder.calls == table.build_calls
    status = table_status(
        fixture_catalog,
        model_name="text-embedding-3-small",
        cache_dir=tmp_path,
        use_summary=True,
        catalog_fingerprint=fixture_catalog.fingerprint(),
        namespace=EMBEDDING_NAMESPACE,
    )
    assert status["final_complete"] is True
    assert status["partial_exists"] is False  # cleaned after final
    assert any(e.get("event") == "checkpoint" for e in events)


def test_vector_count_mismatch_does_not_checkpoint(tmp_path, fixture_catalog):
    backend = _ScriptedBackend(bad_count=True)
    embedder = CountingEmbedder(backend, model_name="text-embedding-3-small")
    # CountingEmbedder fail-closes on count mismatch before checkpoint append.
    # build_embedding_table also validates via IncompleteEmbeddingTableError.
    with pytest.raises((RuntimeError, IncompleteEmbeddingTableError), match="batch"):
        build_embedding_table(
            fixture_catalog,
            embedder,
            cache_dir=tmp_path,
            use_summary=True,
            catalog_fingerprint=fixture_catalog.fingerprint(),
            namespace=EMBEDDING_NAMESPACE,
            batch_size=32,
        )
    # No partial rows should have been written.
    status = table_status(
        fixture_catalog,
        model_name="text-embedding-3-small",
        cache_dir=tmp_path,
        use_summary=True,
        catalog_fingerprint=fixture_catalog.fingerprint(),
        namespace=EMBEDDING_NAMESPACE,
    )
    assert status["completed_units"] == 0
    assert status["final_complete"] is False
    assert embedder.calls == 0


def test_interrupt_resume_skips_completed_units(tmp_path, fixture_catalog, monkeypatch):
    backend = _ScriptedBackend()
    embedder = CountingEmbedder(backend, model_name="text-embedding-3-small")
    # Build first two units only by patching pending mid-flight via small catalog
    # and interrupting after first successful batch using a flaky second call.
    call_n = {"n": 0}
    real_embed = backend.embed

    def flaky(texts):
        call_n["n"] += 1
        if call_n["n"] == 2:
            raise KeyboardInterrupt("stop after first checkpoint")
        return real_embed(texts)

    backend.embed = flaky  # type: ignore[method-assign]
    with pytest.raises(KeyboardInterrupt):
        build_embedding_table(
            fixture_catalog,
            embedder,
            cache_dir=tmp_path,
            use_summary=True,
            catalog_fingerprint=fixture_catalog.fingerprint(),
            namespace=EMBEDDING_NAMESPACE,
            batch_size=2,
        )
    status = table_status(
        fixture_catalog,
        model_name="text-embedding-3-small",
        cache_dir=tmp_path,
        use_summary=True,
        catalog_fingerprint=fixture_catalog.fingerprint(),
        namespace=EMBEDDING_NAMESPACE,
    )
    assert status["partial_exists"] is True
    assert status["completed_units"] == 2
    completed_before = status["completed_units"]

    # Resume with a clean backend; must not re-request the first 2 units.
    backend2 = _ScriptedBackend()
    seen_batches: List[int] = []
    real2 = backend2.embed

    def track(texts):
        seen_batches.append(len(texts))
        return real2(texts)

    backend2.embed = track  # type: ignore[method-assign]
    embedder2 = CountingEmbedder(backend2, model_name="text-embedding-3-small")
    table = build_embedding_table(
        fixture_catalog,
        embedder2,
        cache_dir=tmp_path,
        use_summary=True,
        catalog_fingerprint=fixture_catalog.fingerprint(),
        namespace=EMBEDDING_NAMESPACE,
        batch_size=32,
    )
    assert table.complete
    assert table.index_action == "resume_partial"
    # Total successful embed texts across resume == remaining units only.
    assert sum(seen_batches) == status["expected_units"] - completed_before


def test_batch_size_32_used_by_config():
    cfg = {
        "embedding_build": {
            "batch_size": 32,
            "request_timeout_sec": 120,
            "max_retries": 3,
            "retry_backoff_seconds": [1, 2, 4],
        }
    }
    assert assert_embedding_build_config(cfg)["batch_size"] == 32


def test_index_status_partial_fields(tmp_path, fixture_catalog):
    backend = _ScriptedBackend()
    # Write a tiny partial manually via interrupted-like build of 1 batch
    embedder = CountingEmbedder(backend, model_name="text-embedding-3-small")
    # Build complete then delete final to leave nothing — instead create partial only
    # by building with interrupt after first batch size 4.
    n = {"c": 0}
    real = backend.embed

    def once_then_fail(texts):
        n["c"] += 1
        if n["c"] > 1:
            raise RuntimeError("stop")
        return real(texts)

    backend.embed = once_then_fail  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="stop"):
        build_embedding_table(
            fixture_catalog,
            embedder,
            cache_dir=tmp_path,
            use_summary=True,
            catalog_fingerprint=fixture_catalog.fingerprint(),
            namespace=EMBEDDING_NAMESPACE,
            batch_size=4,
        )
    status = table_status(
        fixture_catalog,
        model_name="text-embedding-3-small",
        cache_dir=tmp_path,
        use_summary=True,
        catalog_fingerprint=fixture_catalog.fingerprint(),
        namespace=EMBEDDING_NAMESPACE,
    )
    assert status["partial_exists"] is True
    assert status["final_complete"] is False
    assert status["completed_units"] == 4
    assert status["expected_units"] == status["total"]
    assert status["identity_sha256"]
    assert status["vector_dim"] == 8
