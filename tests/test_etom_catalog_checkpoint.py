"""Offline tests for recoverable ETOM catalog embedding checkpoints.

No network. No real API keys. Fake/mock embedders only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import pytest

from hk_equity_mcp.etom.embeddings import (
    CatalogIndexError,
    HashEmbeddingClient,
    IncompleteCatalogIndexError,
    build_catalog_index,
    catalog_index_status,
    estimate_tokens,
    load_complete_catalog_index,
    load_or_build_catalog_index,
)
from hk_equity_mcp.etom.runner import EtomPilotRunner, PilotLimitError
from hk_equity_mcp.etom.source import repo_root_from_here
from hk_equity_mcp.operation_resolver.schemas import OperationSpec, ParameterSpec

ROOT = repo_root_from_here()


def _spec(op_id: str) -> OperationSpec:
    server, name = op_id.split("::", 1)
    return OperationSpec(
        operation_id=op_id,
        service_id=server,
        name=name,
        summary=f"Summary for {name}",
        description=f"Description for {name}",
        action_terms=[name, "tool"],
        parameters=[ParameterSpec(name="q", type="string", required=True)],
        invocation_guidance="Input fields: q",
        underlying_tool=name,
    )


def _registry(n: int):
    specs = {
        f"server-{i:03d}::tool_{i:03d}": _spec(f"server-{i:03d}::tool_{i:03d}")
        for i in range(n)
    }
    return specs


class RecordingEmbedder:
    """Wraps HashEmbeddingClient and records every embed request."""

    def __init__(self, batch_size: int = 2, model_name: str = "mock-hash-embedding-v1"):
        self.inner = HashEmbeddingClient(model_name=model_name, batch_size=batch_size)
        self.model_name = model_name
        self.batch_size = batch_size
        self.calls = 0
        self.texts: List[str] = []
        self.fail_after_calls: int | None = None
        self.interrupt_after_calls: int | None = None

    def embed(self, texts: List[str]):
        if self.interrupt_after_calls is not None and self.calls >= self.interrupt_after_calls:
            raise KeyboardInterrupt
        if self.fail_after_calls is not None and self.calls >= self.fail_after_calls:
            raise RuntimeError("forced embed failure")
        self.calls += 1
        self.texts.extend(texts)
        return self.inner.embed(texts)


def test_interrupt_then_resume_only_requests_missing(tmp_path: Path):
    registry = _registry(7)
    embedder = RecordingEmbedder(batch_size=2)
    embedder.interrupt_after_calls = 2  # checkpoint 4 ops, then interrupt

    with pytest.raises(KeyboardInterrupt):
        build_catalog_index(
            registry=registry,
            embedder=embedder,
            registry_sha256="sha-a",
            cache_dir=tmp_path,
        )

    status = catalog_index_status(
        registry=registry,
        model_name=embedder.model_name,
        registry_sha256="sha-a",
        cache_dir=tmp_path,
    )
    assert status["final_complete"] is False
    assert status["partial_completed_count"] == 4
    assert status["partial_present"] is True
    first_calls = embedder.calls
    first_texts = list(embedder.texts)
    assert first_calls == 2
    assert len(first_texts) == 4

    embedder.interrupt_after_calls = None
    embedder.calls = 0
    embedder.texts = []
    index = build_catalog_index(
        registry=registry,
        embedder=embedder,
        registry_sha256="sha-a",
        cache_dir=tmp_path,
    )
    assert len(index.vectors) == 7
    assert set(index.vectors) == set(registry)
    # Only the 3 remaining ops are requested (batch_size=2 -> 2 calls: 2+1).
    assert len(embedder.texts) == 3
    assert embedder.calls == 2
    assert not any(text in first_texts for text in embedder.texts)
    assert status["partial_present"] is True
    status_after = catalog_index_status(
        registry=registry,
        model_name=embedder.model_name,
        registry_sha256="sha-a",
        cache_dir=tmp_path,
    )
    assert status_after["final_complete"] is True
    assert status_after["final_completed_count"] == 7
    assert status_after["partial_present"] is False


def test_completed_operations_are_never_reembedded(tmp_path: Path):
    registry = _registry(5)
    embedder = RecordingEmbedder(batch_size=2)
    first = build_catalog_index(
        registry=registry,
        embedder=embedder,
        registry_sha256="sha-b",
        cache_dir=tmp_path,
    )
    assert first.cache_reused is False
    calls_first = embedder.calls
    assert calls_first > 0

    embedder.calls = 0
    embedder.texts = []
    second = build_catalog_index(
        registry=registry,
        embedder=embedder,
        registry_sha256="sha-b",
        cache_dir=tmp_path,
    )
    assert second.cache_reused is True
    assert embedder.calls == 0
    assert embedder.texts == []
    assert set(second.vectors) == set(registry)


def test_model_mismatch_fail_closed(tmp_path: Path):
    registry = _registry(3)
    embedder = RecordingEmbedder(batch_size=2, model_name="mock-hash-embedding-v1")
    build_catalog_index(
        registry=registry,
        embedder=embedder,
        registry_sha256="sha-c",
        cache_dir=tmp_path,
    )
    # Corrupt meta/model by writing a partial with wrong model after deleting final.
    final = next(tmp_path.glob("catalog-*.json"))
    final.unlink()
    # Rebuild partial under same digest with wrong model rows by hand.
    # Use a different embedder model against existing partial from same digest path:
    # Create partial then reload with mismatched model via direct meta edit.
    embedder2 = RecordingEmbedder(batch_size=2, model_name="mock-hash-embedding-v1")
    embedder2.interrupt_after_calls = 1
    with pytest.raises(KeyboardInterrupt):
        build_catalog_index(
            registry=registry,
            embedder=embedder2,
            registry_sha256="sha-c",
            cache_dir=tmp_path,
        )
    meta = next(tmp_path.glob("*.partial.meta.json"))
    payload = json.loads(meta.read_text(encoding="utf-8"))
    payload["model"] = "other-model"
    meta.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(CatalogIndexError, match="model mismatch"):
        build_catalog_index(
            registry=registry,
            embedder=embedder2,
            registry_sha256="sha-c",
            cache_dir=tmp_path,
        )


def test_registry_hash_mismatch_fail_closed(tmp_path: Path):
    registry = _registry(3)
    embedder = RecordingEmbedder(batch_size=2)
    embedder.interrupt_after_calls = 1
    with pytest.raises(KeyboardInterrupt):
        build_catalog_index(
            registry=registry,
            embedder=embedder,
            registry_sha256="sha-d1",
            cache_dir=tmp_path,
        )
    meta = next(tmp_path.glob("*.partial.meta.json"))
    payload = json.loads(meta.read_text(encoding="utf-8"))
    payload["registry_sha256"] = "sha-d2-tampered"
    meta.write_text(json.dumps(payload), encoding="utf-8")
    embedder.interrupt_after_calls = None
    with pytest.raises(CatalogIndexError, match="registry hash mismatch"):
        build_catalog_index(
            registry=registry,
            embedder=embedder,
            registry_sha256="sha-d1",
            cache_dir=tmp_path,
        )


def test_vector_dimension_mismatch_fail_closed(tmp_path: Path):
    registry = _registry(3)
    embedder = RecordingEmbedder(batch_size=2)
    embedder.interrupt_after_calls = 1
    with pytest.raises(KeyboardInterrupt):
        build_catalog_index(
            registry=registry,
            embedder=embedder,
            registry_sha256="sha-e",
            cache_dir=tmp_path,
        )
    jsonl = next(tmp_path.glob("*.partial.jsonl"))
    meta = next(tmp_path.glob("*.partial.meta.json"))
    lines = [json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    lines[0]["vector"] = lines[0]["vector"] + [0.0]
    lines[0]["vector_dim"] = len(lines[0]["vector"])
    jsonl.write_text("\n".join(json.dumps(row) for row in lines) + "\n", encoding="utf-8")
    meta_payload = json.loads(meta.read_text(encoding="utf-8"))
    # Keep meta dim as original so row/meta disagree or rows disagree.
    with pytest.raises(CatalogIndexError, match="dimension"):
        build_catalog_index(
            registry=registry,
            embedder=embedder,
            registry_sha256="sha-e",
            cache_dir=tmp_path,
        )


def test_corrupted_partial_checkpoint_fail_closed(tmp_path: Path):
    registry = _registry(3)
    embedder = RecordingEmbedder(batch_size=2)
    embedder.interrupt_after_calls = 1
    with pytest.raises(KeyboardInterrupt):
        build_catalog_index(
            registry=registry,
            embedder=embedder,
            registry_sha256="sha-f",
            cache_dir=tmp_path,
        )
    jsonl = next(tmp_path.glob("*.partial.jsonl"))
    jsonl.write_text("{not-json\n", encoding="utf-8")
    with pytest.raises(CatalogIndexError, match="corrupted"):
        build_catalog_index(
            registry=registry,
            embedder=embedder,
            registry_sha256="sha-f",
            cache_dir=tmp_path,
        )


def test_final_cache_written_at_full_completion(tmp_path: Path):
    registry = _registry(5)
    embedder = RecordingEmbedder(batch_size=2)
    index = build_catalog_index(
        registry=registry,
        embedder=embedder,
        registry_sha256="sha-g",
        cache_dir=tmp_path,
    )
    assert len(index.vectors) == 5
    finals = list(tmp_path.glob("catalog-*.json"))
    assert len(finals) == 1
    payload = json.loads(finals[0].read_text(encoding="utf-8"))
    assert payload["complete"] is True
    assert payload["completed_count"] == 5
    assert set(payload["vectors"]) == set(registry)
    assert list(tmp_path.glob("*.partial.jsonl")) == []
    assert list(tmp_path.glob("*.partial.meta.json")) == []


def test_final_cache_hit_does_not_call_embedder(tmp_path: Path):
    registry = _registry(4)
    embedder = RecordingEmbedder(batch_size=2)
    build_catalog_index(
        registry=registry,
        embedder=embedder,
        registry_sha256="sha-h",
        cache_dir=tmp_path,
    )
    embedder.calls = 0
    embedder.texts = []
    loaded = load_complete_catalog_index(
        registry=registry,
        model_name=embedder.model_name,
        registry_sha256="sha-h",
        cache_dir=tmp_path,
    )
    assert len(loaded.vectors) == 4
    assert embedder.calls == 0
    again = load_or_build_catalog_index(
        registry=registry,
        embedder=embedder,
        registry_sha256="sha-h",
        cache_dir=tmp_path,
        require_complete=True,
    )
    assert again.cache_reused is True
    assert embedder.calls == 0


def test_require_complete_refuses_incomplete_index(tmp_path: Path):
    registry = _registry(4)
    embedder = RecordingEmbedder(batch_size=2)
    embedder.interrupt_after_calls = 1
    with pytest.raises(KeyboardInterrupt):
        build_catalog_index(
            registry=registry,
            embedder=embedder,
            registry_sha256="sha-i",
            cache_dir=tmp_path,
        )
    with pytest.raises(IncompleteCatalogIndexError):
        load_or_build_catalog_index(
            registry=registry,
            embedder=embedder,
            registry_sha256="sha-i",
            cache_dir=tmp_path,
            require_complete=True,
        )


def test_usage_accumulates_across_resume_without_double_counting(tmp_path: Path):
    registry = _registry(6)
    embedder = RecordingEmbedder(batch_size=2)
    embedder.interrupt_after_calls = 2
    with pytest.raises(KeyboardInterrupt):
        build_catalog_index(
            registry=registry,
            embedder=embedder,
            registry_sha256="sha-j",
            cache_dir=tmp_path,
        )
    meta = json.loads(next(tmp_path.glob("*.partial.meta.json")).read_text(encoding="utf-8"))
    usage_after_first = meta["build_usage"]
    assert usage_after_first["calls"] == 2
    tokens_first = usage_after_first["tokens"]

    embedder.interrupt_after_calls = None
    embedder.calls = 0
    embedder.texts = []
    index = build_catalog_index(
        registry=registry,
        embedder=embedder,
        registry_sha256="sha-j",
        cache_dir=tmp_path,
    )
    # Remaining 2 ops -> 1 call. Total calls = 2 (checkpointed) + 1 = 3.
    assert embedder.calls == 1
    assert len(embedder.texts) == 2
    assert index.build_usage.calls == 3
    assert index.build_usage.tokens == tokens_first + estimate_tokens(embedder.texts)


def test_sigint_preserves_partial_checkpoint(tmp_path: Path):
    registry = _registry(8)
    embedder = RecordingEmbedder(batch_size=3)
    embedder.interrupt_after_calls = 1
    with pytest.raises(KeyboardInterrupt):
        build_catalog_index(
            registry=registry,
            embedder=embedder,
            registry_sha256="sha-k",
            cache_dir=tmp_path,
        )
    assert list(tmp_path.glob("*.partial.jsonl"))
    assert list(tmp_path.glob("*.partial.meta.json"))
    meta = json.loads(next(tmp_path.glob("*.partial.meta.json")).read_text(encoding="utf-8"))
    assert meta["completed_count"] == 3
    # Partial must not be deleted by the interrupt path.
    assert not list(tmp_path.glob("catalog-*.json")) or not json.loads(
        next(tmp_path.glob("catalog-*.json")).read_text(encoding="utf-8")
    ).get("complete")


def test_build_index_only_does_not_instantiate_llm(tmp_path: Path, monkeypatch):
    pytest.importorskip("yaml")
    created = {"llm": 0}

    class BoomLLM:
        def __init__(self, *args, **kwargs):
            created["llm"] += 1
            raise AssertionError("FailClosedChatClient must not be constructed during build-index")

    monkeypatch.setattr("hk_equity_mcp.etom.runner.FailClosedChatClient", BoomLLM)
    out = ROOT / "outputs" / "_test_build_index_only_tmp"
    out.mkdir(parents=True, exist_ok=True)
    runner = EtomPilotRunner(
        ROOT,
        out,
        mode="mock",
        build_index_only=True,
        max_tasks=10,
        allow_dirty=True,
    )
    runner._cache_dir = tmp_path / "cache"

    def fake_bundle():
        specs = _registry(4)
        return specs, {k: {} for k in specs}, "sha-build-only"

    monkeypatch.setattr(runner, "_registry_bundle", fake_bundle)
    report = runner._build_index_phase(
        {"pin": {"commit_sha": "abc", "repository_url": "x", "license": "y"}, "sha256": {}}
    )
    assert created["llm"] == 0
    assert report["generation_llm_instantiated"] is False
    assert report["generation_llm_calls"] == 0
    assert report["final_complete"] is True
    assert report["completed_count"] == 4


def test_agent_run_rejects_incomplete_final_index(tmp_path: Path):
    cache_dir = tmp_path / "cache"
    embedder = RecordingEmbedder(batch_size=2)
    embedder.interrupt_after_calls = 1
    with pytest.raises(KeyboardInterrupt):
        build_catalog_index(
            registry=_registry(4),
            embedder=embedder,
            registry_sha256="sha-run-reject",
            cache_dir=cache_dir,
        )
    with pytest.raises(IncompleteCatalogIndexError):
        load_complete_catalog_index(
            registry=_registry(4),
            model_name=embedder.model_name,
            registry_sha256="sha-run-reject",
            cache_dir=cache_dir,
        )
    with pytest.raises(IncompleteCatalogIndexError):
        load_or_build_catalog_index(
            registry=_registry(4),
            embedder=embedder,
            registry_sha256="sha-run-reject",
            cache_dir=cache_dir,
            require_complete=True,
        )


def test_unknown_operation_id_in_partial_fail_closed(tmp_path: Path):
    registry = _registry(3)
    embedder = RecordingEmbedder(batch_size=2)
    embedder.interrupt_after_calls = 1
    with pytest.raises(KeyboardInterrupt):
        build_catalog_index(
            registry=registry,
            embedder=embedder,
            registry_sha256="sha-l",
            cache_dir=tmp_path,
        )
    jsonl = next(tmp_path.glob("*.partial.jsonl"))
    lines = [json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    lines[0]["operation_id"] = "not-in-registry::gone"
    jsonl.write_text("\n".join(json.dumps(row) for row in lines) + "\n", encoding="utf-8")
    with pytest.raises(CatalogIndexError, match="unknown operation_id"):
        build_catalog_index(
            registry=registry,
            embedder=embedder,
            registry_sha256="sha-l",
            cache_dir=tmp_path,
        )


def test_pilot_still_refuses_eleventh_task():
    with pytest.raises(PilotLimitError):
        EtomPilotRunner(
            ROOT,
            ROOT / "outputs" / "tmp_should_not_matter",
            mode="mock",
            max_tasks=11,
        )
