"""Catalog and query embeddings for the ETOM routing pilot.

Catalog vectors are built once and cached. Query vectors are counted separately.
API keys are never logged.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import signal
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from hk_equity_mcp.etom import EXPECTED_EMBEDDING_MODEL
from hk_equity_mcp.operation_resolver.retrieval import (
    DeterministicHashEmbeddingBackend,
    cosine_similarity,
    lexical_score,
)
from hk_equity_mcp.operation_resolver.schemas import OperationSpec


class ModelMismatchError(RuntimeError):
    """Raised when a provider echoes a model other than the pinned one."""


@dataclass
class EmbeddingUsage:
    calls: int = 0
    retries: int = 0
    tokens: int = 0
    token_source: str = "provider"
    estimated_tokens: int = 0
    latency_ms: float = 0.0

    def add(self, other: "EmbeddingUsage") -> None:
        self.calls += other.calls
        self.retries += other.retries
        self.tokens += other.tokens
        self.estimated_tokens += other.estimated_tokens
        self.latency_ms += other.latency_ms
        if other.token_source == "estimated" or self.token_source == "estimated":
            if self.tokens and other.token_source == "provider":
                self.token_source = "mixed"
            elif other.token_source == "estimated":
                self.token_source = (
                    "estimated" if self.token_source == "estimated" else "mixed"
                )


def estimate_tokens(texts: Sequence[str]) -> int:
    return sum(max(1, (len(text) + 3) // 4) for text in texts)


def urlopen_with_deadline(request, timeout_sec: float):
    """Read an HTTP response with a hard wall-clock deadline.

    socket timeouts do not always fire while a local proxy holds the connection.
    SIGALRM interrupts that wait. The caller must read the body inside this call.
    """
    if timeout_sec <= 0:
        raise TimeoutError("timeout must be positive")

    def _handler(signum, frame):
        raise TimeoutError(f"request exceeded {timeout_sec:.0f}s")

    previous = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, timeout_sec)
    try:
        with urllib.request.urlopen(request, timeout=timeout_sec) as response:
            payload = response.read()
        return payload
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def scrub_secret(text: str) -> str:
    redacted = text
    for env_name in ("LLM_API_KEY", "EMBEDDING_API_KEY", "OPENAI_API_KEY"):
        value = os.environ.get(env_name) or ""
        if len(value) >= 8:
            redacted = redacted.replace(value, "[REDACTED]")
    return redacted


class OpenAICompatibleEmbeddingClient:
    """Fail-closed embeddings. Retries are bounded and counted."""

    def __init__(
        self,
        *,
        model_name: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout_sec: float = 60.0,
        max_retries: int = 1,
        batch_size: int = 64,
    ):
        self.model_name = model_name
        self.base_url = (base_url or os.environ.get("EMBEDDING_BASE_URL") or "").rstrip("/")
        self.api_key = api_key or os.environ.get("EMBEDDING_API_KEY") or ""
        self.timeout_sec = timeout_sec
        self.max_retries = max_retries
        self.batch_size = batch_size
        if self.model_name != EXPECTED_EMBEDDING_MODEL:
            raise ModelMismatchError(
                "Embedding model mismatch before the request: "
                f"expected {EXPECTED_EMBEDDING_MODEL}."
            )
        if not self.base_url or not self.api_key:
            raise RuntimeError(
                "Missing embedding endpoint configuration. "
                "Load the existing embedding env file. The key is not printed."
            )

    def embed(self, texts: List[str]) -> tuple:
        usage = EmbeddingUsage()
        vectors: List[List[float]] = []
        for start in range(0, len(texts), self.batch_size):
            chunk = texts[start : start + self.batch_size]
            part, part_usage = self._embed_chunk(chunk)
            vectors.extend(part)
            usage.add(part_usage)
        return vectors, usage

    def _embed_chunk(self, texts: List[str]) -> tuple:
        if len(texts) == 1:
            return self._embed_once(texts)
        try:
            return self._embed_once(texts)
        except RuntimeError as exc:
            message = str(exc)
            if "HTTP 400" not in message or len(texts) <= 1:
                raise
            usage = EmbeddingUsage(retries=1)
            mid = len(texts) // 2
            left, left_usage = self._embed_chunk(texts[:mid])
            right, right_usage = self._embed_chunk(texts[mid:])
            usage.add(left_usage)
            usage.add(right_usage)
            return left + right, usage

    def _embed_once(self, texts: List[str]) -> tuple:
        url = f"{self.base_url}/embeddings"
        payload = json.dumps({"model": self.model_name, "input": texts}).encode("utf-8")
        last_error: Optional[Exception] = None
        retries = 0
        for attempt in range(self.max_retries + 1):
            request = urllib.request.Request(
                url,
                data=payload,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
                method="POST",
            )
            started = time.perf_counter()
            try:
                raw = urlopen_with_deadline(request, self.timeout_sec)
                body = json.loads(raw.decode("utf-8"))
                latency_ms = (time.perf_counter() - started) * 1000
                return self._parse_body(body, texts, retries, latency_ms)
            except ModelMismatchError:
                raise
            except TimeoutError:
                last_error = TimeoutError("embedding deadline")
                if attempt >= self.max_retries:
                    raise RuntimeError(
                        f"Embedding request exceeded {self.timeout_sec:.0f}s "
                        f"for model {self.model_name}."
                    ) from None
                retries += 1
            except urllib.error.HTTPError as exc:
                last_error = exc
                if exc.code not in {429, 500, 502, 503, 504} or attempt >= self.max_retries:
                    raise RuntimeError(
                        f"Embedding request failed with HTTP {exc.code} "
                        f"for model {self.model_name}."
                    ) from None
                retries += 1
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt >= self.max_retries:
                    raise RuntimeError(
                        "Embedding request failed for model "
                        f"{self.model_name}: {type(exc).__name__}."
                    ) from None
                retries += 1
        raise RuntimeError(f"Embedding request failed: {type(last_error).__name__}.")

    def _parse_body(self, body: dict, texts: List[str], retries: int, latency_ms: float):
        echoed = body.get("model")
        if echoed and echoed != self.model_name:
            raise ModelMismatchError(
                "Embedding provider/model mismatch: "
                f"expected {self.model_name}, echoed {echoed}."
            )
        data = body.get("data")
        if not isinstance(data, list) or len(data) != len(texts):
            raise RuntimeError("Unexpected embedding response shape.")
        ordered = sorted(data, key=lambda row: int(row.get("index", 0)))
        vectors = []
        for row in ordered:
            emb = row.get("embedding")
            if not isinstance(emb, list) or not emb:
                raise RuntimeError("Embedding response missing vectors.")
            vectors.append([float(value) for value in emb])
        usage_body = body.get("usage") or {}
        provider_tokens = usage_body.get("prompt_tokens", usage_body.get("total_tokens"))
        estimated = estimate_tokens(texts)
        if provider_tokens is None:
            tokens = estimated
            token_source = "estimated"
        else:
            tokens = int(provider_tokens)
            token_source = "provider"
        usage = EmbeddingUsage(
            calls=1,
            retries=retries,
            tokens=tokens,
            token_source=token_source,
            estimated_tokens=estimated,
            latency_ms=latency_ms,
        )
        return vectors, usage


class HashEmbeddingClient:
    """Offline stand-in used by mock runs and unit tests."""

    def __init__(self, model_name: str = "mock-hash-embedding-v1"):
        self.model_name = model_name
        self._inner = DeterministicHashEmbeddingBackend(model_name=model_name)

    def embed(self, texts: List[str]) -> tuple:
        started = time.perf_counter()
        vectors = self._inner.embed(texts)
        usage = EmbeddingUsage(
            calls=1 if texts else 0,
            tokens=estimate_tokens(texts),
            token_source="estimated",
            estimated_tokens=estimate_tokens(texts),
            latency_ms=(time.perf_counter() - started) * 1000,
        )
        return vectors, usage


@dataclass
class CatalogIndex:
    model_name: str
    registry_sha256: str
    vectors: Dict[str, List[float]]
    build_usage: EmbeddingUsage
    cache_reused: bool
    cache_path: Optional[str] = None
    query_usage: EmbeddingUsage = field(default_factory=EmbeddingUsage)

    def rank(
        self,
        query: str,
        registry: Dict[str, OperationSpec],
        embedder,
        top_k: int,
        lexical_weight: float,
    ) -> List[dict]:
        vectors, usage = embedder.embed([query])
        if embedder.model_name != self.model_name:
            raise ModelMismatchError(
                "Query embedding model does not match the catalog index model."
            )
        self.query_usage.add(usage)
        query_vector = vectors[0]
        scored = []
        for op_id, spec in registry.items():
            cosine = cosine_similarity(query_vector, self.vectors[op_id])
            lexical = lexical_score(query, spec)
            score = cosine + lexical_weight * lexical
            scored.append((score, op_id))
        scored.sort(key=lambda row: (-row[0], row[1]))
        results = []
        for score, op_id in scored[:top_k]:
            spec = registry[op_id]
            summary = spec.summary or ""
            if len(summary) > 320:
                summary = summary[:320]
            results.append(
                {
                    "operation_id": spec.operation_id,
                    "score": round(float(score), 6),
                    "server_name": spec.service_id,
                    "service_id": spec.service_id,
                    "name": spec.name,
                    "summary": summary,
                }
            )
        return results


def _cache_file(cache_dir: Path, model_name: str, registry_sha256: str) -> Path:
    digest = hashlib.sha256(f"{model_name}|{registry_sha256}".encode("utf-8")).hexdigest()
    return cache_dir / f"catalog-{digest[:16]}.json"


def _write_catalog_cache(
    path: Path,
    *,
    model_name: str,
    registry_sha256: str,
    vectors: Dict[str, List[float]],
    usage: EmbeddingUsage,
) -> None:
    payload = {
        "model": model_name,
        "registry_sha256": registry_sha256,
        "build_usage": {
            "calls": usage.calls,
            "retries": usage.retries,
            "tokens": usage.tokens,
            "token_source": usage.token_source,
            "estimated_tokens": usage.estimated_tokens,
            "latency_ms": usage.latency_ms,
        },
        "vectors": vectors,
        "complete": False,
    }
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    tmp.replace(path)


def load_or_build_catalog_index(
    *,
    registry: Dict[str, OperationSpec],
    embedder,
    registry_sha256: str,
    cache_dir: Path,
) -> CatalogIndex:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _cache_file(cache_dir, embedder.model_name, registry_sha256)
    vectors: Dict[str, List[float]] = {}
    usage = EmbeddingUsage()
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            payload.get("model") == embedder.model_name
            and payload.get("registry_sha256") == registry_sha256
        ):
            raw = payload.get("vectors") or {}
            vectors = {
                op_id: [float(value) for value in vector]
                for op_id, vector in raw.items()
                if op_id in registry
            }
            build = payload.get("build_usage") or {}
            usage = EmbeddingUsage(
                calls=int(build.get("calls") or 0),
                retries=int(build.get("retries") or 0),
                tokens=int(build.get("tokens") or 0),
                token_source=str(build.get("token_source") or "provider"),
                estimated_tokens=int(build.get("estimated_tokens") or 0),
                latency_ms=float(build.get("latency_ms") or 0.0),
            )
            if set(vectors) == set(registry):
                return CatalogIndex(
                    model_name=embedder.model_name,
                    registry_sha256=registry_sha256,
                    vectors=vectors,
                    build_usage=usage,
                    cache_reused=True,
                    cache_path=str(path),
                )
            print(
                f"catalog cache resume {len(vectors)}/{len(registry)}",
                flush=True,
            )
    missing_ids = sorted(op_id for op_id in registry if op_id not in vectors)
    batch_size = max(1, int(getattr(embedder, "batch_size", 64) or 64))
    for start in range(0, len(missing_ids), batch_size):
        chunk_ids = missing_ids[start : start + batch_size]
        docs = [registry[op_id].index_document() for op_id in chunk_ids]
        print(
            f"embedding {len(vectors) + 1}-{len(vectors) + len(chunk_ids)}/{len(registry)}",
            flush=True,
        )
        vectors_list, part_usage = embedder.embed(docs)
        if len(vectors_list) != len(chunk_ids):
            raise RuntimeError("Catalog embedding batch size mismatch.")
        for op_id, vector in zip(chunk_ids, vectors_list):
            vectors[op_id] = vector
        usage.add(part_usage)
        _write_catalog_cache(
            path,
            model_name=embedder.model_name,
            registry_sha256=registry_sha256,
            vectors=vectors,
            usage=usage,
        )
    if set(vectors) != set(registry):
        raise RuntimeError("Catalog embedding incomplete after resume.")
    dim = len(next(iter(vectors.values()))) if vectors else 0
    if any(len(vector) != dim for vector in vectors.values()):
        raise RuntimeError("Catalog embedding dimension mismatch.")
    _write_catalog_cache(
        path,
        model_name=embedder.model_name,
        registry_sha256=registry_sha256,
        vectors=vectors,
        usage=usage,
    )
    return CatalogIndex(
        model_name=embedder.model_name,
        registry_sha256=registry_sha256,
        vectors=vectors,
        build_usage=usage,
        cache_reused=False,
        cache_path=str(path),
    )


def cosine_self_check(vector: Sequence[float]) -> float:
    return cosine_similarity(vector, vector) if vector else math.nan
