"""Catalog and query embeddings for the ETOM routing pilot.

Catalog vectors are built in batches with an append-only JSONL checkpoint.
Query vectors are counted separately. API keys are never logged.
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
from typing import Dict, List, Optional, Sequence, Tuple

from hk_equity_mcp.etom import EXPECTED_EMBEDDING_MODEL
from hk_equity_mcp.operation_resolver.retrieval import (
    DeterministicHashEmbeddingBackend,
    cosine_similarity,
    lexical_score,
)
from hk_equity_mcp.operation_resolver.schemas import OperationSpec

PARTIAL_FORMAT = "jsonl_v1"
DEFAULT_CATALOG_BATCH_SIZE = 32


class ModelMismatchError(RuntimeError):
    """Raised when a provider echoes a model other than the pinned one."""


class CatalogIndexError(RuntimeError):
    """Fail-closed catalog cache / checkpoint errors."""


class IncompleteCatalogIndexError(CatalogIndexError):
    """Raised when a complete final index is required but missing."""


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

    def to_dict(self) -> dict:
        return {
            "calls": self.calls,
            "retries": self.retries,
            "tokens": self.tokens,
            "token_source": self.token_source,
            "estimated_tokens": self.estimated_tokens,
            "latency_ms": self.latency_ms,
        }

    @classmethod
    def from_dict(cls, payload: Optional[dict]) -> "EmbeddingUsage":
        body = payload or {}
        return cls(
            calls=int(body.get("calls") or 0),
            retries=int(body.get("retries") or 0),
            tokens=int(body.get("tokens") or 0),
            token_source=str(body.get("token_source") or "provider"),
            estimated_tokens=int(body.get("estimated_tokens") or 0),
            latency_ms=float(body.get("latency_ms") or 0.0),
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
        batch_size: int = DEFAULT_CATALOG_BATCH_SIZE,
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

    def __init__(
        self,
        model_name: str = "mock-hash-embedding-v1",
        batch_size: int = DEFAULT_CATALOG_BATCH_SIZE,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self._inner = DeterministicHashEmbeddingBackend(model_name=model_name)

    def embed(self, texts: List[str]) -> tuple:
        usage = EmbeddingUsage()
        vectors: List[List[float]] = []
        if not texts:
            return vectors, usage
        for start in range(0, len(texts), max(1, self.batch_size)):
            chunk = texts[start : start + max(1, self.batch_size)]
            started = time.perf_counter()
            part = self._inner.embed(chunk)
            vectors.extend(part)
            usage.add(
                EmbeddingUsage(
                    calls=1,
                    tokens=estimate_tokens(chunk),
                    token_source="estimated",
                    estimated_tokens=estimate_tokens(chunk),
                    latency_ms=(time.perf_counter() - started) * 1000,
                )
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
    completed_count: Optional[int] = None
    total_count: Optional[int] = None

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


def _digest(model_name: str, registry_sha256: str) -> str:
    return hashlib.sha256(f"{model_name}|{registry_sha256}".encode("utf-8")).hexdigest()[:16]


def _cache_file(cache_dir: Path, model_name: str, registry_sha256: str) -> Path:
    return cache_dir / f"catalog-{_digest(model_name, registry_sha256)}.json"


def _partial_jsonl_path(cache_dir: Path, model_name: str, registry_sha256: str) -> Path:
    return cache_dir / f"catalog-{_digest(model_name, registry_sha256)}.partial.jsonl"


def _partial_meta_path(cache_dir: Path, model_name: str, registry_sha256: str) -> Path:
    return cache_dir / f"catalog-{_digest(model_name, registry_sha256)}.partial.meta.json"


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    data = json.dumps(payload, ensure_ascii=False)
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _write_final_catalog_cache(
    path: Path,
    *,
    model_name: str,
    registry_sha256: str,
    vectors: Dict[str, List[float]],
    usage: EmbeddingUsage,
) -> None:
    dim = len(next(iter(vectors.values()))) if vectors else 0
    payload = {
        "model": model_name,
        "registry_sha256": registry_sha256,
        "complete": True,
        "completed_count": len(vectors),
        "vector_dim": dim,
        "build_usage": usage.to_dict(),
        "vectors": vectors,
    }
    _atomic_write_json(path, payload)


def _validate_vector_dims(vectors: Dict[str, List[float]]) -> int:
    if not vectors:
        return 0
    dim = len(next(iter(vectors.values())))
    if dim <= 0:
        raise CatalogIndexError("Catalog embedding dimension is invalid.")
    for op_id, vector in vectors.items():
        if len(vector) != dim:
            raise CatalogIndexError(
                f"Catalog embedding dimension mismatch for {op_id}: "
                f"{len(vector)} vs {dim}."
            )
    return dim


def _load_final_cache(
    path: Path,
    *,
    model_name: str,
    registry_sha256: str,
    registry: Dict[str, OperationSpec],
    require_complete: bool,
) -> Optional[CatalogIndex]:
    if not path.exists():
        if require_complete:
            raise IncompleteCatalogIndexError(
                f"Final catalog index missing at {path}. "
                "Run --build-index-only until completed_count matches the registry."
            )
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CatalogIndexError(f"Final catalog cache JSON is corrupted: {exc}.") from exc
    if payload.get("model") != model_name:
        raise CatalogIndexError(
            "Final catalog cache model mismatch: "
            f"expected {model_name}, found {payload.get('model')}."
        )
    if payload.get("registry_sha256") != registry_sha256:
        raise CatalogIndexError(
            "Final catalog cache registry hash mismatch."
        )
    raw = payload.get("vectors")
    if not isinstance(raw, dict):
        raise CatalogIndexError("Final catalog cache vectors are missing or incomplete.")
    vectors = {
        op_id: [float(value) for value in vector]
        for op_id, vector in raw.items()
    }
    unknown = set(vectors) - set(registry)
    if unknown:
        sample = sorted(unknown)[:3]
        raise CatalogIndexError(
            f"Final catalog cache contains unknown operation_id values: {sample}."
        )
    dim = _validate_vector_dims(vectors)
    cached_dim = payload.get("vector_dim")
    if cached_dim is not None and int(cached_dim) != dim:
        raise CatalogIndexError("Final catalog cache vector_dim metadata mismatch.")
    complete = bool(payload.get("complete")) and set(vectors) == set(registry)
    if not complete:
        if require_complete:
            raise IncompleteCatalogIndexError(
                f"Final catalog index incomplete: {len(vectors)}/{len(registry)}. "
                "Refuse to auto-rebuild during agent run."
            )
        return None
    usage = EmbeddingUsage.from_dict(payload.get("build_usage"))
    return CatalogIndex(
        model_name=model_name,
        registry_sha256=registry_sha256,
        vectors=vectors,
        build_usage=usage,
        cache_reused=True,
        cache_path=str(path),
        completed_count=len(vectors),
        total_count=len(registry),
    )


def _load_partial_checkpoint(
    *,
    jsonl_path: Path,
    meta_path: Path,
    model_name: str,
    registry_sha256: str,
    registry: Dict[str, OperationSpec],
) -> Tuple[Dict[str, List[float]], EmbeddingUsage, Optional[int]]:
    if not jsonl_path.exists() and not meta_path.exists():
        return {}, EmbeddingUsage(), None
    if jsonl_path.exists() ^ meta_path.exists():
        raise CatalogIndexError(
            "Partial catalog checkpoint is incomplete "
            "(jsonl/meta pair must both exist)."
        )
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CatalogIndexError(f"Partial catalog meta JSON is corrupted: {exc}.") from exc
    if meta.get("format") != PARTIAL_FORMAT:
        raise CatalogIndexError(
            f"Unsupported partial checkpoint format: {meta.get('format')}."
        )
    if meta.get("model") != model_name:
        raise CatalogIndexError(
            "Partial catalog checkpoint model mismatch: "
            f"expected {model_name}, found {meta.get('model')}."
        )
    if meta.get("registry_sha256") != registry_sha256:
        raise CatalogIndexError(
            "Partial catalog checkpoint registry hash mismatch."
        )
    vectors: Dict[str, List[float]] = {}
    expected_dim = meta.get("vector_dim")
    expected_dim_int = int(expected_dim) if expected_dim is not None else None
    try:
        with jsonl_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                text = line.strip()
                if not text:
                    continue
                try:
                    row = json.loads(text)
                except json.JSONDecodeError as exc:
                    raise CatalogIndexError(
                        f"Partial catalog JSONL corrupted at line {line_number}: {exc}."
                    ) from exc
                op_id = row.get("operation_id")
                vector = row.get("vector")
                if not isinstance(op_id, str) or not isinstance(vector, list) or not vector:
                    raise CatalogIndexError(
                        f"Partial catalog JSONL row incomplete at line {line_number}."
                    )
                if op_id not in registry:
                    raise CatalogIndexError(
                        f"Partial catalog checkpoint contains unknown operation_id: {op_id}."
                    )
                if row.get("model") != model_name:
                    raise CatalogIndexError(
                        f"Partial catalog row model mismatch for {op_id}."
                    )
                if row.get("registry_sha256") != registry_sha256:
                    raise CatalogIndexError(
                        f"Partial catalog row registry hash mismatch for {op_id}."
                    )
                values = [float(value) for value in vector]
                row_dim = row.get("vector_dim")
                if row_dim is not None and int(row_dim) != len(values):
                    raise CatalogIndexError(
                        f"Partial catalog row vector_dim mismatch for {op_id}."
                    )
                if expected_dim_int is None:
                    expected_dim_int = len(values)
                elif len(values) != expected_dim_int:
                    raise CatalogIndexError(
                        f"Partial catalog vector dimension mismatch for {op_id}: "
                        f"{len(values)} vs {expected_dim_int}."
                    )
                if op_id in vectors:
                    raise CatalogIndexError(
                        f"Partial catalog checkpoint has duplicate operation_id: {op_id}."
                    )
                vectors[op_id] = values
    except CatalogIndexError:
        raise
    except OSError as exc:
        raise CatalogIndexError(f"Failed reading partial catalog checkpoint: {exc}.") from exc

    completed_meta = int(meta.get("completed_count") or 0)
    if completed_meta != len(vectors):
        raise CatalogIndexError(
            "Partial catalog completed_count does not match JSONL rows: "
            f"{completed_meta} vs {len(vectors)}."
        )
    usage = EmbeddingUsage.from_dict(meta.get("build_usage"))
    return vectors, usage, expected_dim_int


def _checkpoint_batch(
    *,
    jsonl_path: Path,
    meta_path: Path,
    model_name: str,
    registry_sha256: str,
    chunk_ids: Sequence[str],
    vectors_list: Sequence[Sequence[float]],
    usage: EmbeddingUsage,
    vector_dim: int,
    completed_count: int,
) -> None:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("a", encoding="utf-8") as handle:
        for op_id, vector in zip(chunk_ids, vectors_list):
            if len(vector) != vector_dim:
                raise CatalogIndexError(
                    f"Refusing to checkpoint dimension mismatch for {op_id}."
                )
            row = {
                "operation_id": op_id,
                "vector": [float(value) for value in vector],
                "vector_dim": vector_dim,
                "model": model_name,
                "registry_sha256": registry_sha256,
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    meta = {
        "format": PARTIAL_FORMAT,
        "model": model_name,
        "registry_sha256": registry_sha256,
        "vector_dim": vector_dim,
        "completed_count": int(completed_count),
        "build_usage": usage.to_dict(),
    }
    _atomic_write_json(meta_path, meta)


def _remove_partial_checkpoint(jsonl_path: Path, meta_path: Path) -> None:
    for path in (jsonl_path, meta_path):
        if path.exists():
            path.unlink()


def catalog_index_status(
    *,
    registry: Dict[str, OperationSpec],
    model_name: str,
    registry_sha256: str,
    cache_dir: Path,
) -> dict:
    """Offline status for final cache and partial checkpoint. No network."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    final_path = _cache_file(cache_dir, model_name, registry_sha256)
    jsonl_path = _partial_jsonl_path(cache_dir, model_name, registry_sha256)
    meta_path = _partial_meta_path(cache_dir, model_name, registry_sha256)
    total = len(registry)
    final_complete = False
    final_count = 0
    final_dim = None
    if final_path.exists():
        try:
            payload = json.loads(final_path.read_text(encoding="utf-8"))
            vectors = payload.get("vectors") or {}
            final_count = len(vectors)
            final_complete = (
                payload.get("model") == model_name
                and payload.get("registry_sha256") == registry_sha256
                and bool(payload.get("complete"))
                and set(vectors) == set(registry)
            )
            if vectors:
                final_dim = len(next(iter(vectors.values())))
        except (json.JSONDecodeError, TypeError, ValueError):
            final_complete = False
    partial_count = 0
    partial_dim = None
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            partial_count = int(meta.get("completed_count") or 0)
            if meta.get("vector_dim") is not None:
                partial_dim = int(meta["vector_dim"])
        except (json.JSONDecodeError, TypeError, ValueError):
            partial_count = -1
    elif jsonl_path.exists():
        partial_count = -1
    return {
        "model": model_name,
        "registry_sha256": registry_sha256,
        "total": total,
        "final_cache_path": str(final_path),
        "final_complete": final_complete,
        "final_completed_count": final_count,
        "final_vector_dim": final_dim,
        "partial_jsonl_path": str(jsonl_path),
        "partial_meta_path": str(meta_path),
        "partial_completed_count": partial_count,
        "partial_vector_dim": partial_dim,
        "partial_present": jsonl_path.exists() or meta_path.exists(),
        "completed": final_count if final_complete else max(partial_count, 0),
    }


def build_catalog_index(
    *,
    registry: Dict[str, OperationSpec],
    embedder,
    registry_sha256: str,
    cache_dir: Path,
    progress_callback=None,
) -> CatalogIndex:
    """Build or resume the catalog index. Embedding only; never calls an LLM."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    model_name = embedder.model_name
    final_path = _cache_file(cache_dir, model_name, registry_sha256)
    existing = _load_final_cache(
        final_path,
        model_name=model_name,
        registry_sha256=registry_sha256,
        registry=registry,
        require_complete=False,
    )
    if existing is not None:
        if progress_callback:
            progress_callback(
                {
                    "event": "final_cache_hit",
                    "completed": len(registry),
                    "total": len(registry),
                }
            )
        return existing

    jsonl_path = _partial_jsonl_path(cache_dir, model_name, registry_sha256)
    meta_path = _partial_meta_path(cache_dir, model_name, registry_sha256)
    vectors, usage, vector_dim = _load_partial_checkpoint(
        jsonl_path=jsonl_path,
        meta_path=meta_path,
        model_name=model_name,
        registry_sha256=registry_sha256,
        registry=registry,
    )
    ordered_ids = sorted(registry.keys())
    missing_ids = [op_id for op_id in ordered_ids if op_id not in vectors]
    if vectors and not missing_ids:
        dim = _validate_vector_dims(vectors)
        _write_final_catalog_cache(
            final_path,
            model_name=model_name,
            registry_sha256=registry_sha256,
            vectors=vectors,
            usage=usage,
        )
        loaded = _load_final_cache(
            final_path,
            model_name=model_name,
            registry_sha256=registry_sha256,
            registry=registry,
            require_complete=True,
        )
        assert loaded is not None
        _remove_partial_checkpoint(jsonl_path, meta_path)
        loaded.cache_reused = False
        return loaded

    batch_size = max(1, int(getattr(embedder, "batch_size", DEFAULT_CATALOG_BATCH_SIZE) or DEFAULT_CATALOG_BATCH_SIZE))
    total = len(registry)
    if progress_callback:
        progress_callback(
            {
                "event": "resume",
                "completed": len(vectors),
                "total": total,
                "missing": len(missing_ids),
            }
        )
    print(f"catalog index resume {len(vectors)}/{total}", flush=True)

    for start in range(0, len(missing_ids), batch_size):
        chunk_ids = missing_ids[start : start + batch_size]
        docs = [registry[op_id].index_document() for op_id in chunk_ids]
        lo = len(vectors) + 1
        hi = len(vectors) + len(chunk_ids)
        print(f"embedding {lo}-{hi}/{total}", flush=True)
        if progress_callback:
            progress_callback(
                {
                    "event": "batch_start",
                    "completed": len(vectors),
                    "total": total,
                    "batch_first": chunk_ids[0],
                    "batch_last": chunk_ids[-1],
                    "batch_size": len(chunk_ids),
                }
            )
        try:
            vectors_list, part_usage = embedder.embed(docs)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "Catalog embedding batch failed for operation_id range "
                f"{chunk_ids[0]} .. {chunk_ids[-1]} "
                f"(positions {lo}-{hi}/{total}): {type(exc).__name__}: {exc}"
            ) from exc
        if len(vectors_list) != len(chunk_ids):
            raise RuntimeError("Catalog embedding batch size mismatch.")
        batch_dim = len(vectors_list[0]) if vectors_list else 0
        if batch_dim <= 0:
            raise CatalogIndexError("Catalog embedding batch returned empty vectors.")
        if vector_dim is None:
            vector_dim = batch_dim
        elif batch_dim != vector_dim:
            raise CatalogIndexError(
                f"Catalog embedding dimension mismatch in batch "
                f"{chunk_ids[0]} .. {chunk_ids[-1]}: {batch_dim} vs {vector_dim}."
            )
        for op_id, vector in zip(chunk_ids, vectors_list):
            if len(vector) != vector_dim:
                raise CatalogIndexError(
                    f"Catalog embedding dimension mismatch for {op_id}."
                )
            vectors[op_id] = [float(value) for value in vector]
        usage.add(part_usage)
        _checkpoint_batch(
            jsonl_path=jsonl_path,
            meta_path=meta_path,
            model_name=model_name,
            registry_sha256=registry_sha256,
            chunk_ids=chunk_ids,
            vectors_list=vectors_list,
            usage=usage,
            vector_dim=vector_dim,
            completed_count=len(vectors),
        )
        if progress_callback:
            progress_callback(
                {
                    "event": "batch_checkpointed",
                    "completed": len(vectors),
                    "total": total,
                    "usage": usage.to_dict(),
                }
            )

    if set(vectors) != set(registry):
        raise RuntimeError(
            f"Catalog embedding incomplete after build: {len(vectors)}/{len(registry)}."
        )
    _validate_vector_dims(vectors)
    _write_final_catalog_cache(
        final_path,
        model_name=model_name,
        registry_sha256=registry_sha256,
        vectors=vectors,
        usage=usage,
    )
    loaded = _load_final_cache(
        final_path,
        model_name=model_name,
        registry_sha256=registry_sha256,
        registry=registry,
        require_complete=True,
    )
    assert loaded is not None
    _remove_partial_checkpoint(jsonl_path, meta_path)
    loaded.cache_reused = False
    loaded.build_usage = usage
    print(f"catalog index complete {len(vectors)}/{total}", flush=True)
    return loaded


def load_complete_catalog_index(
    *,
    registry: Dict[str, OperationSpec],
    model_name: str,
    registry_sha256: str,
    cache_dir: Path,
) -> CatalogIndex:
    """Load a verified final index. Never rebuilds. Never calls embedding APIs."""
    path = _cache_file(cache_dir, model_name, registry_sha256)
    loaded = _load_final_cache(
        path,
        model_name=model_name,
        registry_sha256=registry_sha256,
        registry=registry,
        require_complete=True,
    )
    assert loaded is not None
    return loaded


def load_or_build_catalog_index(
    *,
    registry: Dict[str, OperationSpec],
    embedder,
    registry_sha256: str,
    cache_dir: Path,
    require_complete: bool = False,
) -> CatalogIndex:
    """Compatibility helper.

    When require_complete=True, only a verified final cache is accepted and the
    embedder is never called. Otherwise build/resume via the JSONL checkpoint.
    """
    if require_complete:
        return load_complete_catalog_index(
            registry=registry,
            model_name=embedder.model_name,
            registry_sha256=registry_sha256,
            cache_dir=cache_dir,
        )
    return build_catalog_index(
        registry=registry,
        embedder=embedder,
        registry_sha256=registry_sha256,
        cache_dir=cache_dir,
    )


def cosine_self_check(vector: Sequence[float]) -> float:
    return cosine_similarity(vector, vector) if vector else math.nan
