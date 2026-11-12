"""Operation retrieval: embedding cosine + 0.12 * lexical (Ghoshal et al. 2026 §4.1).

Embedding backbone in this project is adapted to relay-hosted text-embedding-3-small.
Ranking formula and non-embedding behavior follow the paper.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import math
import os
import re
import socket
import ssl
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from hk_equity_mcp.operation_resolver.registry import registry_fingerprint
from hk_equity_mcp.operation_resolver.schemas import OperationSpec

LEXICAL_WEIGHT = 0.12
DEFAULT_TOP_K = 20
MAX_TOP_K = 20
_TOKEN_RE = re.compile(r"[a-z0-9_.]+", re.IGNORECASE)


def tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError(
            f"Embedding dimension mismatch: {len(a)} vs {len(b)}. "
            "Do not mix embedding models or cache entries."
        )
    if not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def lexical_score(query: str, operation: OperationSpec) -> float:
    """Paper-style lexical blend: overlap, operation coverage, action/exact matches."""
    q_tokens = set(tokenize(query))
    if not q_tokens:
        return 0.0
    doc = operation.index_document().lower()
    d_tokens = set(tokenize(doc))
    overlap = len(q_tokens & d_tokens) / len(q_tokens)

    name_tokens = set(tokenize(operation.name.replace("_", " ")))
    op_cov = len(q_tokens & name_tokens) / max(len(name_tokens), 1)

    action_tokens = set(tokenize(" ".join(operation.action_terms)))
    action = len(q_tokens & action_tokens) / max(len(action_tokens), 1)

    exact_op = 1.0 if operation.name.lower() in query.lower() else 0.0
    exact_client = 1.0 if operation.service_id.lower() in query.lower() else 0.0
    exact_id = 1.0 if operation.operation_id.lower() in query.lower() else 0.0

    # Fixed weights — deterministic, no LLM.
    score = (
        0.40 * overlap
        + 0.25 * op_cov
        + 0.20 * action
        + 0.10 * exact_op
        + 0.03 * exact_client
        + 0.02 * exact_id
    )
    return float(min(1.0, score))


class EmbeddingBackend(ABC):
    """Embedding interface. Production config uses OpenAI-compatible /embeddings."""

    model_name: str

    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError


_RETRYABLE_HTTP = frozenset({429, 500, 502, 503, 504})
_RETRYABLE_DISCONNECT = (
    http.client.RemoteDisconnected,
    http.client.IncompleteRead,
    ConnectionResetError,
    ConnectionAbortedError,
    BrokenPipeError,
)
_SSL_INTERRUPT_MARKERS = (
    "eof",
    "unexpected_eof",
    "connection reset",
    "broken pipe",
    "connection aborted",
    "syscall",
)
_RETRYABLE_TYPE_NAMES = (
    "RemoteDisconnected",
    "IncompleteRead",
    "ConnectionResetError",
    "ConnectionAbortedError",
    "BrokenPipeError",
)


def _is_ssl_connection_interrupt(exc: BaseException) -> bool:
    if not isinstance(exc, ssl.SSLError):
        return False
    text = str(exc).lower()
    return any(marker in text for marker in _SSL_INTERRUPT_MARKERS)


class OpenAICompatibleEmbeddingBackend(EmbeddingBackend):
    """Relay/OpenAI-compatible embeddings with optional transport retries.

    Retries only TimeoutError / URLError / disconnect / 429 / 5xx / SSL EOF.
    Non-429 HTTP 4xx fail closed. Never lexical/mock fallback. Never logs keys/vectors.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout_sec: float = 60.0,
        max_retries: int = 0,
        retry_backoff_seconds: Optional[Sequence[float]] = None,
        on_attempt: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        self.base_url = (base_url or os.environ.get("EMBEDDING_BASE_URL") or "").rstrip(
            "/"
        )
        self.api_key = api_key or os.environ.get("EMBEDDING_API_KEY") or ""
        self.model_name = model or os.environ.get("EMBEDDING_MODEL") or ""
        self.timeout_sec = float(timeout_sec)
        self.max_retries = max(0, int(max_retries))
        self.retry_backoff_seconds = [float(x) for x in (retry_backoff_seconds or [])]
        self.on_attempt = on_attempt
        self.retry_count = 0
        self.call_count = 0
        if not self.base_url:
            raise RuntimeError(
                "Missing EMBEDDING_BASE_URL. Load EMBEDDING_ENV_FILE=.env.embedding-relay "
                "and do not fall back to lexical-only or the generation endpoint."
            )
        if not self.api_key:
            raise RuntimeError(
                "Missing EMBEDDING_API_KEY. Configure .env.embedding-relay; "
                "never print the key value."
            )
        if not self.model_name:
            raise RuntimeError(
                "Missing EMBEDDING_MODEL. Pin text-embedding-3-small for this project."
            )
        if self.model_name.strip().lower() == "auto":
            raise RuntimeError("EMBEDDING_MODEL=auto is not allowed; pin an explicit model.")

    def _emit(self, payload: Dict[str, Any]) -> None:
        if self.on_attempt is not None:
            self.on_attempt(payload)

    def _backoff_delay(self, retry_index: int) -> float:
        if not self.retry_backoff_seconds:
            return float(2 ** retry_index)
        if retry_index < len(self.retry_backoff_seconds):
            return float(self.retry_backoff_seconds[retry_index])
        return float(self.retry_backoff_seconds[-1])

    def _embed_once(self, texts: List[str]) -> List[List[float]]:
        url = f"{self.base_url}/embeddings"
        payload = json.dumps({"model": self.model_name, "input": texts}).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise RuntimeError(
                f"Embedding request failed with HTTP {exc.code} for model "
                f"{self.model_name}. No lexical-only fallback."
            ) from exc
        except (TimeoutError, socket.timeout) as exc:
            raise TimeoutError(
                f"Embedding request failed for model {self.model_name}: TimeoutError. "
                "No lexical-only fallback."
            ) from exc
        except _RETRYABLE_DISCONNECT:
            raise
        except ssl.SSLError as exc:
            if _is_ssl_connection_interrupt(exc):
                raise
            raise RuntimeError(
                f"Embedding request failed for model {self.model_name}: SSLError. "
                "No lexical-only fallback."
            ) from exc
        except urllib.error.URLError as exc:
            raise urllib.error.URLError(
                f"Embedding request failed for model {self.model_name}: URLError. "
                "No lexical-only fallback."
            ) from exc
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"Embedding request failed for model {self.model_name}: {type(exc).__name__}. "
                "No lexical-only fallback."
            ) from exc
        data = body.get("data")
        if not isinstance(data, list) or len(data) != len(texts):
            raise RuntimeError("Unexpected embedding response shape.")
        ordered = sorted(data, key=lambda row: int(row.get("index", 0)))
        vectors: List[List[float]] = []
        for row in ordered:
            emb = row.get("embedding")
            if not isinstance(emb, list) or not emb:
                raise RuntimeError("Embedding response missing vectors.")
            vectors.append([float(x) for x in emb])
        return vectors

    @staticmethod
    def _is_retryable(exc: BaseException) -> bool:
        if isinstance(exc, (TimeoutError, socket.timeout)):
            return True
        if isinstance(exc, _RETRYABLE_DISCONNECT):
            return True
        if _is_ssl_connection_interrupt(exc):
            return True
        if isinstance(exc, urllib.error.HTTPError):
            return int(exc.code) in _RETRYABLE_HTTP
        if isinstance(exc, urllib.error.URLError):
            reason = getattr(exc, "reason", None)
            if isinstance(reason, BaseException) and (
                isinstance(reason, _RETRYABLE_DISCONNECT)
                or _is_ssl_connection_interrupt(reason)
                or isinstance(reason, (TimeoutError, socket.timeout))
            ):
                return True
            return True
        if isinstance(exc, RuntimeError):
            text = str(exc)
            if any(f"HTTP {code}" in text for code in _RETRYABLE_HTTP):
                return True
            if "TimeoutError" in text or "URLError" in text:
                return True
            if any(name in text for name in _RETRYABLE_TYPE_NAMES):
                return True
            if "SSLError" in text and any(
                marker in text.lower() for marker in _SSL_INTERRUPT_MARKERS
            ):
                return True
            if "HTTP 4" in text:
                return False
            cause = exc.__cause__
            if isinstance(cause, BaseException) and cause is not exc:
                return OpenAICompatibleEmbeddingBackend._is_retryable(cause)
            return False
        return False

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        max_attempts = 1 + self.max_retries
        last_exc: Optional[BaseException] = None
        for attempt in range(1, max_attempts + 1):
            self._emit(
                {
                    "phase": "attempt",
                    "attempt": attempt,
                    "max_attempts": max_attempts,
                    "batch_size": len(texts),
                }
            )
            try:
                vectors = self._embed_once(texts)
                self.call_count += 1
                return vectors
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                retryable = self._is_retryable(exc)
                self._emit(
                    {
                        "phase": "error",
                        "attempt": attempt,
                        "max_attempts": max_attempts,
                        "error_type": type(exc).__name__,
                        "retryable": retryable,
                    }
                )
                if (not retryable) or attempt >= max_attempts:
                    if isinstance(exc, RuntimeError):
                        raise
                    raise RuntimeError(
                        f"Embedding request failed for model {self.model_name}: "
                        f"{type(exc).__name__}. No lexical-only fallback."
                    ) from exc
                delay = self._backoff_delay(attempt - 1)
                self.retry_count += 1
                self._emit(
                    {
                        "phase": "retry",
                        "attempt": attempt,
                        "max_attempts": max_attempts,
                        "delay_sec": delay,
                        "error_type": type(exc).__name__,
                    }
                )
                time.sleep(delay)
        assert last_exc is not None
        raise RuntimeError(
            f"Embedding request failed for model {self.model_name}: "
            f"{type(last_exc).__name__}. No lexical-only fallback."
        ) from last_exc


class DeterministicHashEmbeddingBackend(EmbeddingBackend):
    """Offline test backend: hashed bag-of-tokens vectors (not for paid experiments)."""

    def __init__(self, model_name: str = "mock-hash-embedding-v1", dim: int = 64):
        self.model_name = model_name
        self.dim = dim

    def embed(self, texts: List[str]) -> List[List[float]]:
        out: List[List[float]] = []
        for text in texts:
            vec = [0.0] * self.dim
            for tok in tokenize(text):
                h = hashlib.sha256(tok.encode("utf-8")).digest()
                idx = int.from_bytes(h[:4], "big") % self.dim
                sign = 1.0 if h[4] % 2 == 0 else -1.0
                vec[idx] += sign
            # L2 normalize for cosine stability
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / norm for v in vec])
        return out


class EmbeddingCache:
    """Local embedding cache keyed by model + registry fingerprint + text hash."""

    def __init__(self, cache_dir: Path, model_name: str, registry_fp: str):
        self.cache_dir = Path(cache_dir)
        self.model_name = model_name
        self.registry_fp = registry_fp
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key(self, text: str) -> str:
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        raw = f"{self.model_name}|{self.registry_fp}|{text_hash}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, text: str) -> Optional[List[float]]:
        path = self.cache_dir / f"{self._key(text)}.json"
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        # Never store API keys; only model/fingerprint/vector metadata.
        if payload.get("model") != self.model_name:
            return None
        if payload.get("registry_fingerprint") != self.registry_fp:
            return None
        emb = payload.get("embedding")
        if not isinstance(emb, list):
            return None
        return [float(x) for x in emb]

    def put(self, text: str, embedding: List[float]) -> None:
        path = self.cache_dir / f"{self._key(text)}.json"
        payload = {
            "model": self.model_name,
            "registry_fingerprint": self.registry_fp,
            "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "embedding": embedding,
        }
        path.write_text(json.dumps(payload), encoding="utf-8")


class OperationRetriever:
    def __init__(
        self,
        registry: Dict[str, OperationSpec],
        embedding_backend: EmbeddingBackend,
        cache_dir: Optional[Path] = None,
        *,
        preloaded_doc_vectors: Optional[Dict[str, List[float]]] = None,
        ensure_docs: bool = True,
    ):
        self.registry = registry
        self.backend = embedding_backend
        self.registry_fp = registry_fingerprint(registry)
        root = Path(__file__).resolve().parents[3]
        self.cache = EmbeddingCache(
            cache_dir or (root / ".cache" / "operation_embeddings"),
            model_name=embedding_backend.model_name,
            registry_fp=self.registry_fp,
        )
        self._doc_vectors: Dict[str, List[float]] = {}
        if preloaded_doc_vectors is not None:
            missing = sorted(set(registry) - set(preloaded_doc_vectors))
            if missing:
                raise RuntimeError(
                    "preloaded_doc_vectors incomplete for registry; missing "
                    f"{len(missing)} ids (e.g. {missing[:3]})"
                )
            self._doc_vectors = {
                oid: list(preloaded_doc_vectors[oid]) for oid in registry
            }
        elif ensure_docs:
            self._ensure_doc_embeddings()
        else:
            raise RuntimeError(
                "OperationRetriever requires preloaded_doc_vectors when "
                "ensure_docs=False (live runs must use a complete shared index)."
            )

    def _ensure_doc_embeddings(self) -> None:
        missing_ids: List[str] = []
        missing_docs: List[str] = []
        for op_id, op in sorted(self.registry.items()):
            doc = op.index_document()
            cached = self.cache.get(doc)
            if cached is not None:
                self._doc_vectors[op_id] = cached
            else:
                missing_ids.append(op_id)
                missing_docs.append(doc)
        if not missing_docs:
            return
        vectors = self.backend.embed(missing_docs)
        if len(vectors) != len(missing_docs):
            raise RuntimeError("Embedding backend returned unexpected batch size.")
        for op_id, doc, vec in zip(missing_ids, missing_docs, vectors):
            self.cache.put(doc, vec)
            self._doc_vectors[op_id] = vec

    def rank(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        *,
        allowed_ids: Optional[Sequence[str]] = None,
    ) -> List[Dict[str, object]]:
        if top_k is None:
            top_k = DEFAULT_TOP_K
        top_k = int(top_k)
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        if top_k > MAX_TOP_K:
            top_k = MAX_TOP_K

        if allowed_ids is None:
            pool_ids = list(self.registry.keys())
        else:
            allow = {str(x) for x in allowed_ids}
            pool_ids = [oid for oid in self.registry.keys() if oid in allow]

        q_vec = self.backend.embed([query])[0]
        scored: List[Tuple[float, str, float, float]] = []
        for op_id in pool_ids:
            op = self.registry[op_id]
            if op_id not in self._doc_vectors:
                continue
            cos = cosine_similarity(q_vec, self._doc_vectors[op_id])
            lex = lexical_score(query, op)
            score = cos + LEXICAL_WEIGHT * lex
            scored.append((score, op_id, cos, lex))

        # Stable ranking: score desc, then operation_id asc for ties.
        scored.sort(key=lambda row: (-row[0], row[1]))
        results: List[Dict[str, object]] = []
        for score, op_id, cos, lex in scored[:top_k]:
            op = self.registry[op_id]
            results.append(
                {
                    "operation_id": op.operation_id,
                    "score": round(float(score), 6),
                    "cosine_similarity": float(cos),
                    "lexical_score": float(lex),
                    "lexical_weight": float(LEXICAL_WEIGHT),
                    "service_id": op.service_id,
                    "name": op.name,
                    "summary": op.summary,
                }
            )
        return results
