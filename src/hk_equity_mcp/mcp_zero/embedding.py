"""Embedding utilities for MCP-Zero: counting wrapper + server/tool vector table.

Two embedder interfaces exist in this repo:
  * HK ``EmbeddingBackend.embed(texts) -> List[vector]``
  * ETOM clients ``embed(texts) -> (vectors, usage)``
``CountingEmbedder`` accepts both and exposes ``embed(texts) -> List[vector]``
plus call / token counters (query-time accounting).
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from hk_equity_mcp.mcp_zero.metadata import McpZeroCatalog
from hk_equity_mcp.mcp_zero.scoring import SCORE_OFFICIAL_COMBINED


class IncompleteEmbeddingTableError(RuntimeError):
    """A complete table was required but is missing (never builds silently)."""


class CacheNamespaceError(RuntimeError):
    """Refusing to read or write a foreign embedding index."""


# Distinct from Operation-Visible ``operation_embeddings`` and ETOM ``catalog-*.json``.
CACHE_NAMESPACE = "mcp_zero_hk_v1"
_FORBIDDEN_CACHE_DIR_NAMES = frozenset(
    {"operation_embeddings", "etom_embeddings", "etom"}
)


def estimate_tokens(texts: Sequence[str]) -> int:
    return sum(max(1, (len(t) + 3) // 4) for t in texts)


class CountingEmbedder:
    """Wrap an embedder; count calls/tokens; normalise the return type."""

    def __init__(self, inner: Any, model_name: Optional[str] = None):
        self.inner = inner
        self.model_name = model_name or getattr(inner, "model_name", "unknown")
        self.calls = 0
        self.tokens = 0
        self.token_source = "provider"

    def snapshot(self) -> Dict[str, int]:
        return {"calls": self.calls, "tokens": self.tokens}

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        texts = list(texts)
        if not texts:
            return []
        result = self.inner.embed(texts)
        if isinstance(result, tuple):
            vectors, usage = result
            calls = int(getattr(usage, "calls", 0) or 1)
            tokens = int(getattr(usage, "tokens", 0) or 0)
            if getattr(usage, "token_source", "provider") != "provider":
                self.token_source = "estimated"
        else:
            vectors = result
            calls = 1
            tokens = 0
        if tokens <= 0:
            tokens = estimate_tokens(texts)
            self.token_source = "estimated"
        if len(vectors) != len(texts):
            raise RuntimeError("Embedding backend returned unexpected batch size.")
        self.calls += calls
        self.tokens += tokens
        return [list(map(float, v)) for v in vectors]


@dataclass
class EmbeddingTable:
    model_name: str
    fingerprint: str
    server_desc: Dict[str, List[float]] = field(default_factory=dict)
    server_summary: Dict[str, List[float]] = field(default_factory=dict)
    tool: Dict[str, List[float]] = field(default_factory=dict)
    build_calls: int = 0
    build_tokens: int = 0
    build_latency_ms: float = 0.0
    cache_reused: bool = False
    cache_path: Optional[str] = None
    identity: Dict[str, Any] = field(default_factory=dict)
    identity_sha256: str = ""
    index_action: str = ""
    complete: bool = False

    def size(self) -> int:
        return len(self.server_desc) + len(self.server_summary) + len(self.tool)


def _units(catalog: McpZeroCatalog, use_summary: bool):
    units = []
    for server in catalog.servers():
        units.append(("sd:" + server.server_id, server.description))
        if use_summary and server.summary:
            units.append(("ss:" + server.server_id, server.summary))
        for tool in server.tools:
            units.append(("t:" + tool.operation_id, tool.text))
    return units


def assert_mcp_zero_cache_dir(cache_dir: Path) -> Path:
    """Refuse Operation-Visible and ETOM cache directories."""
    resolved = Path(cache_dir).resolve()
    forbidden = [part for part in resolved.parts if part in _FORBIDDEN_CACHE_DIR_NAMES]
    if forbidden:
        raise CacheNamespaceError(
            "Refusing to use cache directory "
            f"{resolved}: path contains {forbidden}. MCP-Zero indexes live only "
            f"under namespace {CACHE_NAMESPACE} and must not overwrite "
            "operation_embeddings or ETOM catalog indexes."
        )
    return Path(cache_dir)


def default_shared_cache_dir(root: Path) -> Path:
    """Worktree-shared catalog index cache (independent of output_dir)."""
    return assert_mcp_zero_cache_dir(Path(root) / ".cache" / CACHE_NAMESPACE)


def resolve_embedding_cache_dir(
    root: Path, embedding_cache_dir: Optional[Path | str] = None
) -> Path:
    if embedding_cache_dir is None or str(embedding_cache_dir).strip() == "":
        return default_shared_cache_dir(root)
    return assert_mcp_zero_cache_dir(Path(embedding_cache_dir))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _vector_dimension(vectors: Dict[str, List[float]]) -> Optional[int]:
    dims = {len(v) for v in vectors.values() if isinstance(v, list)}
    if not dims:
        return None
    if len(dims) != 1:
        raise IncompleteEmbeddingTableError(
            f"Embedding index has inconsistent vector dimensions: {sorted(dims)}"
        )
    return next(iter(dims))


def embedding_cache_identity(
    *,
    embedding_model: str,
    catalog_fingerprint: str,
    metadata_hash: str,
    score_mode: str = SCORE_OFFICIAL_COMBINED,
    server_top_k: int = 3,
    tool_top_k: int = 5,
    similarity_threshold: float = 0.0,
    use_summary: bool = True,
) -> Dict[str, Any]:
    """Identity of one server/tool embedding index.

    Vectors are reused only when the model, unified catalog fingerprint,
    server/tool metadata hash, and scoring config all match.
    """
    return {
        "namespace": CACHE_NAMESPACE,
        "embedding_model": str(embedding_model),
        "catalog_fingerprint": str(catalog_fingerprint),
        "metadata_hash": str(metadata_hash),
        "score_mode": str(score_mode),
        "server_top_k": int(server_top_k),
        "tool_top_k": int(tool_top_k),
        "similarity_threshold": float(similarity_threshold),
        "use_summary": bool(use_summary),
    }


def identity_digest(identity: Dict[str, Any]) -> str:
    blob = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def table_paths(cache_dir: Path, identity: Dict[str, Any]) -> Tuple[Path, Path, str]:
    digest = identity_digest(identity)
    root = assert_mcp_zero_cache_dir(cache_dir)
    final = Path(root) / f"mcp_zero_table_{digest}.json"
    partial = Path(root) / f"mcp_zero_table_{digest}.partial.jsonl"
    return final, partial, digest


def _split_table(
    vectors: Dict[str, List[float]], model_name: str, fingerprint: str
) -> EmbeddingTable:
    table = EmbeddingTable(model_name=model_name, fingerprint=fingerprint)
    for key, vec in vectors.items():
        kind, _, ident = key.partition(":")
        if kind == "sd":
            table.server_desc[ident] = vec
        elif kind == "ss":
            table.server_summary[ident] = vec
        elif kind == "t":
            table.tool[ident] = vec
    return table


def _read_partial(
    partial: Path, expected: set
) -> Dict[str, List[float]]:
    """Load a partial checkpoint. A truncated final line is ignored."""
    vectors: Dict[str, List[float]] = {}
    if not partial.exists():
        return vectors
    lines = partial.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines) - 1:
                continue
            raise
        key = row.get("k")
        if key not in expected or "v" not in row:
            continue
        vectors[str(key)] = row["v"]
    return vectors


def _payload_matches(
    payload: Dict[str, Any], identity: Dict[str, Any], expected: set
) -> bool:
    vectors = payload.get("vectors") or {}
    if not isinstance(vectors, dict):
        return False
    try:
        _vector_dimension(vectors)
    except IncompleteEmbeddingTableError:
        return False
    return (
        payload.get("namespace") == CACHE_NAMESPACE
        and payload.get("identity") == identity
        and payload.get("identity_sha256") == identity_digest(identity)
        and payload.get("model") == identity["embedding_model"]
        and payload.get("fingerprint") == identity["metadata_hash"]
        and bool(payload.get("complete"))
        and set(vectors) == expected
    )


def verify_complete_index_payload(
    payload: Dict[str, Any],
    *,
    identity: Dict[str, Any],
    expected_keys: set,
    expected_identity_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    """Fail-closed verification of an on-disk complete index payload."""
    if not isinstance(payload, dict):
        raise IncompleteEmbeddingTableError("Index payload is not an object.")
    if payload.get("namespace") != CACHE_NAMESPACE:
        raise CacheNamespaceError(
            f"Foreign/missing namespace on index: {payload.get('namespace')!r}"
        )
    if not bool(payload.get("complete")):
        raise IncompleteEmbeddingTableError(
            "Index payload is not marked complete; refusing partial-as-complete."
        )
    digest = identity_digest(identity)
    if expected_identity_sha256 and payload.get("identity_sha256") != expected_identity_sha256:
        raise IncompleteEmbeddingTableError(
            "identity_sha256 mismatch: "
            f"file={payload.get('identity_sha256')!r} expected={expected_identity_sha256!r}"
        )
    if payload.get("identity_sha256") != digest:
        raise IncompleteEmbeddingTableError(
            "identity_sha256 does not match recomputed digest for current identity."
        )
    if payload.get("identity") != identity:
        raise IncompleteEmbeddingTableError(
            "Index identity object does not match expected identity."
        )
    if payload.get("model") != identity["embedding_model"]:
        raise IncompleteEmbeddingTableError(
            f"Index model mismatch: {payload.get('model')!r}"
        )
    if payload.get("fingerprint") != identity["metadata_hash"]:
        raise IncompleteEmbeddingTableError("Index metadata_hash / fingerprint mismatch.")
    vectors = payload.get("vectors")
    if not isinstance(vectors, dict):
        raise IncompleteEmbeddingTableError("Index vectors must be an object.")
    if set(vectors) != expected_keys:
        raise IncompleteEmbeddingTableError(
            "Index vector key set does not match catalog units "
            f"({len(vectors)}/{len(expected_keys)})."
        )
    dim = _vector_dimension(vectors)
    if dim is None or dim <= 0:
        raise IncompleteEmbeddingTableError("Index has no usable vectors.")
    return {
        "namespace": CACHE_NAMESPACE,
        "identity": identity,
        "identity_sha256": digest,
        "vector_count": len(vectors),
        "vector_dimension": dim,
        "complete": True,
        "model": identity["embedding_model"],
    }


def install_verified_index(
    source_path: Path,
    *,
    cache_dir: Path,
    catalog: McpZeroCatalog,
    model_name: str,
    catalog_fingerprint: str,
    score_mode: str = SCORE_OFFICIAL_COMBINED,
    server_top_k: int = 3,
    tool_top_k: int = 5,
    similarity_threshold: float = 0.0,
    use_summary: bool = True,
    expected_identity_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    """Offline copy of a verified complete index into the shared cache.

    Never calls an embedding API. Source file is left untouched.
    """
    source_path = Path(source_path)
    if not source_path.is_file():
        raise IncompleteEmbeddingTableError(f"Source index missing: {source_path}")
    try:
        payload = json.loads(source_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise IncompleteEmbeddingTableError(
            f"Damaged JSON index refused: {source_path}"
        ) from exc
    metadata_hash = catalog.fingerprint()
    identity = embedding_cache_identity(
        embedding_model=model_name,
        catalog_fingerprint=catalog_fingerprint,
        metadata_hash=metadata_hash,
        score_mode=score_mode,
        server_top_k=server_top_k,
        tool_top_k=tool_top_k,
        similarity_threshold=similarity_threshold,
        use_summary=use_summary,
    )
    expected = {k for k, _ in _units(catalog, use_summary)}
    verified = verify_complete_index_payload(
        payload,
        identity=identity,
        expected_keys=expected,
        expected_identity_sha256=expected_identity_sha256,
    )
    cache_dir = assert_mcp_zero_cache_dir(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    final, _partial, digest = table_paths(cache_dir, identity)
    if digest != verified["identity_sha256"]:
        raise IncompleteEmbeddingTableError("Internal identity digest mismatch.")
    source_sha = file_sha256(source_path)
    if final.exists():
        existing_sha = file_sha256(final)
        if existing_sha != source_sha:
            # Re-verify existing before refusing overwrite of a different blob.
            try:
                existing = json.loads(final.read_text(encoding="utf-8"))
                verify_complete_index_payload(
                    existing,
                    identity=identity,
                    expected_keys=expected,
                    expected_identity_sha256=digest,
                )
            except (json.JSONDecodeError, IncompleteEmbeddingTableError, CacheNamespaceError):
                raise CacheNamespaceError(
                    f"Shared cache already has a different file at {final}; "
                    "refusing to overwrite without matching verified content."
                )
            # Existing is also a valid complete index for this identity.
            return {
                **verified,
                "source_path": str(source_path),
                "cache_path": str(final),
                "cache_file_sha256": existing_sha,
                "source_file_sha256": source_sha,
                "index_action": "already_present",
                "cache_reused": True,
            }
        return {
            **verified,
            "source_path": str(source_path),
            "cache_path": str(final),
            "cache_file_sha256": existing_sha,
            "source_file_sha256": source_sha,
            "index_action": "already_present",
            "cache_reused": True,
        }
    tmp = Path(str(final) + ".tmp")
    tmp.write_bytes(source_path.read_bytes())
    tmp.replace(final)
    installed_sha = file_sha256(final)
    if installed_sha != source_sha:
        raise IncompleteEmbeddingTableError(
            "Installed shared cache SHA256 does not match source after copy."
        )
    return {
        **verified,
        "source_path": str(source_path),
        "cache_path": str(final),
        "cache_file_sha256": installed_sha,
        "source_file_sha256": source_sha,
        "index_action": "installed_from_verified_source",
        "cache_reused": True,
    }


def table_status(
    catalog: McpZeroCatalog,
    *,
    model_name: str,
    cache_dir: Path,
    use_summary: bool = False,
    catalog_fingerprint: Optional[str] = None,
    score_mode: str = SCORE_OFFICIAL_COMBINED,
    server_top_k: int = 3,
    tool_top_k: int = 5,
    similarity_threshold: float = 0.0,
) -> Dict[str, Any]:
    """Offline status of the final cache + partial checkpoint (no network)."""
    metadata_hash = catalog.fingerprint()
    identity = embedding_cache_identity(
        embedding_model=model_name,
        catalog_fingerprint=catalog_fingerprint or metadata_hash,
        metadata_hash=metadata_hash,
        score_mode=score_mode,
        server_top_k=server_top_k,
        tool_top_k=tool_top_k,
        similarity_threshold=similarity_threshold,
        use_summary=use_summary,
    )
    final, partial, digest = table_paths(cache_dir, identity)
    expected = {k for k, _ in _units(catalog, use_summary)}
    final_complete = False
    final_count = 0
    foreign_final = False
    damaged_final = False
    vector_dimension = None
    cache_file_sha256 = None
    if final.exists():
        try:
            cache_file_sha256 = file_sha256(final)
            payload = json.loads(final.read_text(encoding="utf-8"))
            vectors = payload.get("vectors") or {}
            final_count = len(vectors) if isinstance(vectors, dict) else 0
            if payload.get("namespace") not in {None, CACHE_NAMESPACE}:
                foreign_final = True
            final_complete = _payload_matches(payload, identity, expected)
            if final_complete and isinstance(vectors, dict):
                vector_dimension = _vector_dimension(vectors)
        except IncompleteEmbeddingTableError:
            final_complete = False
            damaged_final = True
        except (json.JSONDecodeError, TypeError, ValueError, OSError):
            final_complete = False
            damaged_final = True
    partial_vectors = _read_partial(partial, expected) if partial.exists() else {}
    return {
        "namespace": CACHE_NAMESPACE,
        "model": model_name,
        "identity": identity,
        "identity_sha256": digest,
        "catalog_fingerprint": identity["catalog_fingerprint"],
        "metadata_hash": metadata_hash,
        "total": len(expected),
        "vector_count": final_count if final_complete else len(partial_vectors),
        "vector_dimension": vector_dimension,
        "final_cache_path": str(final),
        "final_complete": final_complete,
        "complete": final_complete,
        "final_completed_count": final_count,
        "foreign_final": foreign_final,
        "damaged_final": damaged_final,
        "partial_path": str(partial),
        "partial_completed_count": len(partial_vectors),
        "completed": final_count if final_complete else len(partial_vectors),
        "cache_dir": str(cache_dir),
        "cache_path": str(final),
        "cache_file_sha256": cache_file_sha256,
        "cache_reused": final_complete,
        "index_action": "reuse" if final_complete else (
            "partial_present" if partial_vectors else "missing"
        ),
        "build_calls": 0,
        "build_tokens": 0,
    }


def load_complete_table(
    catalog: McpZeroCatalog,
    *,
    model_name: str,
    cache_dir: Path,
    use_summary: bool = False,
    catalog_fingerprint: Optional[str] = None,
    score_mode: str = SCORE_OFFICIAL_COMBINED,
    server_top_k: int = 3,
    tool_top_k: int = 5,
    similarity_threshold: float = 0.0,
) -> EmbeddingTable:
    status = table_status(
        catalog,
        model_name=model_name,
        cache_dir=cache_dir,
        use_summary=use_summary,
        catalog_fingerprint=catalog_fingerprint,
        score_mode=score_mode,
        server_top_k=server_top_k,
        tool_top_k=tool_top_k,
        similarity_threshold=similarity_threshold,
    )
    if status["foreign_final"]:
        raise CacheNamespaceError(
            "Refusing to read a non-MCP-Zero embedding index at "
            f"{status['final_cache_path']}."
        )
    if not status["final_complete"]:
        raise IncompleteEmbeddingTableError(
            "Complete MCP-Zero embedding table required but missing "
            f"({status['completed']}/{status['total']}) at {status['final_cache_path']}. "
            "Refusing to use a partial or foreign cache. Build the index explicitly."
        )
    path = Path(status["final_cache_path"])
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise IncompleteEmbeddingTableError(
            f"Damaged JSON embedding index refused: {path}"
        ) from exc
    expected = {k for k, _ in _units(catalog, use_summary)}
    verify_complete_index_payload(
        payload,
        identity=dict(status["identity"]),
        expected_keys=expected,
        expected_identity_sha256=status["identity_sha256"],
    )
    table = _split_table(payload["vectors"], model_name, catalog.fingerprint())
    table.cache_reused = True
    table.cache_path = status["final_cache_path"]
    table.identity = dict(status["identity"])
    table.identity_sha256 = status["identity_sha256"]
    table.index_action = "reuse"
    table.complete = True
    table.build_calls = 0
    table.build_tokens = 0
    table.build_latency_ms = 0.0
    return table


def build_embedding_table(
    catalog: McpZeroCatalog,
    embedder: CountingEmbedder,
    *,
    cache_dir: Optional[Path] = None,
    use_summary: bool = True,
    batch_size: int = 64,
    progress: Optional[Callable[[int, int], None]] = None,
    catalog_fingerprint: Optional[str] = None,
    score_mode: str = SCORE_OFFICIAL_COMBINED,
    server_top_k: int = 3,
    tool_top_k: int = 5,
    similarity_threshold: float = 0.0,
) -> EmbeddingTable:
    """Embed servers + tools.

    A missing index is an explicit build. A partial checkpoint for the same
    identity is resumed. A complete matching index is reused. A file from
    another namespace is never overwritten or silently reused.
    """
    metadata_hash = catalog.fingerprint()
    model_name = embedder.model_name
    identity = embedding_cache_identity(
        embedding_model=model_name,
        catalog_fingerprint=catalog_fingerprint or metadata_hash,
        metadata_hash=metadata_hash,
        score_mode=score_mode,
        server_top_k=server_top_k,
        tool_top_k=tool_top_k,
        similarity_threshold=similarity_threshold,
        use_summary=use_summary,
    )
    units = _units(catalog, use_summary)
    expected = {k for k, _ in units}
    vectors: Dict[str, List[float]] = {}
    final = partial = None
    index_action = "build"
    if cache_dir is not None:
        cache_dir = assert_mcp_zero_cache_dir(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        final, partial, _digest = table_paths(cache_dir, identity)
        status = table_status(
            catalog,
            model_name=model_name,
            cache_dir=cache_dir,
            use_summary=use_summary,
            catalog_fingerprint=catalog_fingerprint,
            score_mode=score_mode,
            server_top_k=server_top_k,
            tool_top_k=tool_top_k,
            similarity_threshold=similarity_threshold,
        )
        if status["foreign_final"]:
            raise CacheNamespaceError(
                "Refusing to overwrite a foreign embedding index at "
                f"{status['final_cache_path']}."
            )
        if status["final_complete"]:
            return load_complete_table(
                catalog,
                model_name=model_name,
                cache_dir=cache_dir,
                use_summary=use_summary,
                catalog_fingerprint=catalog_fingerprint,
                score_mode=score_mode,
                server_top_k=server_top_k,
                tool_top_k=tool_top_k,
                similarity_threshold=similarity_threshold,
            )
        vectors = _read_partial(partial, expected)
        if vectors:
            index_action = "resume_partial"
    calls_before, tokens_before = embedder.calls, embedder.tokens
    started = time.perf_counter()
    pending = [(k, t) for k, t in units if k not in vectors]
    step = max(1, int(batch_size))
    for start in range(0, len(pending), step):
        chunk = pending[start : start + step]
        vecs = embedder.embed([t for _, t in chunk])
        for (key, _), vec in zip(chunk, vecs):
            vectors[key] = vec
        if partial is not None:
            with partial.open("a", encoding="utf-8") as handle:
                for (key, _), vec in zip(chunk, vecs):
                    handle.write(json.dumps({"k": key, "v": vec}) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        if progress:
            progress(len(vectors), len(units))
    build_latency_ms = (time.perf_counter() - started) * 1000.0
    if set(vectors) != expected:
        raise IncompleteEmbeddingTableError(
            "Embedding build finished without a complete index "
            f"({len(vectors)}/{len(expected)}). Refusing to mark it complete."
        )
    table = _split_table(vectors, model_name, metadata_hash)
    table.build_calls = embedder.calls - calls_before
    table.build_tokens = embedder.tokens - tokens_before
    table.build_latency_ms = round(build_latency_ms, 3)
    table.identity = identity
    table.identity_sha256 = identity_digest(identity)
    table.index_action = index_action
    table.complete = True
    table.cache_reused = False
    if final is not None:
        if final.exists():
            try:
                existing = json.loads(final.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                raise CacheNamespaceError(
                    f"Refusing to overwrite unreadable cache file {final}."
                ) from exc
            if existing.get("namespace") not in {None, CACHE_NAMESPACE}:
                raise CacheNamespaceError(
                    f"Refusing to overwrite foreign cache file {final}."
                )
            if (
                existing.get("namespace") == CACHE_NAMESPACE
                and existing.get("identity_sha256") not in {None, table.identity_sha256}
            ):
                raise CacheNamespaceError(
                    f"Refusing to overwrite MCP-Zero index with a different identity: {final}."
                )
        tmp = Path(str(final) + ".tmp")
        tmp.write_text(
            json.dumps(
                {
                    "namespace": CACHE_NAMESPACE,
                    "identity": identity,
                    "identity_sha256": table.identity_sha256,
                    "model": model_name,
                    "fingerprint": metadata_hash,
                    "complete": True,
                    "vectors": vectors,
                }
            ),
            encoding="utf-8",
        )
        tmp.replace(final)
        if partial is not None and partial.exists():
            partial.unlink()
        table.cache_path = str(final)
    return table
