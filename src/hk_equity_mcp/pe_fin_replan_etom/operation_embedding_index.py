"""PE-Fin RePlan-ETOM operation embedding index (catalog docs only).

Adapted from ``hk_equity_mcp.pe_fin_lite.operation_embedding_index``:

* cache namespace is ``pe_fin_replan_etom_operation_embeddings_v1`` and lives
  under ``.cache/``;
* the HK ``pe_fin_lite_operation_embeddings_v1`` directory and the
  Operation-Visible ``etom_embeddings`` directory are FORBIDDEN for index writes
  of this condition (their final files use incompatible identities and must
  never be touched);
* ``HashEmbeddingClient`` (offline mock) may build a deterministic index;
* a LIVE (network) embedder is refused unless ``allow_live=True`` is passed.
  Tests never pass it.

Transport hardening (batched embed + retries + partial JSONL checkpoints) is
unchanged. Ranking semantics live in ``tool_retrieval`` (R5).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from hk_equity_mcp.operation_resolver.registry import registry_fingerprint
from hk_equity_mcp.operation_resolver.schemas import OperationSpec

LOGGER = logging.getLogger("hk_equity_mcp.pe_fin_replan_etom.operation_embedding_index")

CACHE_NAMESPACE = "pe_fin_replan_etom_operation_embeddings_v1"
HK_PE_FIN_LITE_NAMESPACE = "pe_fin_lite_operation_embeddings_v1"
_FORBIDDEN_CACHE_DIR_NAMES = frozenset(
    {
        "operation_embeddings",  # OV per-doc cache
        "mcp_zero_hk_v1",
        "etom_embeddings",
        "etom",
        HK_PE_FIN_LITE_NAMESPACE,
    }
)


class IncompleteOperationIndexError(RuntimeError):
    """Complete index required but missing / incomplete."""


class OperationIndexIdentityError(RuntimeError):
    """Index identity mismatch or corrupt checkpoint."""


@dataclass
class OperationIndexStatus:
    namespace: str
    cache_dir: str
    identity: Dict[str, Any]
    identity_sha256: str
    completed: int
    total: int
    complete: bool
    vector_dimension: Optional[int]
    final_path: Optional[str]
    partial_path: Optional[str]
    provenance: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "namespace": self.namespace,
            "cache_dir": self.cache_dir,
            "identity": self.identity,
            "identity_sha256": self.identity_sha256,
            "completed": self.completed,
            "total": self.total,
            "complete": self.complete,
            "vector_dimension": self.vector_dimension,
            "final_path": self.final_path,
            "partial_path": self.partial_path,
            "provenance": dict(self.provenance),
        }


def assert_pe_fin_replan_etom_cache_dir(cache_dir: Path) -> Path:
    resolved = Path(cache_dir).resolve()
    forbidden = [p for p in resolved.parts if p in _FORBIDDEN_CACHE_DIR_NAMES]
    if forbidden:
        raise OperationIndexIdentityError(
            f"Refusing cache directory {resolved}: contains forbidden component(s) "
            f"{forbidden}. PE-Fin RePlan-ETOM indexes use namespace {CACHE_NAMESPACE} only."
        )
    if resolved.name != CACHE_NAMESPACE:
        raise OperationIndexIdentityError(
            f"Cache directory must be named {CACHE_NAMESPACE!r} (got {resolved})."
        )
    return Path(cache_dir)


def default_shared_cache_dir(root: Path) -> Path:
    return assert_pe_fin_replan_etom_cache_dir(Path(root) / ".cache" / CACHE_NAMESPACE)


def resolve_embedding_cache_dir(
    root: Path, embedding_cache_dir: Optional[Path | str] = None
) -> Path:
    if embedding_cache_dir is None or str(embedding_cache_dir).strip() == "":
        return default_shared_cache_dir(root)
    path = Path(embedding_cache_dir)
    if not path.is_absolute():
        path = Path(root) / path
    return assert_pe_fin_replan_etom_cache_dir(path)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


from hk_equity_mcp.pe_fin_replan_etom.operation_documents import (  # noqa: E402
    documents_for_registry,
    documents_hash,
)


def ordered_operation_ids(registry: Dict[str, OperationSpec]) -> List[str]:
    return [oid for oid, _, _ in documents_for_registry(registry)]


def ordered_ids_hash(ids: Sequence[str]) -> str:
    return _sha256_text("\n".join(ids) + "\n")


def build_index_identity(
    *,
    registry: Dict[str, OperationSpec],
    catalog_fingerprint: str,
    embedding_model: str,
    embedding_provider_host: Optional[str],
    expected_vector_dimension: Optional[int] = None,
    namespace: str = CACHE_NAMESPACE,
) -> Dict[str, Any]:
    ids = ordered_operation_ids(registry)
    identity: Dict[str, Any] = {
        "namespace": str(namespace),
        "embedding_provider_host": (
            str(embedding_provider_host).strip().lower()
            if embedding_provider_host
            else None
        ),
        "embedding_model": str(embedding_model),
        "catalog_fingerprint": str(catalog_fingerprint),
        "registry_fingerprint": registry_fingerprint(registry),
        "ordered_operation_ids": list(ids),
        "ordered_operation_ids_hash": ordered_ids_hash(ids),
        "documents_hash": documents_hash(registry),
        "expected_vector_dimension": (
            int(expected_vector_dimension)
            if expected_vector_dimension is not None
            else None
        ),
    }
    return identity


def identity_digest(identity: Dict[str, Any]) -> str:
    # Hash without expected_vector_dimension so first-batch dim discovery does not
    # fork the digest; dimension is still validated against identity once set.
    core = {
        k: v
        for k, v in identity.items()
        if k != "expected_vector_dimension"
    }
    blob = json.dumps(core, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def index_paths(cache_dir: Path, identity: Dict[str, Any]) -> Tuple[Path, Path, Path, str]:
    digest = identity_digest(identity)
    root = assert_pe_fin_replan_etom_cache_dir(cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    final = root / f"pe_fin_replan_etom_ops_{digest}.json"
    partial = root / f"pe_fin_replan_etom_ops_{digest}.partial.jsonl"
    partial_meta = root / f"pe_fin_replan_etom_ops_{digest}.partial.meta.json"
    return final, partial, partial_meta, digest


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    data = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    with tmp.open("w", encoding="utf-8") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _fsync_file(path: Path) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.flush()
        os.fsync(handle.fileno())


def _load_partial(
    partial: Path,
    *,
    expected_ids: Sequence[str],
    expected_model: str,
    expected_doc_hashes: Dict[str, str],
    expected_dim: Optional[int],
) -> Dict[str, List[float]]:
    if not partial.exists():
        return {}
    expected = set(expected_ids)
    vectors: Dict[str, List[float]] = {}
    seen_dims: set = set()
    lines = partial.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            if index == len(lines) - 1:
                # Truncated last line — ignore; fail closed on earlier corruption.
                continue
            raise OperationIndexIdentityError(
                f"Corrupt partial JSONL at line {index + 1}: {exc}"
            ) from exc
        oid = row.get("operation_id")
        if oid is None:
            raise OperationIndexIdentityError("partial row missing operation_id")
        oid = str(oid)
        if oid not in expected:
            raise OperationIndexIdentityError(
                f"partial contains unknown operation_id: {oid}"
            )
        if oid in vectors:
            raise OperationIndexIdentityError(
                f"partial contains duplicate operation_id: {oid}"
            )
        if row.get("model") != expected_model:
            raise OperationIndexIdentityError(
                f"partial model mismatch for {oid}: {row.get('model')!r}"
            )
        doc_hash = row.get("document_sha256")
        if doc_hash != expected_doc_hashes.get(oid):
            raise OperationIndexIdentityError(
                f"partial document hash mismatch for {oid}"
            )
        vec = row.get("vector")
        if not isinstance(vec, list) or not vec:
            raise OperationIndexIdentityError(f"partial missing vector for {oid}")
        dim = int(row.get("vector_dim") or len(vec))
        if dim != len(vec):
            raise OperationIndexIdentityError(
                f"partial vector_dim mismatch for {oid}: {dim} vs {len(vec)}"
            )
        if expected_dim is not None and dim != int(expected_dim):
            raise OperationIndexIdentityError(
                f"partial dimension drift for {oid}: {dim} != {expected_dim}"
            )
        seen_dims.add(dim)
        vectors[oid] = [float(x) for x in vec]
    if len(seen_dims) > 1:
        raise OperationIndexIdentityError(
            f"partial has inconsistent dimensions: {sorted(seen_dims)}"
        )
    return vectors


def _load_final(
    final: Path, identity: Dict[str, Any]
) -> Optional[Dict[str, List[float]]]:
    if not final.exists():
        return None
    try:
        payload = json.loads(final.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise OperationIndexIdentityError(f"corrupt final index JSON: {exc}") from exc
    if not payload.get("complete"):
        raise OperationIndexIdentityError("final index missing complete=true")
    stored = payload.get("identity") or {}
    # Compare core identity (ignore expected_vector_dimension null vs set).
    for key in (
        "namespace",
        "embedding_model",
        "catalog_fingerprint",
        "registry_fingerprint",
        "ordered_operation_ids_hash",
        "documents_hash",
    ):
        if stored.get(key) != identity.get(key):
            raise OperationIndexIdentityError(
                f"final identity mismatch on {key}: "
                f"stored={stored.get(key)!r} expected={identity.get(key)!r}"
            )
    if stored.get("ordered_operation_ids") != identity.get("ordered_operation_ids"):
        raise OperationIndexIdentityError("final ordered_operation_ids mismatch")
    vectors_raw = payload.get("vectors") or {}
    if not isinstance(vectors_raw, dict):
        raise OperationIndexIdentityError("final vectors payload invalid")
    expected_ids = list(identity.get("ordered_operation_ids") or [])
    if set(vectors_raw.keys()) != set(expected_ids):
        raise OperationIndexIdentityError(
            "final vectors IDs incomplete or unexpected"
        )
    if len(vectors_raw) != len(expected_ids):
        raise OperationIndexIdentityError("final vectors not unique/complete")
    vectors: Dict[str, List[float]] = {}
    dims: set = set()
    for oid in expected_ids:
        vec = vectors_raw[oid]
        if not isinstance(vec, list) or not vec:
            raise OperationIndexIdentityError(f"final missing vector for {oid}")
        dims.add(len(vec))
        vectors[oid] = [float(x) for x in vec]
    if len(dims) != 1:
        raise OperationIndexIdentityError(f"final inconsistent dims: {sorted(dims)}")
    dim = next(iter(dims))
    stored_dim = payload.get("vector_dimension")
    if stored_dim is not None and int(stored_dim) != dim:
        raise OperationIndexIdentityError("final vector_dimension field mismatch")
    expected_dim = identity.get("expected_vector_dimension")
    if expected_dim is not None and int(expected_dim) != dim:
        raise OperationIndexIdentityError(
            f"final dim {dim} != identity expected {expected_dim}"
        )
    return vectors


def index_status(
    *,
    registry: Dict[str, OperationSpec],
    identity: Dict[str, Any],
    cache_dir: Path,
) -> OperationIndexStatus:
    """Pure local status — no network."""
    final, partial, _meta, digest = index_paths(cache_dir, identity)
    total = len(identity.get("ordered_operation_ids") or [])
    provenance: Dict[str, Any] = {}
    vectors: Optional[Dict[str, List[float]]] = None
    if final.exists():
        vectors = _load_final(final, identity)
        provenance["source"] = "final_cache"
        provenance["final_path"] = str(final)
    else:
        doc_hashes = {
            oid: dhash for oid, _doc, dhash in documents_for_registry(registry)
        }
        vectors = _load_partial(
            partial,
            expected_ids=identity.get("ordered_operation_ids") or [],
            expected_model=str(identity["embedding_model"]),
            expected_doc_hashes=doc_hashes,
            expected_dim=identity.get("expected_vector_dimension"),
        )
        provenance["source"] = "partial" if vectors else "empty"
        provenance["partial_path"] = str(partial) if partial.exists() else None
    completed = len(vectors or {})
    dim = None
    if vectors:
        dim = len(next(iter(vectors.values())))
    complete = bool(vectors) and completed == total and final.exists()
    return OperationIndexStatus(
        namespace=CACHE_NAMESPACE,
        cache_dir=str(cache_dir),
        identity=identity,
        identity_sha256=digest,
        completed=completed,
        total=total,
        complete=complete,
        vector_dimension=dim,
        final_path=str(final) if final.exists() else None,
        partial_path=str(partial) if partial.exists() else None,
        provenance=provenance,
    )


def load_complete_index(
    *,
    registry: Dict[str, OperationSpec],
    identity: Dict[str, Any],
    cache_dir: Path,
) -> Dict[str, List[float]]:
    status = index_status(registry=registry, identity=identity, cache_dir=cache_dir)
    if not status.complete:
        raise IncompleteOperationIndexError(
            f"PE-Fin RePlan-ETOM operation embedding index incomplete: "
            f"{status.completed}/{status.total} complete={status.complete} "
            f"cache_dir={cache_dir}"
        )
    final, _, _, _ = index_paths(cache_dir, identity)
    vectors = _load_final(final, identity)
    assert vectors is not None
    return vectors


def build_or_resume_index(
    *,
    registry: Dict[str, OperationSpec],
    identity: Dict[str, Any],
    cache_dir: Path,
    embed_fn,  # Callable[[List[str]], List[List[float]]]
    batch_size: int = 16,
    sleep_fn=time.sleep,
) -> Dict[str, Any]:
    """Build/resume the shared operation index. Returns status dict.

    ``embed_fn`` must already implement transport retries. This function never
    falls back to lexical/mock and never instantiates a generation LLM.
    """
    del sleep_fn  # reserved; retries live inside embed_fn
    cache_dir = assert_pe_fin_replan_etom_cache_dir(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    final, partial, partial_meta, digest = index_paths(cache_dir, identity)
    docs = documents_for_registry(registry)
    doc_by_id = {oid: doc for oid, doc, _ in docs}
    doc_hashes = {oid: dhash for oid, _doc, dhash in docs}
    ids = [oid for oid, _, _ in docs]
    provenance: Dict[str, Any] = {"built_in_session": True}

    # Prefer complete final.
    if final.exists():
        vectors = _load_final(final, identity)
        return {
            **index_status(
                registry=registry, identity=identity, cache_dir=cache_dir
            ).to_dict(),
            "action": "reused_final",
            "vectors_loaded": len(vectors or {}),
        }

    vectors = _load_partial(
        partial,
        expected_ids=ids,
        expected_model=str(identity["embedding_model"]),
        expected_doc_hashes=doc_hashes,
        expected_dim=identity.get("expected_vector_dimension"),
    )
    missing = [oid for oid in ids if oid not in vectors]
    vector_dim: Optional[int] = None
    if vectors:
        vector_dim = len(next(iter(vectors.values())))

    batch_size = max(1, int(batch_size))
    completed_before = len(vectors)
    for start in range(0, len(missing), batch_size):
        batch_ids = missing[start : start + batch_size]
        batch_docs = [doc_by_id[oid] for oid in batch_ids]
        LOGGER.info(
            "embedding_index batch=%s-%s attempt_via_backend size=%s completed=%s/%s",
            start + 1,
            start + len(batch_ids),
            len(batch_ids),
            len(vectors),
            len(ids),
        )
        batch_vecs = embed_fn(batch_docs)
        if len(batch_vecs) != len(batch_ids):
            raise RuntimeError(
                f"Embedding batch size mismatch: got {len(batch_vecs)} "
                f"expected {len(batch_ids)}"
            )
        for oid, vec in zip(batch_ids, batch_vecs):
            if not isinstance(vec, list) or not vec:
                raise RuntimeError(f"Empty embedding vector for {oid}")
            dim = len(vec)
            if vector_dim is None:
                vector_dim = dim
            elif dim != vector_dim:
                raise OperationIndexIdentityError(
                    f"dimension mismatch for {oid}: {dim} != {vector_dim}"
                )
            expected_dim = identity.get("expected_vector_dimension")
            if expected_dim is not None and dim != int(expected_dim):
                raise OperationIndexIdentityError(
                    f"dimension mismatch vs identity for {oid}: "
                    f"{dim} != {expected_dim}"
                )
            row = {
                "operation_id": oid,
                "vector": [float(x) for x in vec],
                "vector_dim": dim,
                "model": identity["embedding_model"],
                "document_sha256": doc_hashes[oid],
                "catalog_fingerprint": identity["catalog_fingerprint"],
                "registry_fingerprint": identity["registry_fingerprint"],
                "documents_hash": identity["documents_hash"],
                "namespace": CACHE_NAMESPACE,
            }
            with partial.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            vectors[oid] = row["vector"]
        _atomic_write_json(
            partial_meta,
            {
                "identity": identity,
                "identity_sha256": digest,
                "completed": len(vectors),
                "total": len(ids),
                "vector_dimension": vector_dim,
                "complete": False,
            },
        )
        LOGGER.info(
            "embedding_index checkpoint completed=%s/%s dim=%s",
            len(vectors),
            len(ids),
            vector_dim,
        )

    if len(vectors) != len(ids) or set(vectors) != set(ids):
        raise IncompleteOperationIndexError(
            f"index build finished incomplete: {len(vectors)}/{len(ids)}"
        )
    assert vector_dim is not None
    _write_final(
        final=final,
        partial=partial,
        partial_meta=partial_meta,
        identity=identity,
        vectors=vectors,
        vector_dimension=vector_dim,
        provenance={"source": "build_or_resume", "batches_from": completed_before},
    )
    status = index_status(registry=registry, identity=identity, cache_dir=cache_dir)
    out = status.to_dict()
    out["action"] = "built"
    out["provenance"] = {"source": "build_or_resume"}
    return out


def _write_final(
    *,
    final: Path,
    partial: Path,
    partial_meta: Path,
    identity: Dict[str, Any],
    vectors: Dict[str, List[float]],
    vector_dimension: int,
    provenance: Dict[str, Any],
) -> None:
    ids = list(identity.get("ordered_operation_ids") or [])
    if set(vectors.keys()) != set(ids) or len(vectors) != len(ids):
        raise IncompleteOperationIndexError("cannot finalize: IDs incomplete/duplicate")
    for oid, vec in vectors.items():
        if len(vec) != int(vector_dimension):
            raise OperationIndexIdentityError(
                f"finalize dim mismatch for {oid}: {len(vec)} != {vector_dimension}"
            )
    identity_out = dict(identity)
    identity_out["expected_vector_dimension"] = int(vector_dimension)
    payload = {
        "complete": True,
        "namespace": CACHE_NAMESPACE,
        "identity": identity_out,
        "identity_sha256": identity_digest(identity),
        "vector_dimension": int(vector_dimension),
        "ordered_operation_ids": ids,
        "vectors": {oid: vectors[oid] for oid in ids},
        "provenance": provenance,
        "n_operations": len(ids),
    }
    _atomic_write_json(final, payload)
    # Remove partial only after final is durable.
    if partial.exists():
        partial.unlink()
    if partial_meta.exists():
        partial_meta.unlink()


# ---------------------------------------------------------------------------
# Gated build helpers + CatalogIndex bridge
# ---------------------------------------------------------------------------


class LiveEmbeddingGatedError(RuntimeError):
    """A network embedder was passed without the explicit future-live gate."""


def is_offline_embedder(embedder: Any) -> bool:
    from hk_equity_mcp.etom.embeddings import HashEmbeddingClient

    return isinstance(embedder, HashEmbeddingClient)


def assert_embedder_allowed(embedder: Any, *, allow_live: bool = False) -> None:
    if is_offline_embedder(embedder):
        return
    if not allow_live:
        raise LiveEmbeddingGatedError(
            "Refusing to build/resume a PE-Fin RePlan-ETOM index with a non-offline "
            "embedder. Live embedding build is future-only and requires allow_live=True "
            "plus the future live runbook (docs/pe_fin_replan_etom_v1/FUTURE_LIVE_COMMANDS.md)."
        )


def build_index_with_embedder(
    *,
    registry: Dict[str, OperationSpec],
    identity: Dict[str, Any],
    cache_dir: Path,
    embedder: Any,
    batch_size: int = 16,
    allow_live: bool = False,
) -> Dict[str, Any]:
    """Build/resume with a ``HashEmbeddingClient``-style embedder (or gated live one)."""
    assert_embedder_allowed(embedder, allow_live=allow_live)
    if embedder.model_name != identity["embedding_model"]:
        raise OperationIndexIdentityError(
            f"embedder model {embedder.model_name!r} != identity model "
            f"{identity['embedding_model']!r}"
        )

    def _embed(texts: List[str]) -> List[List[float]]:
        vectors, _usage = embedder.embed(list(texts))
        return vectors

    return build_or_resume_index(
        registry=registry,
        identity=identity,
        cache_dir=cache_dir,
        embed_fn=_embed,
        batch_size=batch_size,
    )


def load_catalog_index(
    *,
    registry: Dict[str, OperationSpec],
    identity: Dict[str, Any],
    cache_dir: Path,
    registry_sha256: str,
):
    """Load a COMPLETE final index as an ``etom.embeddings.CatalogIndex``. Never builds."""
    from hk_equity_mcp.etom.embeddings import CatalogIndex, EmbeddingUsage

    vectors = load_complete_index(registry=registry, identity=identity, cache_dir=cache_dir)
    final, _p, _m, _d = index_paths(cache_dir, identity)
    return CatalogIndex(
        model_name=str(identity["embedding_model"]),
        registry_sha256=registry_sha256,
        vectors=vectors,
        build_usage=EmbeddingUsage(),
        cache_reused=True,
        cache_path=str(final),
        completed_count=len(vectors),
        total_count=len(registry),
    )


def namespace_inventory(root: Path) -> Dict[str, Any]:
    """Read-only inventory of sibling cache dirs (never written by this condition)."""
    cache_root = Path(root) / ".cache"
    names = [CACHE_NAMESPACE, HK_PE_FIN_LITE_NAMESPACE, "etom_embeddings", "operation_embeddings"]
    return {
        name: {
            "path": str(cache_root / name),
            "exists": (cache_root / name).exists(),
            "writable_by_this_condition": name == CACHE_NAMESPACE,
        }
        for name in names
    }
