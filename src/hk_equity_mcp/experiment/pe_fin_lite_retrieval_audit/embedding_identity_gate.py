"""Fail-closed embedding identity / dimension gates for retrieval audit.

Prevents mixing real 1536-d query vectors with host=null mock/hash catalog caches.
Does not modify PE-Fin Lite agent method runtime.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from hk_equity_mcp.experiment.pe_fin_lite_retrieval_audit.live_path import LivePathError
from hk_equity_mcp.pe_fin_lite.identity import EXPECTED_EMBEDDING_MODEL
from hk_equity_mcp.pe_fin_lite.operation_embedding_index import (
    CACHE_NAMESPACE,
    identity_digest,
)


class EmbeddingDimensionMismatch(RuntimeError):
    """Query and catalog vectors differ in length; ranking must fail closed."""


MOCK_HASH_DIM = 64
REAL_TEXT_EMBEDDING_DIM = 1536


def assert_vector_dimensions_match(
    query_vector: Sequence[float],
    catalog_vector: Sequence[float],
    *,
    context: str = "ranking",
) -> None:
    qn = len(query_vector)
    cn = len(catalog_vector)
    if qn != cn:
        raise EmbeddingDimensionMismatch(
            f"{context}: query_dim={qn} != catalog_dim={cn}; "
            "refusing zip/truncate cosine (fail-closed)"
        )
    if qn == 0:
        raise EmbeddingDimensionMismatch(f"{context}: empty embedding vectors")


def classify_catalog_source(
    *,
    embedding_provider_host: Optional[str],
    vector_dimension: Optional[int],
    embedding_model: Optional[str] = None,
) -> str:
    host = (embedding_provider_host or "").strip() or None
    dim = int(vector_dimension) if vector_dimension is not None else None
    if host is None or dim == MOCK_HASH_DIM:
        return "mock_hash"
    if (
        host
        and (embedding_model or EXPECTED_EMBEDDING_MODEL) == EXPECTED_EMBEDDING_MODEL
        and dim == REAL_TEXT_EMBEDDING_DIM
    ):
        return "real"
    if host and dim not in (None, MOCK_HASH_DIM):
        return "real"
    return "unknown"


def assert_real_backend_identity(
    *,
    embedding_backend: str,
    embedding_provider_host: Optional[str],
    vector_dimension: Optional[int],
    source: Optional[str] = None,
) -> None:
    """Refuse real_text_embedding_3_small with host=null / mock-hash / dim=64."""
    backend = str(embedding_backend or "")
    if "real_text_embedding" not in backend and backend != "real":
        return
    host = (embedding_provider_host or "").strip() or None
    src = source or classify_catalog_source(
        embedding_provider_host=host, vector_dimension=vector_dimension
    )
    if host is None:
        raise LivePathError(
            "real embedding backend forbids embedding_provider_host=null (fail-closed)"
        )
    if src == "mock_hash":
        raise LivePathError(
            "real embedding backend forbids mock/hash catalog source (fail-closed)"
        )
    if vector_dimension == MOCK_HASH_DIM:
        raise LivePathError(
            "real embedding backend forbids vector_dimension=64 mock/hash cache "
            "(fail-closed)"
        )


def list_catalog_cache_inventory(cache_dir: Path) -> List[Dict[str, Any]]:
    """List all pe_fin_lite_ops_*.json final caches (zero network)."""
    root = Path(cache_dir)
    out: List[Dict[str, Any]] = []
    if not root.exists():
        return out
    for path in sorted(root.glob("pe_fin_lite_ops_*.json")):
        name = path.name
        if ".partial" in name or name.endswith(".tmp"):
            continue
        digest = name[len("pe_fin_lite_ops_") : -len(".json")]
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            out.append(
                {
                    "path": str(path),
                    "identity_digest": digest,
                    "error": f"{type(exc).__name__}:{exc}",
                }
            )
            continue
        ident = payload.get("identity") or {}
        vectors = payload.get("vectors") or {}
        dim = None
        if isinstance(vectors, dict) and vectors:
            dim = len(next(iter(vectors.values())))
        host = ident.get("embedding_provider_host")
        source = classify_catalog_source(
            embedding_provider_host=host,
            vector_dimension=dim,
            embedding_model=ident.get("embedding_model"),
        )
        out.append(
            {
                "path": str(path),
                "identity_digest": digest,
                "embedding_provider_host": host,
                "embedding_model": ident.get("embedding_model"),
                "vector_dimension": dim,
                "documents_hash": ident.get("documents_hash"),
                "registry_fingerprint": ident.get("registry_fingerprint"),
                "catalog_fingerprint": ident.get("catalog_fingerprint"),
                "complete": bool(isinstance(vectors, dict) and vectors),
                "n_vectors": len(vectors) if isinstance(vectors, dict) else 0,
                "source": source,
                "bytes": path.stat().st_size,
                "namespace": ident.get("namespace") or CACHE_NAMESPACE,
            }
        )
    return out


def resolve_embedding_meta_for_offline(
    *,
    root: Path,
    embedding_env_file: Optional[str],
    query_cache_identity: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Resolve host/model for offline ranking without network calls.

    Preferred: --embedding-env-file (identity only).
    Fallback: complete question/step query cache identity with non-null host.
    """
    from hk_equity_mcp.experiment.pe_fin_lite_retrieval_audit.live_path import (
        load_and_assert_embedding,
    )

    if embedding_env_file:
        meta = load_and_assert_embedding(root, embedding_env_file)
        host = meta.get("host")
        if not host:
            raise LivePathError(
                "embedding env resolved empty host (fail-closed; no mock fallback)"
            )
        return {
            "provider": meta.get("provider"),
            "base_url": meta.get("base_url"),
            "model": meta.get("model") or EXPECTED_EMBEDDING_MODEL,
            "host": host,
            "profile_file": meta.get("profile_file"),
            "identity_source": "embedding_env_file",
            "expected_vector_dimension": REAL_TEXT_EMBEDDING_DIM,
        }

    qid = query_cache_identity or {}
    host = (qid.get("embedding_provider_host") or "").strip() or None
    model = qid.get("embedding_model") or EXPECTED_EMBEDDING_MODEL
    dim = qid.get("embedding_dim")
    if host and model:
        return {
            "provider": "openai_compatible",
            "base_url": f"https://{host}",
            "model": model,
            "host": host,
            "profile_file": None,
            "identity_source": "query_cache_identity",
            "expected_vector_dimension": int(dim)
            if dim is not None
            else REAL_TEXT_EMBEDDING_DIM,
        }

    raise LivePathError(
        "Offline real ranking requires --embedding-env-file (identity only, "
        "zero network) or a complete query-cache identity with non-null "
        "embedding_provider_host. Refusing host=null mock/hash catalog selection."
    )


def select_matching_catalog_cache(
    *,
    inventory: Sequence[Dict[str, Any]],
    expected_identity: Dict[str, Any],
    expected_digest: Optional[str] = None,
    require_real: bool = True,
    require_dim: Optional[int] = None,
) -> Dict[str, Any]:
    """Select exactly one catalog cache matching identity; else fail closed."""
    digest = expected_digest or identity_digest(expected_identity)
    matches = [row for row in inventory if row.get("identity_digest") == digest]
    if len(matches) == 1:
        chosen = matches[0]
    elif len(matches) == 0:
        # Fallback: match host/model/docs/registry/catalog fingerprint uniquely.
        host = expected_identity.get("embedding_provider_host")
        model = expected_identity.get("embedding_model")
        docs = expected_identity.get("documents_hash")
        reg = expected_identity.get("registry_fingerprint")
        cfp = expected_identity.get("catalog_fingerprint")
        soft = []
        for row in inventory:
            if row.get("embedding_provider_host") != host:
                continue
            if row.get("embedding_model") != model:
                continue
            if docs and row.get("documents_hash") != docs:
                continue
            if reg and row.get("registry_fingerprint") != reg:
                continue
            if cfp and row.get("catalog_fingerprint") != cfp:
                continue
            soft.append(row)
        if len(soft) != 1:
            raise LivePathError(
                "No unique catalog cache for expected embedding identity "
                f"digest={digest} host={host!r} model={model!r}; "
                f"digest_matches={len(matches)} field_matches={len(soft)}. "
                "Fail-closed (no silent mock/hash fallback)."
            )
        chosen = soft[0]
    else:
        raise LivePathError(
            f"Multiple catalog caches share identity digest={digest} (fail-closed)"
        )

    if require_real and chosen.get("source") != "real":
        raise LivePathError(
            f"Selected catalog cache source={chosen.get('source')!r} is not real "
            f"(host={chosen.get('embedding_provider_host')!r}, "
            f"dim={chosen.get('vector_dimension')}); fail-closed"
        )
    if require_dim is not None and chosen.get("vector_dimension") != require_dim:
        raise LivePathError(
            f"Selected catalog dim={chosen.get('vector_dimension')} != "
            f"required query/catalog dim={require_dim} (fail-closed)"
        )
    return chosen


def assert_query_catalog_alignment(
    *,
    query_dim: int,
    catalog_dim: int,
    query_host: Optional[str],
    catalog_host: Optional[str],
    query_model: Optional[str],
    catalog_model: Optional[str],
) -> Dict[str, bool]:
    if query_dim != catalog_dim:
        raise EmbeddingDimensionMismatch(
            f"query_dim={query_dim} != catalog_dim={catalog_dim} (fail-closed)"
        )
    qh = (query_host or "").strip() or None
    ch = (catalog_host or "").strip() or None
    if qh != ch:
        raise LivePathError(
            f"query host={qh!r} != catalog host={ch!r} (fail-closed)"
        )
    qm = query_model or EXPECTED_EMBEDDING_MODEL
    cm = catalog_model or EXPECTED_EMBEDDING_MODEL
    if qm != cm:
        raise LivePathError(
            f"query model={qm!r} != catalog model={cm!r} (fail-closed)"
        )
    return {
        "identity_match": True,
        "dimension_match": True,
        "host_match": True,
        "model_match": True,
    }
