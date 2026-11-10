"""Explicit mock / dry-run / live mode resolution for Retrieval Audit."""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

from hk_equity_mcp.agent.llm_client import OpenAICompatibleChatClient
from hk_equity_mcp.common.env_loader import load_embedding_profile, load_generation_profile
from hk_equity_mcp.operation_resolver.retrieval import (
    DeterministicHashEmbeddingBackend,
    OpenAICompatibleEmbeddingBackend,
)
from hk_equity_mcp.pe_fin_lite.identity import (
    EXPECTED_EMBEDDING_MODEL,
    EXPECTED_EMBEDDING_PROVIDER,
    EXPECTED_GENERATION_HOST,
    EXPECTED_GENERATION_MODEL,
    EXPECTED_GENERATION_PROVIDER,
    assert_live_provider_identity,
    host_from_base_url,
)
from hk_equity_mcp.pe_fin_lite.operation_embedding_index import (
    default_shared_cache_dir,
    resolve_embedding_cache_dir,
)


class AuditMode(str, Enum):
    MOCK = "mock"
    DRY_RUN = "dry_run"
    LIVE = "live"
    OFFLINE_RANK = "offline_rank"  # ranking/report from caches; zero network


@dataclass(frozen=True)
class ResolvedMode:
    mode: AuditMode
    mock: bool
    dry_run: bool
    live: bool
    allow_network: bool
    embedding_backend_kind: str  # mock_hash | real_text_embedding_3_small | none


class LivePathError(RuntimeError):
    pass


def resolve_mode(
    *,
    mock: bool = False,
    dry_run: bool = False,
    live: bool = False,
    no_mock: bool = False,
    offline_rank: bool = False,
) -> ResolvedMode:
    """Fail-closed mode resolution. ``--no-mock`` is deprecated and rejected."""
    if no_mock:
        raise LivePathError(
            "--no-mock is deprecated and refused. Use explicit --live for real "
            "services, or --mock / --dry-run for offline modes."
        )
    flags = [bool(mock), bool(dry_run), bool(live), bool(offline_rank)]
    if sum(1 for x in flags if x) > 1:
        # dry-run may combine with tooling flags at CLI layer; runner resolves
        # exclusive primary mode. Reject mock+live and dry_run+live here.
        if live and (mock or dry_run):
            raise LivePathError("Cannot combine --live with --mock or --dry-run")
        if mock and dry_run:
            raise LivePathError("Cannot combine --mock with --dry-run")
        if offline_rank and live:
            raise LivePathError(
                "Offline ranking/report modes are zero-network; do not pass --live"
            )
    if live:
        return ResolvedMode(
            mode=AuditMode.LIVE,
            mock=False,
            dry_run=False,
            live=True,
            allow_network=True,
            embedding_backend_kind="real_text_embedding_3_small",
        )
    if dry_run:
        return ResolvedMode(
            mode=AuditMode.DRY_RUN,
            mock=False,
            dry_run=True,
            live=False,
            allow_network=False,
            embedding_backend_kind="none",
        )
    if offline_rank:
        return ResolvedMode(
            mode=AuditMode.OFFLINE_RANK,
            mock=False,
            dry_run=False,
            live=False,
            allow_network=False,
            embedding_backend_kind="none",
        )
    if mock or not any(flags):
        # Default when only offline tooling without --live: mock.
        return ResolvedMode(
            mode=AuditMode.MOCK,
            mock=True,
            dry_run=False,
            live=False,
            allow_network=False,
            embedding_backend_kind="mock_hash",
        )
    raise LivePathError(
        "Ambiguous mode. Pass exactly one of --mock, --dry-run, or --live "
        "(ranking/report from caches use offline zero-network path)."
    )


def require_live_env_files(
    *,
    env_file: Optional[str],
    embedding_env_file: Optional[str],
    need_generation: bool,
    need_embedding: bool,
) -> None:
    if need_generation and not env_file:
        raise LivePathError(
            "Live generation requires --env-file "
            "(expected .env.relay-v41; fail-closed, no defaults)"
        )
    if need_embedding and not embedding_env_file:
        raise LivePathError(
            "Live embedding requires --embedding-env-file "
            "(expected .env.embedding-relay; fail-closed, no defaults)"
        )


def load_and_assert_generation(
    root: Path, env_file: str
) -> Dict[str, Any]:
    meta = load_generation_profile(root, env_file)
    host = host_from_base_url(meta.get("base_url"))
    provider = str(meta.get("provider") or "")
    model = str(meta.get("model") or "")
    if provider != EXPECTED_GENERATION_PROVIDER:
        raise LivePathError(
            f"generation provider mismatch: expected {EXPECTED_GENERATION_PROVIDER}, "
            f"got {provider!r}"
        )
    if host != EXPECTED_GENERATION_HOST:
        raise LivePathError(
            f"generation host mismatch: expected {EXPECTED_GENERATION_HOST}, got {host!r}"
        )
    if model != EXPECTED_GENERATION_MODEL:
        raise LivePathError(
            f"generation model mismatch: expected {EXPECTED_GENERATION_MODEL}, "
            f"got {model!r}"
        )
    return {
        "provider": provider,
        "base_url": meta["base_url"],
        "model": model,
        "profile_file": meta.get("profile_file"),
        "host": host,
        # never include api_key
    }


def load_and_assert_embedding(
    root: Path, embedding_env_file: str
) -> Dict[str, Any]:
    meta = load_embedding_profile(root, embedding_env_file)
    host = host_from_base_url(meta.get("base_url"))
    model = str(meta.get("model") or "")
    if model != EXPECTED_EMBEDDING_MODEL:
        raise LivePathError(
            f"embedding model mismatch: expected {EXPECTED_EMBEDDING_MODEL}, got {model!r}"
        )
    if not host:
        raise LivePathError("embedding host missing (fail-closed)")
    return {
        "provider": EXPECTED_EMBEDDING_PROVIDER,
        "base_url": meta["base_url"],
        "model": model,
        "profile_file": meta.get("profile_file"),
        "host": host,
    }


def build_generation_llm(*, json_mode: bool = True) -> OpenAICompatibleChatClient:
    """Construct generation client from already-loaded process env."""
    for key in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL"):
        if not os.environ.get(key):
            raise LivePathError(f"Missing {key} after env load (fail-closed)")
    return OpenAICompatibleChatClient(
        base_url=os.environ["LLM_BASE_URL"],
        api_key=os.environ["LLM_API_KEY"],
        model=os.environ["LLM_MODEL"],
        provider=os.environ.get("LLM_PROVIDER") or EXPECTED_GENERATION_PROVIDER,
        json_mode=json_mode,
    )


def build_embedding_backend(
    *,
    mode: ResolvedMode,
    emb_meta: Optional[Dict[str, Any]] = None,
    for_index_build: bool = True,
    timeout_sec: float = 60.0,
    max_retries: int = 5,
) -> Any:
    if mode.mock:
        return DeterministicHashEmbeddingBackend(model_name="mock-hash-embedding-v1")
    if mode.dry_run or mode.mode == AuditMode.OFFLINE_RANK:
        raise LivePathError(
            "Refusing to construct embedding backend in dry-run/offline-rank mode"
        )
    if not mode.live:
        raise LivePathError("Embedding backend requires --live")
    if emb_meta is None:
        raise LivePathError("emb_meta required for live embedding backend")
    kwargs: Dict[str, Any] = {
        "base_url": os.environ.get("EMBEDDING_BASE_URL") or emb_meta.get("base_url"),
        "api_key": os.environ.get("EMBEDDING_API_KEY"),
        "model": os.environ.get("EMBEDDING_MODEL") or emb_meta.get("model"),
        "timeout_sec": timeout_sec,
    }
    if not kwargs["api_key"]:
        raise LivePathError("Missing EMBEDDING_API_KEY (fail-closed; no hash fallback)")
    if for_index_build:
        kwargs["max_retries"] = max_retries
    else:
        kwargs["max_retries"] = min(1, max_retries)
    return OpenAICompatibleEmbeddingBackend(**kwargs)


def resolve_cache_dir(root: Path, embedding_cache_dir: Optional[str | Path]) -> Path:
    return resolve_embedding_cache_dir(root, embedding_cache_dir)


def public_meta_without_secrets(meta: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in (meta or {}).items():
        lk = str(k).lower()
        if any(tok in lk for tok in ("api_key", "token", "password", "secret")):
            continue
        out[k] = v
    return out


# Silence unused import warning helpers used by callers.
_DEFAULT_CACHE = default_shared_cache_dir
