"""Optional external artifact gates for clean-clone offline tests.

Frozen raw predictions, embedding indexes, and the pinned ETOM checkout are
intentionally not committed. Tests that need them must call these helpers
*before* assertions. Missing artifacts → pytest.skip with a precise setup hint.
Never convert assertion failures into skips.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

import pytest

PathLike = Union[str, Path]

ENV_HK_ARTIFACT_ROOT = "MCP_ROUTER_HK_ARTIFACT_ROOT"
ENV_ETOM_ARTIFACT_ROOT = "MCP_ROUTER_ETOM_ARTIFACT_ROOT"
ENV_ETOM_DATA_ROOT = "MCP_ROUTER_ETOM_DATA_ROOT"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _env_path(name: str) -> Optional[Path]:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return None
    return Path(raw).expanduser().resolve()


def hk_artifact_root() -> Path:
    """Root containing HK-155 / PlanRoute frozen ``outputs/`` trees."""
    return _env_path(ENV_HK_ARTIFACT_ROOT) or repo_root()


def etom_artifact_root() -> Path:
    """Root containing ETOM condition frozen outputs / root-raw incident files."""
    return _env_path(ENV_ETOM_ARTIFACT_ROOT) or repo_root()


def etom_data_root() -> Path:
    """Pinned ETOM official checkout directory (``data/etom/raw`` by default)."""
    override = _env_path(ENV_ETOM_DATA_ROOT)
    if override is not None:
        return override
    return repo_root() / "data" / "etom" / "raw"


def _as_path(root: Path, rel_or_abs: PathLike) -> Path:
    path = Path(rel_or_abs)
    if path.is_absolute():
        return path
    return (root / path).resolve()


def require_paths(
    *rel_paths: PathLike,
    root: Optional[Path] = None,
    env_var: str = ENV_HK_ARTIFACT_ROOT,
    reason: Optional[str] = None,
) -> Path:
    """Skip unless every path exists under ``root`` (default: HK artifact root)."""
    base = root or hk_artifact_root()
    missing = [str(p) for p in rel_paths if not _as_path(base, p).exists()]
    if missing:
        hint = (
            reason
            or (
                "Missing frozen run artifact(s) for offline freeze/audit tests: "
                + ", ".join(missing)
                + f". Place them under the repo or set {env_var} to a directory "
                "that contains the relative paths above."
            )
        )
        pytest.skip(hint)
    return base


def require_any_path(
    *rel_paths: PathLike,
    root: Optional[Path] = None,
    env_var: str = ENV_HK_ARTIFACT_ROOT,
    reason: Optional[str] = None,
) -> Path:
    """Skip unless at least one of the paths exists; return the first hit."""
    base = root or hk_artifact_root()
    for rel in rel_paths:
        candidate = _as_path(base, rel)
        if candidate.exists():
            return candidate
    hint = (
        reason
        or (
            "Missing frozen run artifact (any of: "
            + ", ".join(str(p) for p in rel_paths)
            + f"). Set {env_var} to the experiment worktree that holds these paths."
        )
    )
    pytest.skip(hint)


def require_etom_source(reason: Optional[str] = None) -> Path:
    """Skip unless the pinned ETOM checkout (or MCP_ROUTER_ETOM_DATA_ROOT) exists."""
    checkout = etom_data_root()
    registry = checkout / "data" / "mcp_registry.json"
    if not registry.exists():
        pytest.skip(
            reason
            or (
                "Pinned ETOM checkout missing "
                f"(expected registry at {registry}). "
                "Run `bash scripts/fetch_etom_source.sh` or set "
                f"{ENV_ETOM_DATA_ROOT} to the pinned checkout directory."
            )
        )
    return checkout


def require_complete_index(
    cache_glob: str,
    *,
    root: Optional[Path] = None,
    env_var: str = ENV_HK_ARTIFACT_ROOT,
    reason: Optional[str] = None,
) -> Sequence[Path]:
    """Skip unless at least one complete embedding-index JSON matches ``cache_glob``."""
    base = root or hk_artifact_root()
    hits = sorted(base.glob(cache_glob))
    if not hits:
        pytest.skip(
            reason
            or (
                f"Missing complete embedding index matching {cache_glob!r} under {base}. "
                f"Build/copy the index or set {env_var}."
            )
        )
    return hits


def existing_paths(root: Path, rel_paths: Iterable[PathLike]) -> list[Path]:
    return [p for p in (_as_path(root, r) for r in rel_paths) if p.exists()]


def require_hk_artifact(*rel_paths: PathLike, reason: Optional[str] = None) -> Path:
    """Require HK frozen artifact(s); return the resolved path of the first entry."""
    base = require_paths(
        *rel_paths, root=hk_artifact_root(), env_var=ENV_HK_ARTIFACT_ROOT, reason=reason
    )
    return _as_path(base, rel_paths[0])


def require_etom_artifact(*rel_paths: PathLike, reason: Optional[str] = None) -> Path:
    """Require ETOM frozen artifact(s); return the resolved path of the first entry."""
    base = require_paths(
        *rel_paths,
        root=etom_artifact_root(),
        env_var=ENV_ETOM_ARTIFACT_ROOT,
        reason=reason,
    )
    return _as_path(base, rel_paths[0])


def require_embedding_index(
    cache_glob: str = ".cache/pe_fin_replan_etom_operation_embeddings_v1/pe_fin_replan_etom_ops_*.json",
    *,
    reason: Optional[str] = None,
) -> Sequence[Path]:
    return require_complete_index(
        cache_glob,
        root=etom_artifact_root(),
        env_var=ENV_ETOM_ARTIFACT_ROOT,
        reason=reason,
    )
