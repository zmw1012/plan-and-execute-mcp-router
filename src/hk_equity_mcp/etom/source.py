"""Locate the pinned ETOM checkout and refuse modified official files."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Dict


class EtomSourceError(RuntimeError):
    pass


def repo_root_from_here() -> Path:
    return Path(__file__).resolve().parents[3]


def pin_path(root: Path) -> Path:
    return root / "benchmarks" / "etom" / "etom_source_pin_v1.json"


def lock_path(root: Path) -> Path:
    return root / "benchmarks" / "etom" / "cost_pilot_10_v1.lock.json"


def formal_100_lock_path(root: Path) -> Path:
    return root / "benchmarks" / "etom" / "formal_100_v1.lock.json"


def raw_root(root: Path) -> Path:
    return root / "data" / "etom" / "raw"


def load_pin(root: Path) -> Dict[str, Any]:
    path = pin_path(root)
    if not path.exists():
        raise EtomSourceError(f"Missing ETOM pin file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head(checkout: Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(checkout),
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise EtomSourceError(
            f"ETOM checkout at {checkout} is not a git repository."
        ) from exc
    return out.strip()


def verify_source(root: Path) -> Dict[str, Any]:
    """Confirm the raw checkout matches the pinned commit and file hashes.

    Official query, graph, and registry bytes are only read here.
    """
    pin = load_pin(root)
    checkout = raw_root(root)
    registry = checkout / "data" / "mcp_registry.json"
    if not registry.exists():
        raise EtomSourceError(
            "Pinned ETOM checkout is missing. From the worktree root run:\n"
            "  bash scripts/fetch_etom_source.sh\n"
            f"Expected commit {pin['commit_sha']} under {checkout}"
        )
    head = _git_head(checkout)
    expected = pin["commit_sha"]
    if head != expected:
        raise EtomSourceError(
            f"ETOM checkout HEAD {head} != pinned commit {expected}. "
            "Refusing to evaluate a different snapshot."
        )
    hashes: Dict[str, str] = {}
    for rel, meta in pin["dataset_files"].items():
        path = checkout / rel
        if not path.exists():
            raise EtomSourceError(f"Missing pinned ETOM file: {rel}")
        digest = file_sha256(path)
        if digest != meta["sha256"]:
            raise EtomSourceError(
                f"ETOM file hash mismatch for {rel}. "
                "Official data must stay unmodified."
            )
        if path.stat().st_size != int(meta["bytes"]):
            raise EtomSourceError(f"ETOM file size mismatch for {rel}.")
        hashes[rel] = digest
    return {"pin": pin, "checkout": str(checkout), "sha256": hashes}
