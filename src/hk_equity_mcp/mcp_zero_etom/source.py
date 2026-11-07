"""ETOM source resolution for MCP-Zero (pinned checkout or offline fixture)."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from hk_equity_mcp.etom.convert import convert_registry, registry_fingerprint
from hk_equity_mcp.etom.formal_sampler import (
    load_formal_100_lock,
    ordered_task_ids_sha256,
    sample_document_sha256,
    select_formal_locked_tasks,
)
from hk_equity_mcp.etom.loader import EtomTask, task_from_official
from hk_equity_mcp.etom.sampler import load_lock, select_locked_tasks
from hk_equity_mcp.etom.source import (
    EtomSourceError,
    _git_head,
    file_sha256,
    formal_100_lock_path,
    load_pin,
    lock_path,
    pin_path,
    verify_source,
)
from hk_equity_mcp.mcp_zero_etom import (
    ETOM_AUDIT_SOURCE_COMMIT,
    ETOM_FREEZE_COMMIT,
    ETOM_SOURCE_COMMIT,
    EXPECTED_FORMAL_SAMPLE_SHA256,
    EXPECTED_ORDERED_IDS_SHA256,
)


@dataclass(frozen=True)
class EtomSourceConfig:
    root: Path
    raw_dir: Path
    pin_path: Path
    pilot_lock_path: Path
    formal_lock_path: Path
    source_kind: str  # pinned | fixture


def _refuse_symlink(path: Path, *, label: str) -> Path:
    resolved = Path(path)
    if resolved.is_symlink():
        raise EtomSourceError(
            f"Refusing symlink {label} at {resolved}. "
            "Use a real checkout or --etom-source-dir (no cross-worktree symlinks)."
        )
    return resolved


def resolve_source_config(
    root: Path,
    *,
    etom_source_dir: Optional[str] = None,
    fixture_dir: Optional[str] = None,
    mode: str = "mock",
) -> EtomSourceConfig:
    root = Path(root)
    if fixture_dir:
        if mode not in {"mock", "dry-run", "preflight", "index-status"}:
            raise EtomSourceError(
                "Fixture source is only allowed for mock/dry-run/preflight/index-status. "
                "Live and build-index require a pinned ETOM checkout via --etom-source-dir."
            )
        raw = _refuse_symlink(Path(fixture_dir), label="fixture_dir")
        if not raw.exists():
            raise EtomSourceError(f"Fixture dir missing: {raw}")
        return EtomSourceConfig(
            root=root,
            raw_dir=raw,
            pin_path=pin_path(root),
            pilot_lock_path=lock_path(root),
            formal_lock_path=formal_100_lock_path(root),
            source_kind="fixture",
        )
    env_dir = etom_source_dir or os.environ.get("ETOM_SOURCE_DIR")
    if env_dir:
        raw = _refuse_symlink(Path(env_dir), label="etom_source_dir")
    else:
        raw = _refuse_symlink(root / "data" / "etom" / "raw", label="data/etom/raw")
    return EtomSourceConfig(
        root=root,
        raw_dir=raw,
        pin_path=pin_path(root),
        pilot_lock_path=lock_path(root),
        formal_lock_path=formal_100_lock_path(root),
        source_kind="pinned",
    )


def verify_source_config(cfg: EtomSourceConfig) -> Dict[str, Any]:
    if cfg.source_kind == "fixture":
        registry = cfg.raw_dir / "data" / "mcp_registry.json"
        if not registry.exists():
            raise EtomSourceError(f"Fixture registry missing: {registry}")
        return {
            "source_kind": "fixture",
            "checkout": str(cfg.raw_dir),
            "registry_sha256": file_sha256(registry),
        }
    # Reuse audited verify_source but allow alternate raw_dir via temporary cwd layout.
    # Official verify_source hardcodes data/etom/raw under root. If cfg.raw_dir differs,
    # verify file hashes against the pin without requiring that exact relative path.
    pin = load_pin(cfg.root)
    if pin["commit_sha"] != ETOM_SOURCE_COMMIT:
        raise EtomSourceError(
            f"Source pin commit {pin['commit_sha']} != expected {ETOM_SOURCE_COMMIT}"
        )
    if cfg.raw_dir == (cfg.root / "data" / "etom" / "raw"):
        return {"source_kind": "pinned", **verify_source(cfg.root)}
    checkout = cfg.raw_dir
    registry = checkout / "data" / "mcp_registry.json"
    if not registry.exists():
        raise EtomSourceError(
            f"Pinned ETOM checkout missing at {checkout}. "
            "Run bash scripts/fetch_etom_source.sh or pass --etom-source-dir."
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
            raise EtomSourceError(f"ETOM file hash mismatch for {rel}.")
        if path.stat().st_size != int(meta["bytes"]):
            raise EtomSourceError(f"ETOM file size mismatch for {rel}.")
        hashes[rel] = digest
    return {
        "source_kind": "pinned",
        "pin": pin,
        "checkout": str(checkout),
        "source_commit": head,
        "sha256": hashes,
        "note": "verified via --etom-source-dir override (no symlink)",
    }


def load_registry_payload(
    cfg: EtomSourceConfig, *, expect_tool_count: Optional[int] = None
) -> Dict[str, Any]:
    path = cfg.raw_dir / "data" / "mcp_registry.json"
    if not path.exists():
        raise EtomSourceError(f"Missing registry: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    specs, official = convert_registry(payload)
    if expect_tool_count is not None and len(official) != expect_tool_count:
        raise EtomSourceError(
            f"Registry tool count {len(official)} != expected {expect_tool_count}"
        )
    if cfg.source_kind == "pinned" and len(official) != 2375:
        raise EtomSourceError(
            f"Pinned registry must have 2375 tools; got {len(official)}"
        )
    return {
        "payload": payload,
        "specs": specs,
        "official": official,
        "registry_sha256": file_sha256(path),
        "registry_fingerprint": registry_fingerprint(official),
        "tool_count": len(official),
        "server_count": len({v["server_name"] for v in official.values()}),
    }


def load_tasks_from_queries(cfg: EtomSourceConfig) -> List[EtomTask]:
    tasks: List[EtomTask] = []
    for level in range(1, 6):
        rel = f"data/queries/level_{level}.json"
        path = cfg.raw_dir / rel
        if not path.exists():
            raise EtomSourceError(f"Missing query file: {path}")
        rows = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise EtomSourceError(f"Query file must be a list: {rel}")
        for row in rows:
            tasks.append(task_from_official(level, row, rel))
    return tasks


def assert_formal_lock_identity(root: Path) -> Dict[str, Any]:
    payload = load_formal_100_lock(root)
    if payload.get("sample_sha256") != EXPECTED_FORMAL_SAMPLE_SHA256:
        raise EtomSourceError("formal sample_sha256 mismatch; fail closed")
    if payload.get("ordered_task_ids_sha256") != EXPECTED_ORDERED_IDS_SHA256:
        raise EtomSourceError("ordered_task_ids_sha256 mismatch; fail closed")
    recomputed_ids = ordered_task_ids_sha256(payload["ordered_task_ids"])
    if recomputed_ids != EXPECTED_ORDERED_IDS_SHA256:
        raise EtomSourceError("ordered_task_ids hash does not recompute")
    recomputed_sample = sample_document_sha256(payload)
    if recomputed_sample != EXPECTED_FORMAL_SAMPLE_SHA256:
        raise EtomSourceError("sample_sha256 does not recompute")
    if payload.get("seed") != 20261106:
        raise EtomSourceError("formal seed mismatch")
    if payload.get("source_commit") != ETOM_SOURCE_COMMIT:
        raise EtomSourceError("formal lock source_commit mismatch")
    quotas = payload.get("quotas") or {}
    if quotas != {"L1": 20, "L2": 20, "L3": 20, "L4": 20, "L5": 20}:
        raise EtomSourceError(f"formal quotas mismatch: {quotas}")
    if len(payload.get("ordered_task_ids") or []) != 100:
        raise EtomSourceError("formal ordered_task_ids must be length 100")
    from collections import Counter

    level_counts = Counter(tid.split("-")[0] for tid in payload["ordered_task_ids"])
    if dict(level_counts) != {"L1": 20, "L2": 20, "L3": 20, "L4": 20, "L5": 20}:
        raise EtomSourceError(f"formal level counts mismatch: {dict(level_counts)}")
    pilot = load_lock(root)
    pilot_ids = {t["task_id"] for t in pilot.get("tasks") or []}
    overlap = sorted(pilot_ids.intersection(payload["ordered_task_ids"]))
    if overlap:
        raise EtomSourceError(f"pilot/formal overlap must be 0; got {overlap}")
    return {
        "sample_sha256": payload["sample_sha256"],
        "ordered_task_ids_sha256": payload["ordered_task_ids_sha256"],
        "seed": payload["seed"],
        "pilot_overlap": overlap,
        "quotas": quotas,
        "level_counts": dict(level_counts),
    }


def select_tasks(
    cfg: EtomSourceConfig,
    tasks: Sequence[EtomTask],
    *,
    split: str,
    max_tasks: Optional[int] = None,
    ids: Optional[Sequence[str]] = None,
) -> Tuple[List[EtomTask], Dict[str, Any]]:
    if split == "pilot10":
        payload = load_lock(cfg.root)
        selected = select_locked_tasks(list(tasks), payload)
        if max_tasks is not None:
            if max_tasks < 1 or max_tasks > len(selected):
                raise EtomSourceError(
                    f"max_tasks must be in 1..{len(selected)} for pilot lock"
                )
            selected = selected[:max_tasks]
    elif split == "formal100":
        payload = load_formal_100_lock(cfg.root)
        selected = select_formal_locked_tasks(
            list(tasks), payload, max_tasks=max_tasks
        )
    else:
        raise EtomSourceError(f"Unknown split: {split}")
    if ids:
        wanted = list(ids)
        by_id = {t.task_id: t for t in selected}
        missing = [i for i in wanted if i not in by_id]
        if missing and cfg.source_kind != "fixture":
            # Debug-only: allow filtering to subset of locked ids.
            by_all = {t.task_id: t for t in tasks}
            missing = [i for i in wanted if i not in by_all]
            if missing:
                raise EtomSourceError(f"--ids not found: {missing}")
            selected = [by_all[i] for i in wanted]
        else:
            selected = [by_id[i] for i in wanted if i in by_id]
        payload = dict(payload)
        payload["debug_ids_filter"] = list(ids)
    return selected, payload


def build_source_manifest(
    cfg: EtomSourceConfig,
    *,
    verified: Dict[str, Any],
    lock: Dict[str, Any],
    registry_meta: Dict[str, Any],
    catalog_fingerprint: str,
    split: str,
) -> Dict[str, Any]:
    pin = load_pin(cfg.root) if cfg.pin_path.exists() else {}
    import_manifest_path = (
        cfg.root / "benchmarks" / "etom" / "import_manifest_mcp_zero_v1.json"
    )
    import_manifest = None
    if import_manifest_path.exists():
        import_manifest = json.loads(import_manifest_path.read_text(encoding="utf-8"))
    pilot = load_lock(cfg.root)
    formal = load_formal_100_lock(cfg.root)
    pilot_ids = [t["task_id"] for t in pilot.get("tasks") or []]
    overlap = sorted(set(pilot_ids).intersection(formal.get("ordered_task_ids") or []))
    return {
        "source_kind": cfg.source_kind,
        "source_repository": pin.get("repository_url")
        or lock.get("source_repository"),
        "source_commit": pin.get("commit_sha") or ETOM_SOURCE_COMMIT,
        "etom_audit_source_commit": ETOM_AUDIT_SOURCE_COMMIT,
        "etom_freeze_commit": ETOM_FREEZE_COMMIT,
        "source_pin_path": str(cfg.pin_path.relative_to(cfg.root))
        if cfg.pin_path.exists()
        else None,
        "source_pin_sha256": file_sha256(cfg.pin_path)
        if cfg.pin_path.exists()
        else None,
        "raw_dir_label": "fixture"
        if cfg.source_kind == "fixture"
        else "data/etom/raw_or_ETOM_SOURCE_DIR",
        "dataset_file_sha256": (verified.get("sha256") or {}),
        "registry_sha256": registry_meta["registry_sha256"],
        "registry_fingerprint": registry_meta["registry_fingerprint"],
        "registry_tool_count": registry_meta["tool_count"],
        "registry_server_count": registry_meta["server_count"],
        "official_counts": pin.get("official_counts"),
        "split": split,
        "sample_lock_path": str(
            (cfg.formal_lock_path if split == "formal100" else cfg.pilot_lock_path)
            .relative_to(cfg.root)
        ),
        "sample_lock_sha256": file_sha256(
            cfg.formal_lock_path if split == "formal100" else cfg.pilot_lock_path
        ),
        "sample_sha256": lock.get("sample_sha256"),
        "ordered_task_ids_sha256": lock.get("ordered_task_ids_sha256")
        or ordered_task_ids_sha256(lock.get("ordered_task_ids") or []),
        "sample_size": lock.get("sample_size") or len(lock.get("ordered_task_ids") or []),
        "seed": lock.get("seed"),
        "sampler_version": lock.get("sampler_version") or lock.get("sampler"),
        "quotas": lock.get("quotas"),
        "excluded_task_ids": lock.get("excluded_task_ids")
        or [t["task_id"] for t in pilot.get("tasks") or []],
        "excluded_cost_pilot_lock_sha256": file_sha256(cfg.pilot_lock_path),
        "pilot_overlap": overlap,
        "import_manifest_sha256": file_sha256(import_manifest_path)
        if import_manifest_path.exists()
        else None,
        "import_manifest_identity": (import_manifest or {}).get("identity"),
        "catalog_fingerprint": catalog_fingerprint,
        "gold_access_policy": "post_hoc_scoring_and_retrieval_analysis_only",
        "real_mcp_tool_executed": False,
        "verified": {
            k: verified[k]
            for k in ("source_kind", "checkout", "note")
            if k in verified
        },
    }


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
