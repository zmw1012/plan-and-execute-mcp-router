"""Unified dataset manifest loader (pointers only; never mutates frozen gold)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

DEFAULT_MANIFEST = "benchmarks/manifests/hk_unified_dataset_v1.json"
DEFAULT_SMOKE = "benchmarks/smoke/hk_unified_dev_smoke_8_v1.json"
VALID_SPLITS = ("development", "heldout", "exploratory_all")


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    return rows


def normalize_task(row: Dict[str, Any], *, domain: str, split: str) -> Dict[str, Any]:
    """Attach unified fields without modifying source files on disk."""
    out = dict(row)
    task_id = str(out.get("id") or out.get("question_id") or "")
    if not task_id:
        raise ValueError(f"Missing task id in domain={domain}")
    out["task_id"] = task_id
    out["id"] = task_id
    out["domain"] = out.get("domain") or domain
    out["dataset_split"] = out.get("dataset_split") or split
    out["domain_combination"] = (
        out.get("domain_combination")
        or out.get("domain_combo")
        or (
            "+".join(out["domains"])
            if isinstance(out.get("domains"), list)
            else out.get("domain")
        )
    )
    out["project_difficulty"] = out.get("project_difficulty") or out.get(
        "project_difficulty_level"
    )
    if "question" not in out:
        raise ValueError(f"Task {task_id} missing question")
    return out


def load_manifest(root: Path, path: Optional[Path] = None) -> Dict[str, Any]:
    man_path = Path(path) if path else (root / DEFAULT_MANIFEST)
    if not man_path.is_absolute():
        man_path = (root / man_path).resolve()
    return load_json(man_path)


def load_smoke_manifest(root: Path, path: Optional[Path] = None) -> Dict[str, Any]:
    smoke_path = Path(path) if path else (root / DEFAULT_SMOKE)
    if not smoke_path.is_absolute():
        smoke_path = (root / smoke_path).resolve()
    return load_json(smoke_path)


def load_split_tasks(
    root: Path,
    split: str,
    *,
    manifest: Optional[Dict[str, Any]] = None,
    manifest_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    if split not in VALID_SPLITS:
        raise ValueError(f"Unknown split {split!r}; expected {VALID_SPLITS}")
    if split == "exploratory_all":
        from hk_equity_mcp.experiment.unified_exploratory import (
            load_exploratory_tasks_in_locked_order,
        )

        return load_exploratory_tasks_in_locked_order(root)
    man = manifest or load_manifest(root, manifest_path)
    split_cfg = man["splits"][split]
    tasks: List[Dict[str, Any]] = []
    for source in split_cfg["sources"]:
        rel = source["path"]
        path = (root / rel).resolve()
        domain = str(source["domain"])
        rows = load_jsonl(path)
        expected = int(source["expected_count"])
        if len(rows) != expected:
            raise ValueError(
                f"Split {split} domain={domain}: expected {expected} rows in {rel}, "
                f"got {len(rows)}"
            )
        for row in rows:
            tasks.append(normalize_task(row, domain=domain, split=split))
    expected_total = int(split_cfg["expected_count"])
    if len(tasks) != expected_total:
        raise ValueError(
            f"Split {split}: expected {expected_total} tasks, got {len(tasks)}"
        )
    # Stable order: source order already deterministic.
    ids = [t["task_id"] for t in tasks]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate task_id across unified {split} split")
    return tasks


def select_by_ids(
    tasks: Sequence[Dict[str, Any]],
    ids: Sequence[str],
    *,
    preserve_id_order: bool = False,
) -> List[Dict[str, Any]]:
    wanted = list(ids)
    by_id = {t["task_id"]: t for t in tasks}
    missing = [qid for qid in wanted if qid not in by_id]
    if missing:
        raise ValueError(f"Unknown task id(s): {missing}")
    if preserve_id_order:
        return [by_id[qid] for qid in wanted]
    wanted_set = set(wanted)
    return [t for t in tasks if t["task_id"] in wanted_set]


def load_smoke_tasks(
    root: Path,
    *,
    development_tasks: Optional[List[Dict[str, Any]]] = None,
    smoke_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    smoke = load_smoke_manifest(root, smoke_path)
    if smoke.get("dataset_split") != "development":
        raise ValueError("Smoke subset must belong to development split")
    ids = [t["task_id"] for t in smoke["tasks"]]
    if len(ids) != int(smoke.get("expected_count", 8)):
        raise ValueError("Smoke manifest task count mismatch")
    dev = development_tasks or load_split_tasks(root, "development")
    return select_by_ids(dev, ids, preserve_id_order=True)


def assert_heldout_unlocked(*, split: str, confirm_heldout_final: bool) -> Dict[str, Any]:
    """Held-out lock: require --split heldout AND --confirm-heldout-final."""
    status = {
        "split": split,
        "heldout_lock_active": True,
        "confirm_heldout_final": bool(confirm_heldout_final),
        "allowed": False,
        "reason": "",
    }
    if split != "heldout":
        status["heldout_lock_active"] = False
        status["allowed"] = True
        status["reason"] = "development_or_non_heldout"
        return status
    if not confirm_heldout_final:
        status["reason"] = (
            "Held-out is locked. Pass --split heldout and "
            "--confirm-heldout-final together."
        )
        raise RuntimeError(status["reason"])
    status["allowed"] = True
    status["reason"] = "explicit_heldout_confirmation"
    return status
