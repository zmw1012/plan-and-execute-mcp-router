"""Frozen ETOM-100 dataset loader helpers (lock-authoritative; no resample)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

from hk_equity_mcp.pe_fin_replan_etom.identity import (
    FROZEN_CATALOG_TOTAL_TOOLS,
    FROZEN_LOCK_FILE_SHA256,
    FROZEN_ORDERED_TASK_IDS_SHA256,
    FROZEN_REGISTRY_SHA256,
    FROZEN_SAMPLE_SHA256,
    assert_frozen_identity,
    frozen_dataset_identity,
)


def assert_frozen_etom100_identity(root: Path) -> Dict[str, Any]:
    lock = assert_frozen_identity(root)
    frozen = frozen_dataset_identity()
    return {
        "lock_sha256": FROZEN_LOCK_FILE_SHA256,
        "sample_sha256": FROZEN_SAMPLE_SHA256,
        "ordered_task_ids_sha256": FROZEN_ORDERED_TASK_IDS_SHA256,
        "ordered_task_ids": list(lock.get("ordered_task_ids") or []),
        "task_count": len(lock.get("ordered_task_ids") or []),
        "server_count": 491,
        "tool_count": FROZEN_CATALOG_TOTAL_TOOLS,
        "source_commit": frozen["source_commit"],
        "source_pin_sha256": frozen["source_pin_sha256"],
        "registry_sha256": FROZEN_REGISTRY_SHA256,
        "frozen": frozen,
        "lock": lock,
    }


def load_lock_rows(root: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    identity = assert_frozen_etom100_identity(root)
    lock = identity["lock"]
    return identity, list(lock.get("tasks") or [])
