"""Static ETOM capability profile: load / validate / hash / planner view.

The profile is a short, task-agnostic summary of the ETOM catalog capability
kinds. It must NOT contain task ids, gold operations, accepted tool sets, or
answers. This module deliberately does not import the HK-155 profile loader.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml

DEFAULT_PROFILE_REL = "configs/capabilities/etom100_capability_profile_v1.yaml"
EXPECTED_PROFILE_ID = "etom100_capability_profile_v1"

_BANNED_PROFILE_PATTERNS = (
    r"\bgold_operations?\b",
    r"\bgold_tools?\b",
    r"\bground_truth\w*\b",
    r"\baccepted_(tools?|operations?)\b",
    r"\breference_answer\b",
    r"\btask_id\b",
    r"\bL[1-5]-q\d+\b",
    r"\boperation_id\b",
    r"\bselected_order\b",
    r"::",  # server::tool operation ids
)

_REQUIRED_KEYS = (
    "profile_id",
    "version",
    "capability_summary",
    "io_summaries",
    "flags",
    "known_limitations",
)

_BANNED_KEYS = (
    "task_whitelist",
    "task_ids",
    "gold",
    "gold_operations",
    "accepted_tools",
    "answers",
    "ordered_task_ids",
)


class CapabilityProfileError(ValueError):
    pass


def profile_path(root: Path, rel: Optional[str] = None) -> Path:
    return Path(root) / (rel or DEFAULT_PROFILE_REL)


def validate_capability_profile(
    profile: Dict[str, Any],
    *,
    forbidden_literals: Optional[Iterable[str]] = None,
) -> None:
    for key in _REQUIRED_KEYS:
        if key not in profile:
            raise CapabilityProfileError(f"missing required profile field: {key}")
    if profile.get("profile_id") != EXPECTED_PROFILE_ID:
        raise CapabilityProfileError(
            f"profile_id must be {EXPECTED_PROFILE_ID!r}, got {profile.get('profile_id')!r}"
        )
    if "hk" in str(profile.get("profile_id")).lower():
        raise CapabilityProfileError("HK capability profile must not be reused")
    for banned_key in _BANNED_KEYS:
        if banned_key in profile:
            raise CapabilityProfileError(f"banned profile key: {banned_key}")
    blob = json.dumps(profile, ensure_ascii=False)
    for pat in _BANNED_PROFILE_PATTERNS:
        if re.search(pat, blob, flags=re.IGNORECASE):
            raise CapabilityProfileError(f"capability profile contains banned pattern: {pat}")
    for literal in forbidden_literals or ():
        if literal and str(literal) in blob:
            raise CapabilityProfileError(
                f"capability profile leaks forbidden literal: {str(literal)[:60]!r}"
            )


def load_capability_profile(
    root: Path,
    *,
    path: Optional[Path] = None,
    forbidden_literals: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    cfg_path = Path(path) if path is not None else profile_path(root)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise CapabilityProfileError("capability profile must be a mapping")
    validate_capability_profile(raw, forbidden_literals=forbidden_literals)
    return raw


def capability_profile_hash(profile: Dict[str, Any]) -> str:
    blob = json.dumps(profile, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def forbidden_literals_from_lock(lock: Dict[str, Any]) -> List[str]:
    """Task ids and gold operation ids from the lock; used for leakage audits only."""
    out: List[str] = []
    out.extend(str(t) for t in lock.get("ordered_task_ids") or [])
    for row in lock.get("tasks") or []:
        out.extend(str(o) for o in row.get("operation_ids") or [])
    return sorted(set(out))


def profile_for_planner(profile: Dict[str, Any]) -> Dict[str, Any]:
    """Compact static view injected into the planner prompt."""
    validate_capability_profile(profile)
    return {
        "profile_id": profile.get("profile_id"),
        "version": profile.get("version"),
        "capability_areas": list(((profile.get("capability_areas") or {}).get("areas")) or []),
        "capability_summary": dict(profile.get("capability_summary") or {}),
        "server_categories": list(profile.get("server_categories") or []),
        "io_summaries": dict(profile.get("io_summaries") or {}),
        "flags": dict(profile.get("flags") or {}),
        "known_limitations": list(profile.get("known_limitations") or []),
    }
