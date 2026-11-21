"""Deterministic, reversible action_name ↔ operation_id mapping for Flat All Tools.

Provider-safe action names avoid dots and stay within a conservative length.
Original operation_id remains the gold/evaluator identity; action_name is only
an internal model-facing alias.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

# Conservative OpenAI/DeepSeek-compatible function-name pattern.
SAFE_ACTION_NAME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9_]{0,63}$")
MAX_ACTION_NAME_LEN = 64


def _sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def operation_id_needs_alias(operation_id: str) -> bool:
    """True when the raw operation_id is not a safe action name."""
    return not bool(SAFE_ACTION_NAME_RE.match(operation_id or ""))


def _sanitize_base(operation_id: str) -> str:
    # Replace non [A-Za-z0-9_] with underscore; collapse repeats.
    cleaned = re.sub(r"[^a-zA-Z0-9_]", "_", operation_id)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    if not cleaned:
        cleaned = "op"
    if cleaned[0].isdigit():
        cleaned = f"op_{cleaned}"
    return cleaned


def _unique_name(candidate: str, used: set[str], operation_id: str) -> str:
    if SAFE_ACTION_NAME_RE.match(candidate) and candidate not in used:
        return candidate
    digest = _sha256_hex(operation_id)[:8]
    # Keep room for _ + 8-char digest.
    stem = candidate[: MAX_ACTION_NAME_LEN - 9].rstrip("_") or "op"
    base = f"{stem}_{digest}"
    if base not in used and SAFE_ACTION_NAME_RE.match(base):
        return base
    n = 2
    while True:
        suffix = f"_{digest}_{n}"
        stem2 = candidate[: MAX_ACTION_NAME_LEN - len(suffix)].rstrip("_") or "op"
        name = f"{stem2}{suffix}"
        if name not in used and SAFE_ACTION_NAME_RE.match(name):
            return name
        n += 1


@dataclass(frozen=True)
class FlatActionMapping:
    """Frozen bidirectional mapping for one catalog snapshot."""

    action_to_operation: Dict[str, str]
    operation_to_action: Dict[str, str]
    fingerprint: str

    def action_name(self, operation_id: str) -> str:
        try:
            return self.operation_to_action[operation_id]
        except KeyError as exc:
            raise KeyError(f"Unknown operation_id in Flat mapping: {operation_id}") from exc

    def operation_id(self, action_name: str) -> str:
        try:
            return self.action_to_operation[action_name]
        except KeyError as exc:
            raise KeyError(f"Unknown Flat action_name: {action_name}") from exc

    def resolve(self, name: Optional[str]) -> Optional[str]:
        """Resolve model-supplied name to operation_id (accepts either form)."""
        if not name or not isinstance(name, str):
            return None
        if name in self.action_to_operation:
            return self.action_to_operation[name]
        if name in self.operation_to_action:
            return name
        return None

    def to_dict(self) -> Dict[str, object]:
        return {
            "fingerprint": self.fingerprint,
            "action_to_operation": dict(self.action_to_operation),
            "operation_to_action": dict(self.operation_to_action),
            "n_operations": len(self.operation_to_action),
        }


def build_flat_action_mapping(
    operation_ids: Sequence[str],
) -> FlatActionMapping:
    """Build a deterministic, collision-free, reversible mapping.

    Mapping is sorted by operation_id so the fingerprint is stable.
    """
    ids = sorted({str(oid) for oid in operation_ids if oid})
    action_to_operation: Dict[str, str] = {}
    operation_to_action: Dict[str, str] = {}
    used: set[str] = set()

    for oid in ids:
        if oid in operation_to_action:
            continue
        base = oid if SAFE_ACTION_NAME_RE.match(oid) else _sanitize_base(oid)
        if len(base) > MAX_ACTION_NAME_LEN or not SAFE_ACTION_NAME_RE.match(base):
            digest = _sha256_hex(oid)[:8]
            stem = _sanitize_base(oid)[: MAX_ACTION_NAME_LEN - 9].rstrip("_") or "op"
            base = f"{stem}_{digest}"
        name = _unique_name(base, used, oid)
        used.add(name)
        action_to_operation[name] = oid
        operation_to_action[oid] = name

    # Collision / invertibility checks (fail closed).
    if len(action_to_operation) != len(ids):
        raise RuntimeError("Flat action mapping size mismatch after construction")
    if len(set(action_to_operation.values())) != len(action_to_operation):
        raise RuntimeError("Flat action mapping is not injective (collision)")
    for action, oid in action_to_operation.items():
        if operation_to_action.get(oid) != action:
            raise RuntimeError(f"Flat action mapping not reversible for {oid}")
        if not SAFE_ACTION_NAME_RE.match(action):
            raise RuntimeError(f"Unsafe Flat action_name produced: {action}")

    canonical = {
        "operation_to_action": operation_to_action,
        "action_to_operation": action_to_operation,
    }
    fingerprint = _sha256_hex(
        json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )
    return FlatActionMapping(
        action_to_operation=action_to_operation,
        operation_to_action=operation_to_action,
        fingerprint=fingerprint,
    )


def assert_mapping_covers(
    mapping: FlatActionMapping, operation_ids: Iterable[str]
) -> None:
    missing = sorted(set(operation_ids) - set(mapping.operation_to_action))
    if missing:
        raise RuntimeError(
            "Flat action mapping missing operation_ids: " + ", ".join(missing[:10])
        )


def mapping_from_dict(payload: Mapping[str, object]) -> FlatActionMapping:
    a2o = {str(k): str(v) for k, v in dict(payload["action_to_operation"]).items()}
    o2a = {str(k): str(v) for k, v in dict(payload["operation_to_action"]).items()}
    fp = str(payload.get("fingerprint") or "")
    rebuilt = FlatActionMapping(
        action_to_operation=a2o, operation_to_action=o2a, fingerprint=fp
    )
    # Verify fingerprint if present.
    canonical = {
        "operation_to_action": o2a,
        "action_to_operation": a2o,
    }
    expect = _sha256_hex(
        json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )
    if fp and fp != expect:
        raise RuntimeError(
            f"Flat action mapping fingerprint mismatch: stored={fp} recomputed={expect}"
        )
    return rebuilt
