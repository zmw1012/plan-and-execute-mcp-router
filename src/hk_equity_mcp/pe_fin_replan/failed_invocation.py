"""Cross-round failed invocation signatures for PE-Fin RePlan.

signature = operation_id + canonicalized_arguments + failure_class

Block re-execution of the same operation+arguments after blockable upstream
failures (rate limit / transport / unavailable / protocol). Changing step_id
does not bypass the check. Does not invent fallbacks.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

FAILED_INVOCATION_POLICY_VERSION = "pe_fin_replan_failed_invocation_v1"

BLOCKABLE_FAILURE_CLASSES = (
    "upstream_rate_limit",
    "transport_disconnect",
    "upstream_unavailable",
    "protocol_failure",
    # Deterministic no-result / empty-result signatures (cross-round block).
    "no_matching_result",
    "empty_result",
    "verified_zero_rows",
)

# Map legacy EEA classes into blockable policy classes when appropriate.
_EEA_TO_BLOCKABLE = {
    "mcp_transport_or_protocol_failure": "protocol_failure",
    "gateway_envelope_failure": "protocol_failure",
    "no_matching_result": "no_matching_result",
    "empty_result": "empty_result",
    "verified_zero_rows": "verified_zero_rows",
}

_RATE_MARKERS = (
    "rate limit",
    "rate_limit",
    "ratelimit",
    "too many requests",
    "429",
    "限流",
    "频率",
    "访问频率",
    "ip",
)
_TRANSPORT_MARKERS = (
    "timeout",
    "timed out",
    "connection reset",
    "connection aborted",
    "broken pipe",
    "urlerror",
    "disconnect",
    "temporarily unavailable",
)
_UNAVAILABLE_MARKERS = (
    "service unavailable",
    "upstream unavailable",
    "503",
    "502",
    "bad gateway",
    "unavailable",
)
_PROTOCOL_MARKERS = (
    "protocol",
    "envelope",
    "http 500",
    "internal server",
    "gateway",
)


def failed_invocation_policy_hash() -> str:
    blob = json.dumps(
        {
            "version": FAILED_INVOCATION_POLICY_VERSION,
            "blockable": list(BLOCKABLE_FAILURE_CLASSES),
            "rules": [
                "signature_op_args_failure_class",
                "block_same_op_args_after_blockable_failure",
                "block_no_matching_empty_verified_zero",
                "step_id_cannot_bypass",
                "different_operation_allowed",
                "semantically_different_args_allowed",
                "broadened_args_allowed",
                "blocked_invoke_does_not_count_mcp",
                "no_task_or_gold_branching",
            ],
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def canonicalize_arguments(arguments: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    args = arguments or {}
    out: Dict[str, Any] = {}
    for k in sorted(args):
        v = args[k]
        if v in (None, ""):
            continue
        if isinstance(v, dict):
            out[str(k)] = canonicalize_arguments(v)
        elif isinstance(v, list):
            out[str(k)] = [canonicalize_arguments(x) if isinstance(x, dict) else x for x in v]
        else:
            out[str(k)] = v
    return out


def invocation_key(operation_id: str, arguments: Optional[Dict[str, Any]]) -> str:
    blob = json.dumps(
        {
            "operation_id": str(operation_id),
            "arguments": canonicalize_arguments(arguments),
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def failed_signature(
    operation_id: str,
    arguments: Optional[Dict[str, Any]],
    failure_class: str,
) -> str:
    blob = json.dumps(
        {
            "operation_id": str(operation_id),
            "arguments": canonicalize_arguments(arguments),
            "failure_class": str(failure_class),
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def classify_blockable_failure(
    *,
    error: Optional[str] = None,
    error_type: Optional[str] = None,
    eea_failure_class: Optional[str] = None,
    success: Optional[bool] = None,
) -> Optional[str]:
    """Return a blockable failure class, or None if not blockable."""
    eea = str(eea_failure_class or "").strip()
    if eea in BLOCKABLE_FAILURE_CLASSES:
        return eea

    text = f"{error or ''} {error_type or ''}".lower()
    raw = str(error or "")
    if any(m in text for m in _RATE_MARKERS) and (
        "限流" in raw
        or "rate" in text
        or "429" in text
        or "频率" in raw
        or ("ip" in text and ("limit" in text or "限" in raw))
    ):
        return "upstream_rate_limit"
    # Broader IP+limit Chinese patterns
    if "限流" in raw or re.search(r"IP.{0,12}限", raw) or "访问频率" in raw:
        return "upstream_rate_limit"

    if any(m in text for m in _TRANSPORT_MARKERS):
        return "transport_disconnect"
    if any(m in text for m in _UNAVAILABLE_MARKERS):
        return "upstream_unavailable"
    if any(m in text for m in _PROTOCOL_MARKERS) and success is False:
        return "protocol_failure"

    # Textual no-result markers (when EEA class not already provided).
    if (
        "no_matching" in text
        or "no matching" in text
        or "no match" in text
        or "not found" in text
    ):
        return "no_matching_result"
    if "verified_zero_rows" in text or "zero rows" in text or "zero_rows" in text:
        return "verified_zero_rows"
    if "empty_result" in text or "empty result" in text:
        return "empty_result"

    mapped = _EEA_TO_BLOCKABLE.get(eea)
    if mapped:
        # Prefer more specific rate-limit if markers present
        if "429" in text or "rate" in text or "限流" in raw:
            return "upstream_rate_limit"
        return mapped
    return None


@dataclass
class FailedInvocationGuard:
    """Per-task registry of failed invocation signatures."""

    signatures: Set[str] = field(default_factory=set)
    blocked_keys: Set[str] = field(default_factory=set)
    blocked_reason_by_key: Dict[str, str] = field(default_factory=dict)
    signature_blocked_count: int = 0
    block_events: List[Dict[str, Any]] = field(default_factory=list)

    def is_blocked(
        self, operation_id: str, arguments: Optional[Dict[str, Any]]
    ) -> Tuple[bool, Optional[str]]:
        key = invocation_key(operation_id, arguments)
        if key in self.blocked_keys:
            return True, self.blocked_reason_by_key.get(key) or "signature_blocked"
        return False, None

    def record_failure(
        self,
        *,
        operation_id: str,
        arguments: Optional[Dict[str, Any]],
        failure_class: str,
        error: Optional[str] = None,
    ) -> None:
        if failure_class not in BLOCKABLE_FAILURE_CLASSES:
            return
        key = invocation_key(operation_id, arguments)
        sig = failed_signature(operation_id, arguments, failure_class)
        self.signatures.add(sig)
        self.blocked_keys.add(key)
        self.blocked_reason_by_key[key] = failure_class
        self.block_events.append(
            {
                "event": "recorded",
                "operation_id": operation_id,
                "failure_class": failure_class,
                "error": (error or "")[:240],
                "invocation_key": key,
                "signature": sig,
            }
        )

    def observe_invoke_result(
        self,
        *,
        operation_id: str,
        arguments: Optional[Dict[str, Any]],
        invoke_result: Dict[str, Any],
        eea_failure_class: Optional[str] = None,
    ) -> Optional[str]:
        success = invoke_result.get("success")
        eea = str(eea_failure_class or "").strip() or None
        # Successful payloads are not blockable unless an explicit no-result class
        # is supplied by the executor/EEA layer.
        if success is True and eea not in {
            "no_matching_result",
            "empty_result",
            "verified_zero_rows",
        }:
            return None
        cls = classify_blockable_failure(
            error=str(invoke_result.get("error") or ""),
            error_type=str(invoke_result.get("error_type") or ""),
            eea_failure_class=eea,
            success=bool(success) if success is not None else False,
        )
        if cls:
            self.record_failure(
                operation_id=operation_id,
                arguments=arguments,
                failure_class=cls,
                error=str(invoke_result.get("error") or ""),
            )
        return cls

    def mark_blocked_attempt(
        self, *, operation_id: str, arguments: Optional[Dict[str, Any]], reason: str
    ) -> None:
        self.signature_blocked_count += 1
        key = invocation_key(operation_id, arguments)
        self.block_events.append(
            {
                "event": "blocked_invoke",
                "operation_id": operation_id,
                "reason": reason,
                "invocation_key": key,
            }
        )

    def blocked_response(self, *, operation_id: str, reason: str) -> Dict[str, Any]:
        return {
            "success": False,
            "error": f"signature_blocked:{reason}:{operation_id}",
            "error_type": "signature_blocked",
            "result": None,
            "_signature_blocked": True,
            "_blocked_reason": reason,
        }
