"""GENERIC tool-result normalization for PE-Fin RePlan-ETOM.

Accepts JSON text, parsed objects, lists, MCP-style ``{"content": [{"text": ...}]}``
blocks, plain text, scalars, and ``None``. There are NO provider adapters here
(no Sina / Eastmoney / HKEX / HK ticker logic). Large payloads are truncated with
an explicit marker and a content hash; the full raw payload is addressed through
``raw_result_ref`` (``raw:<sha16>``) and optionally held in an in-memory store.

In ETOM routing-only mode ``business_result`` is ``None``; that normalizes to
``kind="routing_only"`` (transport ok, no business data).
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Optional, Tuple

ENVELOPE_VERSION = "pe_fin_replan_etom_envelope_v1"
MAX_INLINE_CHARS = 4000
MAX_FLAT_VALUES = 64
TRUNCATION_MARKER = "…[TRUNCATED]"


def _canonical(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def content_sha256(obj: Any) -> str:
    text = obj if isinstance(obj, str) else _canonical(obj)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def raw_result_ref(obj: Any) -> str:
    return "raw:" + content_sha256(obj)[:16]


class RawResultStore:
    """In-memory raw payload store keyed by ``raw_result_ref`` (never persisted by default)."""

    def __init__(self) -> None:
        self._items: Dict[str, Any] = {}

    def put(self, obj: Any) -> str:
        ref = raw_result_ref(obj)
        self._items[ref] = obj
        return ref

    def get(self, ref: str) -> Any:
        return self._items.get(ref)

    def __len__(self) -> int:
        return len(self._items)


def _mcp_text_blocks(obj: Any) -> Optional[str]:
    if isinstance(obj, dict) and isinstance(obj.get("content"), list):
        texts = []
        for block in obj["content"]:
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                texts.append(block["text"])
            elif isinstance(block, str):
                texts.append(block)
        if texts:
            return "\n".join(texts)
    return None


def _maybe_json(text: str) -> Tuple[Any, bool]:
    stripped = text.strip()
    if stripped[:1] in ("{", "[") and stripped[-1:] in ("}", "]"):
        try:
            return json.loads(stripped), True
        except json.JSONDecodeError:
            return text, False
    return text, False


def flatten_scalars(
    obj: Any, *, prefix: str = "", limit: int = MAX_FLAT_VALUES, max_depth: int = 4
) -> Dict[str, Any]:
    """Scalar leaves by dotted path (generic; no field-name semantics)."""
    out: Dict[str, Any] = {}

    def walk(node: Any, path: str, depth: int) -> None:
        if len(out) >= limit:
            return
        if isinstance(node, dict):
            if depth >= max_depth:
                return
            for key in sorted(node, key=str):
                walk(node[key], f"{path}.{key}" if path else str(key), depth + 1)
        elif isinstance(node, list):
            if depth >= max_depth:
                return
            for i, item in enumerate(node[:16]):
                walk(item, f"{path}[{i}]", depth + 1)
        elif isinstance(node, (str, int, float, bool)) and path:
            out[path] = node if not isinstance(node, str) else node[:240]

    walk(obj, prefix, 0)
    return out


def _is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (list, dict, str)) and len(value) == 0:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return False


def normalize_tool_result(
    raw: Any,
    *,
    max_chars: int = MAX_INLINE_CHARS,
    store: Optional[RawResultStore] = None,
) -> Dict[str, Any]:
    """Normalize one business payload. Pure and deterministic."""
    if raw is None:
        return {
            "envelope_version": ENVELOPE_VERSION,
            "kind": "none",
            "empty": True,
            "value": None,
            "raw_result_ref": None,
            "content_sha256": None,
            "char_count": 0,
            "truncated": False,
            "truncation_marker": None,
            "flat_values": {},
        }
    original = raw
    text_from_blocks = _mcp_text_blocks(raw)
    if text_from_blocks is not None:
        raw = text_from_blocks
    kind = "scalar"
    value: Any = raw
    if isinstance(raw, str):
        parsed, was_json = _maybe_json(raw)
        if was_json:
            value = parsed
            kind = "json_list" if isinstance(parsed, list) else "json_object"
        else:
            kind = "text"
    elif isinstance(raw, list):
        kind = "list"
    elif isinstance(raw, dict):
        kind = "object"
    elif isinstance(raw, (bool, int, float)):
        kind = "scalar"
    else:
        value = str(raw)
        kind = "text"

    serialized = value if isinstance(value, str) else _canonical(value)
    ref = store.put(original) if store is not None else raw_result_ref(original)
    truncated = len(serialized) > int(max_chars)
    inline: Any = value
    marker = None
    if truncated:
        marker = {
            "marker": TRUNCATION_MARKER,
            "original_chars": len(serialized),
            "kept_chars": int(max_chars),
            "sha256": content_sha256(serialized),
        }
        inline = serialized[: int(max_chars)] + TRUNCATION_MARKER
    return {
        "envelope_version": ENVELOPE_VERSION,
        "kind": kind,
        "empty": _is_empty(value),
        "value": inline,
        "raw_result_ref": ref,
        "content_sha256": content_sha256(serialized),
        "char_count": len(serialized),
        "truncated": truncated,
        "truncation_marker": marker,
        "flat_values": flatten_scalars(value) if not isinstance(value, str) else {},
    }


def normalize_invoke_result(invoke_result: Dict[str, Any], *, store: Optional[RawResultStore] = None) -> Dict[str, Any]:
    """Envelope for a gateway/resolver invoke response (transport vs business split)."""
    transport_ok = bool(invoke_result.get("success"))
    business = invoke_result.get("business_result")
    if invoke_result.get("execution_mode") == "etom_routing_only" or (
        transport_ok and business is None
    ):
        env = normalize_tool_result(None)
        env["kind"] = "routing_only"
        env["empty"] = True
    else:
        env = normalize_tool_result(business, store=store)
    env["transport_ok"] = transport_ok
    env["error"] = invoke_result.get("error")
    env["business_result_available"] = business is not None
    return env


def envelope_identity() -> Dict[str, Any]:
    return {
        "version": ENVELOPE_VERSION,
        "max_inline_chars": MAX_INLINE_CHARS,
        "max_flat_values": MAX_FLAT_VALUES,
        "truncation_marker": TRUNCATION_MARKER,
        "provider_adapters": [],
        "generic_only": True,
    }


def envelope_hash() -> str:
    return content_sha256(envelope_identity())
