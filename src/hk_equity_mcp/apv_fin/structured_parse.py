"""Structured JSON parse helpers for APV-Fin (JSON mode, one repair, fail-closed)."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from hk_equity_mcp.apv_fin.schemas import (
    SchemaValidationError,
    extract_json_object,
    format_repair_user_message,
    parse_and_validate,
    validate_against_schema,
    SCHEMA_BY_NAME,
)


def normalize_llm_content(raw: Any) -> Tuple[str, Dict[str, Any]]:
    """Safely normalize provider message content; never treat null as success."""
    meta = {
        "content_was_null": raw is None,
        "content_type": type(raw).__name__,
    }
    if raw is None:
        return "", meta
    if isinstance(raw, (dict, list)):
        # Some gateways return already-parsed JSON; stringify deterministically.
        import json

        return json.dumps(raw, ensure_ascii=False), meta
    return str(raw), meta


def parse_structured(
    content: Any,
    schema_name: str,
    *,
    repair_content: Any = None,
) -> Dict[str, Any]:
    """Parse + validate with optional single repair content.

    Returns dict with keys:
      ok, payload, error, repair_count, raw_content, repair_raw_content, fail_closed
    Empty objects / empty steps fail schema (fail closed).
    """
    text, meta = normalize_llm_content(content)
    repair_text = None
    repair_meta = None
    if repair_content is not None:
        repair_text, repair_meta = normalize_llm_content(repair_content)

    payload, repair_count, error = parse_and_validate(
        text, schema_name, repair_content=repair_text
    )
    out = {
        "ok": error is None,
        "payload": payload if error is None else None,
        "error": error,
        "repair_count": repair_count,
        "raw_content": text,
        "repair_raw_content": repair_text,
        "content_meta": meta,
        "repair_content_meta": repair_meta,
        "fail_closed": error is not None,
        "schema_name": schema_name,
    }
    # Explicit empty-object guard (schema also rejects, but keep clear reason).
    if error is None and isinstance(payload, dict) and payload == {}:
        out.update(
            {
                "ok": False,
                "payload": None,
                "error": "empty_object_not_allowed",
                "fail_closed": True,
            }
        )
    return out


def repair_user_message(schema_name: str, error: str) -> str:
    return format_repair_user_message(schema_name, error)


def assert_schema(payload: Dict[str, Any], schema_name: str) -> None:
    validate_against_schema(payload, SCHEMA_BY_NAME[schema_name])
