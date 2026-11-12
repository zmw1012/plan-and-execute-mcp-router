"""Described-only Tool Binder for PE-Fin RePlan-ETOM.

* ``primary`` required, ``fallback`` optional; both MUST be described candidates.
* Arguments are validated against the official ETOM ``input_schema``; an empty
  ``{}`` is refused whenever the schema has required arguments.
* At most ONE full regeneration repair (format / schema / binding contract).
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from hk_equity_mcp.pe_fin_lite.schemas import SchemaValidationError, validate_against_schema
from hk_equity_mcp.pe_fin_replan_etom.prompts import BINDER_SYSTEM_PROMPT, render_binder_user
from hk_equity_mcp.pe_fin_replan_etom.schemas import (
    ETOM_BINDER_V1,
    extract_json_object,
    format_repair_user_message,
)

BINDER_PROTOCOL_VERSION = "pe_fin_replan_etom_binder_protocol_v1"
MAX_BINDER_FORMAT_REPAIRS = 1


class ToolBindingError(ValueError):
    pass


def binder_protocol_hash() -> str:
    import hashlib

    blob = json.dumps(
        {
            "version": BINDER_PROTOCOL_VERSION,
            "described_only": True,
            "primary_plus_optional_fallback": True,
            "max_format_repairs": MAX_BINDER_FORMAT_REPAIRS,
            "full_regeneration_only": True,
            "refuse_empty_args_when_required": True,
            "no_json_splice": True,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _type_ok(value: Any, expected: Any) -> bool:
    types = expected if isinstance(expected, list) else [expected]
    for t in types:
        if t == "string" and isinstance(value, str):
            return True
        if t == "integer" and isinstance(value, int) and not isinstance(value, bool):
            return True
        if t == "number" and isinstance(value, (int, float)) and not isinstance(value, bool):
            return True
        if t == "boolean" and isinstance(value, bool):
            return True
        if t == "array" and isinstance(value, list):
            return True
        if t == "object" and isinstance(value, dict):
            return True
        if t == "null" and value is None:
            return True
    return False


def validate_arguments(
    arguments: Dict[str, Any], schema: Dict[str, Any], *, label: str = "arguments"
) -> None:
    """Light JSON-schema check against the OFFICIAL input schema."""
    if not isinstance(arguments, dict):
        raise ToolBindingError(f"{label} must be an object")
    schema = schema if isinstance(schema, dict) else {}
    required = [str(r) for r in (schema.get("required") or [])]
    props = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
    if required and not arguments:
        raise ToolBindingError(
            f"{label} is empty but the operation has required arguments {required}"
        )
    missing = [r for r in required if r not in arguments]
    if missing:
        raise ToolBindingError(f"{label} missing required arguments: {missing}")
    for name in required:
        if arguments.get(name) in (None, ""):
            raise ToolBindingError(f"{label} required argument {name!r} is empty")
    if schema.get("additionalProperties") is False and props:
        unknown = [k for k in arguments if k not in props]
        if unknown:
            raise ToolBindingError(f"{label} has unknown arguments {unknown}")
    for name, value in arguments.items():
        spec = props.get(name)
        if not isinstance(spec, dict):
            continue
        if "type" in spec and not _type_ok(value, spec["type"]):
            raise ToolBindingError(
                f"{label} argument {name!r} has wrong type "
                f"(expected {spec['type']}, got {type(value).__name__})"
            )
        if isinstance(spec.get("enum"), list) and value not in spec["enum"]:
            raise ToolBindingError(f"{label} argument {name!r} not in enum")


def _validate_call(
    call: Dict[str, Any],
    *,
    allowed: Set[str],
    contracts: Dict[str, Dict[str, Any]],
    label: str,
) -> Dict[str, Any]:
    oid = call.get("operation_id")
    arguments = call.get("arguments")
    if not isinstance(oid, str) or not oid:
        raise ToolBindingError(f"{label} missing operation_id")
    if oid not in allowed:
        raise ToolBindingError(f"{label} operation_id {oid!r} not in described candidates")
    if not isinstance(arguments, dict):
        raise ToolBindingError(f"{label} arguments must be an object")
    validate_arguments(
        arguments,
        (contracts.get(oid) or {}).get("json_schema") or {},
        label=f"{label}.arguments",
    )
    return {"operation_id": oid, "arguments": dict(arguments)}


def validate_binding_payload(
    payload: Dict[str, Any],
    *,
    described_operation_ids: Sequence[str],
    described_contracts: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    validate_against_schema(payload, ETOM_BINDER_V1)
    allowed = set(described_operation_ids)
    primary = _validate_call(
        payload.get("primary") or {},
        allowed=allowed,
        contracts=described_contracts,
        label="primary",
    )
    fallback = None
    fallback_raw = payload.get("fallback")
    if fallback_raw is not None:
        fallback = _validate_call(
            fallback_raw, allowed=allowed, contracts=described_contracts, label="fallback"
        )
        if fallback["operation_id"] == primary["operation_id"] and (
            fallback["arguments"] == primary["arguments"]
        ):
            fallback = None
    return {"primary": primary, "fallback": fallback, "notes": payload.get("notes")}


def _try_parse(
    content: Any,
    *,
    described_ids: Sequence[str],
    contracts: Dict[str, Dict[str, Any]],
) -> Tuple[Optional[Dict[str, Any]], Optional[str], str]:
    text = "" if content is None else str(content)
    if not text.strip():
        return None, "empty content", "empty"
    try:
        payload = extract_json_object(text)
        validated = validate_binding_payload(
            payload, described_operation_ids=described_ids, described_contracts=contracts
        )
        return validated, None, "ok"
    except SchemaValidationError as exc:
        return None, str(exc), "schema_invalid"
    except ToolBindingError as exc:
        return None, str(exc), "binding_invalid"
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}:{exc}", "parse_error"


def _add_usage(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(a or {})
    for key, val in (b or {}).items():
        if isinstance(val, (int, float)) and isinstance(out.get(key), (int, float)):
            out[key] = out[key] + val
        else:
            out[key] = val
    return out


def run_tool_binder(
    *,
    llm: Any,
    question: str,
    step: Dict[str, Any],
    described_candidates: Sequence[Dict[str, Any]],
    described_contracts: Dict[str, Dict[str, Any]],
    evidence: Dict[str, Any],
    temperature: float = 0.0,
    max_tokens: int = 1024,
) -> Dict[str, Any]:
    """One binder LLM call plus at most one full-regeneration repair."""
    described_ids = [
        str(c.get("operation_id")) for c in described_candidates if c.get("operation_id")
    ]
    compact: List[Dict[str, Any]] = []
    for oid in described_ids:
        contract = described_contracts.get(oid) or {}
        compact.append(
            {
                "operation_id": oid,
                "summary": contract.get("summary") or contract.get("description"),
                "json_schema": contract.get("json_schema"),
                "required": contract.get("required"),
            }
        )
    messages = [
        {"role": "system", "content": BINDER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": render_binder_user(
                question=question, step=step, candidates=compact, evidence=evidence
            ),
        },
    ]
    completion = llm.complete(
        messages, temperature=temperature, max_tokens=max_tokens, stage="tool_binding"
    )
    usage = dict(completion.get("usage") or {})
    content = completion.get("content")
    validated, error, label = _try_parse(
        content, described_ids=described_ids, contracts=described_contracts
    )
    observability = [{"stage": "tool_binding", "parse_result": label, "schema_error": error}]
    repair_count = 0
    initial_error: Optional[str] = None
    if validated is None and repair_count < MAX_BINDER_FORMAT_REPAIRS:
        initial_error = error or "binder_parse_failed"
        repair_msg = format_repair_user_message(
            "tool_binding", initial_error, previous_content=str(content or "")
        )
        repair = llm.complete(
            messages
            + [
                {"role": "assistant", "content": str(content or "")},
                {"role": "user", "content": repair_msg},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            stage="tool_binding_repair",
        )
        repair_count = 1
        usage = _add_usage(usage, repair.get("usage") or {})
        content = repair.get("content")
        validated, error, label = _try_parse(
            content, described_ids=described_ids, contracts=described_contracts
        )
        observability.append(
            {"stage": "tool_binding_repair", "parse_result": label, "schema_error": error}
        )
    return {
        "ok": validated is not None,
        "binding": validated,
        "error": None if validated is not None else (error or initial_error or "binder_failed"),
        "usage": usage,
        "repair_count": repair_count,
        "initial_parse_error": initial_error,
        "call_observability": observability,
    }
