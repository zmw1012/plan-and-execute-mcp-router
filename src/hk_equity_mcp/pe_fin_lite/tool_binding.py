"""Tool Binder: one LLM call returns primary + fallback from described candidates."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence, Set

from hk_equity_mcp.agent.llm_client import LLMClient
from hk_equity_mcp.pe_fin_lite.prompts import (
    PE_FIN_LITE_SYSTEM_PROMPT,
    render_binder_user,
)
from hk_equity_mcp.pe_fin_lite.schemas import (
    PE_FIN_LITE_TOOL_BINDING_V1,
    SchemaValidationError,
    extract_json_object,
    format_repair_user_message,
    validate_against_schema,
)


class ToolBindingError(ValueError):
    pass


def _validate_call(
    call: Dict[str, Any],
    *,
    allowed: Set[str],
    described_contracts: Dict[str, Dict[str, Any]],
    label: str,
) -> Dict[str, Any]:
    oid = call.get("operation_id")
    arguments = call.get("arguments")
    if not isinstance(oid, str) or not oid:
        raise ToolBindingError(f"{label} missing operation_id")
    if oid not in allowed:
        raise ToolBindingError(
            f"{label} operation_id {oid!r} not in described candidates"
        )
    if not isinstance(arguments, dict):
        raise ToolBindingError(f"{label} arguments must be an object")
    contract = described_contracts.get(oid) or {}
    json_schema = contract.get("json_schema") or {}
    props = json_schema.get("properties") or {}
    required = list(json_schema.get("required") or [])
    for key in arguments:
        if props and key not in props:
            raise ToolBindingError(f"{label} argument {key!r} not in schema")
    missing = [name for name in required if name not in arguments]
    if missing:
        raise ToolBindingError(f"{label} missing required arguments: {missing}")
    if required and arguments == {}:
        raise ToolBindingError(
            f"{label} refuses empty {{}} when schema has required fields"
        )
    for name, value in arguments.items():
        schema = props.get(name) or {}
        if "enum" in schema and value not in schema["enum"]:
            raise ToolBindingError(f"{label} argument {name!r} not in enum")
    return {"operation_id": oid, "arguments": dict(arguments)}


def validate_binding_payload(
    payload: Dict[str, Any],
    *,
    described_operation_ids: Sequence[str],
    described_contracts: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    validate_against_schema(payload, PE_FIN_LITE_TOOL_BINDING_V1)
    allowed = set(described_operation_ids)
    primary = _validate_call(
        payload.get("primary") or {},
        allowed=allowed,
        described_contracts=described_contracts,
        label="primary",
    )
    fallback_raw = payload.get("fallback")
    fallback = None
    if fallback_raw is not None:
        fallback = _validate_call(
            fallback_raw,
            allowed=allowed,
            described_contracts=described_contracts,
            label="fallback",
        )
        if fallback["operation_id"] == primary["operation_id"]:
            # Allow same op only if arguments differ; otherwise drop fallback.
            if fallback["arguments"] == primary["arguments"]:
                fallback = None
    return {
        "primary": primary,
        "fallback": fallback,
        "notes": payload.get("notes"),
    }


def run_tool_binder(
    *,
    llm: LLMClient,
    question: str,
    step: Dict[str, Any],
    described_candidates: Sequence[Dict[str, Any]],
    described_contracts: Dict[str, Dict[str, Any]],
    evidence: Dict[str, Any],
    temperature: float = 0.0,
    max_tokens: int = 1024,
) -> Dict[str, Any]:
    """One binder call; no binder repair (primary+fallback in same response)."""
    described_ids = [
        str(c.get("operation_id"))
        for c in described_candidates
        if c.get("operation_id")
    ]
    compact_candidates = []
    for oid in described_ids:
        contract = described_contracts.get(oid) or {}
        compact_candidates.append(
            {
                "operation_id": oid,
                "summary": contract.get("summary") or contract.get("description"),
                "json_schema": contract.get("json_schema"),
                "constraints": contract.get("constraints"),
            }
        )
    user = render_binder_user(
        question=question,
        step=step,
        candidates=compact_candidates,
        evidence=evidence,
    )
    messages = [
        {"role": "system", "content": PE_FIN_LITE_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
    completion = llm.complete(
        messages, temperature=temperature, max_tokens=max_tokens, stage="tool_binding"
    )
    content = completion.get("content")
    usage = completion.get("usage") or {}
    try:
        payload = extract_json_object(str(content or ""))
        validated = validate_binding_payload(
            payload,
            described_operation_ids=described_ids,
            described_contracts=described_contracts,
        )
        return {
            "ok": True,
            "binding": validated,
            "error": None,
            "usage": usage,
            "raw_content": content,
            "repair_count": 0,
        }
    except (SchemaValidationError, ToolBindingError, json.JSONDecodeError) as exc:
        # Binder has no repair loop per budgets (primary+fallback in one shot).
        return {
            "ok": False,
            "binding": None,
            "error": str(exc),
            "usage": usage,
            "raw_content": content,
            "repair_count": 0,
        }


def mock_binding_from_candidates(
    described_candidates: Sequence[Dict[str, Any]],
    described_contracts: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Deterministic offline binder when scripted LLM is not used for binding."""
    ids = [str(c.get("operation_id")) for c in described_candidates if c.get("operation_id")]
    if not ids:
        raise ToolBindingError("no described candidates")

    def _args_for(oid: str) -> Dict[str, Any]:
        contract = described_contracts.get(oid) or {}
        schema = contract.get("json_schema") or {}
        required = list(schema.get("required") or [])
        props = schema.get("properties") or {}
        args: Dict[str, Any] = {}
        for name in required:
            prop = props.get(name) or {}
            if "default" in prop:
                args[name] = prop["default"]
            elif "enum" in prop and prop["enum"]:
                args[name] = prop["enum"][0]
            elif prop.get("type") == "integer":
                args[name] = 0
            elif prop.get("type") == "number":
                args[name] = 0.0
            elif prop.get("type") == "boolean":
                args[name] = False
            else:
                args[name] = "mock"
        return args

    primary = {"operation_id": ids[0], "arguments": _args_for(ids[0])}
    fallback = None
    if len(ids) > 1:
        fallback = {"operation_id": ids[1], "arguments": _args_for(ids[1])}
    return validate_binding_payload(
        {"primary": primary, "fallback": fallback},
        described_operation_ids=ids,
        described_contracts=described_contracts,
    )
