"""Tool Binder: one LLM call returns primary + fallback from described candidates.

Format repair: at most one full re-generation for null/empty/truncated/invalid
JSON/schema invalid. Semantic selection failure is not infinitely repaired.
Still subject to the global LLM≤8 budget via BudgetedLLMClient.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Optional, Sequence, Set

from hk_equity_mcp.agent.llm_client import LLMClient, build_response_observability
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

BINDER_PROTOCOL_VERSION = "pe_fin_lite_binder_protocol_v1_2"
MAX_BINDER_FORMAT_REPAIRS = 1


class ToolBindingError(ValueError):
    pass


def binder_protocol_hash() -> str:
    blob = json.dumps(
        {
            "version": BINDER_PROTOCOL_VERSION,
            "primary_plus_fallback": True,
            "max_format_repairs": MAX_BINDER_FORMAT_REPAIRS,
            "full_regeneration_only": True,
            "no_json_splice": True,
            "observability": [
                "content_present",
                "content_length",
                "finish_reason",
                "parse_result",
                "schema_error",
                "repair_count",
            ],
            "no_hidden_cot": True,
            "no_secrets": True,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


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


def _obs_from_completion(
    *,
    stage: str,
    completion: Dict[str, Any],
    max_tokens: int,
    parse_result: Optional[str] = None,
    schema_error: Optional[str] = None,
) -> Dict[str, Any]:
    base = dict(completion.get("response_observability") or {})
    if not base:
        usage = dict(completion.get("usage") or {})
        content = str(completion.get("content") or "")
        base = build_response_observability(
            stage=stage,
            content=content,
            usage=usage,
            latency_ms=float(completion.get("latency_ms") or 0.0),
            max_tokens_requested=max_tokens,
            provider=str(completion.get("provider") or ""),
            model=str(completion.get("model") or ""),
            host=completion.get("host"),
            finish_reason=completion.get("finish_reason"),
            message_meta={
                "message_was_null": completion.get("message_was_null"),
                "content_was_null": completion.get("content_was_null"),
                "message_keys": [],
                "reasoning_content_present": False,
                "reasoning_content_length": 0,
                "content_shape": None,
            },
            parse_result=parse_result,
            schema_error=schema_error,
        )
    else:
        if parse_result is not None:
            base["parse_result"] = parse_result
        if schema_error is not None:
            base["schema_error"] = schema_error
    base["stage"] = stage
    # Never persist secrets / reasoning text.
    for banned in ("api_key", "authorization", "reasoning_content", "reasoning_text"):
        base.pop(banned, None)
    return base


def _try_parse_binding(
    content: Any,
    *,
    described_ids: Sequence[str],
    described_contracts: Dict[str, Dict[str, Any]],
) -> tuple[Optional[Dict[str, Any]], Optional[str], str]:
    """Return (validated, error, parse_result_label)."""
    text = "" if content is None else str(content)
    if not text.strip():
        return None, "empty content", "empty"
    try:
        payload = extract_json_object(text)
        validated = validate_binding_payload(
            payload,
            described_operation_ids=described_ids,
            described_contracts=described_contracts,
        )
        return validated, None, "ok"
    except SchemaValidationError as exc:
        return None, str(exc), "schema_invalid"
    except json.JSONDecodeError as exc:
        return None, f"invalid JSON:{exc}", "invalid_json"
    except ToolBindingError as exc:
        # Semantic / contract selection failure — eligible for one format repair
        # only; never an infinite loop.
        return None, str(exc), "binding_invalid"
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}:{exc}", "parse_error"


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
    """Binder call with ≤1 format repair; primary+fallback design unchanged."""
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
    call_observability: List[Dict[str, Any]] = []
    usage: Dict[str, Any] = {}
    repair_count = 0
    content: Any = None
    initial_error: Optional[str] = None

    completion = llm.complete(
        messages, temperature=temperature, max_tokens=max_tokens, stage="tool_binding"
    )
    usage = dict(completion.get("usage") or {})
    content = completion.get("content")
    validated, error, parse_label = _try_parse_binding(
        content,
        described_ids=described_ids,
        described_contracts=described_contracts,
    )
    call_observability.append(
        _obs_from_completion(
            stage="tool_binding",
            completion=completion,
            max_tokens=max_tokens,
            parse_result=parse_label,
            schema_error=error,
        )
    )

    if validated is None and repair_count < MAX_BINDER_FORMAT_REPAIRS:
        initial_error = error or "binder_parse_failed"
        repair_msg = format_repair_user_message(
            "tool_binding",
            initial_error,
            previous_content=str(content or ""),
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
        for key, val in (repair.get("usage") or {}).items():
            if isinstance(val, (int, float)) and isinstance(usage.get(key), (int, float)):
                usage[key] = usage[key] + val
            else:
                usage[key] = val
        content = repair.get("content")
        validated, error, parse_label = _try_parse_binding(
            content,
            described_ids=described_ids,
            described_contracts=described_contracts,
        )
        call_observability.append(
            _obs_from_completion(
                stage="tool_binding_repair",
                completion=repair,
                max_tokens=max_tokens,
                parse_result=parse_label,
                schema_error=error,
            )
        )

    if validated is not None:
        return {
            "ok": True,
            "binding": validated,
            "error": None,
            "usage": usage,
            "raw_content": content,
            "repair_count": repair_count,
            "runtime_recovered": repair_count > 0,
            "initial_parse_error": initial_error,
            "call_observability": call_observability,
        }

    # Second failure: structured binder failure; caller must not invoke MCP.
    final_error = error or initial_error or "binder_failed"
    return {
        "ok": False,
        "binding": None,
        "error": final_error,
        "usage": usage,
        "raw_content": content,
        "repair_count": repair_count,
        "runtime_recovered": False,
        "initial_parse_error": initial_error,
        "call_observability": call_observability,
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
