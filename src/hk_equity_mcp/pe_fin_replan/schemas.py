"""Strict Replanner output schema for PE-Fin RePlan v1.2."""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any, Dict, Optional, Tuple

from hk_equity_mcp.pe_fin_lite.schemas import (
    PLAN_STEP_SCHEMA,
    SchemaValidationError,
    extract_json_object,
    validate_against_schema,
)

REPLAN_SCHEMA_VERSION = "pe_fin_replan_v1_2"

LOOP_POLICY = {
    "version": "pe_fin_replan_loop_policy_v1_2",
    "executor_one_step_per_round": True,
    "no_executor_to_executor_edge": True,
    "planner_exactly_once": True,
    "planner_not_counted_as_execution_round": True,
    "final_response_from_replanner_only": True,
    "max_new_steps_per_replan": 2,
    "max_execution_rounds": 7,
    "max_executed_steps": 7,
    "max_replan_rounds": 7,
    "max_no_progress_rounds": 2,
    "one_replanner_repair": True,
    "no_json_splice": True,
    "round7_response_only": True,
    "simplified_completion": True,
    "decision_reason_optional": True,
    "completion_enum": ["answered", "insufficient_evidence"],
    "one_transient_transport_retry": True,
}

_REPLAN_STEP = deepcopy(PLAN_STEP_SCHEMA)

_DECISION_REASON = {
    "type": ["string", "null"],
    "maxLength": 400,
}

REPLANNER_PLAN_BRANCH: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "schema_version",
        "action",
        "remaining_steps",
    ],
    "properties": {
        "schema_version": {"type": "string", "const": REPLAN_SCHEMA_VERSION},
        "action": {"type": "string", "const": "plan"},
        "decision_reason": _DECISION_REASON,
        "remaining_steps": {
            "type": "array",
            "minItems": 0,
            "maxItems": 7,
            "items": _REPLAN_STEP,
        },
    },
}

_RESPONSE_OBJECT: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["completion", "final_answer", "reasoning_brief"],
    "properties": {
        "completion": {
            "type": "string",
            "enum": ["answered", "insufficient_evidence"],
        },
        "final_answer": {"type": "string", "maxLength": 4000},
        "reasoning_brief": {"type": "string", "maxLength": 800},
    },
}

REPLANNER_RESPONSE_BRANCH: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "schema_version",
        "action",
        "response",
    ],
    "properties": {
        "schema_version": {"type": "string", "const": REPLAN_SCHEMA_VERSION},
        "action": {"type": "string", "const": "response"},
        "decision_reason": _DECISION_REASON,
        "response": _RESPONSE_OBJECT,
    },
}

REPLANNER_OUTPUT_SCHEMA: Dict[str, Any] = {
    "anyOf": [REPLANNER_PLAN_BRANCH, REPLANNER_RESPONSE_BRANCH]
}

REPLANNER_RESPONSE_ONLY_SCHEMA: Dict[str, Any] = {
    "anyOf": [REPLANNER_RESPONSE_BRANCH]
}

BANNED_RESPONSE_FIELDS = (
    "status",
    "clarification_needed",
    "refusal_reason",
    "clarification_question",
    "limitations",
)

PLAN_TEMPLATE: Dict[str, Any] = {
    "schema_version": REPLAN_SCHEMA_VERSION,
    "action": "plan",
    "decision_reason": "optional short reason",
    "remaining_steps": [],
}

RESPONSE_ANSWERED_TEMPLATE: Dict[str, Any] = {
    "schema_version": REPLAN_SCHEMA_VERSION,
    "action": "response",
    "decision_reason": "optional short reason",
    "response": {
        "completion": "answered",
        "final_answer": "answer grounded in evidence",
        "reasoning_brief": "short explanation",
    },
}

RESPONSE_INSUFFICIENT_TEMPLATE: Dict[str, Any] = {
    "schema_version": REPLAN_SCHEMA_VERSION,
    "action": "response",
    "decision_reason": "optional short reason",
    "response": {
        "completion": "insufficient_evidence",
        "final_answer": (
            "The answer cannot be reliably determined from the available evidence."
        ),
        "reasoning_brief": "short explanation",
    },
}


def _template_block(*, response_only: bool) -> str:
    if response_only:
        return (
            "Required shape (FINAL ROUND — action=response only):\n"
            + json.dumps(RESPONSE_ANSWERED_TEMPLATE, ensure_ascii=False, indent=2)
            + "\n\nOR:\n"
            + json.dumps(RESPONSE_INSUFFICIENT_TEMPLATE, ensure_ascii=False, indent=2)
        )
    return (
        "Plan template:\n"
        + json.dumps(PLAN_TEMPLATE, ensure_ascii=False, indent=2)
        + "\n\nResponse template (answered):\n"
        + json.dumps(RESPONSE_ANSWERED_TEMPLATE, ensure_ascii=False, indent=2)
        + "\n\nResponse template (insufficient_evidence):\n"
        + json.dumps(RESPONSE_INSUFFICIENT_TEMPLATE, ensure_ascii=False, indent=2)
    )


def normalize_decision_reason(payload: Dict[str, Any]) -> Dict[str, Any]:
    """If model omitted decision_reason, store null — do not invent text."""
    out = dict(payload)
    if "decision_reason" not in out:
        out["decision_reason"] = None
    return out


def validate_replanner_payload(
    payload: Any, *, response_only: bool = False
) -> None:
    if not isinstance(payload, dict):
        raise SchemaValidationError("replanner payload must be object")
    if payload.get("schema_version") != REPLAN_SCHEMA_VERSION:
        raise SchemaValidationError("schema_version mismatch")
    action = payload.get("action")
    if response_only and action != "response":
        raise SchemaValidationError("round7_requires_action_response")
    schema = (
        REPLANNER_RESPONSE_ONLY_SCHEMA if response_only else REPLANNER_OUTPUT_SCHEMA
    )
    validate_against_schema(payload, schema, "$")
    if action == "plan":
        if "response" in payload:
            raise SchemaValidationError("plan action must not include response")
        steps = payload.get("remaining_steps")
        if not isinstance(steps, list):
            raise SchemaValidationError("remaining_steps must be array")
    elif action == "response":
        if "remaining_steps" in payload:
            raise SchemaValidationError(
                "response action must not include remaining_steps"
            )
        resp = payload.get("response") or {}
        if not isinstance(resp, dict):
            raise SchemaValidationError("response must be object")
        for banned in BANNED_RESPONSE_FIELDS:
            if banned in resp:
                raise SchemaValidationError(f"banned_response_field:{banned}")
        if resp.get("completion") not in {"answered", "insufficient_evidence"}:
            raise SchemaValidationError("invalid response.completion")
    else:
        raise SchemaValidationError(f"unknown action: {action}")


def parse_replanner_output(
    content: str,
    *,
    repair_content: Optional[str] = None,
    response_only: bool = False,
) -> Tuple[Optional[Dict[str, Any]], Optional[str], Optional[str]]:
    """Parse + validate. No JSON splice — full object only."""
    candidates = [content]
    if repair_content is not None:
        candidates.append(repair_content)
    last_err = None
    for raw in candidates:
        try:
            obj = extract_json_object(str(raw or ""))
        except Exception as exc:  # noqa: BLE001
            last_err = f"json_extract:{exc}"
            continue
        try:
            validate_replanner_payload(obj, response_only=response_only)
            return normalize_decision_reason(obj), None, None
        except SchemaValidationError as exc:
            last_err = str(exc)
            continue
    return None, None, last_err or "invalid_replanner_output"


def format_replanner_repair_message(
    error: str,
    *,
    response_only: bool = False,
    json_path: str = "$",
) -> str:
    allowed = "action=response only" if response_only else "action=plan|response"
    required = (
        '["schema_version","action","response"]'
        if response_only
        else (
            'plan: ["schema_version","action","remaining_steps"] ; '
            'response: ["schema_version","action","response"]'
        )
    )
    return (
        "Your previous Replanner output failed schema validation.\n"
        f"JSON path: {json_path}\n"
        f"Validation error: {error}\n"
        f"Current round response_only={bool(response_only)}\n"
        f"Allowed actions: {allowed}\n"
        f"Required fields: {required}\n"
        "decision_reason is OPTIONAL (string|null); omit or null is OK — "
        "do not invent a fake reason.\n"
        "Disallowed response fields: status, clarification_needed, "
        "refusal_reason, clarification_question, limitations.\n"
        "Regenerate ONE complete JSON object. No markdown. No splice.\n\n"
        + _template_block(response_only=response_only)
    )


def map_completion_to_predicted_status(completion: str) -> str:
    """Evaluator-facing mapping only; method logic uses completion."""
    if completion == "answered":
        return "answered"
    return "unavailable"
