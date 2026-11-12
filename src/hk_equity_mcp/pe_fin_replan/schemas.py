"""Strict Replanner output schema for PE-Fin RePlan v1."""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

from hk_equity_mcp.pe_fin_lite.schemas import (
    PLAN_STEP_SCHEMA,
    SchemaValidationError,
    extract_json_object,
    validate_against_schema,
)

REPLAN_SCHEMA_VERSION = "pe_fin_replan_v1"

LOOP_POLICY = {
    "version": "pe_fin_replan_loop_policy_v1",
    "executor_one_step_per_round": True,
    "no_executor_to_executor_edge": True,
    "planner_exactly_once": True,
    "final_response_from_replanner_only": True,
    "max_new_steps_per_replan": 2,
    "max_executed_steps": 6,
    "max_replan_rounds": 6,
    "max_no_progress_rounds": 2,
    "one_replanner_repair": True,
    "no_json_splice": True,
}

# Reuse plan-step shape but allow Replanner-generated step ids (r*).
_REPLAN_STEP = deepcopy(PLAN_STEP_SCHEMA)

REPLANNER_PLAN_BRANCH: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "schema_version",
        "action",
        "decision_reason",
        "remaining_steps",
    ],
    "properties": {
        "schema_version": {"type": "string", "const": REPLAN_SCHEMA_VERSION},
        "action": {"type": "string", "const": "plan"},
        "decision_reason": {"type": "string", "minLength": 1, "maxLength": 400},
        "remaining_steps": {
            "type": "array",
            "minItems": 0,
            "maxItems": 6,
            "items": _REPLAN_STEP,
        },
    },
}

REPLANNER_RESPONSE_BRANCH: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "schema_version",
        "action",
        "decision_reason",
        "response",
    ],
    "properties": {
        "schema_version": {"type": "string", "const": REPLAN_SCHEMA_VERSION},
        "action": {"type": "string", "const": "response"},
        "decision_reason": {"type": "string", "minLength": 1, "maxLength": 400},
        "response": {
            "type": "object",
            "additionalProperties": False,
            "required": ["status", "final_answer", "reasoning_brief"],
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["answered", "clarification_needed", "unavailable"],
                },
                "final_answer": {"type": "string", "maxLength": 4000},
                "reasoning_brief": {"type": "string", "maxLength": 800},
                "clarification_question": {"type": ["string", "null"], "maxLength": 400},
                "refusal_reason": {"type": ["string", "null"], "maxLength": 400},
                "limitations": {
                    "type": ["array", "null"],
                    "items": {"type": "string", "maxLength": 200},
                    "maxItems": 8,
                },
            },
        },
    },
}

REPLANNER_OUTPUT_SCHEMA: Dict[str, Any] = {
    "anyOf": [REPLANNER_PLAN_BRANCH, REPLANNER_RESPONSE_BRANCH]
}


def validate_replanner_payload(payload: Any) -> None:
    validate_against_schema(payload, REPLANNER_OUTPUT_SCHEMA, "$")
    if not isinstance(payload, dict):
        raise SchemaValidationError("replanner payload must be object")
    if payload.get("schema_version") != REPLAN_SCHEMA_VERSION:
        raise SchemaValidationError("schema_version mismatch")
    action = payload.get("action")
    if action == "plan":
        if "response" in payload:
            raise SchemaValidationError("plan action must not include response")
        steps = payload.get("remaining_steps")
        if not isinstance(steps, list):
            raise SchemaValidationError("remaining_steps must be array")
    elif action == "response":
        if "remaining_steps" in payload:
            raise SchemaValidationError("response action must not include remaining_steps")
        resp = payload.get("response") or {}
        if resp.get("status") not in {"answered", "clarification_needed", "unavailable"}:
            raise SchemaValidationError("invalid response.status")
    else:
        raise SchemaValidationError(f"unknown action: {action}")


def parse_replanner_output(
    content: str,
    *,
    repair_content: Optional[str] = None,
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
            validate_replanner_payload(obj)
            return obj, None, None
        except SchemaValidationError as exc:
            last_err = str(exc)
            continue
    return None, None, last_err or "invalid_replanner_output"


def format_replanner_repair_message(error: str) -> str:
    return (
        "Your previous Replanner output failed schema validation.\n"
        f"Error: {error}\n"
        "Return ONE complete JSON object matching pe_fin_replan_v1 with "
        "action=plan|response, additionalProperties=false. No markdown."
    )
