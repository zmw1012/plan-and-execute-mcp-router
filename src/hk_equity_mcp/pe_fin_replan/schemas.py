"""Strict Replanner output schema for PE-Fin RePlan v1.3."""

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
from hk_equity_mcp.pe_fin_replan.step_canonicalizer import (
    canonicalize_replanner_json_text,
    canonicalize_replanner_payload,
)

REPLAN_SCHEMA_VERSION = "pe_fin_replan_v1_3"

LOOP_POLICY = {
    "version": "pe_fin_replan_loop_policy_v1_3",
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
    "step_canonicalizer": "pe_fin_replan_step_canonicalizer_v1",
    "grounding_contract": "pe_fin_replan_grounding_v2",
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

# Concrete PlanStep examples — fields taken only from PLAN_STEP_SCHEMA.
RETRIEVE_STEP_TEMPLATE: Dict[str, Any] = {
    "id": "s1",
    "type": "retrieve",
    "domain": "market",
    "objective": "retrieve the required market fields for the scoped entity/date",
    "required_outputs": ["close", "trade_date", "currency"],
    "depends_on": [],
    "entity": None,
    "period": None,
    "date": None,
    "scope_constraints": None,
    "compute_spec": None,
    "workflow": None,
}

COMPUTE_STEP_TEMPLATE: Dict[str, Any] = {
    "id": "s2",
    "type": "compute",
    "domain": "compute",
    "objective": "compute a derived metric from verified operand refs",
    "required_outputs": ["revenue"],
    "depends_on": ["s1"],
    "entity": None,
    "period": None,
    "date": None,
    "scope_constraints": None,
    "compute_spec": {
        "expression": "(a-b)/b",
        "operand_refs": {"a": "s1.revenue", "b": "s1.basic_eps"},
        "output_name": "derived_metric",
        "output_unit": None,
    },
    "workflow": None,
}

REASON_STEP_TEMPLATE: Dict[str, Any] = {
    "id": "s3",
    "type": "reason",
    "domain": "general",
    "objective": "summarize verified evidence into the final answer",
    "required_outputs": [],
    "depends_on": ["s1"],
    "entity": None,
    "period": None,
    "date": None,
    "scope_constraints": None,
    "compute_spec": None,
    "workflow": None,
}

PLAN_TEMPLATE: Dict[str, Any] = {
    "schema_version": REPLAN_SCHEMA_VERSION,
    "action": "plan",
    "decision_reason": "optional short reason",
    "remaining_steps": [
        RETRIEVE_STEP_TEMPLATE,
        COMPUTE_STEP_TEMPLATE,
        REASON_STEP_TEMPLATE,
    ],
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


def _plan_step_templates_block() -> str:
    return (
        "PlanStep templates (fields from PLAN_STEP_SCHEMA only):\n"
        "retrieve step:\n"
        + json.dumps(RETRIEVE_STEP_TEMPLATE, ensure_ascii=False, indent=2)
        + "\n\ncompute step:\n"
        + json.dumps(COMPUTE_STEP_TEMPLATE, ensure_ascii=False, indent=2)
        + "\n\nreason step:\n"
        + json.dumps(REASON_STEP_TEMPLATE, ensure_ascii=False, indent=2)
    )


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
        + "\n\n"
        + _plan_step_templates_block()
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
    """Parse + canonicalize + strict-validate. No JSON splice — full object only."""
    candidates = [content]
    if repair_content is not None:
        candidates.append(repair_content)
    last_err = None
    for raw in candidates:
        try:
            normalized, _ = canonicalize_replanner_json_text(str(raw or ""))
            obj = extract_json_object(normalized if normalized else str(raw or ""))
        except Exception as exc:  # noqa: BLE001
            last_err = f"json_extract:{exc}"
            continue
        if not isinstance(obj, dict):
            last_err = "replanner payload must be object"
            continue
        try:
            obj, _canon_rules = canonicalize_replanner_payload(obj)
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
    step_required = (
        'Each PlanStep requires: ["id","type","domain","objective",'
        '"required_outputs","depends_on"]. '
        "Use id/type (not step_id/step_type). "
        "Root/first remaining step must include depends_on (use [] if none)."
    )
    return (
        "Your previous Replanner output failed schema validation.\n"
        f"JSON path: {json_path}\n"
        f"Validation error: {error}\n"
        f"Current round response_only={bool(response_only)}\n"
        f"Allowed actions: {allowed}\n"
        f"Required fields: {required}\n"
        f"{step_required}\n"
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
