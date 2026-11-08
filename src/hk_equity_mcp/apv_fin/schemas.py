"""Versioned APV-Fin structured-output schemas (JSON Schema + deterministic parse).

No MCP-Zero ATR protocol. Provider structured output may be unavailable; fallback
is JSON mode → deterministic parse → at most one format repair → fail closed.
"""

from __future__ import annotations

import json
import re
from copy import deepcopy
from typing import Any, Dict, List, Optional, Sequence, Tuple

SCHEMA_VERSIONS = {
    "plan": "apv_fin_plan_v1",
    "monitor": "apv_fin_monitor_v1",
    "compute": "apv_fin_compute_v1",
    "final": "apv_fin_final_v1",
}

STEP_TYPES = (
    "retrieve",
    "compute",
    "reason",
    "verify",
    "answer",
    "clarify",
    "refuse",
)

STEP_STATUSES = (
    "pending",
    "ready",
    "running",
    "resolved",
    "failed",
    "skipped",
)

MONITOR_DECISIONS = (
    "continue",
    "replan",
    "verify",
    "answer",
    "clarify",
    "refuse",
    "fail",
)

ANSWER_STATUSES = ("answered", "clarification_needed", "unavailable")

PLAN_STEP_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "step_id",
        "step_type",
        "objective",
        "depends_on",
        "mandatory",
        "success_criteria",
    ],
    "properties": {
        "step_id": {"type": "string", "minLength": 1},
        "step_type": {"type": "string", "enum": list(STEP_TYPES)},
        "objective": {"type": "string", "minLength": 1},
        "depends_on": {
            "type": "array",
            "items": {"type": "string"},
        },
        "mandatory": {"type": "boolean"},
        "success_criteria": {"type": "string"},
        "retrieval_query_hint": {"type": ["string", "null"]},
        "notes": {"type": ["string", "null"]},
    },
}

APV_FIN_PLAN_V1: Dict[str, Any] = {
    "$id": "apv_fin_plan_v1",
    "type": "object",
    "additionalProperties": False,
    "required": ["schema_version", "task_spec", "initial_plan"],
    "properties": {
        "schema_version": {"type": "string", "const": "apv_fin_plan_v1"},
        "task_spec": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "goal",
                "entities",
                "periods",
                "metrics",
                "constraints",
                "ambiguities",
                "answerability_hypothesis",
            ],
            "properties": {
                "goal": {"type": "string", "minLength": 1},
                "entities": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "periods": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "metrics": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "constraints": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "ambiguities": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "answerability_hypothesis": {
                    "type": "string",
                    "enum": [
                        "answerable",
                        "needs_clarification",
                        "likely_unavailable",
                        "unknown",
                    ],
                },
                "cutoff_date": {"type": ["string", "null"]},
                "currency_unit_hints": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "metric_basis_hints": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
        },
        "initial_plan": {
            "type": "object",
            "additionalProperties": False,
            "required": ["steps"],
            "properties": {
                "steps": {
                    "type": "array",
                    "minItems": 1,
                    "items": PLAN_STEP_SCHEMA,
                },
                "planning_notes": {"type": ["string", "null"]},
            },
        },
    },
}

MONITOR_PATCH_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "completed_steps",
        "updated_steps",
        "new_steps",
        "removed_steps",
    ],
    "properties": {
        "completed_steps": {
            "type": "array",
            "items": {"type": "string"},
        },
        "updated_steps": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["step_id"],
                "properties": {
                    "step_id": {"type": "string"},
                    "objective": {"type": ["string", "null"]},
                    "depends_on": {
                        "type": ["array", "null"],
                        "items": {"type": "string"},
                    },
                    "mandatory": {"type": ["boolean", "null"]},
                    "success_criteria": {"type": ["string", "null"]},
                    "step_type": {
                        "type": ["string", "null"],
                        "enum": list(STEP_TYPES) + [None],
                    },
                    "retrieval_query_hint": {"type": ["string", "null"]},
                    "notes": {"type": ["string", "null"]},
                },
            },
        },
        "new_steps": {
            "type": "array",
            "items": PLAN_STEP_SCHEMA,
        },
        "removed_steps": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
}

APV_FIN_MONITOR_V1: Dict[str, Any] = {
    "$id": "apv_fin_monitor_v1",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "schema_version",
        "decision",
        "decision_reason",
        "ready_step_ids",
        "patch",
    ],
    "properties": {
        "schema_version": {"type": "string", "const": "apv_fin_monitor_v1"},
        "decision": {"type": "string", "enum": list(MONITOR_DECISIONS)},
        "decision_reason": {"type": "string", "minLength": 1},
        "ready_step_ids": {
            "type": "array",
            "items": {"type": "string"},
        },
        "patch": MONITOR_PATCH_SCHEMA,
        "clarify_question": {"type": ["string", "null"]},
        "refuse_reason": {"type": ["string", "null"]},
        "fail_reason": {"type": ["string", "null"]},
    },
}

APV_FIN_COMPUTE_V1: Dict[str, Any] = {
    "$id": "apv_fin_compute_v1",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "schema_version",
        "expression",
        "operands",
        "unit",
        "notes",
    ],
    "properties": {
        "schema_version": {"type": "string", "const": "apv_fin_compute_v1"},
        "expression": {"type": "string", "minLength": 1},
        "operands": {
            "type": "object",
            "additionalProperties": {
                "type": ["number", "integer"],
            },
        },
        "unit": {"type": ["string", "null"]},
        "notes": {"type": ["string", "null"]},
    },
}

APV_FIN_FINAL_V1: Dict[str, Any] = {
    "$id": "apv_fin_final_v1",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "schema_version",
        "status",
        "final_answer",
        "reasoning_brief",
        "evidence_ids",
    ],
    "properties": {
        "schema_version": {"type": "string", "const": "apv_fin_final_v1"},
        "status": {"type": "string", "enum": list(ANSWER_STATUSES)},
        "final_answer": {"type": "string"},
        "reasoning_brief": {"type": "string"},
        "evidence_ids": {
            "type": "array",
            "items": {"type": "string"},
        },
        "clarification_question": {"type": ["string", "null"]},
        "refusal_reason": {"type": ["string", "null"]},
        "currency": {"type": ["string", "null"]},
        "unit": {"type": ["string", "null"]},
        "metric_basis": {"type": ["string", "null"]},
        "period": {"type": ["string", "null"]},
        "entity": {"type": ["string", "null"]},
    },
}

SCHEMA_BY_NAME = {
    "apv_fin_plan_v1": APV_FIN_PLAN_V1,
    "apv_fin_monitor_v1": APV_FIN_MONITOR_V1,
    "apv_fin_compute_v1": APV_FIN_COMPUTE_V1,
    "apv_fin_final_v1": APV_FIN_FINAL_V1,
}


class SchemaValidationError(ValueError):
    def __init__(self, message: str, path: str = "$"):
        super().__init__(f"{path}: {message}")
        self.path = path
        self.message = message


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def validate_against_schema(
    instance: Any, schema: Dict[str, Any], path: str = "$"
) -> None:
    """Minimal JSON Schema subset validator (draft-like, fail-closed)."""
    types = schema.get("type")
    if types is not None:
        allowed = types if isinstance(types, list) else [types]
        ok = False
        for t in allowed:
            if t == "null" and instance is None:
                ok = True
            elif t == "object" and isinstance(instance, dict):
                ok = True
            elif t == "array" and isinstance(instance, list):
                ok = True
            elif t == "string" and isinstance(instance, str):
                ok = True
            elif t == "boolean" and isinstance(instance, bool):
                ok = True
            elif t == "integer" and isinstance(instance, int) and not isinstance(
                instance, bool
            ):
                ok = True
            elif t == "number" and _is_number(instance):
                ok = True
        if not ok:
            raise SchemaValidationError(
                f"type mismatch: expected {allowed}, got {type(instance).__name__}",
                path,
            )
        if instance is None:
            return

    if "const" in schema and instance != schema["const"]:
        raise SchemaValidationError(
            f"const mismatch: expected {schema['const']!r}", path
        )

    if "enum" in schema:
        # Allow explicit null members when present in enum.
        if instance not in schema["enum"]:
            raise SchemaValidationError(
                f"enum mismatch: {instance!r} not in {schema['enum']}", path
            )

    if isinstance(instance, str) and "minLength" in schema:
        if len(instance) < int(schema["minLength"]):
            raise SchemaValidationError(
                f"minLength {schema['minLength']} violated", path
            )

    if isinstance(instance, list):
        if "minItems" in schema and len(instance) < int(schema["minItems"]):
            raise SchemaValidationError(
                f"minItems {schema['minItems']} violated", path
            )
        item_schema = schema.get("items")
        if item_schema:
            for i, item in enumerate(instance):
                validate_against_schema(item, item_schema, f"{path}[{i}]")

    if isinstance(instance, dict):
        props = schema.get("properties") or {}
        required = schema.get("required") or []
        for key in required:
            if key not in instance:
                raise SchemaValidationError(f"missing required property {key!r}", path)
        additional = schema.get("additionalProperties", True)
        for key, value in instance.items():
            if key in props:
                validate_against_schema(value, props[key], f"{path}.{key}")
            elif additional is False:
                raise SchemaValidationError(
                    f"additional property not allowed: {key!r}", path
                )
            elif isinstance(additional, dict):
                validate_against_schema(value, additional, f"{path}.{key}")


def strip_json_fences(text: str) -> str:
    raw = text.strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        raw = "\n".join(lines).strip()
    return raw


def extract_json_object(text: str) -> Any:
    """Deterministic JSON object extract: full parse, else first {...} block."""
    cleaned = strip_json_fences(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def parse_and_validate(
    text: str,
    schema_name: str,
    *,
    repair_content: Optional[str] = None,
) -> Tuple[Dict[str, Any], int, Optional[str]]:
    """Parse JSON and validate. Returns (payload, repair_count, error).

    Caller supplies an optional already-repaired content for the single allowed
    repair attempt. This function itself does not call any LLM.
    """
    schema = SCHEMA_BY_NAME[schema_name]
    repair_count = 0
    last_error: Optional[str] = None
    candidates: List[str] = [text]
    if repair_content is not None:
        candidates.append(repair_content)

    for idx, candidate in enumerate(candidates):
        if idx > 0:
            repair_count = 1
        try:
            parsed = extract_json_object(candidate)
            if not isinstance(parsed, dict):
                raise SchemaValidationError("top-level value must be an object")
            validate_against_schema(parsed, schema)
            return parsed, repair_count, None
        except (json.JSONDecodeError, SchemaValidationError, ValueError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            continue
    return {}, repair_count, last_error or "parse_failed"


def format_repair_user_message(schema_name: str, error: str) -> str:
    return (
        f"Your previous response was not valid {schema_name} JSON. "
        f"Validation error: {error}. "
        "Reply again with a single JSON object that matches the schema exactly. "
        "No markdown fences."
    )


def schema_catalog() -> Dict[str, Any]:
    return {
        name: deepcopy(schema) for name, schema in SCHEMA_BY_NAME.items()
    }


def assert_no_operation_id_in_plan(plan_payload: Dict[str, Any]) -> None:
    """Plans must not hard-code operation_id (retrieve steps go through resolver)."""

    def _walk(node: Any, path: str = "$") -> None:
        if isinstance(node, dict):
            if "operation_id" in node:
                raise SchemaValidationError(
                    "initial plan must not specify operation_id; use retrieve steps instead",
                    path=f"{path}.operation_id",
                )
            for key, value in node.items():
                _walk(value, f"{path}.{key}")
        elif isinstance(node, list):
            for i, value in enumerate(node):
                _walk(value, f"{path}[{i}]")

    _walk(plan_payload)
