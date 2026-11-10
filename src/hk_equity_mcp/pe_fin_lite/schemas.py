"""PE-Fin Lite structured-output schemas (plan / binder / final).

No Monitor schema. No capability-compiler ontology enums.
Planner emits semantic steps only — never operation_id.
"""

from __future__ import annotations

import json
import re
from copy import deepcopy
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

SCHEMA_VERSIONS = {
    "plan": "pe_fin_lite_plan_v1",
    "tool_binding": "pe_fin_lite_tool_binding_v1",
    "final": "pe_fin_lite_final_v1",
    "reason": "pe_fin_lite_reason_v1",
    "compute": "pe_fin_lite_compute_v1",
}

STEP_TYPES = ("retrieve", "compute", "reason")
STEP_DOMAINS = (
    "financial",
    "market",
    "disclosure",
    "news",
    "compute",
    "general",
)
ANSWER_STATUSES = ("answered", "clarification_needed", "unavailable")
MAX_PLAN_STEPS = 6

PLAN_STEP_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "id",
        "type",
        "domain",
        "objective",
        "required_outputs",
        "depends_on",
    ],
    "properties": {
        "id": {"type": "string", "minLength": 1, "maxLength": 64},
        "type": {"type": "string", "enum": list(STEP_TYPES)},
        "domain": {"type": "string", "enum": list(STEP_DOMAINS)},
        "objective": {"type": "string", "minLength": 1, "maxLength": 240},
        "required_outputs": {
            "type": "array",
            "maxItems": 12,
            "items": {"type": "string", "maxLength": 80},
        },
        "depends_on": {
            "type": "array",
            "maxItems": 8,
            "items": {"type": "string", "maxLength": 64},
        },
        "entity": {"type": ["string", "null"], "maxLength": 120},
        "period": {"type": ["string", "null"], "maxLength": 120},
        "date": {"type": ["string", "null"], "maxLength": 40},
        "compute_spec": {
            "type": ["object", "null"],
            "additionalProperties": False,
            "required": ["expression", "operand_refs", "output_name"],
            "properties": {
                "expression": {"type": "string", "minLength": 1, "maxLength": 240},
                "operand_refs": {
                    "type": "object",
                    "additionalProperties": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 120,
                    },
                },
                "output_name": {"type": "string", "minLength": 1, "maxLength": 80},
                "output_unit": {"type": ["string", "null"], "maxLength": 40},
            },
        },
        "workflow": {
            "type": ["string", "null"],
            "enum": [None, "retrieve_official_disclosure_evidence"],
        },
    },
}

PE_FIN_LITE_PLAN_V1: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["steps"],
    "properties": {
        "schema_version": {
            "type": ["string", "null"],
            "enum": [None, SCHEMA_VERSIONS["plan"]],
        },
        "steps": {
            "type": "array",
            "minItems": 1,
            "maxItems": MAX_PLAN_STEPS,
            "items": PLAN_STEP_SCHEMA,
        },
    },
}

BOUND_CALL_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["operation_id", "arguments"],
    "properties": {
        "operation_id": {"type": "string", "minLength": 1, "maxLength": 160},
        "arguments": {"type": "object"},
    },
}

PE_FIN_LITE_TOOL_BINDING_V1: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["primary", "fallback"],
    "properties": {
        "schema_version": {
            "type": ["string", "null"],
            "enum": [None, SCHEMA_VERSIONS["tool_binding"]],
        },
        "primary": BOUND_CALL_SCHEMA,
        "fallback": {
            "anyOf": [
                BOUND_CALL_SCHEMA,
                {"type": "null"},
            ]
        },
        "notes": {"type": ["string", "null"], "maxLength": 200},
    },
}

PE_FIN_LITE_REASON_V1: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["summary"],
    "properties": {
        "summary": {"type": "string", "minLength": 1, "maxLength": 600},
        "claims": {
            "type": ["array", "null"],
            "maxItems": 8,
            "items": {"type": "string", "maxLength": 200},
        },
        "gaps": {
            "type": ["array", "null"],
            "maxItems": 8,
            "items": {"type": "string", "maxLength": 200},
        },
    },
}

PE_FIN_LITE_FINAL_V1: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["status", "final_answer", "reasoning_brief"],
    "properties": {
        "schema_version": {
            "type": ["string", "null"],
            "enum": [None, SCHEMA_VERSIONS["final"]],
        },
        "status": {"type": "string", "enum": list(ANSWER_STATUSES)},
        "final_answer": {"type": "string", "maxLength": 2000},
        "reasoning_brief": {"type": "string", "maxLength": 400},
        "evidence_ids": {
            "type": ["array", "null"],
            "maxItems": 20,
            "items": {"type": "string", "maxLength": 80},
        },
        "clarification_question": {"type": ["string", "null"], "maxLength": 400},
        "refusal_reason": {"type": ["string", "null"], "maxLength": 400},
        "currency": {"type": ["string", "null"], "maxLength": 40},
        "unit": {"type": ["string", "null"], "maxLength": 40},
        "metric_basis": {"type": ["string", "null"], "maxLength": 80},
        "period": {"type": ["string", "null"], "maxLength": 80},
        "entity": {"type": ["string", "null"], "maxLength": 120},
    },
}

SCHEMA_BY_NAME: Dict[str, Dict[str, Any]] = {
    "plan": PE_FIN_LITE_PLAN_V1,
    "tool_binding": PE_FIN_LITE_TOOL_BINDING_V1,
    "final": PE_FIN_LITE_FINAL_V1,
    "reason": PE_FIN_LITE_REASON_V1,
}


class SchemaValidationError(ValueError):
    pass


def validate_against_schema(payload: Any, schema: Dict[str, Any], path: str = "$") -> None:
    """Minimal JSON-Schema subset validator (object/array/enum/required)."""
    if not isinstance(schema, dict):
        raise SchemaValidationError(f"{path}: invalid schema")
    if "anyOf" in schema:
        errors = []
        for i, option in enumerate(schema["anyOf"]):
            try:
                validate_against_schema(payload, option, f"{path}.anyOf[{i}]")
                return
            except SchemaValidationError as exc:
                errors.append(str(exc))
        raise SchemaValidationError("%s: no anyOf branch matched (%s)" % (path, errors[0]))
    expected_type = schema.get("type")
    types = expected_type if isinstance(expected_type, list) else [expected_type]
    if expected_type is not None:
        ok = False
        for t in types:
            if t == "null" and payload is None:
                ok = True
            elif t == "object" and isinstance(payload, dict):
                ok = True
            elif t == "array" and isinstance(payload, list):
                ok = True
            elif t == "string" and isinstance(payload, str):
                ok = True
            elif t == "integer" and isinstance(payload, int) and not isinstance(payload, bool):
                ok = True
            elif t == "number" and isinstance(payload, (int, float)) and not isinstance(
                payload, bool
            ):
                ok = True
            elif t == "boolean" and isinstance(payload, bool):
                ok = True
        if not ok:
            raise SchemaValidationError(
                f"{path}: expected type {expected_type}, got {type(payload).__name__}"
            )
    if payload is None:
        return
    if "enum" in schema and payload not in schema["enum"]:
        raise SchemaValidationError(f"{path}: value {payload!r} not in enum")
    if isinstance(payload, str):
        if "minLength" in schema and len(payload) < int(schema["minLength"]):
            raise SchemaValidationError(f"{path}: string shorter than minLength")
        if "maxLength" in schema and len(payload) > int(schema["maxLength"]):
            raise SchemaValidationError(f"{path}: string longer than maxLength")
    if isinstance(payload, list):
        if "minItems" in schema and len(payload) < int(schema["minItems"]):
            raise SchemaValidationError(f"{path}: fewer than minItems")
        if "maxItems" in schema and len(payload) > int(schema["maxItems"]):
            raise SchemaValidationError(f"{path}: more than maxItems")
        item_schema = schema.get("items")
        if item_schema:
            for i, item in enumerate(payload):
                validate_against_schema(item, item_schema, f"{path}[{i}]")
    if isinstance(payload, dict):
        required = schema.get("required") or []
        for key in required:
            if key not in payload:
                raise SchemaValidationError(f"{path}: missing required property {key!r}")
        props = schema.get("properties") or {}
        additional = schema.get("additionalProperties", True)
        for key, value in payload.items():
            if key in props:
                validate_against_schema(value, props[key], f"{path}.{key}")
            elif additional is False:
                raise SchemaValidationError(f"{path}: unexpected property {key!r}")
            elif isinstance(additional, dict):
                validate_against_schema(value, additional, f"{path}.{key}")


def extract_json_object(text: str) -> Any:
    if not isinstance(text, str) or not text.strip():
        raise SchemaValidationError("empty content")
    raw = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", raw, flags=re.DOTALL)
    if fence:
        raw = fence.group(1)
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end < 0 or end <= start:
        raise SchemaValidationError("no JSON object found")
    try:
        return json.loads(raw[start : end + 1])
    except json.JSONDecodeError as exc:
        raise SchemaValidationError(f"invalid JSON: {exc}") from exc


def parse_and_validate(
    content: str,
    schema_name: str,
    *,
    repair_content: Optional[str] = None,
) -> Tuple[Optional[Dict[str, Any]], int, Optional[str]]:
    schema = SCHEMA_BY_NAME[schema_name]
    repair_count = 0
    try:
        payload = extract_json_object(content)
        validate_against_schema(payload, schema)
        if schema_name == "plan":
            validate_plan_semantics(payload)
        return payload, repair_count, None
    except (SchemaValidationError, PlanValidationError) as exc:
        err = str(exc)
    if repair_content is None:
        return None, repair_count, err
    repair_count = 1
    try:
        payload = extract_json_object(repair_content)
        validate_against_schema(payload, schema)
        if schema_name == "plan":
            validate_plan_semantics(payload)
        return payload, repair_count, None
    except (SchemaValidationError, PlanValidationError) as exc:
        return None, repair_count, str(exc)


def format_repair_user_message(schema_name: str, error: str) -> str:
    return (
        f"Your previous JSON failed validation for schema {schema_name}. "
        f"Error: {error}. Return one corrected JSON object only — no markdown."
    )


class PlanValidationError(ValueError):
    pass


def _has_cycle(steps: Sequence[Dict[str, Any]]) -> bool:
    ids = {str(s["id"]) for s in steps}
    deps: Dict[str, List[str]] = {
        str(s["id"]): [d for d in (s.get("depends_on") or []) if d in ids] for s in steps
    }
    visiting: Set[str] = set()
    visited: Set[str] = set()

    def dfs(node: str) -> bool:
        if node in visiting:
            return True
        if node in visited:
            return False
        visiting.add(node)
        for nxt in deps.get(node) or []:
            if dfs(nxt):
                return True
        visiting.remove(node)
        visited.add(node)
        return False

    return any(dfs(n) for n in deps)


def validate_plan_semantics(payload: Dict[str, Any]) -> None:
    steps = payload.get("steps") or []
    if not steps:
        raise PlanValidationError("plan must contain at least one step")
    if len(steps) > MAX_PLAN_STEPS:
        raise PlanValidationError(f"plan exceeds max {MAX_PLAN_STEPS} steps")
    ids = [str(s.get("id") or "") for s in steps]
    if len(ids) != len(set(ids)):
        raise PlanValidationError("duplicate step ids")
    id_set = set(ids)
    blob = json.dumps(payload, ensure_ascii=False)
    # Fail closed if planner smuggles concrete operation ids / gold tooling.
    banned = ("operation_id", "gold_operations", "reference_answer", "task_id")
    for token in banned:
        if token in blob:
            raise PlanValidationError(f"plan must not contain {token}")
    for step in steps:
        sid = str(step["id"])
        for dep in step.get("depends_on") or []:
            if dep not in id_set:
                raise PlanValidationError(f"step {sid} depends on unknown {dep}")
        stype = step.get("type")
        if stype == "retrieve":
            if not step.get("domain"):
                raise PlanValidationError(f"retrieve step {sid} missing domain")
            if not step.get("objective"):
                raise PlanValidationError(f"retrieve step {sid} missing objective")
            if step.get("required_outputs") is None:
                raise PlanValidationError(
                    f"retrieve step {sid} missing required_outputs"
                )
        if stype == "compute":
            spec = step.get("compute_spec")
            if not isinstance(spec, dict):
                raise PlanValidationError(f"compute step {sid} missing compute_spec")
            if not (step.get("depends_on") or []):
                raise PlanValidationError(
                    f"compute step {sid} must depend on prior evidence steps"
                )
        if stype == "reason":
            if step.get("compute_spec") is not None:
                raise PlanValidationError(f"reason step {sid} must not have compute_spec")
    if _has_cycle(steps):
        raise PlanValidationError("plan DAG contains a cycle")


def exported_schemas() -> Dict[str, Dict[str, Any]]:
    return {name: deepcopy(schema) for name, schema in SCHEMA_BY_NAME.items()}
