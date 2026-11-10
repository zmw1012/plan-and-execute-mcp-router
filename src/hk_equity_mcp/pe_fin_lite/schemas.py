"""PE-Fin Lite structured-output schemas (plan / binder / final).

No Monitor schema. No capability-compiler ontology enums.
Planner emits semantic steps only — never operation_id.
"""

from __future__ import annotations

import json
import re
from copy import deepcopy
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from hk_equity_mcp.pe_fin_lite.output_vocabulary import (
    CANONICAL_OUTPUT_FIELDS,
    validate_required_outputs,
)
from hk_equity_mcp.pe_fin_lite.plan_canonicalizer import (
    canonicalize_plan_payload,
    normalize_json_text,
)

SCHEMA_VERSIONS = {
    "plan": "pe_fin_lite_plan_v2",
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

# Explicit planner string budgets (must match prompt + repair messaging).
OBJECTIVE_MAX_LENGTH = 480
REQUIRED_OUTPUT_ITEM_MAX_LENGTH = 64
COMPUTE_EXPRESSION_MAX_LENGTH = 240

SCOPE_CONSTRAINTS_SCHEMA: Dict[str, Any] = {
    "type": ["object", "null"],
    "additionalProperties": False,
    "properties": {
        "entity": {
            "type": ["string", "object", "null"],
            "maxLength": 120,
            "additionalProperties": False,
            "properties": {
                "name": {"type": ["string", "null"], "maxLength": 120},
                "code": {"type": ["string", "null"], "maxLength": 32},
                "company_name": {"type": ["string", "null"], "maxLength": 120},
                "stock_code": {"type": ["string", "null"], "maxLength": 32},
                "issuer_code": {"type": ["string", "null"], "maxLength": 32},
            },
        },
        "period": {"type": ["string", "null"], "maxLength": 80},
        "trade_date": {"type": ["string", "null"], "maxLength": 40},
        "from_date": {"type": ["string", "null"], "maxLength": 40},
        "to_date": {"type": ["string", "null"], "maxLength": 40},
        "metric_basis": {"type": ["string", "null"], "maxLength": 80},
        "adjustment": {"type": ["string", "null"], "maxLength": 40},
        "title_keyword": {"type": ["string", "null"], "maxLength": 160},
    },
}

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
        "objective": {
            "type": "string",
            "minLength": 1,
            "maxLength": OBJECTIVE_MAX_LENGTH,
        },
        "required_outputs": {
            "type": "array",
            "maxItems": 12,
            "items": {
                "type": "string",
                "enum": list(CANONICAL_OUTPUT_FIELDS),
                "maxLength": REQUIRED_OUTPUT_ITEM_MAX_LENGTH,
            },
        },
        "depends_on": {
            "type": "array",
            "maxItems": 8,
            "items": {"type": "string", "maxLength": 64},
        },
        "entity": {"type": ["string", "null"], "maxLength": 120},
        "period": {"type": ["string", "null"], "maxLength": 120},
        "date": {"type": ["string", "null"], "maxLength": 40},
        "scope_constraints": SCOPE_CONSTRAINTS_SCHEMA,
        "compute_spec": {
            "type": ["object", "null"],
            "additionalProperties": False,
            "required": ["expression", "operand_refs", "output_name"],
            "properties": {
                "expression": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": COMPUTE_EXPRESSION_MAX_LENGTH,
                },
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

# Runtime-owned identity keys — LLM may emit/omit/stale-version these; runtime
# strips them before semantic validation and re-injects canonical values after.
RUNTIME_OWNED_FIELDS_BY_SCHEMA: Dict[str, Tuple[str, ...]] = {
    "plan": (
        "schema_version",
        "method_version",
        "protocol_version",
        "status",
        "timestamps",
        "created_at",
        "updated_at",
        "timestamp",
    ),
    "tool_binding": ("schema_version", "method_version", "protocol_version"),
    "final": ("schema_version", "method_version", "protocol_version"),
    "reason": ("schema_version", "method_version", "protocol_version"),
}

PE_FIN_LITE_PLAN_V1: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["steps"],
    "properties": {
        # Optional for documentation only; validation strips LLM-provided values
        # and runtime injects SCHEMA_VERSIONS["plan"] after accept.
        "schema_version": {"type": ["string", "null"], "maxLength": 80},
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
        "schema_version": {"type": ["string", "null"], "maxLength": 80},
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
        "schema_version": {"type": ["string", "null"], "maxLength": 80},
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
            raise SchemaValidationError(
                f"{path}: string length {len(payload)} shorter than minLength "
                f"{schema['minLength']}"
            )
        if "maxLength" in schema and len(payload) > int(schema["maxLength"]):
            raise SchemaValidationError(
                f"{path}: string length {len(payload)} exceeds maxLength "
                f"{schema['maxLength']} (allowed={schema['maxLength']}, actual={len(payload)})"
            )
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
        # When schema allows object|string|null union, only apply object rules for dicts.
        if "object" not in types and expected_type is not None and "object" not in (
            expected_type if isinstance(expected_type, list) else [expected_type]
        ):
            # string-only branch already handled
            pass
        required = schema.get("required") or []
        for key in required:
            if key not in payload:
                raise SchemaValidationError(f"{path}: missing required property {key!r}")
        props = schema.get("properties") or {}
        additional = schema.get("additionalProperties", True)
        # Only enforce additionalProperties when payload is validated as object.
        enforce_object = "object" in types or expected_type is None or expected_type == "object"
        if enforce_object or isinstance(payload, dict):
            for key, value in payload.items():
                if key in props:
                    validate_against_schema(value, props[key], f"{path}.{key}")
                elif additional is False:
                    raise SchemaValidationError(f"{path}: unexpected property {key!r}")
                elif isinstance(additional, dict):
                    validate_against_schema(value, additional, f"{path}.{key}")


def extract_json_object(text: str) -> Any:
    if text is None:
        raise SchemaValidationError("null content")
    if not isinstance(text, str) or not text.strip():
        raise SchemaValidationError("empty content")
    raw, _norm_applied = normalize_json_text(text)
    if not raw.strip():
        raise SchemaValidationError("empty content")
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end < 0 or end <= start:
        raise SchemaValidationError("no JSON object found (truncated or non-JSON content)")
    candidate = raw[start : end + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as exc:
        # Fail closed: never splice truncated fragments into invented JSON.
        raise SchemaValidationError(f"invalid JSON: {exc}") from exc


def strip_runtime_owned_fields(
    payload: Dict[str, Any], schema_name: str
) -> Dict[str, Any]:
    """Remove runtime identity fields so LLM stale/missing versions cannot fail semantic plans."""
    if not isinstance(payload, dict):
        return payload
    owned = set(RUNTIME_OWNED_FIELDS_BY_SCHEMA.get(schema_name) or ())
    if not owned:
        return payload
    return {k: v for k, v in payload.items() if k not in owned}


def inject_runtime_owned_fields(
    payload: Dict[str, Any], schema_name: str
) -> Dict[str, Any]:
    """Attach canonical runtime identity after semantic validation succeeds."""
    out = dict(payload)
    if schema_name in SCHEMA_VERSIONS or schema_name in SCHEMA_BY_NAME:
        version = SCHEMA_VERSIONS.get(schema_name)
        if version:
            out["schema_version"] = version
    return out


def _validate_payload(payload: Any, schema_name: str) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise SchemaValidationError("payload must be a JSON object")
    semantic = strip_runtime_owned_fields(payload, schema_name)
    if schema_name == "plan":
        semantic, _applied = canonicalize_plan_payload(semantic)
    schema = SCHEMA_BY_NAME[schema_name]
    # Validate the semantic payload; schema_version is optional/stripped.
    validate_against_schema(semantic, schema)
    if schema_name == "plan":
        validate_plan_semantics(semantic)
    return inject_runtime_owned_fields(semantic, schema_name)


def parse_and_validate(
    content: str,
    schema_name: str,
    *,
    repair_content: Optional[str] = None,
) -> Tuple[Optional[Dict[str, Any]], int, Optional[str]]:
    repair_count = 0
    try:
        payload = extract_json_object(content)
        return _validate_payload(payload, schema_name), repair_count, None
    except (SchemaValidationError, PlanValidationError) as exc:
        err = str(exc)
    if repair_content is None:
        return None, repair_count, err
    repair_count = 1
    try:
        payload = extract_json_object(repair_content)
        return _validate_payload(payload, schema_name), repair_count, None
    except (SchemaValidationError, PlanValidationError) as exc:
        return None, repair_count, str(exc)


def _collect_length_violations(payload: Any, schema: Dict[str, Any], path: str = "$") -> List[str]:
    """Best-effort scan for maxLength violations to include in repair hints."""
    notes: List[str] = []
    if not isinstance(schema, dict) or payload is None:
        return notes
    if isinstance(payload, str) and "maxLength" in schema:
        if len(payload) > int(schema["maxLength"]):
            notes.append(
                f"{path}: actual_length={len(payload)} allowed_length={schema['maxLength']}"
            )
    if isinstance(payload, list) and isinstance(schema.get("items"), dict):
        for i, item in enumerate(payload):
            notes.extend(_collect_length_violations(item, schema["items"], f"{path}[{i}]"))
    if isinstance(payload, dict):
        props = schema.get("properties") or {}
        for key, value in payload.items():
            if key in props:
                notes.extend(
                    _collect_length_violations(value, props[key], f"{path}.{key}")
                )
        items = payload.get("steps") if "steps" in payload else None
        step_schema = (props.get("steps") or {}).get("items") if props else None
        if isinstance(items, list) and isinstance(step_schema, dict):
            for i, step in enumerate(items):
                notes.extend(
                    _collect_length_violations(step, step_schema, f"$.steps[{i}]")
                )
    return notes


def _parse_error_path_and_value(error: str) -> Tuple[Optional[str], Optional[str]]:
    """Best-effort extract JSON path and invalid value from validator error text."""
    path = None
    actual = None
    m = re.match(r"^(\$[.\w\[\]'\"-]+)\s*:\s*(.*)$", error or "")
    if m:
        path = m.group(1)
        rest = m.group(2)
        vm = re.search(r"value\s+('.*?'|\".*?\"|[^\s]+)\s+not in enum", rest)
        if vm:
            actual = vm.group(1)
        else:
            lm = re.search(r"actual=(\d+)", rest)
            if lm:
                actual = lm.group(1)
            elif "missing required property" in rest:
                pm = re.search(r"missing required property\s+'([^']+)'", rest)
                actual = f"<missing:{pm.group(1)}>" if pm else "<missing>"
            else:
                actual = rest[:160]
    return path, actual


def format_repair_user_message(
    schema_name: str,
    error: str,
    *,
    previous_content: Optional[str] = None,
) -> str:
    """Repair instruction: full re-generation only; never splice/truncate silently."""
    length_notes: List[str] = []
    if schema_name == "plan" and previous_content:
        try:
            prev = extract_json_object(previous_content)
            length_notes = _collect_length_violations(prev, SCHEMA_BY_NAME["plan"])
        except SchemaValidationError:
            length_notes = []
    json_path, actual_value = _parse_error_path_and_value(error)
    lines = [
        f"Your previous JSON failed validation for schema {schema_name}.",
        f"Error: {error}",
    ]
    if json_path:
        lines.append(f"JSON path: {json_path}")
    if actual_value is not None:
        lines.append(f"Actual invalid value: {actual_value}")
    lines.append(
        "Expected: valid field presence, enum membership, and maxLength as defined "
        "by the frozen schema. Regenerate a COMPLETE closed JSON object."
    )
    if length_notes:
        lines.append("Length violations (JSON path / actual / allowed):")
        lines.extend(f"- {n}" for n in length_notes[:12])
    if schema_name == "plan":
        lines.extend(
            [
                "Every step MUST contain depends_on. First/root step uses depends_on=[].",
                "domain MUST be one of: financial|market|disclosure|news|compute|general.",
                "reason steps use domain=general; compute steps use domain=compute.",
                f"Hard limits: objective maxLength={OBJECTIVE_MAX_LENGTH}; "
                f"compute_spec.expression maxLength={COMPUTE_EXPRESSION_MAX_LENGTH}.",
                "required_outputs must be canonical field IDs only "
                f"(enum={list(CANONICAL_OUTPUT_FIELDS)}).",
                "Put entity/period/date/basis into scope_constraints — not required_outputs.",
                "Compute steps MUST include compute_spec with non-empty expression "
                "and typed operand_refs (map of operand name → evidence ref string).",
                "Return a COMPLETE new plan JSON object with all braces closed. "
                "Do NOT splice fragments. Do NOT truncate mid-JSON. "
                "Do NOT silently truncate overlength strings — rewrite within limits "
                "while preserving meaning.",
            ]
        )
    lines.append("Return one corrected JSON object only — no markdown.")
    return "\n".join(lines)


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
        req = list(step.get("required_outputs") or [])
        bad_req = validate_required_outputs(req)
        if bad_req:
            raise PlanValidationError(
                f"step {sid} required_outputs not canonical: {bad_req}"
            )
        if stype == "retrieve":
            if not step.get("domain"):
                raise PlanValidationError(f"retrieve step {sid} missing domain")
            if not step.get("objective"):
                raise PlanValidationError(f"retrieve step {sid} missing objective")
            if step.get("required_outputs") is None:
                raise PlanValidationError(
                    f"retrieve step {sid} missing required_outputs"
                )
            obj = str(step.get("objective") or "")
            if len(obj) > OBJECTIVE_MAX_LENGTH:
                raise PlanValidationError(
                    f"$.steps[id={sid}].objective: actual_length={len(obj)} "
                    f"allowed_length={OBJECTIVE_MAX_LENGTH}"
                )
        if stype == "compute":
            spec = step.get("compute_spec")
            if not isinstance(spec, dict):
                raise PlanValidationError(
                    f"compute step {sid} missing compute_spec "
                    "(need expression + typed operand_refs + output_name)"
                )
            expr = str(spec.get("expression") or "").strip()
            if not expr:
                raise PlanValidationError(
                    f"compute step {sid} missing compute_spec.expression"
                )
            refs = spec.get("operand_refs")
            if not isinstance(refs, dict) or not refs:
                raise PlanValidationError(
                    f"compute step {sid} missing typed compute_spec.operand_refs"
                )
            for op_name, ref in refs.items():
                if not isinstance(op_name, str) or not str(op_name).strip():
                    raise PlanValidationError(
                        f"compute step {sid} operand_refs has invalid operand name"
                    )
                if not isinstance(ref, str) or not str(ref).strip():
                    raise PlanValidationError(
                        f"compute step {sid} operand_refs[{op_name!r}] must be a "
                        "non-empty evidence ref string"
                    )
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
