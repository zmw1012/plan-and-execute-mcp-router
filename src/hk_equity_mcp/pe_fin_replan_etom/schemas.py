"""ETOM planner / binder / reason / replanner schemas for PE-Fin RePlan-ETOM v1.

Design notes (see docs/pe_fin_replan_etom_v1/architecture.md):

* Step types: ``retrieve | compute | reason`` (unchanged from HK-155).
* ``domain`` is a GENERIC ETOM capability area, NOT the HK
  financial/market/news/disclosure set. Allowed areas:
  search, read, write, list, compute, reason, calendar, code, docs, data,
  communication, general. These were derived from the public ETOM registry
  server categories (documentation-access, file-system, code-analysis,
  communication, database, search, ...) and from the verb distribution of
  public tool names; they do not encode task identities or gold tools.
* ``required_outputs`` are FREE strings (short noun-phrase labels of what a step
  must yield). There is no HK canonical output enum. Each item is bounded
  (``REQUIRED_OUTPUT_ITEM_MAX_LENGTH``); overlength strings fail closed and
  require full regeneration — never silent truncation.
* Planner plans never contain ``operation_id``; tools are selected only through
  R5 retrieval + described-only Binder.
* Binder returns ``primary`` plus an OPTIONAL ``fallback`` chosen only from
  described candidates.
* Replanner schema version is distinct from HK: ``pe_fin_replan_etom_v1``.
"""

from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Domain-agnostic reuse (no HK semantics inside these helpers).
from hk_equity_mcp.pe_fin_lite.plan_canonicalizer import normalize_json_text
from hk_equity_mcp.pe_fin_lite.schemas import (
    PlanValidationError,
    SchemaValidationError,
    _has_cycle,
    validate_against_schema,
)

PLAN_SCHEMA_VERSION = "pe_fin_replan_etom_plan_v2"
BINDER_SCHEMA_VERSION = "pe_fin_replan_etom_binder_v1"
REASON_SCHEMA_VERSION = "pe_fin_replan_etom_reason_v1"
REPLAN_SCHEMA_VERSION = "pe_fin_replan_etom_v1"
SCHEMA_SET_VERSION = "pe_fin_replan_etom_schema_set_v2"

STEP_TYPES = ("retrieve", "compute", "reason")
ETOM_DOMAINS = (
    "search",
    "read",
    "write",
    "list",
    "compute",
    "reason",
    "calendar",
    "code",
    "docs",
    "data",
    "communication",
    "general",
)
COMPLETIONS = ("answered", "insufficient_evidence")
MAX_PLAN_STEPS = 6
MAX_REMAINING_STEPS = 7
OBJECTIVE_MAX_LENGTH = 480
# Was 64; Smoke-v4 L5-q68 planner_invalid_after_repair with actual=70.
# Cross-domain free-text semantic labels; 128 remains a strict short bound.
REQUIRED_OUTPUT_ITEM_MAX_LENGTH = 128
COMPUTE_EXPRESSION_MAX_LENGTH = 240
PLANNER_MAX_TOKENS = 4096

BANNED_PLAN_KEYS = frozenset(
    {
        "operation_id",
        "operation_ids",
        "tool_name",
        "server_name",
        "gold",
        "gold_operations",
        "gold_tools",
        "ground_truth",
        "ground_truth_tools",
        "accepted_tools",
        "accepted_operations",
        "reference_answer",
        "task_id",
    }
)

_SCOPE_VALUE_SCHEMA: Dict[str, Any] = {
    "type": ["string", "number", "boolean", "null"],
    "maxLength": 240,
}

SCOPE_CONSTRAINTS_SCHEMA: Dict[str, Any] = {
    "type": ["object", "null"],
    "additionalProperties": _SCOPE_VALUE_SCHEMA,
}

COMPUTE_SPEC_SCHEMA: Dict[str, Any] = {
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
}

ETOM_PLAN_STEP_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["id", "type", "domain", "objective", "required_outputs", "depends_on"],
    "properties": {
        "id": {"type": "string", "minLength": 1, "maxLength": 64},
        "type": {"type": "string", "enum": list(STEP_TYPES)},
        "domain": {"type": "string", "enum": list(ETOM_DOMAINS)},
        "objective": {
            "type": "string",
            "minLength": 1,
            "maxLength": OBJECTIVE_MAX_LENGTH,
        },
        "required_outputs": {
            "type": "array",
            "maxItems": 12,
            # Free strings: NOT an HK canonical enum.
            "items": {
                "type": "string",
                "minLength": 1,
                "maxLength": REQUIRED_OUTPUT_ITEM_MAX_LENGTH,
            },
        },
        "depends_on": {
            "type": "array",
            "maxItems": 8,
            "items": {"type": "string", "maxLength": 64},
        },
        "scope_constraints": SCOPE_CONSTRAINTS_SCHEMA,
        "compute_spec": COMPUTE_SPEC_SCHEMA,
    },
}

ETOM_PLAN_V1: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["steps"],
    "properties": {
        "schema_version": {"type": ["string", "null"], "maxLength": 80},
        "steps": {
            "type": "array",
            "minItems": 1,
            "maxItems": MAX_PLAN_STEPS,
            "items": ETOM_PLAN_STEP_SCHEMA,
        },
    },
}

# Public aliases (HK-style / parent-agent spellings).
PLAN_SCHEMA = ETOM_PLAN_V1
ETOM_PLAN = ETOM_PLAN_V1

BOUND_CALL_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["operation_id", "arguments"],
    "properties": {
        "operation_id": {"type": "string", "minLength": 1, "maxLength": 240},
        "arguments": {"type": "object"},
    },
}

ETOM_BINDER_V1: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["primary"],
    "properties": {
        "schema_version": {"type": ["string", "null"], "maxLength": 80},
        "primary": BOUND_CALL_SCHEMA,
        "fallback": {"anyOf": [BOUND_CALL_SCHEMA, {"type": "null"}]},
        "notes": {"type": ["string", "null"], "maxLength": 200},
    },
}

BINDER_SCHEMA = ETOM_BINDER_V1

ETOM_REASON_V1: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["summary"],
    "properties": {
        "schema_version": {"type": ["string", "null"], "maxLength": 80},
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

SCHEMA_BY_NAME: Dict[str, Dict[str, Any]] = {
    "plan": ETOM_PLAN_V1,
    "tool_binding": ETOM_BINDER_V1,
    "reason": ETOM_REASON_V1,
}
SCHEMA_VERSIONS = {
    "plan": PLAN_SCHEMA_VERSION,
    "tool_binding": BINDER_SCHEMA_VERSION,
    "reason": REASON_SCHEMA_VERSION,
    "replanner": REPLAN_SCHEMA_VERSION,
}

# ---------------------------------------------------------------------------
# Replanner schema (action = plan | response)
# ---------------------------------------------------------------------------

LOOP_POLICY = {
    "version": "pe_fin_replan_etom_loop_policy_v1",
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
    "one_semantic_or_schema_repair": True,
    "no_silent_step_truncate": True,
    "no_json_splice": True,
    "round7_response_only": True,
    "no_eighth_round": True,
    "completion_enum": list(COMPLETIONS),
    "one_transient_transport_retry": True,
    "no_monitor": True,
    "no_global_replan": True,
    "no_replan_round_4": True,
    "allowed_domains": list(ETOM_DOMAINS),
}

SEMANTIC_PLAN_REPAIR_POLICY = {
    "version": "pe_fin_replan_etom_semantic_plan_repair_v1",
    "max_repairs": 1,
    "shared_with_schema_repair": True,
    "max_new_steps_per_replan": 2,
    "no_silent_truncate": True,
    "no_runtime_step_deletion": True,
    "revalidate_schema_and_semantics": True,
    "counts_toward_llm_budget": True,
}

_DECISION_REASON = {"type": ["string", "null"], "maxLength": 400}

REPLANNER_PLAN_BRANCH: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["schema_version", "action", "remaining_steps"],
    "properties": {
        "schema_version": {"type": "string", "const": REPLAN_SCHEMA_VERSION},
        "action": {"type": "string", "const": "plan"},
        "decision_reason": _DECISION_REASON,
        "remaining_steps": {
            "type": "array",
            "minItems": 0,
            "maxItems": MAX_REMAINING_STEPS,
            "items": ETOM_PLAN_STEP_SCHEMA,
        },
    },
}

_RESPONSE_OBJECT: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["completion", "final_answer", "reasoning_brief"],
    "properties": {
        "completion": {"type": "string", "enum": list(COMPLETIONS)},
        "final_answer": {"type": "string", "maxLength": 4000},
        "reasoning_brief": {"type": "string", "maxLength": 800},
    },
}

REPLANNER_RESPONSE_BRANCH: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["schema_version", "action", "response"],
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
REPLANNER_RESPONSE_ONLY_SCHEMA: Dict[str, Any] = {"anyOf": [REPLANNER_RESPONSE_BRANCH]}

BANNED_RESPONSE_FIELDS = (
    "status",
    "clarification_needed",
    "refusal_reason",
    "clarification_question",
    "limitations",
)

RETRIEVE_STEP_TEMPLATE: Dict[str, Any] = {
    "id": "s1",
    "type": "retrieve",
    "domain": "search",
    "objective": "find the information or perform the lookup that the request needs",
    "required_outputs": ["result summary"],
    "depends_on": [],
    "scope_constraints": {"target": None, "location": None, "time_range": None},
    "compute_spec": None,
}

COMPUTE_STEP_TEMPLATE: Dict[str, Any] = {
    "id": "s2",
    "type": "compute",
    "domain": "compute",
    "objective": "compute a derived number from verified operand refs",
    "required_outputs": ["derived value"],
    "depends_on": ["s1"],
    "scope_constraints": None,
    "compute_spec": {
        "expression": "(a-b)/b",
        "operand_refs": {"a": "s1.value_a", "b": "s1.value_b"},
        "output_name": "derived_value",
        "output_unit": None,
    },
}

REASON_STEP_TEMPLATE: Dict[str, Any] = {
    "id": "s3",
    "type": "reason",
    "domain": "reason",
    "objective": "summarize verified evidence (or the absence of a suitable tool)",
    "required_outputs": [],
    "depends_on": ["s1"],
    "scope_constraints": None,
    "compute_spec": None,
}

PLAN_TEMPLATE: Dict[str, Any] = {
    "schema_version": REPLAN_SCHEMA_VERSION,
    "action": "plan",
    "decision_reason": "optional short reason",
    "remaining_steps": [RETRIEVE_STEP_TEMPLATE, COMPUTE_STEP_TEMPLATE, REASON_STEP_TEMPLATE],
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
        "final_answer": "The request cannot be completed with the available evidence or tools.",
        "reasoning_brief": "short explanation",
    },
}

PLANNER_PLAN_TEMPLATE: Dict[str, Any] = {
    "steps": [RETRIEVE_STEP_TEMPLATE, COMPUTE_STEP_TEMPLATE, REASON_STEP_TEMPLATE]
}


def _sha(obj: Any) -> str:
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def schema_set_hash() -> str:
    return _sha(
        {
            "version": SCHEMA_SET_VERSION,
            "domains": list(ETOM_DOMAINS),
            "step_types": list(STEP_TYPES),
            "plan": ETOM_PLAN_V1,
            "tool_binding": ETOM_BINDER_V1,
            "reason": ETOM_REASON_V1,
            "versions": SCHEMA_VERSIONS,
        }
    )


def replanner_schema_hash() -> str:
    return _sha({"schema_version": REPLAN_SCHEMA_VERSION, "schema": REPLANNER_OUTPUT_SCHEMA})


def loop_policy_hash() -> str:
    return _sha(LOOP_POLICY)


def semantic_plan_repair_policy_hash() -> str:
    return _sha(SEMANTIC_PLAN_REPAIR_POLICY)


# ---------------------------------------------------------------------------
# JSON parsing + canonicalization
# ---------------------------------------------------------------------------


def extract_json_object(text: Optional[str]) -> Any:
    if text is None:
        raise SchemaValidationError("null content")
    if not isinstance(text, str) or not text.strip():
        raise SchemaValidationError("empty content")
    raw, _applied = normalize_json_text(text)
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end <= start:
        raise SchemaValidationError("no JSON object found (truncated or non-JSON content)")
    try:
        return json.loads(raw[start : end + 1])
    except json.JSONDecodeError as exc:
        raise SchemaValidationError(f"invalid JSON: {exc}") from exc


_TYPE_ALIASES = {"analysis": "reason", "reasoning": "reason", "calculation": "compute"}
_DOMAIN_ALIASES = {
    "filesystem": "read",
    "file": "read",
    "files": "list",
    "web": "search",
    "browser": "search",
    "documentation": "docs",
    "documents": "docs",
    "database": "data",
    "code_execution": "code",
    "messaging": "communication",
    "email": "communication",
    "date": "calendar",
}


def canonicalize_step(step: Any) -> Any:
    """Unambiguous structural fixes only. Never invents semantic content."""
    if not isinstance(step, dict):
        return step
    out = dict(step)
    if "step_id" in out and "id" not in out:
        out["id"] = out.pop("step_id")
    if "step_type" in out and "type" not in out:
        out["type"] = out.pop("step_type")
    stype = out.get("type")
    if isinstance(stype, str) and stype in _TYPE_ALIASES:
        out["type"] = _TYPE_ALIASES[stype]
    dom = out.get("domain")
    if isinstance(dom, str) and dom.lower() in _DOMAIN_ALIASES:
        out["domain"] = _DOMAIN_ALIASES[dom.lower()]
    elif isinstance(dom, str) and dom != dom.lower() and dom.lower() in ETOM_DOMAINS:
        out["domain"] = dom.lower()
    if "depends_on" not in out and not out.get("compute_spec"):
        out["depends_on"] = []
    return out


def canonicalize_plan_payload(payload: Any) -> Any:
    if not isinstance(payload, dict):
        return payload
    out = dict(payload)
    if isinstance(out.get("steps"), list):
        out["steps"] = [canonicalize_step(s) for s in out["steps"]]
    return out


def canonicalize_replanner_payload(payload: Any) -> Any:
    if not isinstance(payload, dict):
        return payload
    out = dict(payload)
    if isinstance(out.get("remaining_steps"), list):
        out["remaining_steps"] = [canonicalize_step(s) for s in out["remaining_steps"]]
    return out


def step_canonicalizer_hash() -> str:
    return _sha(
        {
            "version": "pe_fin_replan_etom_step_canonicalizer_v1",
            "type_aliases": _TYPE_ALIASES,
            "domain_aliases": _DOMAIN_ALIASES,
            "rules": [
                "step_id_to_id",
                "step_type_to_type",
                "missing_depends_on_to_empty_array_when_no_compute_spec",
                "no_objective_rewrite",
                "no_required_outputs_invention",
                "no_operation_id",
                "no_task_or_gold_branching",
                "no_json_splice",
            ],
        }
    )


# ---------------------------------------------------------------------------
# Plan semantics
# ---------------------------------------------------------------------------


def _scan_banned_keys(node: Any, path: str = "$") -> Optional[str]:
    if isinstance(node, dict):
        for key, value in node.items():
            if str(key).lower() in BANNED_PLAN_KEYS:
                return f"{path}.{key}"
            hit = _scan_banned_keys(value, f"{path}.{key}")
            if hit:
                return hit
    elif isinstance(node, list):
        for i, item in enumerate(node):
            hit = _scan_banned_keys(item, f"{path}[{i}]")
            if hit:
                return hit
    return None


def validate_steps_semantics(
    steps: Sequence[Dict[str, Any]],
    *,
    allow_empty: bool = False,
    allow_external_deps: bool = False,
) -> None:
    """Plan semantics. ``allow_external_deps`` is for Replanner ``remaining_steps``,
    whose steps may depend on already-executed steps that are no longer in the list
    (the agent re-validates those refs against executed ids)."""
    if not steps:
        if allow_empty:
            return
        raise PlanValidationError("plan must contain at least one step")
    ids = [str(s.get("id") or "") for s in steps]
    if len(ids) != len(set(ids)):
        raise PlanValidationError("duplicate step ids")
    id_set = set(ids)
    banned_hit = _scan_banned_keys(list(steps))
    if banned_hit:
        raise PlanValidationError(f"plan must not contain banned key at {banned_hit}")
    for step in steps:
        sid = str(step["id"])
        for dep in step.get("depends_on") or []:
            if dep not in id_set and not allow_external_deps:
                raise PlanValidationError(f"step {sid} depends on unknown {dep}")
            if dep == sid:
                raise PlanValidationError(f"step {sid} depends on itself")
        stype = step.get("type")
        if stype == "retrieve":
            if step.get("compute_spec") is not None:
                raise PlanValidationError(f"retrieve step {sid} must not have compute_spec")
            if not str(step.get("objective") or "").strip():
                raise PlanValidationError(f"retrieve step {sid} missing objective")
        elif stype == "compute":
            spec = step.get("compute_spec")
            if not isinstance(spec, dict):
                raise PlanValidationError(
                    f"compute step {sid} missing compute_spec "
                    "(expression + operand_refs + output_name)"
                )
            if not str(spec.get("expression") or "").strip():
                raise PlanValidationError(f"compute step {sid} missing expression")
            refs = spec.get("operand_refs")
            if not isinstance(refs, dict) or not refs:
                raise PlanValidationError(f"compute step {sid} missing operand_refs")
            for name, ref in refs.items():
                if not isinstance(name, str) or not name.strip():
                    raise PlanValidationError(f"compute step {sid} bad operand name")
                if not isinstance(ref, str) or not ref.strip():
                    raise PlanValidationError(
                        f"compute step {sid} operand_refs[{name!r}] must be a non-empty ref"
                    )
            if not (step.get("depends_on") or []):
                raise PlanValidationError(
                    f"compute step {sid} must depend on prior evidence steps"
                )
        elif stype == "reason":
            if step.get("compute_spec") is not None:
                raise PlanValidationError(f"reason step {sid} must not have compute_spec")
    if _has_cycle(list(steps)):
        raise PlanValidationError("plan DAG contains a cycle")


def validate_plan_semantics(payload: Dict[str, Any]) -> None:
    steps = payload.get("steps") or []
    if len(steps) > MAX_PLAN_STEPS:
        raise PlanValidationError(f"plan exceeds max {MAX_PLAN_STEPS} steps")
    validate_steps_semantics(steps)


def _validate_payload(payload: Any, schema_name: str) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise SchemaValidationError("payload must be a JSON object")
    semantic = {k: v for k, v in payload.items() if k not in ("schema_version", "method_version", "protocol_version")}
    if schema_name == "plan":
        semantic = canonicalize_plan_payload(semantic)
    validate_against_schema(semantic, SCHEMA_BY_NAME[schema_name])
    if schema_name == "plan":
        validate_plan_semantics(semantic)
    out = dict(semantic)
    out["schema_version"] = SCHEMA_VERSIONS[schema_name]
    return out


def validate_plan_payload(payload: Any) -> Dict[str, Any]:
    """Validate a Planner payload (schema + semantics). Returns the canonical payload."""
    return _validate_payload(payload, "plan")


def parse_and_validate(
    content: Optional[str],
    schema_name: str,
    *,
    repair_content: Optional[str] = None,
) -> Tuple[Optional[Dict[str, Any]], int, Optional[str]]:
    """Return (payload, repair_count_used, error). Full-object only, no splice."""
    try:
        return _validate_payload(extract_json_object(content), schema_name), 0, None
    except (SchemaValidationError, PlanValidationError) as exc:
        err = str(exc)
    if repair_content is None:
        return None, 0, err
    try:
        return _validate_payload(extract_json_object(repair_content), schema_name), 1, None
    except (SchemaValidationError, PlanValidationError) as exc:
        return None, 1, str(exc)


def format_repair_user_message(
    schema_name: str, error: str, *, previous_content: Optional[str] = None
) -> str:
    lines = [
        f"Your previous JSON failed validation for schema {schema_name}.",
        f"Error: {error}",
        "Regenerate a COMPLETE closed JSON object (full re-generation, no splice, "
        "no truncation of overlength strings).",
    ]
    if schema_name == "plan":
        lines.extend(
            [
                "Every step MUST contain id, type, domain, objective, required_outputs, depends_on.",
                f"type must be one of {list(STEP_TYPES)}; domain must be one of {list(ETOM_DOMAINS)}.",
                "required_outputs items MUST be short noun phrases "
                f"(max {REQUIRED_OUTPUT_ITEM_MAX_LENGTH} characters each). "
                "Do NOT write full explanations, reasoning, or tool names. "
                "Do NOT include operation_id.",
                "Compute steps need compute_spec{expression, operand_refs, output_name} and depends_on.",
            ]
        )
    elif schema_name == "tool_binding":
        lines.extend(
            [
                "primary.operation_id MUST be one of the described candidate operation ids.",
                "arguments MUST satisfy the described input schema: include every required "
                "argument with the right type; never return an empty {} when required "
                "arguments exist.",
                "fallback is optional (object or null) and must also be a described candidate.",
            ]
        )
    elif schema_name == "reason":
        lines.append('Return {"summary": "...", "claims": [...], "gaps": [...]}.')
    lines.append("Return one corrected JSON object only, no markdown.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Replanner validation / parsing
# ---------------------------------------------------------------------------


def normalize_decision_reason(payload: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(payload)
    if "decision_reason" not in out:
        out["decision_reason"] = None
    return out


def validate_replanner_payload(payload: Any, *, response_only: bool = False) -> None:
    if not isinstance(payload, dict):
        raise SchemaValidationError("replanner payload must be object")
    if payload.get("schema_version") != REPLAN_SCHEMA_VERSION:
        raise SchemaValidationError("schema_version mismatch")
    action = payload.get("action")
    if response_only and action != "response":
        raise SchemaValidationError("final_round_requires_action_response")
    schema = REPLANNER_RESPONSE_ONLY_SCHEMA if response_only else REPLANNER_OUTPUT_SCHEMA
    validate_against_schema(payload, schema, "$")
    if action == "plan":
        if "response" in payload:
            raise SchemaValidationError("plan action must not include response")
        if not isinstance(payload.get("remaining_steps"), list):
            raise SchemaValidationError("remaining_steps must be array")
    elif action == "response":
        if "remaining_steps" in payload:
            raise SchemaValidationError("response action must not include remaining_steps")
        resp = payload.get("response") or {}
        for banned in BANNED_RESPONSE_FIELDS:
            if banned in resp:
                raise SchemaValidationError(f"banned_response_field:{banned}")
        if resp.get("completion") not in COMPLETIONS:
            raise SchemaValidationError("invalid response.completion")
    else:
        raise SchemaValidationError(f"unknown action: {action}")


def parse_replanner_output(
    content: Optional[str],
    *,
    repair_content: Optional[str] = None,
    response_only: bool = False,
) -> Tuple[Optional[Dict[str, Any]], Optional[str], Optional[str]]:
    candidates = [content]
    if repair_content is not None:
        candidates.append(repair_content)
    last_err: Optional[str] = None
    for raw in candidates:
        try:
            obj = extract_json_object(str(raw or ""))
        except Exception as exc:  # noqa: BLE001
            last_err = f"json_extract:{exc}"
            continue
        try:
            obj = canonicalize_replanner_payload(obj)
            validate_replanner_payload(obj, response_only=response_only)
            if obj.get("action") == "plan":
                validate_steps_semantics(
                    obj.get("remaining_steps") or [],
                    allow_empty=True,
                    allow_external_deps=True,
                )
            return normalize_decision_reason(obj), None, None
        except (SchemaValidationError, PlanValidationError) as exc:
            last_err = str(exc)
            continue
    return None, None, last_err or "invalid_replanner_output"


def _template_block(*, response_only: bool) -> str:
    if response_only:
        return (
            "Required shape (FINAL ROUND - action=response only):\n"
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


def format_replanner_repair_message(
    error: str, *, response_only: bool = False, json_path: str = "$"
) -> str:
    allowed = "action=response only" if response_only else "action=plan|response"
    return (
        "Your previous Replanner output failed schema validation.\n"
        f"JSON path: {json_path}\n"
        f"Validation error: {error}\n"
        f"Current round response_only={bool(response_only)}\n"
        f"Allowed actions: {allowed}\n"
        f"schema_version must be {REPLAN_SCHEMA_VERSION}.\n"
        f"Each PlanStep requires id, type, domain (one of {list(ETOM_DOMAINS)}), "
        "objective, required_outputs (short noun phrases, "
        f"max {REQUIRED_OUTPUT_ITEM_MAX_LENGTH} chars each; no explanations/tool names), "
        "depends_on. Never emit operation_id.\n"
        "decision_reason is OPTIONAL (string|null).\n"
        "Disallowed response fields: status, clarification_needed, refusal_reason, "
        "clarification_question, limitations.\n"
        "Regenerate ONE complete JSON object. No markdown. No splice.\n\n"
        + _template_block(response_only=response_only)
    )


def format_semantic_plan_repair_message(
    error: str, *, max_new_steps: int = 2, execution_round: int = 0, response_only: bool = False
) -> str:
    return (
        "Your previous Replanner Plan failed plan semantic validation.\n"
        f"Semantic error: {error}\n"
        f"max_new_steps_per_replan={int(max_new_steps)}\n"
        f"current execution round={int(execution_round)}\n"
        f"Current round response_only={bool(response_only)}\n"
        "Do NOT silently truncate steps. Do NOT rely on runtime to delete steps. "
        f"Regenerate one complete Plan JSON with at most {int(max_new_steps)} newly "
        "added steps versus the previous remaining plan.\n"
        "decision_reason is OPTIONAL (string|null).\n"
        "Regenerate ONE complete JSON object. No markdown. No splice.\n\n"
        + _template_block(response_only=False)
    )


def map_completion_to_predicted_status(completion: str) -> str:
    return "answered" if completion == "answered" else "unavailable"


def exported_schemas() -> Dict[str, Dict[str, Any]]:
    out = {name: deepcopy(schema) for name, schema in SCHEMA_BY_NAME.items()}
    out["replanner"] = deepcopy(REPLANNER_OUTPUT_SCHEMA)
    return out
