"""Versioned structural Replanner step canonicalizer v2 (unambiguous rules only).

Must run before strict schema validation. Never invents answers, operations,
required_outputs, entities, periods, dates, or non-root depends_on.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

from hk_equity_mcp.pe_fin_lite.plan_canonicalizer import normalize_json_text

CANONICALIZER_VERSION = "pe_fin_replan_step_canonicalizer_v2"

_TYPE_ALIASES = {
    "analysis": "reason",
    "reasoning": "reason",
    "calculation": "compute",
}

_DOMAIN_ALIASES = {
    "filing": "disclosure",
}


class CanonicalizerConflictError(ValueError):
    """Raised when alias and canonical values conflict — fail closed."""


def step_canonicalizer_hash() -> str:
    payload = {
        "version": CANONICALIZER_VERSION,
        "type_aliases": dict(_TYPE_ALIASES),
        "domain_aliases": dict(_DOMAIN_ALIASES),
        "scope_aliases": {
            "ticker": "scope_constraints.entity.code",
            "entity_name": "scope_constraints.entity.name",
            "adjust": "scope_constraints.adjustment",
        },
        "rules": [
            "step_id_to_id",
            "step_type_to_type",
            "first_or_root_step_missing_depends_on_to_empty_array",
            "type_alias_analysis_reasoning_to_reason",
            "type_alias_calculation_to_compute",
            "domain_alias_filing_to_disclosure",
            "top_level_entity_dict_to_scope_constraints_entity",
            "scope_ticker_to_entity_code",
            "scope_entity_name_to_entity_name",
            "scope_adjust_to_adjustment",
            "conflict_alias_fail_closed",
            "no_overwrite_existing_canonical",
            "json_text_whitespace_fence_unicode_quote_trailing_comma",
            "no_objective_rewrite",
            "no_required_outputs_invention",
            "no_entity_period_date_invention",
            "no_operation_id",
            "no_task_or_gold_branching",
            "no_silent_truncate",
            "no_json_splice",
            "no_free_text_scope_guessing",
            "unknown_semantic_fields_fail_closed_after_validate",
        ],
    }
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def canonicalize_replanner_json_text(text: Optional[str]) -> Tuple[str, List[str]]:
    return normalize_json_text(text)


def _values_equal(a: Any, b: Any) -> bool:
    return json.dumps(a, sort_keys=True, default=str) == json.dumps(
        b, sort_keys=True, default=str
    )


def _ensure_scope_dict(step: Dict[str, Any]) -> Dict[str, Any]:
    scope = step.get("scope_constraints")
    if scope is None:
        step["scope_constraints"] = {}
        return step["scope_constraints"]
    if isinstance(scope, str):
        # Free-text scope string: do not guess meaning; leave for schema fail-closed.
        raise CanonicalizerConflictError(
            "scope_constraints_free_text_string_not_canonicalized"
        )
    if not isinstance(scope, dict):
        raise CanonicalizerConflictError("scope_constraints_must_be_object_or_null")
    return scope


def _ensure_entity_object(scope: Dict[str, Any]) -> Dict[str, Any]:
    ent = scope.get("entity")
    if ent is None:
        scope["entity"] = {}
        return scope["entity"]
    if isinstance(ent, str):
        # Preserve string entity as name if moving structured aliases in.
        scope["entity"] = {"name": ent}
        return scope["entity"]
    if not isinstance(ent, dict):
        raise CanonicalizerConflictError("scope_constraints.entity_invalid_type")
    return ent


def _set_if_absent_or_equal(
    target: Dict[str, Any],
    key: str,
    value: Any,
    *,
    path: str,
    applied: List[str],
    rule: str,
) -> None:
    if key not in target or target[key] in (None, ""):
        target[key] = value
        applied.append(rule)
        return
    if _values_equal(target[key], value):
        applied.append(f"{rule}_redundant")
        return
    raise CanonicalizerConflictError(f"alias_conflict:{path}")


def _canonicalize_scope_aliases(
    step: Dict[str, Any], applied: List[str]
) -> None:
    scope = step.get("scope_constraints")
    if scope is None:
        return
    if isinstance(scope, str):
        # Leave free-text for schema validation (do not guess).
        return
    if not isinstance(scope, dict):
        raise CanonicalizerConflictError("scope_constraints_invalid_type")

    # ticker → entity.code
    if "ticker" in scope:
        ticker = scope.pop("ticker")
        ent = _ensure_entity_object(scope)
        _set_if_absent_or_equal(
            ent,
            "code",
            ticker,
            path="scope_constraints.entity.code",
            applied=applied,
            rule="scope_ticker_to_entity_code",
        )

    # entity_name → entity.name
    if "entity_name" in scope:
        name = scope.pop("entity_name")
        ent = _ensure_entity_object(scope)
        _set_if_absent_or_equal(
            ent,
            "name",
            name,
            path="scope_constraints.entity.name",
            applied=applied,
            rule="scope_entity_name_to_entity_name",
        )

    # adjust → adjustment
    if "adjust" in scope:
        adj = scope.pop("adjust")
        _set_if_absent_or_equal(
            scope,
            "adjustment",
            adj,
            path="scope_constraints.adjustment",
            applied=applied,
            rule="scope_adjust_to_adjustment",
        )


def _canonicalize_top_level_entity(
    step: Dict[str, Any], applied: List[str]
) -> None:
    ent = step.get("entity")
    if not isinstance(ent, dict):
        return
    scope = _ensure_scope_dict(step)
    existing = scope.get("entity")
    if existing in (None, "", {}):
        scope["entity"] = deepcopy(ent)
        step["entity"] = None
        applied.append("top_level_entity_dict_to_scope_constraints_entity")
        return
    if isinstance(existing, str):
        # Conflict unless dict only adds the same name
        if list(ent.keys()) == ["name"] and _values_equal(ent.get("name"), existing):
            scope["entity"] = deepcopy(ent)
            step["entity"] = None
            applied.append("top_level_entity_dict_merge_string_name")
            return
        raise CanonicalizerConflictError(
            "alias_conflict:scope_constraints.entity_vs_top_level_entity"
        )
    if isinstance(existing, dict):
        # Merge only absent keys; conflict on differing values
        merged = deepcopy(existing)
        for k, v in ent.items():
            if k not in merged or merged[k] in (None, ""):
                merged[k] = v
            elif not _values_equal(merged[k], v):
                raise CanonicalizerConflictError(
                    f"alias_conflict:scope_constraints.entity.{k}"
                )
        scope["entity"] = merged
        step["entity"] = None
        applied.append("top_level_entity_dict_merged_into_scope")
        return
    raise CanonicalizerConflictError("alias_conflict:scope_constraints.entity_type")


def _canonicalize_one_step(
    step: Dict[str, Any],
    *,
    is_first: bool,
) -> Tuple[Dict[str, Any], List[str]]:
    out = dict(step)
    applied: List[str] = []

    if "id" not in out and "step_id" in out:
        out["id"] = out.pop("step_id")
        applied.append("step_id_to_id")
    elif "step_id" in out and "id" in out:
        out.pop("step_id", None)
        applied.append("drop_alias_step_id")

    if "type" not in out and "step_type" in out:
        out["type"] = out.pop("step_type")
        applied.append("step_type_to_type")
    elif "step_type" in out and "type" in out:
        out.pop("step_type", None)
        applied.append("drop_alias_step_type")

    stype = out.get("type")
    if isinstance(stype, str):
        alias = _TYPE_ALIASES.get(stype.strip().lower())
        if alias is not None:
            out["type"] = alias
            applied.append(f"type_alias_{stype.strip().lower()}_to_{alias}")

    domain = out.get("domain")
    if isinstance(domain, str):
        d_alias = _DOMAIN_ALIASES.get(domain.strip().lower())
        if d_alias is not None:
            out["domain"] = d_alias
            applied.append(f"domain_alias_{domain.strip().lower()}_to_{d_alias}")

    if is_first and "depends_on" not in out:
        out["depends_on"] = []
        applied.append("first_or_root_step_missing_depends_on_to_empty_array")

    _canonicalize_top_level_entity(out, applied)
    _canonicalize_scope_aliases(out, applied)

    return out, applied


def canonicalize_replanner_payload(
    payload: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[str]]:
    """Apply unambiguous structural rules to Replanner plan/response objects."""
    if not isinstance(payload, dict):
        return payload, []
    out = deepcopy(payload)
    applied: List[str] = []
    steps = out.get("remaining_steps")
    if not isinstance(steps, list):
        return out, applied
    new_steps: List[Any] = []
    for i, step in enumerate(steps):
        if not isinstance(step, dict):
            new_steps.append(step)
            continue
        fixed, rules = _canonicalize_one_step(step, is_first=(i == 0))
        new_steps.append(fixed)
        applied.extend(rules)
    out["remaining_steps"] = new_steps
    return out, applied


def describe_step_canonicalizer() -> Dict[str, Any]:
    return {
        "version": CANONICALIZER_VERSION,
        "hash": step_canonicalizer_hash(),
        "allowed": [
            "step_id→id",
            "step_type→type",
            "first/root missing depends_on→[]",
            "analysis|reasoning→reason",
            "calculation→compute",
            "filing→disclosure",
            "top-level entity dict→scope_constraints.entity",
            "ticker→scope_constraints.entity.code",
            "entity_name→scope_constraints.entity.name",
            "adjust→adjustment",
            "JSON fence / unicode quotes / trailing comma normalization",
        ],
        "forbidden": [
            "overwrite existing canonical values",
            "conflicting alias/canonical (fail closed)",
            "rewrite objective semantics",
            "invent required_outputs",
            "invent entity/period/date",
            "add operation_id",
            "guess free-text scope_constraints strings",
            "branch on task_id/company/gold",
            "silent truncate",
            "JSON splice",
        ],
    }
