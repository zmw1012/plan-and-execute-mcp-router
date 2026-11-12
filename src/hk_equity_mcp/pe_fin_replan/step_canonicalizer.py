"""Versioned structural Replanner step canonicalizer (unambiguous rules only).

Must run before strict schema validation. Never invents answers, operations,
required_outputs, entities, periods, dates, or non-root depends_on.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

from hk_equity_mcp.pe_fin_lite.plan_canonicalizer import normalize_json_text

CANONICALIZER_VERSION = "pe_fin_replan_step_canonicalizer_v1"

# Structural type aliases only (not semantic invention).
_TYPE_ALIASES = {
    "analysis": "reason",
    "reasoning": "reason",
    "calculation": "compute",
}


def step_canonicalizer_hash() -> str:
    payload = {
        "version": CANONICALIZER_VERSION,
        "type_aliases": dict(_TYPE_ALIASES),
        "rules": [
            "step_id_to_id",
            "step_type_to_type",
            "first_or_root_step_missing_depends_on_to_empty_array",
            "type_alias_analysis_reasoning_to_reason",
            "type_alias_calculation_to_compute",
            "json_text_whitespace_fence_unicode_quote_trailing_comma",
            "no_objective_rewrite",
            "no_required_outputs_invention",
            "no_entity_period_date_invention",
            "no_operation_id",
            "no_task_or_gold_branching",
            "no_silent_truncate",
            "no_json_splice",
            "unknown_semantic_fields_fail_closed_after_validate",
        ],
    }
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def canonicalize_replanner_json_text(text: Optional[str]) -> Tuple[str, List[str]]:
    """Reuse Lite text normalizer (fence / unicode quotes / trailing commas)."""
    return normalize_json_text(text)


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
        # Prefer canonical key; drop alias only (same value or not — do not merge inventively).
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

    # Only first/root remaining step may receive depends_on=[] when omitted.
    if is_first and "depends_on" not in out:
        out["depends_on"] = []
        applied.append("first_or_root_step_missing_depends_on_to_empty_array")

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
            "JSON fence / unicode quotes / trailing comma normalization",
        ],
        "forbidden": [
            "rewrite objective semantics",
            "invent required_outputs",
            "invent entity/period/date",
            "add operation_id",
            "branch on task_id/company/gold",
            "silent truncate",
            "JSON splice",
            "accept unknown semantic fields without schema fail",
        ],
    }
