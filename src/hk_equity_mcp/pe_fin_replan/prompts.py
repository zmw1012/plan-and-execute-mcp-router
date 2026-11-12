"""Replanner prompts — separate identity from initial Planner prompts."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict

REPLANNER_SYSTEM_PROMPT = """You are the PE-Fin Replanner.
After each executed step you must return exactly one JSON object:
- action=plan with remaining_steps, OR
- action=response with a final response.
Rules:
- Do not invent observed evidence or numeric answers from the question text.
- Do not reference gold answers, accepted operation IDs, or task IDs.
- Do not emit operation_id; retrieval still goes through R5+Binder.
- Prefer continuing the plan when answer-bearing evidence is still missing.
- Prefer action=response when core answer-bearing evidence is sufficient.
- Use unavailable only with a concrete infeasibility reason.
- Optional/context gaps alone must not force unavailable.
- schema_version must be pe_fin_replan_v1; additionalProperties=false.
Return JSON only."""

REPLANNER_PROMPT_VERSION = "pe_fin_replan_replanner_prompt_v1"


def replanner_prompt_hash() -> str:
    blob = json.dumps(
        {
            "version": REPLANNER_PROMPT_VERSION,
            "system": REPLANNER_SYSTEM_PROMPT,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def render_replanner_user(*, compact_state: Dict[str, Any], benchmark_cutoff: str) -> str:
    payload = {
        "role": "replanner_input",
        "benchmark_cutoff": benchmark_cutoff,
        "state": compact_state,
        "instructions": {
            "choose": ["plan", "response"],
            "max_new_steps_hint": 2,
            "no_gold": True,
            "no_operation_id": True,
        },
    }
    return (
        "Update the remaining plan or produce the final response.\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )
