"""Replanner prompts — separate identity from initial Planner prompts."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict

REPLANNER_SYSTEM_PROMPT = """You are the PE-Fin Replanner (v1.1).
After each executed step you must return exactly one JSON object:
- action=plan with remaining_steps, OR
- action=response with completion answered|insufficient_evidence.
Rules:
- Do not invent observed evidence or numeric answers from the question text.
- Do not reference gold answers, accepted operation IDs, or task IDs.
- Do not emit operation_id; retrieval still goes through R5+Binder.
- Prefer continuing the plan when answer-bearing evidence is still missing.
- Prefer action=response with completion=answered when core evidence is sufficient.
- Use completion=insufficient_evidence only when truly exhausted or infeasible.
- Do not give up early while recoverable retrieve/compute paths remain.
- Optional/context gaps alone must not force insufficient_evidence.
- Do not emit clarification_needed, refusal_reason, or clarification_question.
- schema_version must be pe_fin_replan_v1_1; additionalProperties=false.
Return JSON only."""

REPLANNER_FINAL_ROUND_SYSTEM_PROMPT = """You are the PE-Fin Replanner on the FINAL round.
You MUST return action=response only (no plan).
response.completion must be answered or insufficient_evidence.
Do not invent numbers from the question. No clarification/refusal fields.
schema_version=pe_fin_replan_v1_1; additionalProperties=false. JSON only."""

REPLANNER_PROMPT_VERSION = "pe_fin_replan_replanner_prompt_v1_1"


def replanner_prompt_hash() -> str:
    blob = json.dumps(
        {
            "version": REPLANNER_PROMPT_VERSION,
            "system": REPLANNER_SYSTEM_PROMPT,
            "final_round_system": REPLANNER_FINAL_ROUND_SYSTEM_PROMPT,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def render_replanner_user(
    *,
    compact_state: Dict[str, Any],
    benchmark_cutoff: str,
    response_only: bool = False,
    execution_round: int = 0,
    max_execution_rounds: int = 7,
) -> str:
    choose = ["response"] if response_only else ["plan", "response"]
    payload = {
        "role": "replanner_input",
        "benchmark_cutoff": benchmark_cutoff,
        "execution_round": execution_round,
        "max_execution_rounds": max_execution_rounds,
        "forced_final_round": bool(response_only),
        "state": compact_state,
        "instructions": {
            "choose": choose,
            "max_new_steps_hint": 0 if response_only else 2,
            "completion_enum": ["answered", "insufficient_evidence"],
            "no_gold": True,
            "no_operation_id": True,
            "no_clarification_fields": True,
            "no_premature_insufficient": not response_only,
        },
    }
    if response_only:
        header = (
            "FINAL ROUND: produce the final response only "
            "(action=response; completion=answered|insufficient_evidence).\n"
        )
    else:
        header = "Update the remaining plan or produce the final response.\n"
    return header + json.dumps(payload, ensure_ascii=False, indent=2)
