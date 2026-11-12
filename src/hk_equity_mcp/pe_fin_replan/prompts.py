"""Replanner prompts — separate identity from initial Planner prompts."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict

from hk_equity_mcp.pe_fin_replan.schemas import (
    PLAN_TEMPLATE,
    REPLAN_SCHEMA_VERSION,
    RESPONSE_ANSWERED_TEMPLATE,
    RESPONSE_INSUFFICIENT_TEMPLATE,
)

_TEMPLATES_BLOCK = (
    "You must return exactly one complete JSON object matching one template.\n\n"
    "Plan template:\n"
    + json.dumps(PLAN_TEMPLATE, ensure_ascii=False, indent=2)
    + "\n\nResponse template (answered):\n"
    + json.dumps(RESPONSE_ANSWERED_TEMPLATE, ensure_ascii=False, indent=2)
    + "\n\nResponse template (insufficient_evidence):\n"
    + json.dumps(RESPONSE_INSUFFICIENT_TEMPLATE, ensure_ascii=False, indent=2)
)

_FINAL_TEMPLATES_BLOCK = (
    "FINAL ROUND: you must return action=response only.\n\n"
    "Response template (answered):\n"
    + json.dumps(RESPONSE_ANSWERED_TEMPLATE, ensure_ascii=False, indent=2)
    + "\n\nResponse template (insufficient_evidence):\n"
    + json.dumps(RESPONSE_INSUFFICIENT_TEMPLATE, ensure_ascii=False, indent=2)
)

REPLANNER_SYSTEM_PROMPT = f"""You are the PE-Fin Replanner (v1.2).
{_TEMPLATES_BLOCK}

Rules:
- schema_version must be {REPLAN_SCHEMA_VERSION}; additionalProperties=false.
- decision_reason is OPTIONAL (string or null). Omit or null is allowed.
- Do not invent observed evidence or numeric answers from the question text.
- Do not reference gold answers, accepted operation IDs, or task IDs.
- Do not emit operation_id; retrieval still goes through R5+Binder.
- Prefer continuing the plan when answer-bearing evidence is still missing.
- Prefer action=response with completion=answered when core evidence is sufficient.
- Use completion=insufficient_evidence only when truly exhausted or infeasible.
- Do not give up early while recoverable retrieve/compute paths remain.
- Optional/context gaps alone must not force insufficient_evidence.
- Do not emit clarification_needed, refusal_reason, or clarification_question.
- Return JSON only. No markdown. No splice.
"""

REPLANNER_FINAL_ROUND_SYSTEM_PROMPT = f"""You are the PE-Fin Replanner on the FINAL round.
{_FINAL_TEMPLATES_BLOCK}

Rules:
- schema_version must be {REPLAN_SCHEMA_VERSION}; additionalProperties=false.
- action MUST be response (plan is forbidden).
- decision_reason is OPTIONAL (string or null).
- completion must be answered or insufficient_evidence.
- Do not invent numbers from the question.
- No clarification/refusal fields.
- Return JSON only. No markdown. No splice.
"""

REPLANNER_PROMPT_VERSION = "pe_fin_replan_replanner_prompt_v1_2"


def replanner_prompt_hash() -> str:
    blob = json.dumps(
        {
            "version": REPLANNER_PROMPT_VERSION,
            "system": REPLANNER_SYSTEM_PROMPT,
            "final_round_system": REPLANNER_FINAL_ROUND_SYSTEM_PROMPT,
            "plan_template": PLAN_TEMPLATE,
            "response_answered_template": RESPONSE_ANSWERED_TEMPLATE,
            "response_insufficient_template": RESPONSE_INSUFFICIENT_TEMPLATE,
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
            "decision_reason_optional": True,
            "no_gold": True,
            "no_operation_id": True,
            "no_clarification_fields": True,
            "no_premature_insufficient": not response_only,
            "templates": {
                "plan": PLAN_TEMPLATE,
                "response_answered": RESPONSE_ANSWERED_TEMPLATE,
                "response_insufficient": RESPONSE_INSUFFICIENT_TEMPLATE,
            },
        },
    }
    if response_only:
        header = (
            "FINAL ROUND: produce ONE complete JSON object using a Response "
            "template (action=response only).\n"
        )
    else:
        header = (
            "Update the remaining plan or produce the final response using "
            "ONE complete JSON object from the templates.\n"
        )
    return header + json.dumps(payload, ensure_ascii=False, indent=2)
