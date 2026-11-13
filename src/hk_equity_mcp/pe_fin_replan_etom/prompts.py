"""Planner / Binder / Reason / Replanner prompts for PE-Fin RePlan-ETOM v1.

No gold, no task ids, no accepted-tool hints. Prompts are static text; only the
user question and runtime evidence are interpolated into user messages.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Dict, List, Sequence

from hk_equity_mcp.pe_fin_replan_etom.schemas import (
    COMPUTE_STEP_TEMPLATE,
    ETOM_DOMAINS,
    PLAN_TEMPLATE,
    PLANNER_PLAN_TEMPLATE,
    REASON_STEP_TEMPLATE,
    REPLAN_SCHEMA_VERSION,
    REQUIRED_OUTPUT_ITEM_MAX_LENGTH,
    RESPONSE_ANSWERED_TEMPLATE,
    RESPONSE_INSUFFICIENT_TEMPLATE,
    RETRIEVE_STEP_TEMPLATE,
)

PROMPT_SET_VERSION = "pe_fin_replan_etom_prompt_set_v2"

SYSTEM_PROMPT = """You are the PE-Fin RePlan-ETOM agent (Planner / Executor / Replanner).
This is a tool-routing evaluation over a large external MCP tool catalog.
No external tool is actually executed: a tool call only records a predicted route
node and returns no business data. Never invent tool outputs, records, files,
statuses, or numbers. Never reference gold answers, accepted tools, or task ids.
Tools are selected only by retrieval + a described-only binder; do not name tools.
Return JSON only. No markdown. No splice."""

PLANNER_SYSTEM_PROMPT = SYSTEM_PROMPT + """

You are the Planner. You run exactly once. Produce a short plan of semantic steps.
Step types: retrieve (use an external tool), compute (safe arithmetic over verified
numbers), reason (summarize verified evidence).
- Use the smallest plan that can satisfy the request (1-6 steps).
- Never emit operation_id, tool names, or server names in the plan.
- domain is a generic capability area, one of: """ + ", ".join(ETOM_DOMAINS) + """.
- required_outputs are short free-text noun phrases describing what the step must
  yield. Each item is at most """ + str(REQUIRED_OUTPUT_ITEM_MAX_LENGTH) + """ characters.
  Do not write full explanations, reasoning text, or tool names in required_outputs.
- A later step that needs an earlier result lists it in depends_on.
- Put concrete values from the question (names, paths, urls, dates) into the step
  objective or scope_constraints; do not invent values.
- If NO available capability can satisfy the request, return a single reason step
  and do not plan any retrieve step.
- Each retrieve step maps to exactly one tool call later; do not bundle several
  independent lookups into one step."""

REPLANNER_SYSTEM_PROMPT = SYSTEM_PROMPT + f"""

You are the Replanner (schema_version={REPLAN_SCHEMA_VERSION}).
After every executed step you either update the remaining plan (action=plan) or
finish (action=response). You are the only component that writes the final response.
Plan template:
{json.dumps(PLAN_TEMPLATE, ensure_ascii=False, indent=2)}

Response template (answered):
{json.dumps(RESPONSE_ANSWERED_TEMPLATE, ensure_ascii=False, indent=2)}

Response template (insufficient_evidence):
{json.dumps(RESPONSE_INSUFFICIENT_TEMPLATE, ensure_ascii=False, indent=2)}

Rules:
- schema_version must be {REPLAN_SCHEMA_VERSION}; additionalProperties=false.
- decision_reason is OPTIONAL (string or null).
- remaining_steps use PlanStep fields id/type/domain/objective/required_outputs/depends_on.
- required_outputs items MUST be short noun phrases (max {REQUIRED_OUTPUT_ITEM_MAX_LENGTH}
  characters each); no full explanations, reasoning, or tool names.
- domain is one of: {", ".join(ETOM_DOMAINS)}.
- Never emit operation_id. Do not invent evidence or numbers.
- Prefer continuing the plan while answer-bearing steps remain; prefer
  action=response (answered) when the request has been routed/answered.
- Use completion=insufficient_evidence only when the request is infeasible or exhausted.
- Do not emit clarification_needed, refusal_reason, clarification_question, status.
- Do not repeat a step that already completed, and do not re-issue a call that failed
  with a blocked signature.
- Return JSON only."""

REPLANNER_FINAL_ROUND_SYSTEM_PROMPT = SYSTEM_PROMPT + f"""

You are the Replanner on the FINAL round (schema_version={REPLAN_SCHEMA_VERSION}).
action MUST be response; plan is forbidden. completion is answered or insufficient_evidence.
Response template (answered):
{json.dumps(RESPONSE_ANSWERED_TEMPLATE, ensure_ascii=False, indent=2)}

Response template (insufficient_evidence):
{json.dumps(RESPONSE_INSUFFICIENT_TEMPLATE, ensure_ascii=False, indent=2)}
Do not invent numbers. No clarification/refusal fields. Return JSON only."""

BINDER_SYSTEM_PROMPT = SYSTEM_PROMPT + """

You are the Tool Binder. Given ONE retrieve step and a few DESCRIBED candidate
operations, choose a primary operation_id and arguments, and optionally one fallback.
- Choose ONLY from the described candidates. Never invent an operation_id.
- arguments MUST satisfy the candidate's input schema: include every required
  argument with the correct type. Never return an empty {} when required arguments
  exist. Use values from the question or verified evidence only.
- fallback is optional (object or null).
Output: {"primary": {"operation_id": "...", "arguments": {...}}, "fallback": null, "notes": null}"""

REASON_SYSTEM_PROMPT = SYSTEM_PROMPT + """

You are the Reasoner. Summarize ONLY verified evidence. If evidence is missing,
say so in gaps. Output: {"summary": "...", "claims": [], "gaps": []}"""

_BANNED_PROMPT_MARKERS = (
    "ground_truth",
    "gold_tool",
    "gold_operation",
    "accepted_tools",
    "reference_graph",
    "hkex",
    "eastmoney",
    "sina",
    "stock_code",
    "hk155",
    "benchmark_cutoff",
)


def _all_static_prompts() -> Dict[str, str]:
    return {
        "system": SYSTEM_PROMPT,
        "planner_system": PLANNER_SYSTEM_PROMPT,
        "replanner_system": REPLANNER_SYSTEM_PROMPT,
        "replanner_final_round_system": REPLANNER_FINAL_ROUND_SYSTEM_PROMPT,
        "binder_system": BINDER_SYSTEM_PROMPT,
        "reason_system": REASON_SYSTEM_PROMPT,
    }


def assert_prompts_clean() -> None:
    for name, text in _all_static_prompts().items():
        lowered = text.lower()
        for marker in _BANNED_PROMPT_MARKERS:
            if marker in lowered:
                raise RuntimeError(f"ETOM prompt {name} contains banned marker {marker!r}")
        if re.search(r"\bL[1-5]-q\d+\b", text):
            raise RuntimeError(f"ETOM prompt {name} contains a task-id-like token")


def prompt_hashes() -> Dict[str, str]:
    return {
        name: hashlib.sha256(text.encode("utf-8")).hexdigest()
        for name, text in _all_static_prompts().items()
    }


def pe_fin_replan_etom_prompt_hash() -> str:
    blob = json.dumps(
        {
            "version": PROMPT_SET_VERSION,
            "prompts": prompt_hashes(),
            "plan_template": PLANNER_PLAN_TEMPLATE,
            "retrieve_step_template": RETRIEVE_STEP_TEMPLATE,
            "compute_step_template": COMPUTE_STEP_TEMPLATE,
            "reason_step_template": REASON_STEP_TEMPLATE,
            "allowed_domains": list(ETOM_DOMAINS),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def etom_prompt_bundle_hash() -> str:
    """Alias used by identity/resume payloads."""
    return pe_fin_replan_etom_prompt_hash()


etom_prompt_bundle_hash = pe_fin_replan_etom_prompt_hash


def replanner_prompt_hash() -> str:
    blob = json.dumps(
        {
            "version": PROMPT_SET_VERSION,
            "system": REPLANNER_SYSTEM_PROMPT,
            "final": REPLANNER_FINAL_ROUND_SYSTEM_PROMPT,
            "plan_template": PLAN_TEMPLATE,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def render_planner_user(
    *,
    question: str,
    capability_profile: Dict[str, Any],
    limits: Dict[str, Any],
    max_plan_steps: int = 6,
) -> str:
    payload = {
        "role": "planner_input",
        "question": question,
        "capability_profile": capability_profile,
        "limits": {
            "max_plan_steps": int(max_plan_steps),
            "max_total_llm_calls_per_task": limits.get("max_total_llm_calls_per_task"),
            "max_total_mcp_invocations": limits.get("max_total_mcp_invocations"),
            "max_execution_rounds": limits.get("max_execution_rounds"),
            "no_global_replan": True,
            "no_monitor": True,
        },
        "output_template": PLANNER_PLAN_TEMPLATE,
        "instructions": [
            "Return one JSON object with a steps array.",
            "Do not output operation_id, tool names, or server names.",
            "The question gives scope only; never fabricate observed values.",
            (
                "required_outputs items MUST be short noun phrases "
                f"(max {REQUIRED_OUTPUT_ITEM_MAX_LENGTH} characters each); "
                "no full explanations, reasoning, or tool names."
            ),
        ],
    }
    return "Produce the plan as ONE complete JSON object.\n" + json.dumps(
        payload, ensure_ascii=False, indent=2
    )


def render_binder_user(
    *,
    question: str,
    step: Dict[str, Any],
    candidates: Sequence[Dict[str, Any]],
    evidence: Dict[str, Any],
) -> str:
    payload = {
        "role": "binder_input",
        "question": question,
        "step": {
            k: step.get(k)
            for k in ("id", "type", "domain", "objective", "required_outputs", "scope_constraints")
        },
        "described_candidates": list(candidates),
        "verified_evidence": evidence,
        "instructions": [
            "primary.operation_id MUST be one of described_candidates[*].operation_id.",
            "Fill every required argument of the chosen candidate; never return {} when required args exist.",
            "fallback is optional.",
        ],
    }
    return "Bind the step to a described candidate.\n" + json.dumps(
        payload, ensure_ascii=False, indent=2
    )


def render_reason_user(*, question: str, step: Dict[str, Any], evidence: Dict[str, Any]) -> str:
    payload = {
        "role": "reason_input",
        "question": question,
        "step": {k: step.get(k) for k in ("id", "objective", "required_outputs")},
        "verified_evidence": evidence,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def render_replanner_user(
    *,
    compact_state: Dict[str, Any],
    response_only: bool = False,
    execution_round: int = 0,
    max_execution_rounds: int = 7,
) -> str:
    choose: List[str] = ["response"] if response_only else ["plan", "response"]
    payload = {
        "role": "replanner_input",
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
            "allowed_domains": list(ETOM_DOMAINS),
            "plan_step_required_fields": [
                "id",
                "type",
                "domain",
                "objective",
                "required_outputs",
                "depends_on",
            ],
        },
    }
    header = (
        "FINAL ROUND: produce ONE complete JSON object using a Response template "
        "(action=response only).\n"
        if response_only
        else "Update the remaining plan or produce the final response using ONE complete JSON object.\n"
    )
    return header + json.dumps(payload, ensure_ascii=False, indent=2)
