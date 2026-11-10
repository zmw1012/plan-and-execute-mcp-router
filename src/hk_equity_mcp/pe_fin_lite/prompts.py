"""PE-Fin Lite prompts — planner / binder / reason / final.

No gold, reference answers, task_id branching, or full catalog dump.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Sequence

from hk_equity_mcp.experiment.schema import FINAL_ANSWER_CONTRACT

PE_FIN_LITE_SYSTEM_PROMPT = """You are PE-Fin Lite, a careful Hong Kong equity research agent.
Follow the plan-first step-level retrieval protocol.
Never invent numbers, periods, entities, or accounting bases.
Never emit hidden chain-of-thought — return only the requested JSON fields.
Do not hard-code issuer-specific shortcuts."""

PE_FIN_LITE_PLANNER_USER_TEMPLATE = """User question:
{question}

Static capability profile (not a tool list; do not invent operations):
{capability_profile_json}

Execution limits:
{limits_json}

Plan schema rules:
- Return JSON with key "steps" only (optional schema_version).
- Each step: id, type (retrieve|compute|reason), domain, objective, required_outputs, depends_on.
- Retrieve steps are semantic: never write operation_id or concrete MCP tool names.
- For official HKEX disclosure evidence, set workflow to retrieve_official_disclosure_evidence on a retrieve step (domain=disclosure).
- Compute steps must include compute_spec and depend on prior evidence.
- Reason steps only analyze verified evidence; no tools.
- Max {max_plan_steps} steps. DAG must be acyclic.
- Final synthesis is NOT a plan step.

Return one JSON object only."""

PE_FIN_LITE_BINDER_USER_TEMPLATE = """Bind one primary and one fallback tool call for this retrieve step.

Question:
{question}

Step:
{step_json}

Described candidate contracts (you may ONLY select from these operation_ids):
{candidates_json}

Evidence store (verified values only):
{evidence_json}

Rules:
- Return JSON: primary {{operation_id, arguments}}, fallback {{operation_id, arguments}} or null.
- Both operation_ids MUST be from the described candidates.
- Arguments MUST satisfy required schema fields; never return empty {{}} when required args exist.
- Prefer arguments grounded in the question, step fields, or evidence store.
- No secrets. No gold. No invented identifiers.

Return one JSON object only."""

PE_FIN_LITE_REASON_USER_TEMPLATE = """Analyze the verified evidence for this reason step. Do not call tools. Do not invent missing values.

Question:
{question}

Step:
{step_json}

Verified evidence store:
{evidence_json}

Return JSON with keys: summary, claims (optional), gaps (optional)."""

PE_FIN_LITE_FINAL_USER_TEMPLATE = """Produce the final answer under the shared contract.

Question:
{question}

Plan step statuses:
{step_statuses_json}

Verified values / provenance:
{evidence_json}

Compute results:
{compute_json}

Reason artifacts:
{reason_json}

Missing / unavailable notes:
{missing_json}

{FINAL_ANSWER_CONTRACT}

Additional rules:
- If required evidence is missing, use status unavailable (or clarification_needed when ambiguous) — never fabricate.
- Cite actual sources via evidence_ids when present.
- Tool invoke success alone does not imply a correct answer.
- Return one JSON object only."""


def pe_fin_lite_prompt_hash() -> str:
    blob = "\n".join(
        [
            PE_FIN_LITE_SYSTEM_PROMPT,
            PE_FIN_LITE_PLANNER_USER_TEMPLATE,
            PE_FIN_LITE_BINDER_USER_TEMPLATE,
            PE_FIN_LITE_REASON_USER_TEMPLATE,
            PE_FIN_LITE_FINAL_USER_TEMPLATE,
            FINAL_ANSWER_CONTRACT,
        ]
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


_BANNED = (
    "gold_operations",
    "reference_answer",
    "task_id=",
    "expected_tool",
)


def assert_prompts_clean() -> None:
    joined = "\n".join(
        [
            PE_FIN_LITE_SYSTEM_PROMPT,
            PE_FIN_LITE_PLANNER_USER_TEMPLATE,
            PE_FIN_LITE_BINDER_USER_TEMPLATE,
            PE_FIN_LITE_REASON_USER_TEMPLATE,
            PE_FIN_LITE_FINAL_USER_TEMPLATE,
        ]
    )
    for token in _BANNED:
        if token in joined:
            raise RuntimeError(f"PE-Fin Lite prompt contains banned token: {token}")


def render_planner_user(
    *,
    question: str,
    capability_profile: Dict[str, Any],
    limits: Dict[str, Any],
    max_plan_steps: int,
) -> str:
    return PE_FIN_LITE_PLANNER_USER_TEMPLATE.format(
        question=question,
        capability_profile_json=json.dumps(
            capability_profile, ensure_ascii=False, indent=2
        ),
        limits_json=json.dumps(limits, ensure_ascii=False, indent=2),
        max_plan_steps=max_plan_steps,
    )


def render_binder_user(
    *,
    question: str,
    step: Dict[str, Any],
    candidates: Sequence[Dict[str, Any]],
    evidence: Dict[str, Any],
) -> str:
    return PE_FIN_LITE_BINDER_USER_TEMPLATE.format(
        question=question,
        step_json=json.dumps(step, ensure_ascii=False, indent=2),
        candidates_json=json.dumps(list(candidates), ensure_ascii=False, indent=2),
        evidence_json=json.dumps(evidence, ensure_ascii=False, indent=2),
    )


def render_reason_user(
    *,
    question: str,
    step: Dict[str, Any],
    evidence: Dict[str, Any],
) -> str:
    return PE_FIN_LITE_REASON_USER_TEMPLATE.format(
        question=question,
        step_json=json.dumps(step, ensure_ascii=False, indent=2),
        evidence_json=json.dumps(evidence, ensure_ascii=False, indent=2),
    )


def render_final_user(
    *,
    question: str,
    step_statuses: List[Dict[str, Any]],
    evidence: Dict[str, Any],
    compute: List[Dict[str, Any]],
    reason: List[Dict[str, Any]],
    missing: List[str],
) -> str:
    return PE_FIN_LITE_FINAL_USER_TEMPLATE.format(
        question=question,
        step_statuses_json=json.dumps(step_statuses, ensure_ascii=False, indent=2),
        evidence_json=json.dumps(evidence, ensure_ascii=False, indent=2),
        compute_json=json.dumps(compute, ensure_ascii=False, indent=2),
        reason_json=json.dumps(reason, ensure_ascii=False, indent=2),
        missing_json=json.dumps(missing, ensure_ascii=False, indent=2),
        FINAL_ANSWER_CONTRACT=FINAL_ANSWER_CONTRACT,
    )
