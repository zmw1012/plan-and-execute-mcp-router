"""PE-Fin Lite prompts — planner / binder / reason / final.

No gold, reference answers, task_id branching, or full catalog dump.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Sequence

from hk_equity_mcp.experiment.schema import FINAL_ANSWER_CONTRACT
from hk_equity_mcp.pe_fin_lite.output_vocabulary import (
    CANONICAL_OUTPUT_FIELDS,
    CANONICAL_OUTPUT_VOCABULARY_VERSION,
    SCOPE_CONSTRAINT_KEYS,
    SCOPE_CONTRACT_VERSION,
)
from hk_equity_mcp.pe_fin_lite.calculator import (
    COMPUTE_FUNCTION_CONTRACT_VERSION,
    SAFE_FUNCTION_NAMES,
)
from hk_equity_mcp.pe_fin_lite.schemas import (
    ADJUSTMENT_MAX_LENGTH,
    COMPUTE_EXPRESSION_MAX_LENGTH,
    CURRENCY_MAX_LENGTH,
    METRIC_BASIS_MAX_LENGTH,
    OBJECTIVE_MAX_LENGTH,
    PLANNER_MAX_TOKENS,
)

PE_FIN_LITE_SYSTEM_PROMPT = """You are PE-Fin Lite, a careful Hong Kong equity research agent.
Follow the plan-first step-level retrieval protocol.
Never invent numbers, periods, entities, or accounting bases.
Never emit hidden chain-of-thought — return only the requested JSON fields.
Do not hard-code issuer-specific shortcuts.
Return compact minified JSON only (no Markdown, no pretty-print)."""

# Compact generic example (not company/task specific). All required step fields present.
_COMPACT_PLAN_EXAMPLE = (
    '{"steps":['
    '{"id":"s1","type":"retrieve","domain":"financial",'
    '"objective":"Retrieve statement metrics for the asked issuer/period.",'
    '"required_outputs":["revenue","basic_eps","currency","report_date"],'
    '"depends_on":[],'
    '"scope_constraints":{"entity":{"name":"Issuer","code":"00000"},'
    '"period":"FY2025","metric_basis":"reported","currency":"HKD"}},'
    '{"id":"s2","type":"compute","domain":"compute",'
    '"objective":"Compute the requested ratio from retrieved metrics.",'
    '"required_outputs":["revenue"],"depends_on":["s1"],'
    '"compute_spec":{"expression":"(a-b)/b","operand_refs":{"a":"s1.revenue","b":"s1.basic_eps"},'
    '"output_name":"ratio","output_unit":"ratio"}},'
    '{"id":"s3","type":"reason","domain":"general",'
    '"objective":"Summarize verified evidence only.",'
    '"required_outputs":[],"depends_on":["s1","s2"]}]}'
)

PE_FIN_LITE_PLANNER_USER_TEMPLATE = """User question:
{question}

Static capability profile (not a tool list; do not invent operations):
{capability_profile_json}

Execution limits:
{limits_json}

Compact plan JSON rules (protocol v3):
- Return ONE minified JSON object only. No Markdown fences. No pretty-print whitespace.
- Top-level key: "steps" only (optional schema_version).
- Every step MUST include all required fields: id, type, domain, objective, required_outputs, depends_on.
- First/root step MUST use depends_on=[]. Later steps list prior step ids.
- domain enum ONLY: financial|market|disclosure|news|compute|general.
  reason → domain=general; compute → domain=compute. Never analysis/reasoning/calculation.
- objective: one short sentence <= {objective_max_length} chars. Do NOT restate the full user question.
  Do NOT repeat all scope/required_outputs inside objective.
- required_outputs: canonical field IDs only from {canonical_outputs_json}
  (vocab {canonical_vocab_version}).
- scope_constraints keys ({scope_contract_version}): {scope_keys_json}
  Use short canonical values. Optional currency maxLength={currency_max_length}.
  metric_basis maxLength={metric_basis_max_length}; adjustment maxLength={adjustment_max_length}.
  Example: {{"entity":{{"name":"Issuer","code":"00000"}},"period":"FY2025","metric_basis":"reported","currency":"HKD","adjustment":""}}
- Omit null optional fields. Do not emit explanations, CoT, or selection reasons.
- Compute steps need compute_spec (expression maxLength={compute_expression_max_length}) and depends_on prior evidence.
  Allowed operators: + - * / ** % and parentheses. Allowed functions ONLY: abs, min, max, round_half_up.
  Operand names in expression must match compute_spec.operand_refs keys (verified evidence only).
  No attribute access, imports, or other function calls.
- For latest/current/today market requests without an explicit historical date: plan for unavailable/refusal under dataset cutoff — do not treat an earlier session as current.
- Disclosure retrieve may set workflow=retrieve_official_disclosure_evidence (domain=disclosure).
- Max {max_plan_steps} steps. JSON must be complete and closed.
- Planner output budget: max_tokens={planner_max_tokens} for this stage only.
- If repairing: regenerate FULL closed JSON; never splice/truncate mid-JSON.

Compact generic example (illustrative structure only; do not copy numbers/entities as answers):
{_compact_plan_example}

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
- Prefer arguments grounded in the question, step.scope_constraints, step fields, or evidence store.
- No secrets. No gold. No invented identifiers.

Return one JSON object only."""

PE_FIN_LITE_REASON_USER_TEMPLATE = """Analyze the verified evidence for this reason step. Do not call tools. Do not invent missing values.

Question:
{question}

Step:
{step_json}

Verified evidence store:
{evidence_json}

Return ONE complete closed JSON object only (no markdown fences, no trailing text).
Required keys:
- summary: non-empty string
Optional keys:
- claims: array (use [] if none)
- gaps: array (use [] if none)
Null/empty/truncated output is invalid; if evidence is insufficient, still return valid JSON with gaps listing what is missing.
Do not invent numbers not present in verified evidence."""

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
            f"objective_max={OBJECTIVE_MAX_LENGTH}",
            f"compute_expr_max={COMPUTE_EXPRESSION_MAX_LENGTH}",
            f"metric_basis_max={METRIC_BASIS_MAX_LENGTH}",
            f"adjustment_max={ADJUSTMENT_MAX_LENGTH}",
            f"currency_max={CURRENCY_MAX_LENGTH}",
            f"planner_max_tokens={PLANNER_MAX_TOKENS}",
            f"scope={SCOPE_CONTRACT_VERSION}",
            f"vocab={CANONICAL_OUTPUT_VOCABULARY_VERSION}",
            f"compute_fns={COMPUTE_FUNCTION_CONTRACT_VERSION}:{','.join(sorted(SAFE_FUNCTION_NAMES))}",
            ",".join(CANONICAL_OUTPUT_FIELDS),
            ",".join(SCOPE_CONSTRAINT_KEYS),
            _COMPACT_PLAN_EXAMPLE,
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
        objective_max_length=OBJECTIVE_MAX_LENGTH,
        compute_expression_max_length=COMPUTE_EXPRESSION_MAX_LENGTH,
        canonical_outputs_json=json.dumps(list(CANONICAL_OUTPUT_FIELDS)),
        canonical_vocab_version=CANONICAL_OUTPUT_VOCABULARY_VERSION,
        scope_contract_version=SCOPE_CONTRACT_VERSION,
        scope_keys_json=json.dumps(list(SCOPE_CONSTRAINT_KEYS)),
        currency_max_length=CURRENCY_MAX_LENGTH,
        metric_basis_max_length=METRIC_BASIS_MAX_LENGTH,
        adjustment_max_length=ADJUSTMENT_MAX_LENGTH,
        planner_max_tokens=PLANNER_MAX_TOKENS,
        _compact_plan_example=_COMPACT_PLAN_EXAMPLE,
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
    step_statuses: Sequence[Dict[str, Any]],
    evidence: Dict[str, Any],
    compute: Dict[str, Any],
    reason: Dict[str, Any],
    missing: Any,
) -> str:
    return PE_FIN_LITE_FINAL_USER_TEMPLATE.format(
        question=question,
        step_statuses_json=json.dumps(list(step_statuses), ensure_ascii=False, indent=2),
        evidence_json=json.dumps(evidence, ensure_ascii=False, indent=2),
        compute_json=json.dumps(compute, ensure_ascii=False, indent=2),
        reason_json=json.dumps(reason, ensure_ascii=False, indent=2),
        missing_json=json.dumps(missing, ensure_ascii=False, indent=2),
        FINAL_ANSWER_CONTRACT=FINAL_ANSWER_CONTRACT,
    )
