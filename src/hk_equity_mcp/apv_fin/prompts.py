"""APV-Fin prompts (no gold/reference leakage; company-generic)."""

from __future__ import annotations

from hk_equity_mcp.experiment.prompts import prompt_hash
from hk_equity_mcp.experiment.schema import FINAL_ANSWER_CONTRACT
from hk_equity_mcp.experiment.unified_schema import BANNED_PROMPT_SUBSTRINGS

APV_FIN_SYSTEM_PROMPT = f"""You are APV-Fin, an Adaptive Plan-Visible Finance Router for Hong Kong equity Q&A
across Financial, Market, News, and Cross-domain tasks.

Method rules:
- Produce structured plans with step types only from:
  retrieve, compute, reason, verify, answer, clarify, refuse.
- Never put operation_id values inside the plan. Only retrieve steps may later
  call the Operation-Resolver meta-tools (find_operations / describe_operation /
  invoke_api) through the runtime router.
- compute steps must propose an explicit arithmetic expression and named operands;
  the runtime evaluates them deterministically (no eval in the model).
- reason / verify steps may use short decision summaries only — do not emit private
  chain-of-thought.
- Prefer official disclosure and curated MCP sources. Dataset cutoff is 2026-08-12
  (Asia/Hong_Kong) unless the question states otherwise. Do not use post-cutoff facts.
- If the question is ambiguous, clarify. If evidence cannot support the ask, refuse.
  Never fabricate figures, citations, or tool outputs.
- Be company-generic for any HK listed issuer / index. Do not assume a fixed issuer
  whitelist.

{FINAL_ANSWER_CONTRACT}
"""

APV_FIN_PLANNER_USER_TEMPLATE = """Question:
{question}

Return ONE JSON object matching schema apv_fin_plan_v1 with:
- schema_version: "apv_fin_plan_v1"
- task_spec: goal, entities, periods, metrics, constraints, ambiguities,
  answerability_hypothesis, optional cutoff_date / currency_unit_hints /
  metric_basis_hints
- initial_plan.steps: a mutable DAG of steps (unique step_id, depends_on,
  step_type, objective, mandatory, success_criteria). Do NOT include operation_id.
No markdown fences.
"""

APV_FIN_MONITOR_USER_TEMPLATE = """Question:
{question}

Current plan revision: {revision}
Plan steps JSON:
{plan_json}

Recent observations JSON:
{observations_json}

Evidence ledger summary JSON:
{evidence_summary_json}

Budget counters JSON:
{budget_json}

Return ONE JSON object matching schema apv_fin_monitor_v1 with decision in
continue|replan|verify|answer|clarify|refuse|fail and an incremental patch
(completed_steps/updated_steps/new_steps/removed_steps). No operation_id in new
steps. No markdown fences.
"""

APV_FIN_COMPUTE_USER_TEMPLATE = """Question:
{question}

Compute step_id={step_id}
Objective: {objective}
Available evidence JSON:
{evidence_json}

Return ONE JSON object matching schema apv_fin_compute_v1 with expression,
operands (numeric map), unit, notes. Expression may only use operand names and
+ - * / ** parentheses. No markdown fences.
"""

APV_FIN_REASON_USER_TEMPLATE = """Question:
{question}

Reason/verify step_id={step_id} type={step_type}
Objective: {objective}
Evidence JSON:
{evidence_json}

Return ONE JSON object:
{{
  "summary": "concise decision reason (not private chain-of-thought)",
  "decision": "continue|answer|clarify|refuse|verify",
  "resolved": true,
  "claims": ["short claim"]
}}
No markdown fences.
"""

APV_FIN_FINAL_USER_TEMPLATE = """Question:
{question}

Plan coverage JSON:
{coverage_json}
Evidence ledger JSON:
{evidence_json}
Validator notes JSON:
{validator_json}

Return ONE JSON object matching schema apv_fin_final_v1.
status must be answered|clarification_needed|unavailable.
Cite only evidence_ids actually present. No markdown fences.
"""


def apv_fin_prompt_hash() -> str:
    # Hash the stable system prompt only (templates are versioned separately).
    return prompt_hash(APV_FIN_SYSTEM_PROMPT)


def assert_prompts_clean() -> None:
    blobs = [
        APV_FIN_SYSTEM_PROMPT,
        APV_FIN_PLANNER_USER_TEMPLATE,
        APV_FIN_MONITOR_USER_TEMPLATE,
        APV_FIN_COMPUTE_USER_TEMPLATE,
        APV_FIN_REASON_USER_TEMPLATE,
        APV_FIN_FINAL_USER_TEMPLATE,
    ]
    for blob in blobs:
        for banned in BANNED_PROMPT_SUBSTRINGS:
            if banned in blob:
                raise RuntimeError(f"APV-Fin prompt contains banned substring {banned}")
        for needle in (
            "Tencent",
            "HSBC",
            "Alibaba",
            "Pop Mart",
            "Xiaomi",
            "fin_cf_",
            "mkt_dev_",
            "gold_operations",
            "reference_answer",
        ):
            if needle in blob:
                raise RuntimeError(f"APV-Fin prompt must stay company/task generic; found {needle}")
