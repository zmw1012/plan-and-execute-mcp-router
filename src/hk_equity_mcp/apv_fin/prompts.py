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

Output ONE compact JSON object only matching schema apv_fin_plan_v1.
Do not write prose in enum fields. Keep the plan minimal. Prefer 3–6 steps
(max 8). Never include operation_id. Do not include markdown fences.
Do not repeat the full question. Do not write chain-of-thought.
Do not repeat global constraints (cutoff/source/currency) in every step —
put globals in task_spec; steps only hold step-specific content.

Required fields:
- schema_version: "apv_fin_plan_v1"
- task_spec.goal (≤180 chars)
- task_spec.entities/periods/metrics/constraints/ambiguities (compact arrays)
- task_spec.answerability_hypothesis: EXACTLY one of
  answerable | needs_clarification | likely_unavailable | unknown
- task_spec.answerability_reason: one short sentence (≤180 chars) explaining
  the hypothesis — NEVER put that explanation inside answerability_hypothesis
- initial_plan.steps: unique step_id, depends_on, step_type, objective (≤180),
  mandatory, success_criteria (≤160)

Company-generic compact example (illustrative only; adapt to the question):
{{
  "schema_version": "apv_fin_plan_v1",
  "task_spec": {{
    "goal": "Obtain the requested HK equity metric from supported sources.",
    "entities": ["hk_listed_issuer"],
    "periods": ["FY2025"],
    "metrics": ["requested_metric"],
    "constraints": ["cutoff_2026-08-12"],
    "ambiguities": [],
    "answerability_hypothesis": "answerable",
    "answerability_reason": "Required evidence is available from supported sources.",
    "cutoff_date": "2026-08-12",
    "currency_unit_hints": [],
    "metric_basis_hints": []
  }},
  "initial_plan": {{
    "steps": [
      {{
        "step_id": "s1_retrieve",
        "step_type": "retrieve",
        "objective": "Retrieve supporting evidence for the metric.",
        "depends_on": [],
        "mandatory": true,
        "success_criteria": "Evidence rows obtained",
        "retrieval_query_hint": "hong kong equity metric",
        "notes": null
      }},
      {{
        "step_id": "s2_reason",
        "step_type": "reason",
        "objective": "Summarize evidence for the answer.",
        "depends_on": ["s1_retrieve"],
        "mandatory": true,
        "success_criteria": "Concise summary ready",
        "retrieval_query_hint": null,
        "notes": null
      }},
      {{
        "step_id": "s3_answer",
        "step_type": "answer",
        "objective": "Emit final answer.",
        "depends_on": ["s2_reason"],
        "mandatory": true,
        "success_criteria": "Final JSON ready",
        "retrieval_query_hint": null,
        "notes": null
      }}
    ],
    "planning_notes": null
  }}
}}
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

Output ONE compact JSON object only matching schema apv_fin_monitor_v1.
Required fields:
- schema_version: MUST be "apv_fin_monitor_v1"
- decision: continue|replan|verify|answer|clarify|refuse|fail
- decision_reason: short non-empty string (do NOT use reasoning_brief)
- ready_step_ids: array
- patch: {{completed_steps, updated_steps, new_steps, removed_steps}}

Rules:
- schema_version must exist.
- decision_reason must exist; never substitute reasoning_brief.
- No markdown fences. No chain-of-thought.
- No operation_id in new/updated steps.
- Prefer compact patches. On failed retrieve, replan may update
  retrieval_query_hint and reset the step rather than repeating the same invoke.

Minimal example:
{{
  "schema_version": "apv_fin_monitor_v1",
  "decision": "continue",
  "decision_reason": "Ready steps remain after successful observation.",
  "ready_step_ids": [],
  "patch": {{
    "completed_steps": [],
    "updated_steps": [],
    "new_steps": [],
    "removed_steps": []
  }},
  "clarify_question": null,
  "refuse_reason": null,
  "fail_reason": null
}}
"""

APV_FIN_TOOL_BINDING_USER_TEMPLATE = """Question:
{question}

Normalized task_spec JSON:
{task_spec_json}

Current retrieve step JSON:
{step_json}

Grounding candidates JSON (normalized; do not invent missing facts):
{grounding_json}

Reranked candidate summaries (Top-K; not full schemas):
{candidates_json}

Described operation contracts (full input schemas for these only):
{described_json}

Budget JSON:
{budget_json}

Output ONE compact JSON object matching schema apv_fin_tool_binding_v1:
{{
  "schema_version": "apv_fin_tool_binding_v1",
  "decision": "select | expand_search | unsupported",
  "selected_operation_id": "string or null",
  "arguments": {{}},
  "argument_sources": {{"argument_name": "task_spec|plan_step|evidence_ledger|default"}},
  "expected_output": "short string",
  "decision_reason": "short string",
  "reformulated_query": null
}}

Rules:
- selected_operation_id MUST be one of the described candidates when decision=select.
- Never choose blocked/unavailable ops. Prefer Tier A then B; Tier C only if no A/B.
- arguments may only use schema-allowed fields; all required fields must be present.
- argument_sources must cover every provided argument.
- If a required argument cannot be reliably grounded, use expand_search or unsupported.
- Never select with empty {{}} arguments when required fields exist.
- No gold operations / reference answers. No markdown fences.

Minimal select example (illustrative ids only):
{{
  "schema_version": "apv_fin_tool_binding_v1",
  "decision": "select",
  "selected_operation_id": "example.market_history",
  "arguments": {{"symbol": "00700", "adjust": ""}},
  "argument_sources": {{"symbol": "task_spec", "adjust": "default"}},
  "expected_output": "OHLCV rows",
  "decision_reason": "Compatible market-history op with grounded symbol.",
  "reformulated_query": null
}}
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
    # System + planner + tool-binding + monitor contracts (protocol v3).
    return prompt_hash(
        APV_FIN_SYSTEM_PROMPT
        + "\n---\n"
        + APV_FIN_PLANNER_USER_TEMPLATE
        + "\n---\n"
        + APV_FIN_TOOL_BINDING_USER_TEMPLATE
        + "\n---\n"
        + APV_FIN_MONITOR_USER_TEMPLATE
    )


def assert_prompts_clean() -> None:
    blobs = [
        APV_FIN_SYSTEM_PROMPT,
        APV_FIN_PLANNER_USER_TEMPLATE,
        APV_FIN_MONITOR_USER_TEMPLATE,
        APV_FIN_TOOL_BINDING_USER_TEMPLATE,
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
