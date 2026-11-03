"""Shared and condition-specific prompts (no gold/reference leakage)."""

from __future__ import annotations

import hashlib
import json
from typing import Dict, List, Optional

from hk_equity_mcp.experiment.schema import FINAL_ANSWER_CONTRACT
from hk_equity_mcp.operation_resolver.registry import build_default_registry
from hk_equity_mcp.operation_resolver.schemas import OperationSpec, ParameterSpec

MCP_FREE_SYSTEM_PROMPT = f"""You are taking a closed-book financial question-answering benchmark.
You have no MCP tools, no web search, no retrieval system, no uploaded reports, and no
external memory. Answer only from your model knowledge and the information explicitly
included in the question.

{FINAL_ANSWER_CONTRACT}
"""

OPERATION_VISIBLE_SYSTEM_PROMPT = f"""You are an Operation-Visible enterprise agent for HK equity Q&A.
This is a simplified reimplementation adapted from Ghoshal et al. (2026).
You may call ONLY these meta-tools:
- find_operations(query, top_k?): search the operation registry
- describe_operation(operation_id): inspect schema/constraints
- invoke_api(operation_id, arguments): validate and execute

Do NOT invent underlying tool names. Do NOT claim hidden planning.
Underlying Financial tools are not directly callable.
Use find_operations first for a subgoal, then describe_operation before invoke_api
unless the contract is already known from this trajectory.
You may issue multiple independent read-only invoke_api calls in one turn when
arguments are already known and calls have no dependency.
If validation/execution errors occur, recover using the visible structured error.

Respond with a single JSON object of the form:
{{
  "thought": "brief visible plan (not private chain-of-thought)",
  "tool_calls": [
     {{"name": "find_operations", "arguments": {{"query": "...", "top_k": 20}}}},
     {{"name": "describe_operation", "arguments": {{"operation_id": "..."}}}},
     {{"name": "invoke_api", "arguments": {{"operation_id": "...", "arguments": {{...}}}}}}
  ],
  "final_answer": null
}}
When ready to answer, set tool_calls to [] and set final_answer to the shared answer object.

{FINAL_ANSWER_CONTRACT}
"""


def _format_param(p: ParameterSpec) -> str:
    bits = [f"- {p.name} ({p.type})", "required" if p.required else "optional"]
    if p.enum is not None:
        bits.append(f"enum={json.dumps(p.enum, ensure_ascii=False)}")
    if p.default is not None:
        bits.append(f"default={json.dumps(p.default, ensure_ascii=False)}")
    line = ", ".join(bits)
    if p.description:
        line += f"\n  description: {p.description}"
    return line


def format_operation_contract_text(op: OperationSpec) -> str:
    """Render one OperationSpec as Flat-visible tool documentation."""
    lines: List[str] = [
        f"### Tool: {op.name}",
        f"operation_id: {op.operation_id}",
        f"service_id: {op.service_id}",
        f"summary: {op.summary}",
        f"description: {op.description}",
        "required_parameters:",
    ]
    req = op.required_params()
    opt = op.optional_params()
    if req:
        lines.extend(_format_param(p) for p in req)
    else:
        lines.append("- (none)")
    lines.append("optional_parameters:")
    if opt:
        lines.extend(_format_param(p) for p in opt)
    else:
        lines.append("- (none)")
    lines.append("json_schema:")
    lines.append(json.dumps(op.json_schema(), ensure_ascii=False, indent=2))
    lines.append("constraints:")
    if op.constraints:
        lines.extend(f"- {c}" for c in op.constraints)
    else:
        lines.append("- (none)")
    lines.append(f"invocation_guidance: {op.invocation_guidance or '(none)'}")
    lines.append(f"read_only: {op.read_only}")
    return "\n".join(lines)


def build_flat_system_prompt(
    registry: Optional[Dict[str, OperationSpec]] = None,
) -> str:
    """Flat prompt with full contracts generated from the OperationSpec registry.

    Runtime still calls FinancialServer directly (no resolver / embedding).
    """
    registry = registry or build_default_registry()
    # Stable order matching registry build order.
    preferred = [
        "financial.get_financial_metric",
        "financial.get_filing_summary",
    ]
    ops = [registry[oid] for oid in preferred if oid in registry]
    if not ops:
        ops = list(registry.values())
    contracts = "\n\n".join(format_operation_contract_text(op) for op in ops)
    tool_names = ", ".join(op.name for op in ops)
    return f"""You are a Flat-tool HK equity Q&A agent.
You may call ONLY these Financial Server tools directly by name: {tool_names}.
Do not invent other tool names. Do not claim access to Market or News tools in this condition.
You may issue multiple independent read-only tool calls in one turn when arguments are known.

Full tool contracts (same underlying OperationSpec registry used by Operation-Visible):

{contracts}

Respond with a single JSON object of the form:
{{
  "thought": "brief visible plan",
  "tool_calls": [
     {{"name": "get_financial_metric", "arguments": {{"ticker": "0700.HK", "metric": "revenue"}}}}
  ],
  "final_answer": null
}}
When ready to answer, set tool_calls to [] and set final_answer to the shared answer object.

{FINAL_ANSWER_CONTRACT}
"""


# Evaluated at import for hash stability in simple cases; prefer build_flat_system_prompt().
FLAT_SYSTEM_PROMPT = build_flat_system_prompt()


def prompt_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
