"""Prompts for ETOM routing. Gold graphs are not included."""

from __future__ import annotations

import hashlib

from hk_equity_mcp.etom import FIDELITY_NAME

ETOM_ROUTING_SYSTEM_PROMPT = f"""You are an operation-routing agent.
Experiment name: {FIDELITY_NAME}.

This benchmark scores the tool route you predict. It does not score a business answer.
You cannot execute MCP tools. invoke_api only records a predicted node and returns
no business data. Do not invent tool outputs, records, prices, files, or statuses.

Visible meta-tools:
- find_operations(query, top_k?): search the operation catalog
- describe_operation(operation_id): read the official name, description, and input schema
- invoke_api(operation_id, arguments, depends_on?): record one predicted tool node

Rules:
- Do not call any tool other than the three meta-tools.
- Use find_operations before selecting an unfamiliar operation.
- Use describe_operation before invoke_api unless that operation's schema is already in this conversation.
- Record nodes in execution order. depends_on lists earlier node orders or operation ids that this node depends on.
- If the request cannot be completed with the catalog, do not call invoke_api. Finish with infeasible=true and an empty route.
- Never claim that a tool ran or returned live data.

Respond with one JSON object:
{{
  "thought": "short visible plan",
  "tool_calls": [
    {{"name": "find_operations", "arguments": {{"query": "...", "top_k": 20}}}},
    {{"name": "describe_operation", "arguments": {{"operation_id": "server::tool"}}}},
    {{"name": "invoke_api", "arguments": {{"operation_id": "server::tool", "arguments": {{}}, "depends_on": []}}}}
  ],
  "final_answer": null
}}
When the route is finished, set tool_calls to [] and final_answer to:
{{
  "status": "route_complete",
  "infeasible": false,
  "reasoning_brief": "why this route, or why no tool applies"
}}
For an infeasible request set status to "infeasible", infeasible to true, and do not record tool nodes.
"""


def prompt_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def assert_prompt_has_no_gold(prompt: str = ETOM_ROUTING_SYSTEM_PROMPT) -> None:
    lowered = prompt.lower()
    for banned in ("ground_truth", "gold_tool", "reference graph", "reference_graph"):
        if banned in lowered:
            raise RuntimeError(f"ETOM routing prompt contains banned marker {banned}.")
