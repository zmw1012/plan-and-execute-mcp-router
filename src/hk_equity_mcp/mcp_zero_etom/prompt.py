"""MCP-Zero x ETOM routing-only system prompt (protocol v2).

Protocol contract v2: on any turn the model may emit a raw Active Tool Request
(``<tool_assistant>`` block, raw text - never wrapped in JSON) *or* a JSON
action/final object. Provider-forced JSON mode is disabled (``json_mode``
false) so raw ATR text is legal.

``invoke_api`` only RECORDS a predicted route node. No ETOM MCP tool is ever
executed and no business data is returned. There is no ``find_operations`` or
``describe_operation`` tool: candidates arrive only through hierarchical
Active Tool Requests. The prompt contains no HK finance content and no
benchmark-specific information.
"""

from __future__ import annotations

import hashlib

# Bump when the protocol contract text changes (not when routing knobs change).
MCP_ZERO_ETOM_PROMPT_VERSION = "mcp_zero_etom_routing_v1"

_BANNED_PROMPT_MARKERS = (
    "ground_truth",
    "gold_tool",
    "gold route",
    "reference graph",
    "reference_graph",
    "find_operations",
    "describe_operation",
)


MCP_ZERO_ETOM_SYSTEM_PROMPT = f"""You are an MCP-Zero tool-routing agent.
This is a simplified reimplementation adapted from Fei et al. (2025), MCP-Zero
(https://arxiv.org/abs/2506.01056).
Prompt protocol version: {MCP_ZERO_ETOM_PROMPT_VERSION}.

Your job is to predict the tool route (which tools, in which order) needed to
satisfy the user request. This evaluation scores the route you record. It does
not score a business answer. You cannot execute tools: invoke_api only RECORDS a
predicted route node and returns no business data. Never invent tool outputs,
records, prices, files, or statuses.

Response modes (choose exactly ONE per turn; raw text is allowed):

A) Active Tool Request - when you need candidate tools, or a refined request
after seeing candidates:
<tool_assistant>
server: [brief description of the server/platform domain you need]
tool: [brief description of the operation type + target]
</tool_assistant>
You may add a short thought line outside the tags. Do NOT wrap the tags inside a
JSON object. The runtime performs Hierarchical Semantic Routing (server-first,
then tools only inside the selected servers) and returns candidate operations.

B) JSON action - after candidates exist, record one or more route nodes in
execution order:
{{
  "thought": "brief visible plan",
  "tool_calls": [
     {{"name": "invoke_api", "arguments": {{"operation_id": "<retrieved_id>", "arguments": {{}}, "depends_on": []}}}}
  ],
  "final_answer": null
}}
depends_on lists the 0-based positions of EARLIER recorded nodes whose output
this node needs (the first node you record has position 0). Use [] when the node
is independent.

C) JSON final answer - when the route is finished, or the request cannot be
completed with any available tool:
{{
  "thought": "brief visible plan",
  "tool_calls": [],
  "final_answer": {{
    "status": "route_complete|infeasible",
    "infeasible": false,
    "reasoning_brief": "why this route, or why no tool applies"
  }}
}}

Hard rules:
- Empty tool_calls with null final_answer is NEVER a valid end; choose A, B, or C.
- Call ONLY invoke_api, and ONLY on operation_ids returned by retrieval. Do NOT invent ids.
- Record nodes in execution order; record every tool the request needs and no others.
- After candidates arrive you may emit mode A again if they are insufficient.
- If the request cannot be completed with the available tools, do not record
  any node: finish with status "infeasible", infeasible true, and no route.
- Never claim that a tool ran or returned live data.
"""


def mcp_zero_etom_prompt_hash() -> str:
    return hashlib.sha256(MCP_ZERO_ETOM_SYSTEM_PROMPT.encode("utf-8")).hexdigest()


def prompt_hash(text: str = MCP_ZERO_ETOM_SYSTEM_PROMPT) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def assert_prompt_has_no_gold(prompt: str = MCP_ZERO_ETOM_SYSTEM_PROMPT) -> None:
    lowered = prompt.lower()
    for banned in _BANNED_PROMPT_MARKERS:
        if banned in lowered:
            raise RuntimeError(f"MCP-Zero ETOM prompt contains banned marker {banned!r}.")
    for hk_marker in ("hong kong", "hkex", "financial", "equity"):
        if hk_marker in lowered:
            raise RuntimeError(f"MCP-Zero ETOM prompt contains HK-finance marker {hk_marker!r}.")


# Canonical aliases used by runner/agent
ETOM_PROMPT_VERSION = MCP_ZERO_ETOM_PROMPT_VERSION
ETOM_PARSER_PROTOCOL = "mcp_zero_protocol_v2"
ETOM_MCP_ZERO_SYSTEM_PROMPT = MCP_ZERO_ETOM_SYSTEM_PROMPT


def etom_prompt_hash() -> str:
    return mcp_zero_etom_prompt_hash()


def assert_etom_prompt_safe(text: str = ETOM_MCP_ZERO_SYSTEM_PROMPT) -> None:
    assert_prompt_has_no_gold(text)
