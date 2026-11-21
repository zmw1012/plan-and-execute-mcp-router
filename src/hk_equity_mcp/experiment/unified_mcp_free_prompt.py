"""Unified HK MCP-Free (closed-book) system prompt.

Frozen, domain-agnostic, no gold/reference/domain/task leakage.
"""

from __future__ import annotations

from hk_equity_mcp.experiment.prompts import prompt_hash
from hk_equity_mcp.experiment.schema import FINAL_ANSWER_CONTRACT

UNIFIED_HK_MCP_FREE_SYSTEM_PROMPT = f"""You have no external tools, web access, retrieval system, database, filings,
market feed, or news source. Answer only from model-internal knowledge. For
time-sensitive or exact company data that you cannot verify, return unavailable
instead of fabricating. Follow the required JSON answer contract.

{FINAL_ANSWER_CONTRACT}
"""


def unified_mcp_free_prompt_hash() -> str:
    return prompt_hash(UNIFIED_HK_MCP_FREE_SYSTEM_PROMPT)
