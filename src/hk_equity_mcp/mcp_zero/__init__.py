"""MCP-Zero baseline (adapted) for the HK Real-MCP experiment.

This package is a *simplified, adapted* implementation of the MCP-Zero idea
(Fei et al., 2025, https://arxiv.org/abs/2506.01056): active tool request ->
server-first hierarchical retrieval -> iterative capability extension.

It is NOT a full reproduction of the official ``xfey/MCP-Zero`` code. See
``docs/mcp_zero_fidelity_audit.md`` for the exact / adapted / omitted table.

ETOM-100 runner integration is intentionally out of scope for the HK-155
MCP-Zero baseline commit.
"""

from hk_equity_mcp.mcp_zero.scoring import SCORE_OFFICIAL_COMBINED, SCORE_TOOL_ONLY

FIDELITY_NAME = "MCP-Zero adapted baseline (hierarchical active tool retrieval)"
CONDITION = "mcp_zero"

# Official matcher defaults (pinned reference: official_matcher.py).
OFFICIAL_TOP_SERVERS = 5
OFFICIAL_TOP_TOOLS = 3
OFFICIAL_EMBEDDING_MODEL = "text-embedding-3-large"

# ETOM MCP0 eval defaults (pinned reference: eval.py / utils).
ETOM_SERVER_TOP_K = 3
ETOM_TOOL_TOP_K = 5
ETOM_UTILS_SIMILARITY_THRESHOLD = 0.3

# Defaults used by this project (documented adaptation).
DEFAULT_SERVER_TOP_K = ETOM_SERVER_TOP_K
DEFAULT_TOOL_TOP_K = ETOM_TOOL_TOP_K
DEFAULT_SIMILARITY_THRESHOLD = 0.0
DEFAULT_MAX_EXTENSION_ROUNDS = 3
DEFAULT_SCORE_MODE = SCORE_OFFICIAL_COMBINED

__all__ = [
    "FIDELITY_NAME",
    "CONDITION",
    "OFFICIAL_TOP_SERVERS",
    "OFFICIAL_TOP_TOOLS",
    "OFFICIAL_EMBEDDING_MODEL",
    "ETOM_SERVER_TOP_K",
    "ETOM_TOOL_TOP_K",
    "ETOM_UTILS_SIMILARITY_THRESHOLD",
    "DEFAULT_SERVER_TOP_K",
    "DEFAULT_TOOL_TOP_K",
    "DEFAULT_SIMILARITY_THRESHOLD",
    "DEFAULT_MAX_EXTENSION_ROUNDS",
    "DEFAULT_SCORE_MODE",
    "SCORE_OFFICIAL_COMBINED",
    "SCORE_TOOL_ONLY",
]
