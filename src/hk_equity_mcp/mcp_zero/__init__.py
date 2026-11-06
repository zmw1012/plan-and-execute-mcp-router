"""MCP-Zero baseline (adapted) for the HK Real-MCP experiment.

This package is a *simplified, adapted* implementation of the MCP-Zero idea
(Fei et al., 2025, https://arxiv.org/abs/2506.01056): active tool request ->
server-first hierarchical retrieval -> iterative capability extension.

It is NOT a full reproduction of the official ``xfey/MCP-Zero`` code. See
``docs/mcp_zero_fidelity_audit.md`` for the exact / adapted / omitted table.

ETOM-100 runner integration is intentionally out of scope for the HK-155
MCP-Zero baseline commit.
"""

import hashlib
import json
from typing import Any, Dict

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

# Frozen before any HK-155 real result. Paper/official matcher is 5/3.
# This baseline keeps the ETOM pinned adaptation 3/5. A later 5/3 run is an
# independent ablation and must not replace these primary numbers.
FROZEN_EVALUATOR = "hk_unified_evaluator_v1.2"
FROZEN_GENERATION_MODEL = "deepseek-v4.1-flash"
FROZEN_EMBEDDING_MODEL = "text-embedding-3-small"
FROZEN_TIE_BREAK = ("-score", "server_id", "tool_name", "operation_id")
ABLATION_POLICY = (
    "server_top_k=5 and tool_top_k=3 may be reported only as an independent "
    "ablation. They must not replace this frozen 3/5 baseline."
)


def frozen_method_identity() -> Dict[str, Any]:
    """Canonical MCP-Zero method card. Hash this; do not retune from HK-155."""
    return {
        "condition": CONDITION,
        "server_top_k": DEFAULT_SERVER_TOP_K,
        "tool_top_k": DEFAULT_TOOL_TOP_K,
        "max_extension_rounds": DEFAULT_MAX_EXTENSION_ROUNDS,
        "tie_break": list(FROZEN_TIE_BREAK),
        "score_mode": DEFAULT_SCORE_MODE,
        "similarity_threshold": DEFAULT_SIMILARITY_THRESHOLD,
        "generation_model": FROZEN_GENERATION_MODEL,
        "embedding_model": FROZEN_EMBEDDING_MODEL,
        "temperature": 0.0,
        "max_tokens": 1024,
        "max_llm_turns": 8,
        "evaluator": FROZEN_EVALUATOR,
        "topk_provenance": "etom_pinned_adaptation",
        "paper_official_server_top_k": OFFICIAL_TOP_SERVERS,
        "paper_official_tool_top_k": OFFICIAL_TOP_TOOLS,
        "paper_official_embedding_model": OFFICIAL_EMBEDDING_MODEL,
        "frozen_before_hk155_real_results": True,
        "ablation_policy": ABLATION_POLICY,
    }


def method_identity_sha256() -> str:
    blob = json.dumps(
        frozen_method_identity(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def assert_frozen_config(config: Dict[str, Any]) -> None:
    """Fail closed if the loaded YAML drifts from the frozen method card."""
    ident = frozen_method_identity()
    retrieval = dict(config.get("retrieval") or {})
    limits = dict(config.get("limits") or {})
    observed = {
        "server_top_k": int(retrieval.get("server_top_k")),
        "tool_top_k": int(retrieval.get("tool_top_k")),
        "score_mode": str(retrieval.get("score_mode")),
        "similarity_threshold": float(retrieval.get("similarity_threshold")),
        "max_extension_rounds": int(limits.get("max_extension_rounds")),
        "generation_model": str(config.get("generation_model")),
        "embedding_model": str(config.get("embedding_model")),
        "temperature": float(config.get("temperature")),
        "max_tokens": int(config.get("max_tokens")),
        "max_llm_turns": int(limits.get("max_llm_turns")),
    }
    mismatches = [
        f"{key}: config={observed[key]!r} frozen={ident[key]!r}"
        for key in observed
        if observed[key] != ident[key]
    ]
    if mismatches:
        raise RuntimeError(
            "MCP-Zero config is not the frozen pre-result baseline:\n"
            + "\n".join(mismatches)
        )


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
    "FROZEN_EVALUATOR",
    "FROZEN_GENERATION_MODEL",
    "FROZEN_EMBEDDING_MODEL",
    "FROZEN_TIE_BREAK",
    "ABLATION_POLICY",
    "SCORE_OFFICIAL_COMBINED",
    "SCORE_TOOL_ONLY",
    "frozen_method_identity",
    "method_identity_sha256",
    "assert_frozen_config",
]
