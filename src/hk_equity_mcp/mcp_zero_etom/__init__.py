"""MCP-Zero × ETOM routing-only baseline.

Hierarchical Active Tool Request routing against the frozen ETOM formal-100 /
cost-pilot locks. Does not execute external MCP tools. Does not score
natural-language final answers as accuracy.

HK MCP-Zero routing knobs stay frozen (method-card hash
``77cc3b07a4c2b9af00ad3f6bfb7cba59c82084d94b7592f2c820f46f1d3e8d3f``).
ETOM adds a dataset-specific output contract with its own prompt hash.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict

from hk_equity_mcp.mcp_zero import (
    ABLATION_POLICY,
    DEFAULT_MAX_EXTENSION_ROUNDS,
    DEFAULT_SCORE_MODE,
    DEFAULT_SERVER_TOP_K,
    DEFAULT_SIMILARITY_THRESHOLD,
    DEFAULT_TOOL_TOP_K,
    FROZEN_EMBEDDING_MODEL,
    FROZEN_GENERATION_HOST,
    FROZEN_GENERATION_MODEL,
    FROZEN_GENERATION_PROVIDER,
    FROZEN_TIE_BREAK,
    OFFICIAL_EMBEDDING_MODEL,
    OFFICIAL_TOP_SERVERS,
    OFFICIAL_TOP_TOOLS,
    frozen_method_identity,
    method_identity_sha256,
)
from hk_equity_mcp.mcp_zero.scoring import SCORE_OFFICIAL_COMBINED

FIDELITY_NAME = "MCP-Zero adapted baseline for ETOM routing-only evaluation"
CONDITION = "mcp_zero"
EMBEDDING_NAMESPACE = "mcp_zero_etom_v1"
CACHE_NAMESPACE = EMBEDDING_NAMESPACE
EXPERIMENT_PILOT = "etom_external_cost_pilot_mcp_zero_v1"
EXPERIMENT_FORMAL = "etom_external_formal_100_mcp_zero_v1"

ETOM_AUDIT_SOURCE_COMMIT = "d1934e052a32c0677a43f3b46ceaab2609e16cbe"
ETOM_FREEZE_COMMIT = "f4bfefec49eb28ed749bf600679819cb54d0e1a6"
ETOM_SOURCE_COMMIT = "790f28727f567b88918420d163c524d46644a694"

EXPECTED_FORMAL_SAMPLE_SHA256 = (
    "99cb671ea3528c1c69ab6fc2a3d622c4d6f6bb07058728071aae5eae8720cbb1"
)
EXPECTED_ORDERED_IDS_SHA256 = (
    "3be67c36e4eec58920864a401bbaa73db55136930b0f406e2da56c4988734479"
)
EXPECTED_HK_METHOD_CARD_SHA256 = (
    "77cc3b07a4c2b9af00ad3f6bfb7cba59c82084d94b7592f2c820f46f1d3e8d3f"
)

FROZEN_ETOM_EVALUATOR = "etom_official_routing_v1"
FROZEN_ETOM_MAX_TOKENS = 2048

# Catalog index transport only. Never part of routing method-card / prompt hashes.
FROZEN_EMBEDDING_BUILD: Dict[str, Any] = {
    "batch_size": 32,
    "request_timeout_sec": 120,
    "max_retries": 3,
    "retry_backoff_seconds": [1, 2, 4],
}


def frozen_embedding_build() -> Dict[str, Any]:
    """Return a copy of the frozen catalog-index transport knobs."""
    return {
        "batch_size": int(FROZEN_EMBEDDING_BUILD["batch_size"]),
        "request_timeout_sec": int(FROZEN_EMBEDDING_BUILD["request_timeout_sec"]),
        "max_retries": int(FROZEN_EMBEDDING_BUILD["max_retries"]),
        "retry_backoff_seconds": list(FROZEN_EMBEDDING_BUILD["retry_backoff_seconds"]),
    }


def assert_embedding_build_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate optional embedding_build block; defaults to frozen transport."""
    block = dict(config.get("embedding_build") or {})
    expected = frozen_embedding_build()
    if not block:
        return expected
    observed = {
        "batch_size": int(block.get("batch_size")),
        "request_timeout_sec": int(block.get("request_timeout_sec")),
        "max_retries": int(block.get("max_retries")),
        "retry_backoff_seconds": [
            int(x) for x in list(block.get("retry_backoff_seconds") or [])
        ],
    }
    mismatches = [
        f"{key}: config={observed[key]!r} frozen={expected[key]!r}"
        for key in expected
        if observed[key] != expected[key]
    ]
    if mismatches:
        raise RuntimeError(
            "MCP-Zero×ETOM embedding_build transport drifted from freeze:\n"
            + "\n".join(mismatches)
        )
    return expected


def shared_retrieval_knobs() -> Dict[str, Any]:
    """Routing knobs shared with the HK MCP-Zero method card (no evaluator)."""
    return {
        "server_top_k": DEFAULT_SERVER_TOP_K,
        "tool_top_k": DEFAULT_TOOL_TOP_K,
        "max_extension_rounds": DEFAULT_MAX_EXTENSION_ROUNDS,
        "tie_break": list(FROZEN_TIE_BREAK),
        "score_mode": DEFAULT_SCORE_MODE,
        "similarity_threshold": DEFAULT_SIMILARITY_THRESHOLD,
        "generation_model": FROZEN_GENERATION_MODEL,
        "embedding_model": FROZEN_EMBEDDING_MODEL,
        "temperature": 0.0,
        "max_llm_turns": 8,
    }


def retrieval_method_sha256() -> str:
    blob = json.dumps(
        shared_retrieval_knobs(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def assert_shared_knobs_match_hk() -> None:
    hk = frozen_method_identity()
    shared = shared_retrieval_knobs()
    for key, value in shared.items():
        if hk.get(key) != value:
            raise RuntimeError(
                f"ETOM shared routing knob drifted from HK method card: {key}"
            )
    if method_identity_sha256() != EXPECTED_HK_METHOD_CARD_SHA256:
        raise RuntimeError(
            "HK MCP-Zero method-card hash changed; refuse to proceed."
        )


def frozen_etom_method_identity() -> Dict[str, Any]:
    """ETOM experiment method card (routing knobs + ETOM-specific contract)."""
    assert_shared_knobs_match_hk()
    return {
        "condition": CONDITION,
        "experiment_family": "mcp_zero_etom",
        "routing_only": True,
        "real_tool_executions": 0,
        "server_top_k": DEFAULT_SERVER_TOP_K,
        "tool_top_k": DEFAULT_TOOL_TOP_K,
        "max_extension_rounds": DEFAULT_MAX_EXTENSION_ROUNDS,
        "tie_break": list(FROZEN_TIE_BREAK),
        "score_mode": DEFAULT_SCORE_MODE,
        "similarity_threshold": DEFAULT_SIMILARITY_THRESHOLD,
        "generation_model": FROZEN_GENERATION_MODEL,
        "embedding_model": FROZEN_EMBEDDING_MODEL,
        "temperature": 0.0,
        "max_tokens": FROZEN_ETOM_MAX_TOKENS,
        "max_llm_turns": 8,
        "evaluator": FROZEN_ETOM_EVALUATOR,
        "topk_provenance": "etom_pinned_adaptation",
        "paper_official_server_top_k": OFFICIAL_TOP_SERVERS,
        "paper_official_tool_top_k": OFFICIAL_TOP_TOOLS,
        "paper_official_embedding_model": OFFICIAL_EMBEDDING_MODEL,
        "hk_method_identity_sha256": EXPECTED_HK_METHOD_CARD_SHA256,
        "hk_method_knobs_unchanged": True,
        "ablation_policy": ABLATION_POLICY,
        "embedding_namespace": EMBEDDING_NAMESPACE,
        "score_formula": SCORE_OFFICIAL_COMBINED,
    }


def etom_method_identity_sha256() -> str:
    blob = json.dumps(
        frozen_etom_method_identity(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def frozen_etom_runtime_env_identity() -> Dict[str, Any]:
    return {
        "generation_provider": FROZEN_GENERATION_PROVIDER,
        "generation_base_url_host": FROZEN_GENERATION_HOST,
        "generation_model": FROZEN_GENERATION_MODEL,
        "embedding_model": FROZEN_EMBEDDING_MODEL,
    }


def assert_frozen_etom_config(config: Dict[str, Any]) -> None:
    ident = frozen_etom_method_identity()
    runtime = frozen_etom_runtime_env_identity()
    # Transport knobs are validated separately and never enter method identity.
    assert_embedding_build_config(config)
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
    expected = {
        "server_top_k": ident["server_top_k"],
        "tool_top_k": ident["tool_top_k"],
        "score_mode": ident["score_mode"],
        "similarity_threshold": ident["similarity_threshold"],
        "max_extension_rounds": ident["max_extension_rounds"],
        "generation_model": ident["generation_model"],
        "embedding_model": ident["embedding_model"],
        "temperature": ident["temperature"],
        "max_tokens": ident["max_tokens"],
        "max_llm_turns": ident["max_llm_turns"],
    }
    mismatches = [
        f"{key}: config={observed[key]!r} frozen={expected[key]!r}"
        for key in expected
        if observed[key] != expected[key]
    ]
    provider = str(config.get("generation_provider") or "")
    if provider != runtime["generation_provider"]:
        mismatches.append(
            "generation_provider: "
            f"config={provider!r} frozen={runtime['generation_provider']!r}"
        )
    host = str(config.get("generation_base_url_host") or "")
    if host and host != runtime["generation_base_url_host"]:
        mismatches.append(
            "generation_base_url_host: "
            f"config={host!r} frozen={runtime['generation_base_url_host']!r}"
        )
    if bool(config.get("json_mode")):
        mismatches.append("json_mode must be false for ATR protocol")
    if mismatches:
        raise RuntimeError(
            "MCP-Zero×ETOM config is not the frozen baseline:\n"
            + "\n".join(mismatches)
        )


__all__ = [
    "FIDELITY_NAME",
    "CONDITION",
    "EMBEDDING_NAMESPACE",
    "CACHE_NAMESPACE",
    "EXPERIMENT_PILOT",
    "EXPERIMENT_FORMAL",
    "ETOM_AUDIT_SOURCE_COMMIT",
    "ETOM_FREEZE_COMMIT",
    "ETOM_SOURCE_COMMIT",
    "EXPECTED_FORMAL_SAMPLE_SHA256",
    "EXPECTED_ORDERED_IDS_SHA256",
    "EXPECTED_HK_METHOD_CARD_SHA256",
    "FROZEN_EMBEDDING_BUILD",
    "frozen_embedding_build",
    "assert_embedding_build_config",
    "shared_retrieval_knobs",
    "retrieval_method_sha256",
    "assert_shared_knobs_match_hk",
    "frozen_etom_method_identity",
    "etom_method_identity_sha256",
    "frozen_etom_runtime_env_identity",
    "assert_frozen_etom_config",
]
