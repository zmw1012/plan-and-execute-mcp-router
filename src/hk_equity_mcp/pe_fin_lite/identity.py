"""Formal live identity + schema hashes for PE-Fin Lite resume/fail-closed checks."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict

from hk_equity_mcp.pe_fin_lite.schemas import SCHEMA_BY_NAME

EXPECTED_GENERATION_PROVIDER = "openai_compatible"
EXPECTED_GENERATION_HOST = "api.zhizengzeng.com"
EXPECTED_GENERATION_MODEL = "deepseek-v4.1-flash"
EXPECTED_EMBEDDING_MODEL = "text-embedding-3-small"
EXPECTED_EMBEDDING_PROVIDER = "openai_compatible"

FORMAL_OUTPUT_DIRS = {
    "dry_run": "outputs/pe_fin_lite_dry_run_v1",
    "smoke5_mock": "outputs/pe_fin_lite_smoke5_mock_v1",
    "smoke5_live": "outputs/pe_fin_lite_smoke5_live_v3",
    "smoke5_live_v1_invalidated": "outputs/pe_fin_lite_smoke5_live_v1",
    "smoke5_live_v2_invalidated": "outputs/pe_fin_lite_smoke5_live_v2",
    "diagnostic12_live": "outputs/pe_fin_lite_diagnostic12_live_v1",
}

METRICS_VERSION = "pe_fin_lite_metrics_v3"
ARCHITECTURE_ID = "plan_first_step_level_retrieval_v1"


def protocol_identity_hashes() -> Dict[str, Any]:
    """Hashes for protocol identities that may change across engineering smokes."""
    from hk_equity_mcp.pe_fin_lite.argument_grounding import argument_grounding_hash
    from hk_equity_mcp.pe_fin_lite.disclosure_workflow import disclosure_workflow_hash
    from hk_equity_mcp.pe_fin_lite.evidence_provenance import (
        evidence_acceptance_hash,
        provenance_contract_hash,
    )
    from hk_equity_mcp.pe_fin_lite.output_vocabulary import (
        canonical_output_vocabulary_hash,
        scope_contract_hash,
    )
    from hk_equity_mcp.pe_fin_lite.plan_canonicalizer import plan_canonicalizer_hash
    from hk_equity_mcp.pe_fin_lite.prompts import pe_fin_lite_prompt_hash
    from hk_equity_mcp.pe_fin_lite.tool_binding import binder_protocol_hash

    return {
        "combined_schema_hash": combined_schema_hash(),
        "schema_hashes": schema_hashes(),
        "prompt_hash": pe_fin_lite_prompt_hash(),
        "plan_canonicalizer_hash": plan_canonicalizer_hash(),
        "canonical_output_vocabulary_hash": canonical_output_vocabulary_hash(),
        "scope_contract_hash": scope_contract_hash(),
        "argument_grounding_hash": argument_grounding_hash(),
        "provenance_contract_hash": provenance_contract_hash(),
        "evidence_acceptance_hash": evidence_acceptance_hash(),
        "disclosure_workflow_hash": disclosure_workflow_hash(),
        "binder_protocol_hash": binder_protocol_hash(),
        "reason_protocol_hash": hashlib.sha256(
            b"pe_fin_lite_reason_protocol_v1_2:"
            b"direct_ok|repair_ok_runtime_recovered|repair_fail_stage_fail"
        ).hexdigest(),
        "metrics_version": METRICS_VERSION,
    }


def schema_hashes() -> Dict[str, str]:
    out: Dict[str, str] = {}
    for name, schema in SCHEMA_BY_NAME.items():
        blob = json.dumps(schema, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        out[name] = hashlib.sha256(blob.encode("utf-8")).hexdigest()
    return out


def combined_schema_hash() -> str:
    parts = [f"{k}:{v}" for k, v in sorted(schema_hashes().items())]
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


def host_from_base_url(base_url: str | None) -> str | None:
    if not base_url:
        return None
    raw = str(base_url).strip()
    if "://" in raw:
        raw = raw.split("://", 1)[1]
    raw = raw.split("/", 1)[0]
    return raw.lower() or None


def assert_live_provider_identity(
    *,
    generation_provider: str,
    generation_host: str | None,
    generation_model: str,
    embedding_model: str,
    embedding_host: str | None = None,
    embedding_provider: str | None = None,
) -> None:
    if generation_provider != EXPECTED_GENERATION_PROVIDER:
        raise RuntimeError(
            f"generation provider mismatch: expected {EXPECTED_GENERATION_PROVIDER}, "
            f"got {generation_provider!r} (fail-closed; no Aliyun fallback)"
        )
    if generation_host != EXPECTED_GENERATION_HOST:
        raise RuntimeError(
            f"generation host mismatch: expected {EXPECTED_GENERATION_HOST}, "
            f"got {generation_host!r}"
        )
    if generation_model != EXPECTED_GENERATION_MODEL:
        raise RuntimeError(
            f"generation model mismatch: expected {EXPECTED_GENERATION_MODEL}, "
            f"got {generation_model!r}"
        )
    if embedding_model != EXPECTED_EMBEDDING_MODEL:
        raise RuntimeError(
            f"embedding model mismatch: expected {EXPECTED_EMBEDDING_MODEL}, "
            f"got {embedding_model!r}"
        )
    if embedding_provider and embedding_provider != EXPECTED_EMBEDDING_PROVIDER:
        raise RuntimeError(
            f"embedding provider mismatch: expected {EXPECTED_EMBEDDING_PROVIDER}, "
            f"got {embedding_provider!r}"
        )
    if embedding_host is not None and not str(embedding_host).strip():
        raise RuntimeError("embedding host missing (fail-closed)")


def worst_case_budget_estimate(limits: Dict[str, Any]) -> Dict[str, Any]:
    """Offline worst-case call budget from config limits."""
    max_mcp = int(limits.get("max_total_mcp_invocations", 8))
    hard_cap = int(limits.get("max_total_llm_calls_per_task", 8))
    max_plan_steps = int(limits.get("max_plan_steps", 6))
    describe_n = int(limits.get("describe_top_n", 3))
    retrieve_top_k = int(limits.get("retrieval_top_k", 5))
    # Planner 1+1, binder up to ~4 retrieve steps (+ optional ≤1 format repair each),
    # reason optional (+≤1 repair), final 1+1. Hard-capped at 8.
    planner_calls = 2
    binding_calls = max_plan_steps * 2  # primary+fallback call + ≤1 format repair
    reason_calls = 2
    final_calls = 2
    structural_upper = planner_calls + binding_calls + reason_calls + final_calls
    max_llm = min(structural_upper, hard_cap)
    return {
        "per_task_max_planner_calls": planner_calls,
        "per_task_max_binding_calls": binding_calls,
        "per_task_max_reason_calls": reason_calls,
        "per_task_max_final_calls_including_repair": final_calls,
        "per_task_structural_llm_upper_bound": structural_upper,
        "per_task_max_llm_turns": max_llm,
        "max_total_llm_calls_per_task": hard_cap,
        "per_task_max_mcp_invocations": max_mcp,
        "per_task_max_embedding_query_calls": max_plan_steps,
        "max_plan_steps": max_plan_steps,
        "retrieval_top_k": retrieve_top_k,
        "describe_top_n": describe_n,
        "max_local_fallback_per_retrieve": 1,
        "global_replan_rounds": 0,
        "monitor_calls": 0,
        "notes": (
            "PE-Fin Lite: no Monitor, no global Replan. Allowed repairs: planner "
            "JSON once, binder format once, reason JSON once, final JSON once. "
            "Binder still returns primary+fallback in one response. Hard LLM≤8."
        ),
        "limits_snapshot": dict(limits),
        "architecture": ARCHITECTURE_ID,
    }
