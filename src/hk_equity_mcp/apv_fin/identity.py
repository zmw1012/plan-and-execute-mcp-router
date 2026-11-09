"""Formal live identity + schema hashes for APV-Fin resume/fail-closed checks."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict

from hk_equity_mcp.apv_fin.schemas import SCHEMA_BY_NAME

EXPECTED_GENERATION_PROVIDER = "openai_compatible"
EXPECTED_GENERATION_HOST = "api.zhizengzeng.com"
EXPECTED_GENERATION_MODEL = "deepseek-v4.1-flash"
EXPECTED_EMBEDDING_MODEL = "text-embedding-3-small"
EXPECTED_EMBEDDING_PROVIDER = "openai_compatible"

FORMAL_OUTPUT_DIRS = {
    "dry_run": "outputs/apv_fin_dry_run_v1",
    "real_smoke4": "outputs/apv_fin_real_smoke4_v1",
    "diagnostic12_live": "outputs/apv_fin_diagnostic12_live_v1",
}

# Accounting label for this protocol fix (not an accuracy retune).
PLANNER_PROTOCOL_CHANGE = "planner_protocol_v2_before_valid_method_evaluation"
TOOL_BINDING_PROTOCOL_CHANGE = (
    "tool_binding_and_monitor_protocol_v3_before_valid_method_evaluation"
)
DETERMINISTIC_CONTROLLER_PROTOCOL_CHANGE = (
    "deterministic_controller_refactor_before_valid_method_evaluation"
)


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
    # Embedding host is recorded from env; if present must not be empty for live.
    if embedding_host is not None and not str(embedding_host).strip():
        raise RuntimeError("embedding host missing (fail-closed)")


def worst_case_budget_estimate(limits: Dict[str, Any]) -> Dict[str, Any]:
    """Offline worst-case call budget from config limits (no retuning)."""
    max_revisions = int(limits.get("max_plan_revisions", 2))
    max_exception = int(limits.get("max_exception_replan_calls_per_task", 2))
    max_monitor = int(limits.get("max_llm_monitor_rounds", max_exception))
    max_mcp = int(limits.get("max_total_mcp_invocations", 8))
    max_retrievals_per_step = int(limits.get("max_targeted_retrievals_per_step", 2))
    max_final_repair = int(limits.get("max_final_repair_rounds", 1))
    hard_cap = int(limits.get("max_total_llm_calls_per_task", 10))
    # Planner: 1 + optional 1 repair
    planner_calls = 2
    # Binding per retrieve: 1 + optional 1 repair (budgeted separately in hard cap)
    binding_calls = 4
    # Exceptional replan only: 1 + optional 1 repair each
    monitor_calls = max_exception * 2
    # Reason/verify optional; compute is deterministic.
    reason_verify_calls = 2
    # Final synthesize: 1 + repair
    final_calls = 2 + max_final_repair
    structural_upper = (
        planner_calls + binding_calls + monitor_calls + reason_verify_calls + final_calls
    )
    max_llm_turns = min(structural_upper, hard_cap)
    max_embedding_queries = max_mcp * max_retrievals_per_step
    return {
        "per_task_max_planner_calls": planner_calls,
        "per_task_max_binding_calls_including_repair": binding_calls,
        "per_task_max_exception_replan_calls_including_repair": monitor_calls,
        "per_task_max_monitor_calls_including_repair": monitor_calls,
        "per_task_max_reason_verify_calls_including_repair": reason_verify_calls,
        "per_task_max_final_calls_including_repair": final_calls,
        "per_task_structural_llm_upper_bound": structural_upper,
        "per_task_max_llm_turns": max_llm_turns,
        "max_total_llm_calls_per_task": hard_cap,
        "max_exception_replan_calls_per_task": max_exception,
        "per_task_max_mcp_invocations": max_mcp,
        "per_task_max_embedding_query_calls": max_embedding_queries,
        "diagnostic12_worst_case_llm_turns": 12 * max_llm_turns,
        "diagnostic12_worst_case_mcp_invocations": 12 * max_mcp,
        "diagnostic12_worst_case_embedding_queries": 12 * max_embedding_queries,
        "budget_before": {
            "max_total_llm_calls_per_task": 14,
            "max_llm_monitor_rounds": 6,
        },
        "budget_after": {
            "max_total_llm_calls_per_task": hard_cap,
            "max_exception_replan_calls_per_task": max_exception,
            "max_llm_monitor_rounds": max_monitor,
        },
        "notes": (
            "v1.1 deterministic controller: normal progress has no Monitor LLM. "
            "Hard-capped by max_total_llm_calls_per_task. Compute does not consume "
            "LLM budget. Engineering contraction before valid method evaluation."
        ),
        "limits_snapshot": dict(limits),
        "max_plan_revisions": max_revisions,
        "protocol_change": DETERMINISTIC_CONTROLLER_PROTOCOL_CHANGE,
    }
