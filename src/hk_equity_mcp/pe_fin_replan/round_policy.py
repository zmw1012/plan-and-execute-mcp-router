"""RePlan round-cap ablation policy (depth only; not a new architecture).

Default remains RePlan-7 (pe_fin_replan_v1_4_1). Cap=4 is an independent
posthoc exploratory ablation identity that must not overwrite RePlan-7.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any, Dict, Mapping, Optional

from hk_equity_mcp.pe_fin_replan import (
    PE_FIN_REPLAN_ARCHITECTURE,
    PE_FIN_REPLAN_CONDITION,
    PE_FIN_REPLAN_METHOD_VERSION,
    PE_FIN_REPLAN_PROTOCOL_VERSION,
    REPLAN_FLAGS,
)
from hk_equity_mcp.pe_fin_replan.schemas import LOOP_POLICY

DEFAULT_REPLAN_ROUND_CAP = 7
ALLOWED_REPLAN_ROUND_CAPS = frozenset({4, 7})

# Independent ablation identity — do not mutate pe_fin_replan_v1_4_1 constants.
PE_FIN_REPLAN_R4_CONDITION = "pe_fin_replan_r4"
PE_FIN_REPLAN_R4_METHOD_VERSION = "pe_fin_replan_v1_4_1_round4_ablation"
PE_FIN_REPLAN_R4_PROTOCOL_VERSION = (
    "planner_executor_replanner_four_round_ablation_v1"
)
PE_FIN_REPLAN_R4_ARCHITECTURE = PE_FIN_REPLAN_ARCHITECTURE
PE_FIN_REPLAN_R4_BASE_METHOD = "pe_fin_replan_v1_4_1"

ROUND_POLICY_VERSION = "pe_fin_replan_round_policy_v1"

_BASE_BUDGETS = {
    "max_total_llm_calls_per_task": 16,
    "max_total_mcp_invocations": 8,
    "max_execution_rounds": 7,
    "max_executed_steps": 7,
    "max_replan_rounds": 7,
    "max_new_steps_per_replan": 2,
    "max_local_fallback_per_retrieve": 1,
    "retrieval_top_k": 5,
    "describe_top_n": 3,
}


def _sha(obj: Any) -> str:
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def validate_replan_round_cap(cap: Any) -> int:
    try:
        value = int(cap)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"replan_round_cap must be int in {sorted(ALLOWED_REPLAN_ROUND_CAPS)}; "
            f"got {cap!r}"
        ) from exc
    if value not in ALLOWED_REPLAN_ROUND_CAPS:
        raise ValueError(
            f"replan_round_cap must be one of {sorted(ALLOWED_REPLAN_ROUND_CAPS)}; "
            f"got {value}"
        )
    return value


def response_only_round(cap: int) -> int:
    return validate_replan_round_cap(cap)


def budgets_for_round_cap(cap: int = DEFAULT_REPLAN_ROUND_CAP) -> Dict[str, Any]:
    cap = validate_replan_round_cap(cap)
    out = dict(_BASE_BUDGETS)
    out["max_execution_rounds"] = cap
    out["max_executed_steps"] = cap
    out["max_replan_rounds"] = cap
    return out


def loop_policy_for_round_cap(cap: int = DEFAULT_REPLAN_ROUND_CAP) -> Dict[str, Any]:
    """Return LOOP_POLICY copy with round caps applied (default=7 unchanged)."""
    cap = validate_replan_round_cap(cap)
    policy = deepcopy(LOOP_POLICY)
    policy["max_execution_rounds"] = cap
    policy["max_executed_steps"] = cap
    policy["max_replan_rounds"] = cap
    # Keep round7_response_only True only for the historical 7-round policy.
    policy["round7_response_only"] = cap == 7
    policy["round4_response_only"] = cap == 4
    policy["final_round_response_only"] = True
    policy["response_only_round"] = cap
    policy["replan_round_cap"] = cap
    if cap == 4:
        policy["version"] = "pe_fin_replan_loop_policy_v1_4_1_round4_ablation"
    return policy


def replan_flags_for_round_cap(cap: int = DEFAULT_REPLAN_ROUND_CAP) -> Dict[str, Any]:
    cap = validate_replan_round_cap(cap)
    # Preserve exact historical flag dict for RePlan-7.
    if cap == DEFAULT_REPLAN_ROUND_CAP:
        return dict(REPLAN_FLAGS)
    flags = dict(REPLAN_FLAGS)
    flags["max_execution_rounds"] = 4
    flags["round7_response_only"] = False
    flags["round4_response_only"] = True
    flags["final_round_response_only"] = True
    flags["response_only_round"] = 4
    flags["replan_round_cap"] = 4
    flags["base_method"] = PE_FIN_REPLAN_R4_BASE_METHOD
    flags["posthoc_exploratory"] = True
    flags["formal_primary_result"] = False
    flags["round_depth_ablation"] = True
    return flags


def round_policy_descriptor(cap: int = DEFAULT_REPLAN_ROUND_CAP) -> Dict[str, Any]:
    cap = validate_replan_round_cap(cap)
    return {
        "round_policy_version": ROUND_POLICY_VERSION,
        "replan_round_cap": cap,
        "max_execution_rounds": cap,
        "max_executed_steps": cap,
        "max_replan_rounds": cap,
        "response_only_round": cap,
        "final_round_response_only": True,
        "allowed_round_caps": sorted(ALLOWED_REPLAN_ROUND_CAPS),
        "default_round_cap": DEFAULT_REPLAN_ROUND_CAP,
        "llm_hard_ceiling": 16,
        "mcp_hard_ceiling": 8,
        "isolation_note": (
            "depth ablation only; planner/replanner schema and R5/EEA unchanged"
        ),
    }


def round_policy_hash(cap: int = DEFAULT_REPLAN_ROUND_CAP) -> str:
    return _sha(round_policy_descriptor(cap))


def identity_labels_for_round_cap(cap: int = DEFAULT_REPLAN_ROUND_CAP) -> Dict[str, Any]:
    cap = validate_replan_round_cap(cap)
    if cap == 4:
        return {
            "condition": PE_FIN_REPLAN_R4_CONDITION,
            "method_version": PE_FIN_REPLAN_R4_METHOD_VERSION,
            "protocol_version": PE_FIN_REPLAN_R4_PROTOCOL_VERSION,
            "architecture": PE_FIN_REPLAN_R4_ARCHITECTURE,
            "base_method": PE_FIN_REPLAN_R4_BASE_METHOD,
            "replan_round_cap": 4,
            "response_only_round": 4,
            "posthoc_exploratory": True,
            "formal_primary_result": False,
        }
    return {
        "condition": PE_FIN_REPLAN_CONDITION,
        "method_version": PE_FIN_REPLAN_METHOD_VERSION,
        "protocol_version": PE_FIN_REPLAN_PROTOCOL_VERSION,
        "architecture": PE_FIN_REPLAN_ARCHITECTURE,
        "base_method": REPLAN_FLAGS.get("base_method"),
        "replan_round_cap": 7,
        "response_only_round": 7,
        "posthoc_exploratory": True,
        "formal_primary_result": False,
    }


def apply_round_cap_to_limits(
    limits: Mapping[str, Any],
    *,
    cap: int,
) -> Dict[str, Any]:
    """Force round-cap budgets onto runner limits (no silent override of ceilings)."""
    cap = validate_replan_round_cap(cap)
    forced = budgets_for_round_cap(cap)
    out = dict(limits)
    # Round-depth fields are forced by ablation identity.
    for key in (
        "max_execution_rounds",
        "max_executed_steps",
        "max_replan_rounds",
    ):
        out[key] = forced[key]
    # Preserve hard ceilings; fill defaults if missing.
    out["max_total_llm_calls_per_task"] = int(
        out.get("max_total_llm_calls_per_task", forced["max_total_llm_calls_per_task"])
    )
    out["max_total_mcp_invocations"] = int(
        out.get("max_total_mcp_invocations", forced["max_total_mcp_invocations"])
    )
    if out["max_total_llm_calls_per_task"] != forced["max_total_llm_calls_per_task"]:
        raise ValueError(
            "replan round-cap ablation keeps LLM hard ceiling at "
            f"{forced['max_total_llm_calls_per_task']}; "
            f"got {out['max_total_llm_calls_per_task']}"
        )
    if out["max_total_mcp_invocations"] != forced["max_total_mcp_invocations"]:
        raise ValueError(
            "replan round-cap ablation keeps MCP hard ceiling at "
            f"{forced['max_total_mcp_invocations']}; "
            f"got {out['max_total_mcp_invocations']}"
        )
    for key in (
        "max_new_steps_per_replan",
        "max_local_fallback_per_retrieve",
        "retrieval_top_k",
        "describe_top_n",
    ):
        out.setdefault(key, forced[key])
    return out


def resume_round_identity_fields(cap: int = DEFAULT_REPLAN_ROUND_CAP) -> Dict[str, Any]:
    cap = validate_replan_round_cap(cap)
    labels = identity_labels_for_round_cap(cap)
    return {
        "replan_round_cap": cap,
        "max_execution_rounds": cap,
        "max_replan_rounds": cap,
        "response_only_round": cap,
        "round_policy_hash": round_policy_hash(cap),
        "condition": labels["condition"],
        "method_version": labels["method_version"],
        "protocol_version": labels["protocol_version"],
    }


def assert_round_cap_cli_allowed(
    *,
    replan_enabled: bool,
    replan_round_cap: Optional[int],
) -> int:
    """CLI gate: --replan-round-cap only with --replan; default 7."""
    if replan_round_cap is None:
        return DEFAULT_REPLAN_ROUND_CAP
    cap = validate_replan_round_cap(replan_round_cap)
    if not replan_enabled:
        raise ValueError(
            "--replan-round-cap requires --replan / --replan-smoke5 / "
            "--diagnostic20-replan"
        )
    return cap
