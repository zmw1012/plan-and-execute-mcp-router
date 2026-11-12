"""Identity hashes for PE-Fin RePlan v1 (separate from Lite Planner identity)."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict

from hk_equity_mcp.pe_fin_replan import (
    PE_FIN_REPLAN_ARCHITECTURE,
    PE_FIN_REPLAN_CONDITION,
    PE_FIN_REPLAN_METHOD_VERSION,
    PE_FIN_REPLAN_PROTOCOL_VERSION,
    REPLAN_FLAGS,
)
from hk_equity_mcp.pe_fin_replan.schemas import (
    REPLAN_SCHEMA_VERSION,
    REPLANNER_OUTPUT_SCHEMA,
    LOOP_POLICY,
)

DEFAULT_BUDGETS = {
    "max_total_llm_calls_per_task": 8,
    "max_total_mcp_invocations": 8,
    "max_executed_steps": 6,
    "max_replan_rounds": 6,
    "max_new_steps_per_replan": 2,
    "max_local_fallback_per_retrieve": 1,
    "retrieval_top_k": 5,
    "describe_top_n": 3,
}


def _sha(obj: Any) -> str:
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def replanner_schema_hash() -> str:
    return _sha(
        {
            "schema_version": REPLAN_SCHEMA_VERSION,
            "schema": REPLANNER_OUTPUT_SCHEMA,
        }
    )


def loop_policy_hash() -> str:
    return _sha(LOOP_POLICY)


def budget_identity_hash(budgets: Dict[str, Any] | None = None) -> str:
    return _sha(budgets or DEFAULT_BUDGETS)


def replan_protocol_identity() -> Dict[str, Any]:
    from hk_equity_mcp.pe_fin_lite.eea_identity import eea_protocol_hash
    from hk_equity_mcp.pe_fin_lite.retrieval_formula import (
        RETRIEVAL_FORMULA_NAME,
        retrieval_formula_hash,
    )
    from hk_equity_mcp.pe_fin_lite.prompts import pe_fin_lite_prompt_hash
    from hk_equity_mcp.pe_fin_replan.prompts import replanner_prompt_hash

    return {
        "condition": PE_FIN_REPLAN_CONDITION,
        "method_version": PE_FIN_REPLAN_METHOD_VERSION,
        "protocol_version": PE_FIN_REPLAN_PROTOCOL_VERSION,
        "architecture": PE_FIN_REPLAN_ARCHITECTURE,
        "flags": dict(REPLAN_FLAGS),
        "planner_prompt_hash": pe_fin_lite_prompt_hash(),
        "replanner_prompt_hash": replanner_prompt_hash(),
        "replanner_schema_hash": replanner_schema_hash(),
        "loop_policy_hash": loop_policy_hash(),
        "budget_identity_hash": budget_identity_hash(),
        "eea_identity_hash": eea_protocol_hash(),
        "retrieval_formula_name": RETRIEVAL_FORMULA_NAME,
        "retrieval_formula_hash": retrieval_formula_hash(),
        "budgets": dict(DEFAULT_BUDGETS),
    }
