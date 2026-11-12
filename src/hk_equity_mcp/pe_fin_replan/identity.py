"""Identity hashes for PE-Fin RePlan v1.4 (separate from Lite Planner identity)."""

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

LLM_BUDGET_RATIONALE = {
    "initial_planner": 1,
    "max_executor_internal_llm": 7,
    "max_replanner": 7,
    "repair_slack": 1,
    "hard_ceiling": 16,
    "note": "transport retries also count toward the hard ceiling",
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
    from hk_equity_mcp.pe_fin_lite.output_vocabulary import (
        CANONICAL_OUTPUT_VOCABULARY_VERSION,
        canonical_output_vocabulary_hash,
    )
    from hk_equity_mcp.pe_fin_lite.retrieval_formula import (
        RETRIEVAL_FORMULA_NAME,
        retrieval_formula_hash,
    )
    from hk_equity_mcp.pe_fin_lite.prompts import pe_fin_lite_prompt_hash
    from hk_equity_mcp.pe_fin_replan.failed_invocation import (
        FAILED_INVOCATION_POLICY_VERSION,
        failed_invocation_policy_hash,
    )
    from hk_equity_mcp.pe_fin_replan.grounding import (
        GROUNDING_CONTRACT_VERSION,
        grounding_contract_hash,
    )
    from hk_equity_mcp.pe_fin_replan.prompts import replanner_prompt_hash
    from hk_equity_mcp.pe_fin_replan.step_canonicalizer import (
        CANONICALIZER_VERSION,
        step_canonicalizer_hash,
    )

    return {
        "condition": PE_FIN_REPLAN_CONDITION,
        "method_version": PE_FIN_REPLAN_METHOD_VERSION,
        "protocol_version": PE_FIN_REPLAN_PROTOCOL_VERSION,
        "architecture": PE_FIN_REPLAN_ARCHITECTURE,
        "replanner_schema_version": REPLAN_SCHEMA_VERSION,
        "flags": dict(REPLAN_FLAGS),
        "planner_prompt_hash": pe_fin_lite_prompt_hash(),
        "replanner_prompt_hash": replanner_prompt_hash(),
        "replanner_schema_hash": replanner_schema_hash(),
        "step_canonicalizer_version": CANONICALIZER_VERSION,
        "step_canonicalizer_hash": step_canonicalizer_hash(),
        "grounding_contract_version": GROUNDING_CONTRACT_VERSION,
        "grounding_contract_hash": grounding_contract_hash(),
        "canonical_output_vocabulary_version": CANONICAL_OUTPUT_VOCABULARY_VERSION,
        "canonical_output_vocabulary_hash": canonical_output_vocabulary_hash(),
        "failed_invocation_policy_version": FAILED_INVOCATION_POLICY_VERSION,
        "failed_invocation_policy_hash": failed_invocation_policy_hash(),
        "loop_policy_hash": loop_policy_hash(),
        "budget_identity_hash": budget_identity_hash(),
        "llm_budget_rationale": dict(LLM_BUDGET_RATIONALE),
        "eea_identity_hash": eea_protocol_hash(),
        "retrieval_formula_name": RETRIEVAL_FORMULA_NAME,
        "retrieval_formula_hash": retrieval_formula_hash(),
        "budgets": dict(DEFAULT_BUDGETS),
    }
