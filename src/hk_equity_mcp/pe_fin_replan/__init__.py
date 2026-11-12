"""PE-Fin RePlan v1.4 — Planner → Execute-one-step → Replanner (contract fix).

Post-hoc exploratory. Not the formal HK-155 primary result.
Reuses PE-Fin Lite Planner + R5 + EEA; does not introduce LangGraph.
"""

from __future__ import annotations

PE_FIN_REPLAN_CONDITION = "pe_fin_replan"
PE_FIN_REPLAN_METHOD_VERSION = "pe_fin_replan_v1_4"
PE_FIN_REPLAN_PROTOCOL_VERSION = "planner_executor_replanner_contract_v4"
PE_FIN_REPLAN_ARCHITECTURE = "plan_execute_replan"

REPLAN_FLAGS = {
    "posthoc_exploratory": True,
    "formal_primary_result": False,
    "developed_after_hk155_result_inspection": True,
    "base_method": "pe_fin_lite_v1_2_eea",
    "no_monitor": True,
    "no_capability_compiler": True,
    "no_langgraph_dependency": True,
    "r5_unchanged": True,
    "executor_one_step_per_round": True,
    "final_response_from_replanner_only": True,
    "max_execution_rounds": 7,
    "simplified_completion_response": True,
    "round7_response_only": True,
    "decision_reason_optional": True,
    "replanner_transport_retry_once": True,
    "step_canonicalizer_v2": True,
    "grounding_v3": True,
    "ohlc_vocabulary_v2": True,
    "failed_invocation_signature_v1": True,
}

__all__ = [
    "PE_FIN_REPLAN_CONDITION",
    "PE_FIN_REPLAN_METHOD_VERSION",
    "PE_FIN_REPLAN_PROTOCOL_VERSION",
    "PE_FIN_REPLAN_ARCHITECTURE",
    "REPLAN_FLAGS",
]
