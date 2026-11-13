"""PE-Fin RePlan -> ETOM-100 offline transfer package (v1).

Plan-Execute-Replan agent transferred from HK-155 PE-Fin RePlan v1.4.1 to the
external ETOM-100 formal sample. Routing-only evaluation: no ETOM MCP tool is
executed; ``invoke`` records a predicted route node through
``hk_equity_mcp.etom.adapter.EtomRoutingResolver``.

Post-hoc exploratory. NOT a formal primary result. This package is the offline
build only; no live LLM / embedding / MCP call is made by its tests.
"""

from __future__ import annotations

PE_FIN_REPLAN_ETOM_CONDITION = "pe_fin_replan_etom"
PE_FIN_REPLAN_ETOM_METHOD_VERSION = "pe_fin_replan_etom_v1"
PE_FIN_REPLAN_ETOM_PROTOCOL_VERSION = "planner_executor_replanner_etom_seven_round_v1"
PE_FIN_REPLAN_ETOM_ARCHITECTURE = "plan_execute_replan"
PE_FIN_REPLAN_ETOM_BASE_METHOD = "pe_fin_replan_v1_4_1"
PE_FIN_REPLAN_ETOM_TRANSFER_SOURCE = "hk155_pe_fin_replan"

# Embedding cache namespace (directory name under .cache/). Must never be the HK
# pe_fin_lite namespace nor the Operation-Visible ``etom_embeddings`` directory.
EMBEDDING_CACHE_NAMESPACE = "pe_fin_replan_etom_operation_embeddings_v1"

ETOM_FLAGS = {
    "posthoc_exploratory": True,
    "formal_primary_result": False,
    "cross_benchmark_transfer_evaluation": True,
    "base_method": PE_FIN_REPLAN_ETOM_BASE_METHOD,
    "transfer_source": PE_FIN_REPLAN_ETOM_TRANSFER_SOURCE,
    "no_monitor": True,
    "no_global_replan": True,
    "no_replan_round_4": True,
    "no_replan4": True,
    "no_langgraph_dependency": True,
    "routing_only_evaluation": True,
    "real_mcp_tool_executed": False,
    "r5_unchanged": True,
    "executor_one_step_per_round": True,
    "no_executor_to_executor_edge": True,
    "planner_exactly_once": True,
    "final_response_from_replanner_only": True,
    "round7_response_only": True,
    "no_eighth_round": True,
    "max_execution_rounds": 7,
    "no_hk_capability_profile": True,
    "no_hk_provider_adapters": True,
    "hk_provider_paths_disabled": True,
    "generic_etom_envelope_v1": True,
    "generic_etom_grounding_v1": False,
    "generic_etom_grounding_v2": True,
}

# Alias kept for identity/resume imports.
ETOM_REPLAN_FLAGS = ETOM_FLAGS

# Alias used by identity.py (kept so both spellings resolve).
ETOM_REPLAN_FLAGS = ETOM_FLAGS

__all__ = [
    "ETOM_REPLAN_FLAGS",
    "PE_FIN_REPLAN_ETOM_CONDITION",
    "PE_FIN_REPLAN_ETOM_METHOD_VERSION",
    "PE_FIN_REPLAN_ETOM_PROTOCOL_VERSION",
    "PE_FIN_REPLAN_ETOM_ARCHITECTURE",
    "PE_FIN_REPLAN_ETOM_BASE_METHOD",
    "PE_FIN_REPLAN_ETOM_TRANSFER_SOURCE",
    "EMBEDDING_CACHE_NAMESPACE",
    "ETOM_FLAGS",
    "ETOM_REPLAN_FLAGS",
]
