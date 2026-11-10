"""PE-Fin Lite HK-155 Retrieval Audit (evaluation-only; exploratory/descriptive).

Does not modify PE-Fin Lite Agent Planner, ranking, or runtime budgets.
Gold/accepted operation sets are evaluation-reference only and must never enter
Planner prompts, step queries, operation documents, or Agent runtime.
"""

from __future__ import annotations

AUDIT_VERSION = "pe_fin_lite_retrieval_audit_v1"
RESULT_ROLE = "exploratory_descriptive"
DEFAULT_OUTPUT_DIR = "outputs/pe_fin_lite_retrieval_all155_v1"
TOP_K = 5
DESCRIBE_N = 3
HYBRID_ALPHAS = (1.0, 0.9, 0.8, 0.7)
QUERY_CONSTRUCTION_VERSION = "pe_fin_lite_step_query_v1"
OPERATION_DOCUMENT_VERSION = "pe_fin_lite_op_doc_v1"
DOMAIN_FILTER_VERSION = "pe_fin_lite_domain_filter_v1"
PLANNER_ONLY_PROTOCOL_VERSION = "pe_fin_lite_planner_only_v2"
PLANNER_ONLY_PROTOCOL_VERSION_V1 = "pe_fin_lite_planner_only_v1"
DEFAULT_PLANNER_V2_OUTPUT_DIR = "outputs/pe_fin_lite_retrieval_all155_planner_v2"
DEFAULT_PLANNER_DIAGNOSTIC10_OUTPUT_DIR = (
    "outputs/pe_fin_lite_planner_protocol_diagnostic10_v2"
)
RANKING_PROTOCOL_VERSION = "pe_fin_lite_retrieval_rank_v1"

__all__ = [
    "AUDIT_VERSION",
    "RESULT_ROLE",
    "DEFAULT_OUTPUT_DIR",
    "TOP_K",
    "DESCRIBE_N",
    "HYBRID_ALPHAS",
]
