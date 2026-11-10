"""PE-Fin Lite End-to-End Checkpoint Audit v1 (offline, read-only, passive).

Does not alter Planner/Binder/Reason/Final decisions, Top-K, budgets, or
raw prediction artifacts. Network/LLM/Embedding/MCP are forbidden.
"""

from __future__ import annotations

AUDIT_VERSION = "pe_fin_lite_checkpoint_audit_v1"
CHECKPOINT_TRACE_CONTRACT_VERSION = "pe_fin_lite_checkpoint_trace_v1"
RESULT_ROLE = "engineering_checkpoint_audit_only"
EXPLICITLY_NOT_FORMAL_ACCURACY = True
DEFAULT_AUDIT_SUBDIR = "final_checkpoint_audit_v1"

CHECKPOINT_IDS = (
    "C01_planner",
    "C02_filter",
    "C03_embedding",
    "C04_describe",
    "C05_binder",
    "C06_arguments",
    "C07_mcp",
    "C08_normalize",
    "C09_evidence_acceptance",
    "C10_fallback_unavailable",
    "C11_compute",
    "C12_reason",
    "C13_final",
    "C14_evaluator",
)

CHECKPOINT_STATUSES = ("pass", "fail", "N/A", "unknown", "not_evaluated")

__all__ = [
    "AUDIT_VERSION",
    "CHECKPOINT_TRACE_CONTRACT_VERSION",
    "RESULT_ROLE",
    "EXPLICITLY_NOT_FORMAL_ACCURACY",
    "DEFAULT_AUDIT_SUBDIR",
    "CHECKPOINT_IDS",
    "CHECKPOINT_STATUSES",
]
