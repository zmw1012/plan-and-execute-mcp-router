"""PE-Fin Lite v1 — plan-first step-level retrieval (less-is-more).

Architecture: plan_first_step_level_retrieval_v1

Reuses UnifiedRealMcpGateway, frozen catalog, find/describe/invoke,
dataset loader, and evaluator v1.2. Does not copy APV-Fin Monitor /
Capability Compiler / multi-round Replan controllers.
"""

from __future__ import annotations

PE_FIN_LITE_CONDITION = "pe_fin_lite"
PE_FIN_LITE_METHOD_VERSION = "pe_fin_lite_v1.2"
PE_FIN_LITE_ARCHITECTURE = "plan_first_step_level_retrieval_v1"

__all__ = [
    "PE_FIN_LITE_CONDITION",
    "PE_FIN_LITE_METHOD_VERSION",
    "PE_FIN_LITE_ARCHITECTURE",
]
