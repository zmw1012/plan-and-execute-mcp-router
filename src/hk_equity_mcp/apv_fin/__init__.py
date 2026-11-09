"""Adaptive Plan-Visible Finance Router (APV-Fin) proposed method v1.

Separated from the Operation-Visible baseline: new condition, prompts,
configs, and runner. Reuses UnifiedRealMcpGateway, frozen catalog,
find/describe/invoke boundary, dataset loader, and evaluator v1.2.
"""

from __future__ import annotations

APV_FIN_CONDITION = "apv_fin"
APV_FIN_METHOD_VERSION = "apv_fin_v1_4_1_crash_safe_runtime"

__all__ = ["APV_FIN_CONDITION", "APV_FIN_METHOD_VERSION"]
