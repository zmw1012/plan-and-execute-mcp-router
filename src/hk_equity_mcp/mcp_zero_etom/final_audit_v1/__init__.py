"""Offline MCP-Zero × ETOM Formal-100 final audit (final_audit_v1).

Never modifies raw predictions / run_metadata / gold / locks / caches.
Never calls LLM / Embedding / MCP / paid APIs.
"""

from hk_equity_mcp.mcp_zero_etom.final_audit_v1.audit import (
    AUDIT_VERSION,
    RAW_ARTIFACT_NAMES,
    run_final_audit,
)

__all__ = ["AUDIT_VERSION", "RAW_ARTIFACT_NAMES", "run_final_audit"]
