"""Shared result types for Real MCP adapters."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RealMcpResult:
    success: bool
    server_id: str
    operation_id: str
    arguments: Dict[str, Any]
    raw: Any = None
    normalized: Any = None
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None
    error_type: Optional[str] = None
    latency_ms: Optional[float] = None
    real_mcp: bool = True
    transport: str = "stdio"
    server_version: Optional[str] = None
    protocol_success: bool = False
    execution_success: bool = False
    # Optional offline check filled by smoke/reconcile helpers only.
    reconciliation_success: Optional[bool] = None
    reconciliation_note: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
