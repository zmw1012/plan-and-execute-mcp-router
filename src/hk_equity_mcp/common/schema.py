"""Unified response schema for all MCP tools."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


REQUIRED_FIELDS = (
    "success",
    "source",
    "ticker",
    "as_of_date",
    "data",
    "evidence",
    "error",
)


@dataclass
class ToolResponse:
    """Standard payload returned by every mock MCP tool."""

    success: bool
    source: str
    ticker: str
    as_of_date: str
    data: Any
    evidence: List[str] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def make_response(
    *,
    success: bool,
    source: str,
    ticker: str,
    as_of_date: str,
    data: Any = None,
    evidence: Optional[List[str]] = None,
    error: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a schema-compliant tool response dict."""
    response = ToolResponse(
        success=success,
        source=source,
        ticker=ticker,
        as_of_date=as_of_date,
        data=data,
        evidence=evidence or [],
        error=error,
    )
    return response.to_dict()


def validate_response(payload: Dict[str, Any]) -> bool:
    """Return True if payload contains all required top-level fields."""
    return all(key in payload for key in REQUIRED_FIELDS)
