"""Read-only Real MCP client adapters (AKTools + mcp-hkexnews)."""

from hk_equity_mcp.real_mcp.catalog import (
    DEFAULT_CATALOG_PATH,
    load_real_mcp_catalog,
    load_real_mcp_registry,
)
from hk_equity_mcp.real_mcp.gateway import RealMcpGateway
from hk_equity_mcp.real_mcp.types import RealMcpResult

__all__ = [
    "DEFAULT_CATALOG_PATH",
    "RealMcpGateway",
    "RealMcpResult",
    "load_real_mcp_catalog",
    "load_real_mcp_registry",
]
