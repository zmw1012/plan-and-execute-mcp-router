"""Read-only Real MCP client adapters (AKTools + mcp-hkexnews + Market + Unified)."""

from hk_equity_mcp.real_mcp.catalog import (
    DEFAULT_CATALOG_PATH,
    load_real_mcp_catalog,
    load_real_mcp_registry,
)
from hk_equity_mcp.real_mcp.gateway import RealMcpGateway
from hk_equity_mcp.real_mcp.market_gateway import (
    DEFAULT_MARKET_CATALOG,
    MarketRealMcpGateway,
    require_market_real_mcp_ready,
)
from hk_equity_mcp.real_mcp.readiness import require_real_mcp_ready
from hk_equity_mcp.real_mcp.types import RealMcpResult
from hk_equity_mcp.real_mcp.unified_catalog import (
    build_unified_catalog,
    load_unified_runtime_registry,
)
from hk_equity_mcp.real_mcp.unified_gateway import (
    ScriptedMockTransport,
    UnifiedRealMcpGateway,
)

__all__ = [
    "DEFAULT_CATALOG_PATH",
    "DEFAULT_MARKET_CATALOG",
    "MarketRealMcpGateway",
    "RealMcpGateway",
    "RealMcpResult",
    "ScriptedMockTransport",
    "UnifiedRealMcpGateway",
    "build_unified_catalog",
    "load_real_mcp_catalog",
    "load_real_mcp_registry",
    "load_unified_runtime_registry",
    "require_market_real_mcp_ready",
    "require_real_mcp_ready",
]
