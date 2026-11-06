"""Build MCP-Zero hierarchical catalog from ETOM registry."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from hk_equity_mcp.etom.convert import convert_registry
from hk_equity_mcp.mcp_zero.metadata import (
    McpZeroCatalog,
    ServerMeta,
    ToolMeta,
    tool_embedding_text,
)


def build_etom_catalog(
    payload: Dict[str, Any],
    *,
    official: Optional[Dict[str, Dict[str, Any]]] = None,
    include_summary: bool = True,
) -> McpZeroCatalog:
    if official is None:
        _specs, official = convert_registry(payload)
    by_server: Dict[str, List[ToolMeta]] = {}
    server_meta: Dict[str, Dict[str, Any]] = {}
    for oid, row in official.items():
        server_name = str(row["server_name"])
        tool_name = str(row["tool_name"])
        desc = str(row.get("description") or "").strip()
        text = tool_embedding_text(tool_name, desc)
        if not text:
            text = tool_name
        tool = ToolMeta(
            server_id=server_name,
            tool_name=tool_name,
            operation_id=str(oid),
            description=desc or tool_name,
            text=text,
            spec=None,
            schema=dict(row.get("input_schema") or {}),
        )
        by_server.setdefault(server_name, []).append(tool)
        server_meta[server_name] = row
    servers: List[ServerMeta] = []
    for server_name, tools in by_server.items():
        row = server_meta[server_name]
        description = (
            str(row.get("server_description") or "").strip()
            or str(row.get("server_summary") or "").strip()
            or server_name
        )
        summary = str(row.get("server_summary") or "").strip() or None
        if not include_summary:
            summary = None
        elif summary is None:
            names = ", ".join(t.tool_name for t in tools[:12])
            summary = f"{server_name}; tools: {names}."
        servers.append(
            ServerMeta(
                server_id=server_name,
                description=description,
                summary=summary,
                tools=tools,
            )
        )
    return McpZeroCatalog(servers, source="etom_registry")
