"""Server / tool metadata for MCP-Zero retrieval.

HK: built from the frozen ``UnifiedCatalogBundle`` runtime registry (blocked /
rejected / non-runtime operations are excluded). Servers are grouped by
``server_id`` (``aktools``, ``mcp_hkexnews``, ``ah_disclosure``).

Embedded text intentionally contains only name + description (+ an optional
server summary). It never includes gold flags, roles, or distractor labels that
the catalog keeps in ``OperationSpec.constraints``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

from hk_equity_mcp.operation_resolver.schemas import OperationSpec
from hk_equity_mcp.real_mcp.unified_catalog import UnifiedCatalogBundle

# Bump when the embedded-text convention changes (invalidates caches).
METADATA_DOC_VERSION = "mcp_zero_metadata_v1"

# Neutral, catalog-level server descriptions (no task / gold information).
HK_SERVER_DESCRIPTIONS: Dict[str, str] = {
    "aktools": (
        "AKTools MCP server exposing AKShare-backed Hong Kong and China equity "
        "data: company financial statements and indicators, market quotes, "
        "historical prices, announcements and news."
    ),
    "mcp_hkexnews": (
        "HKEXnews MCP server for Hong Kong Exchanges disclosure documents: "
        "issuer lookup, announcement and filing search and retrieval."
    ),
    "ah_disclosure": (
        "A+H disclosure MCP server: company resolution and disclosure / "
        "announcement data for dual-listed (A and H share) issuers."
    ),
}


@dataclass
class ToolMeta:
    server_id: str
    tool_name: str
    operation_id: str
    description: str
    text: str
    spec: Optional[OperationSpec] = None
    schema: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ServerMeta:
    server_id: str
    description: str
    summary: Optional[str] = None
    tools: List[ToolMeta] = field(default_factory=list)


class McpZeroCatalog:
    """Immutable-ish server → tools index used by both retrievers."""

    def __init__(self, servers: Iterable[ServerMeta], *, source: str = "unknown"):
        self.source = source
        self._servers: Dict[str, ServerMeta] = {}
        self._tools: Dict[str, ToolMeta] = {}
        for server in sorted(servers, key=lambda s: s.server_id):
            if server.server_id in self._servers:
                raise ValueError(f"Duplicate server_id in catalog: {server.server_id}")
            server.tools = sorted(
                server.tools, key=lambda t: (t.tool_name, t.operation_id)
            )
            self._servers[server.server_id] = server
            for tool in server.tools:
                if tool.operation_id in self._tools:
                    raise ValueError(f"Duplicate operation_id: {tool.operation_id}")
                self._tools[tool.operation_id] = tool

    # -- accessors ---------------------------------------------------------
    def servers(self) -> List[ServerMeta]:
        return list(self._servers.values())

    def server_ids(self) -> List[str]:
        return list(self._servers)

    def server(self, server_id: str) -> Optional[ServerMeta]:
        return self._servers.get(server_id)

    def tools_for(self, server_id: str) -> List[ToolMeta]:
        server = self._servers.get(server_id)
        return list(server.tools) if server else []

    def tool(self, operation_id: str) -> Optional[ToolMeta]:
        return self._tools.get(operation_id)

    def all_tools(self) -> List[ToolMeta]:
        return list(self._tools.values())

    def __len__(self) -> int:
        return len(self._tools)

    def fingerprint(self) -> str:
        payload = [
            {
                "server_id": s.server_id,
                "description": s.description,
                "summary": s.summary,
                "tools": [
                    {"op": t.operation_id, "text": t.text} for t in s.tools
                ],
            }
            for s in self.servers()
        ]
        blob = json.dumps(
            {"version": METADATA_DOC_VERSION, "servers": payload},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def tool_description_text(spec: OperationSpec) -> str:
    """name + description; falls back to summary, combines when both differ."""
    summary = (spec.summary or "").strip()
    desc = (spec.description or "").strip()
    if summary and desc and summary != desc:
        body = f"{summary} {desc}"
    else:
        body = desc or summary
    return body


def tool_embedding_text(name: str, description: str) -> str:
    """ETOM MCP0 convention: ``f"{tool_name} {tool_desc}"``."""
    return f"{name} {description}".strip()


def _server_summary(
    server_id: str, domains: Sequence[str], tools: Sequence[ToolMeta]
) -> str:
    names = ", ".join(t.tool_name for t in tools)
    dom = ", ".join(sorted(set(domains))) or "unspecified"
    return f"{server_id} MCP server; catalog domains: {dom}; tools: {names}."


def build_hk_metadata(
    bundle: UnifiedCatalogBundle, *, include_summary: bool = True
) -> McpZeroCatalog:
    """Group the runtime-exposed, non-blocked registry by ``server_id``."""
    by_server: Dict[str, List[ToolMeta]] = {}
    domains: Dict[str, List[str]] = {}
    for oid in sorted(bundle.registry):
        rec = bundle.records.get(oid)
        # Defensive: registry already excludes these; never trust, re-check.
        if rec is not None and (rec.is_blocked or not rec.runtime_exposed):
            continue
        spec = bundle.registry[oid]
        server_id = spec.service_id
        desc = tool_description_text(spec)
        tool = ToolMeta(
            server_id=server_id,
            tool_name=spec.name,
            operation_id=oid,
            description=desc,
            text=tool_embedding_text(spec.name, desc),
            spec=spec,
            schema=spec.json_schema(),
        )
        by_server.setdefault(server_id, []).append(tool)
        if rec is not None:
            domains.setdefault(server_id, []).extend(rec.domains or [])
    servers: List[ServerMeta] = []
    for server_id, tools in by_server.items():
        base = HK_SERVER_DESCRIPTIONS.get(server_id, f"{server_id} MCP server.")
        summary = (
            _server_summary(server_id, domains.get(server_id, []), tools)
            if include_summary
            else None
        )
        servers.append(
            ServerMeta(
                server_id=server_id,
                description=base,
                summary=summary,
                tools=tools,
            )
        )
    return McpZeroCatalog(servers, source="hk_unified_catalog")
