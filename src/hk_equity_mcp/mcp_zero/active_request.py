"""Active Tool Request parsing (paper §3 / official matcher).

Parses the structured ``<tool_assistant>`` block that the model emits when it
identifies a capability gap. Reimplemented from the paper format + the pinned
``official_matcher.extract_tool_assistant`` / ETOM ``_extract_tool_assistant_content``
patterns — not a verbatim copy of either file.
"""

from __future__ import annotations

import re
from typing import Optional, Tuple

from hk_equity_mcp.mcp_zero.schemas import ActiveToolRequest

TOOL_ASSISTANT_RE = re.compile(
    r"<tool_assistant>\s*server:\s*(.*?)\s*tool:\s*(.*?)\s*</tool_assistant>",
    re.DOTALL | re.IGNORECASE,
)

# Looser fallback used by the ETOM pinned MCP0 orchestrator.
_FALLBACK_BLOCK_RE = re.compile(
    r"<tool_assistant>(.*?)</tool_assistant>", re.DOTALL | re.IGNORECASE
)
_SERVER_LINE_RE = re.compile(r"server:\s*(.+)", re.IGNORECASE)
_TOOL_LINE_RE = re.compile(r"tool:\s*(.+)", re.IGNORECASE)


def extract_tool_assistant(text: str) -> Tuple[Optional[str], Optional[str]]:
    """Return ``(server_desc, tool_desc)`` or ``(None, None)``."""
    if not text:
        return None, None
    match = TOOL_ASSISTANT_RE.search(text)
    if match:
        server = match.group(1).strip()
        tool = match.group(2).strip()
        if server and tool:
            return server, tool
        return None, None
    block = _FALLBACK_BLOCK_RE.search(text)
    if not block:
        return None, None
    content = block.group(1).strip()
    server_m = _SERVER_LINE_RE.search(content)
    tool_m = _TOOL_LINE_RE.search(content)
    if server_m and tool_m:
        server = server_m.group(1).strip()
        tool = tool_m.group(1).strip()
        if server and tool:
            return server, tool
    return None, None


def parse_active_tool_request(
    text: str,
    *,
    round_index: int = 0,
    source: str = "llm",
) -> Optional[ActiveToolRequest]:
    server, tool = extract_tool_assistant(text)
    if not server or not tool:
        return None
    raw_match = _FALLBACK_BLOCK_RE.search(text or "")
    raw = raw_match.group(0) if raw_match else text
    return ActiveToolRequest(
        server=server,
        tool=tool,
        raw=raw.strip(),
        round_index=round_index,
        source=source,
    )


def format_active_tool_request(server: str, tool: str) -> str:
    return (
        "<tool_assistant>\n"
        f"server: {server.strip()}\n"
        f"tool: {tool.strip()}\n"
        "</tool_assistant>"
    )
