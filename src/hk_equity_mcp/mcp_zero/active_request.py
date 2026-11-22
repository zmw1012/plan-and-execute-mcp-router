"""Active Tool Request parsing (paper §3 / official matcher).

Parses the structured ``<tool_assistant>`` block that the model emits when it
identifies a capability gap. Reimplemented from the paper format + the pinned
``official_matcher.extract_tool_assistant`` / ETOM ``_extract_tool_assistant_content``
patterns — not a verbatim copy of either file.

Parsing is line-oriented after normalizing escaped newlines so a server field
never swallows a following ``tool:`` line (including when the block is embedded
inside a JSON string with ``\\n`` escapes).
"""

from __future__ import annotations

import re
from typing import Optional, Tuple

from hk_equity_mcp.mcp_zero.schemas import ActiveToolRequest

_FALLBACK_BLOCK_RE = re.compile(
    r"<tool_assistant>(.*?)</tool_assistant>", re.DOTALL | re.IGNORECASE
)


def _normalize_escapes(text: str) -> str:
    """Turn common JSON-style escapes into real whitespace for line parsing."""
    return (
        text.replace("\\r\\n", "\n")
        .replace("\\n", "\n")
        .replace("\\r", "\n")
        .replace("\\t", "\t")
    )


def _parse_server_tool_fields(content: str) -> Tuple[Optional[str], Optional[str]]:
    """Line-oriented server:/tool: extraction. Fail closed if either is missing."""
    normalized = _normalize_escapes(content).replace("\r\n", "\n").replace("\r", "\n")
    server_parts: list[str] = []
    tool_parts: list[str] = []
    mode: Optional[str] = None
    for raw_line in normalized.split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        lower = line.lower()
        if lower.startswith("server:"):
            mode = "server"
            server_parts = [line.split(":", 1)[1].strip()]
            continue
        if lower.startswith("tool:"):
            mode = "tool"
            tool_parts = [line.split(":", 1)[1].strip()]
            continue
        if mode == "server":
            server_parts.append(line)
        elif mode == "tool":
            tool_parts.append(line)
    server = "\n".join(part for part in server_parts if part).strip()
    tool = "\n".join(part for part in tool_parts if part).strip()
    if not server or not tool:
        return None, None
    # Guard: server must not still contain a tool: label after normalization.
    if re.search(r"(?im)^\s*tool\s*:", server):
        return None, None
    return server, tool


def extract_tool_assistant(text: str) -> Tuple[Optional[str], Optional[str]]:
    """Return ``(server_desc, tool_desc)`` or ``(None, None)``."""
    if not text:
        return None, None
    for candidate in (text, _normalize_escapes(text)):
        block = _FALLBACK_BLOCK_RE.search(candidate)
        if not block:
            continue
        server, tool = _parse_server_tool_fields(block.group(1))
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
    raw_match = _FALLBACK_BLOCK_RE.search(_normalize_escapes(text or ""))
    if raw_match is None:
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
