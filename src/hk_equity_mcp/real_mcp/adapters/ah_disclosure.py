"""ah-disclosure-kit Real MCP adapter — stdio only."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Set

from hk_equity_mcp.real_mcp.client import (
    PersistentStdioMcpClient,
    RealMcpError,
    StdioMcpClient,
)
from hk_equity_mcp.real_mcp.redaction import redact_obj
from hk_equity_mcp.real_mcp.types import RealMcpResult

SERVER_ID = "ah_disclosure"
ALLOWED_TOOLS: Set[str] = {
    "resolve_company",
    "search_annual_report",
    "search_filings",
}


def default_ah_disclosure_command() -> str:
    env = os.environ.get("AH_DISCLOSURE_MCP_COMMAND")
    if env:
        return env
    root = Path(__file__).resolve().parents[4]
    candidate = (
        root.parent
        / "hk-equity-mcp-agent"
        / ".venv-ahdisclosure-discovery"
        / "bin"
        / "ah-disclosure-mcp"
    )
    if candidate.exists():
        return str(candidate)
    local = root / ".venv-ahdisclosure-discovery" / "bin" / "ah-disclosure-mcp"
    if local.exists():
        return str(local)
    return "ah-disclosure-mcp"


class AhDisclosureAdapter:
    def __init__(
        self,
        command: Optional[str] = None,
        timeout_sec: float = 180.0,
        cwd: Optional[str] = None,
        *,
        persistent: bool = True,
    ):
        self.command = command or default_ah_disclosure_command()
        self.timeout_sec = timeout_sec
        self.cwd = cwd or str(Path(__file__).resolve().parents[4])
        self.persistent = persistent
        if persistent:
            self._client: Any = PersistentStdioMcpClient(
                server_id=SERVER_ID,
                command=self.command,
                args=[],
                cwd=self.cwd,
                timeout_sec=timeout_sec,
            )
        else:
            self._client = StdioMcpClient(
                server_id=SERVER_ID,
                command=self.command,
                args=[],
                cwd=self.cwd,
                timeout_sec=timeout_sec,
            )

    def open(self) -> "AhDisclosureAdapter":
        if self.persistent:
            self._client.start()
        return self

    def close(self) -> None:
        if self.persistent and hasattr(self._client, "close"):
            self._client.close()

    def __enter__(self) -> "AhDisclosureAdapter":
        return self.open()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def call_mcp_tool(
        self,
        tool: str,
        arguments: Optional[Dict[str, Any]] = None,
        *,
        operation_id: str = "",
    ) -> RealMcpResult:
        arguments = dict(arguments or {})
        if tool not in ALLOWED_TOOLS:
            return RealMcpResult(
                success=False,
                server_id=SERVER_ID,
                operation_id=operation_id or tool,
                arguments=arguments,
                error=f"ah_disclosure tool not allowed: {tool}",
                error_type="blocked_operation",
                protocol_success=False,
                execution_success=False,
            )
        try:
            if self.persistent:
                raw = self._client.call_tool(tool, arguments)
                version = getattr(self._client, "server_version", None)
            else:
                with self._client as client:
                    raw = client.call_tool(tool, arguments)
                    version = getattr(client, "server_version", None)
            return RealMcpResult(
                success=True,
                server_id=SERVER_ID,
                operation_id=operation_id or tool,
                arguments=arguments,
                raw=redact_obj(raw),
                normalized=redact_obj(raw),
                protocol_success=True,
                execution_success=True,
                server_version=version,
            )
        except RealMcpError as exc:
            return RealMcpResult(
                success=False,
                server_id=SERVER_ID,
                operation_id=operation_id or tool,
                arguments=arguments,
                error=str(exc),
                error_type="underlying_tool_error",
                protocol_success=False,
                execution_success=False,
            )
        except Exception as exc:  # noqa: BLE001
            return RealMcpResult(
                success=False,
                server_id=SERVER_ID,
                operation_id=operation_id or tool,
                arguments=arguments,
                error=f"{type(exc).__name__}: {exc}",
                error_type="execution_error",
                protocol_success=False,
                execution_success=False,
            )
