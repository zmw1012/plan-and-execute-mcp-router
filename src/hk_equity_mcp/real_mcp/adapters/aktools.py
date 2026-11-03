"""AKTools Real MCP adapter — stdio aktools-mcp only (no direct akshare import)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

from hk_equity_mcp.real_mcp.client import (
    PersistentStdioMcpClient,
    RealMcpError,
    StdioMcpClient,
    run_coro,
)
from hk_equity_mcp.real_mcp.redaction import redact_obj
from hk_equity_mcp.real_mcp.types import RealMcpResult

SERVER_ID = "aktools"


def default_aktools_command() -> str:
    env = os.environ.get("AKTOOLS_MCP_COMMAND")
    if env:
        return env
    root = Path(__file__).resolve().parents[4]
    candidate = root / ".venv-aktools-discovery" / "bin" / "aktools-mcp"
    if candidate.exists():
        return str(candidate)
    return "aktools-mcp"


class AktoolsAdapter:
    def __init__(
        self,
        command: Optional[str] = None,
        timeout_sec: float = 120.0,
        cwd: Optional[str] = None,
        *,
        persistent: bool = True,
    ):
        self.command = command or default_aktools_command()
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

    def open(self) -> "AktoolsAdapter":
        if self.persistent:
            self._client.start()
        return self

    def close(self) -> None:
        if self.persistent and hasattr(self._client, "close"):
            self._client.close()

    def __enter__(self) -> "AktoolsAdapter":
        return self.open()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def list_tools(self) -> Dict[str, Any]:
        if self.persistent:
            tools = self._client.list_tools()
            version = self._client.server_version
        else:
            tools = run_coro(self._client.list_tools())
            version = self._client.server_version
        return {
            "success": True,
            "server_id": SERVER_ID,
            "tools": tools,
            "server_version": version,
            "real_mcp": True,
        }

    def call_mcp_tool(
        self, tool_name: str, arguments: Optional[Dict[str, Any]] = None
    ) -> RealMcpResult:
        arguments = arguments or {}
        try:
            if self.persistent:
                payload = self._client.call_tool(tool_name, arguments)
            else:
                payload = run_coro(self._client.call_tool(tool_name, arguments))
        except RealMcpError as exc:
            return RealMcpResult(
                success=False,
                server_id=SERVER_ID,
                operation_id=f"aktools.{tool_name}",
                arguments=redact_obj(arguments),
                error=str(exc),
                error_type=exc.error_type,
                protocol_success=False,
                execution_success=False,
            )
        except Exception as exc:  # noqa: BLE001
            return RealMcpResult(
                success=False,
                server_id=SERVER_ID,
                operation_id=f"aktools.{tool_name}",
                arguments=redact_obj(arguments),
                error=f"{type(exc).__name__}: {exc}",
                error_type="execution_error",
                protocol_success=False,
                execution_success=False,
            )

        is_error = bool(payload.get("is_error"))
        normalized = self._extract_normalized(payload)
        return RealMcpResult(
            success=not is_error,
            server_id=SERVER_ID,
            operation_id=f"aktools.{tool_name}",
            arguments=redact_obj(arguments),
            raw=redact_obj(payload),
            normalized=normalized,
            evidence=[],
            error=None if not is_error else self._error_text(payload),
            error_type=None if not is_error else "underlying_tool_error",
            latency_ms=payload.get("latency_ms"),
            real_mcp=True,
            transport="stdio",
            server_version=payload.get("server_version") or getattr(
                self._client, "server_version", None
            ),
            protocol_success=bool(payload.get("protocol_success")),
            execution_success=not is_error,
        )

    def get_data(
        self,
        *,
        interface: str,
        params: Optional[Dict[str, Any]] = None,
        offset: int = 0,
        max_rows: int = 20,
        operation_id: Optional[str] = None,
    ) -> RealMcpResult:
        """Named curated operation → MCP get_data (never import akshare)."""
        args = {
            "name": interface,
            "params": params or {},
            "offset": offset,
            "max_rows": max_rows,
        }
        result = self.call_mcp_tool("get_data", args)
        result.operation_id = operation_id or f"aktools.{interface}"
        result.evidence = [
            {
                "type": "aktools_mcp_trace",
                "exposed_mcp_tool": "get_data",
                "underlying_aktools_interface": interface,
                "params": redact_obj(params or {}),
            }
        ]
        return result

    def _extract_normalized(self, payload: Dict[str, Any]) -> Any:
        for block in payload.get("content") or []:
            if isinstance(block, dict) and "json" in block:
                return block["json"]
            if isinstance(block, dict) and "text" in block:
                return {"text": block["text"]}
        return payload.get("structured_content")

    def _error_text(self, payload: Dict[str, Any]) -> str:
        for block in payload.get("content") or []:
            if isinstance(block, dict) and block.get("text"):
                return str(block["text"])[:2000]
            if isinstance(block, dict) and isinstance(block.get("json"), dict):
                err = block["json"].get("error") or block["json"]
                return str(err)[:2000]
        return "AKTools MCP tool returned isError=true"
