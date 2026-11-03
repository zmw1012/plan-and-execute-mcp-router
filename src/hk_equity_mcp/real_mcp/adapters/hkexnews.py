"""mcp-hkexnews Real MCP adapter — stdio only; no bulk download."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Set

from hk_equity_mcp.real_mcp.client import RealMcpError, StdioMcpClient, run_coro
from hk_equity_mcp.real_mcp.redaction import redact_obj
from hk_equity_mcp.real_mcp.types import RealMcpResult

SERVER_ID = "mcp_hkexnews"
ALLOWED_TOOLS: Set[str] = {
    "lookup_stock_code",
    "list_hkexnews_announcements",
    "download_hkexnews_announcement",
    "parse_hkexnews_announcement",
    "keyword_search",
    "get_announcement_pages",
}


def default_hkexnews_command() -> str:
    env = os.environ.get("HKEXNEWS_MCP_COMMAND")
    if env:
        return env
    root = Path(__file__).resolve().parents[4]
    candidate = root / ".venv-hkexnews-discovery" / "bin" / "agentladle-mcp-hkexnews"
    if candidate.exists():
        return str(candidate)
    return "agentladle-mcp-hkexnews"


def default_cache_dir() -> Path:
    env = os.environ.get("HKEXNEWS_CACHE_DIR")
    if env:
        return Path(env)
    root = Path(__file__).resolve().parents[4]
    path = root / "outputs" / "hkexnews_cache"
    path.mkdir(parents=True, exist_ok=True)
    return path


class HkexnewsAdapter:
    def __init__(
        self,
        command: Optional[str] = None,
        timeout_sec: float = 180.0,
        cwd: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        allow_download: bool = False,
    ):
        self.command = command or default_hkexnews_command()
        self.timeout_sec = timeout_sec
        self.cwd = cwd or str(Path(__file__).resolve().parents[4])
        self.cache_dir = Path(cache_dir) if cache_dir else default_cache_dir()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.allow_download = allow_download
        self._download_count = 0
        self._client = StdioMcpClient(
            server_id=SERVER_ID,
            command=self.command,
            args=[],
            cwd=self.cwd,
            env={
                # Best-effort; package may ignore unknown env vars.
                "HKEXNEWS_CACHE_DIR": str(self.cache_dir),
            },
            timeout_sec=timeout_sec,
        )

    def list_tools(self) -> Dict[str, Any]:
        tools = run_coro(self._client.list_tools())
        return {
            "success": True,
            "server_id": SERVER_ID,
            "tools": tools,
            "server_version": self._client.server_version,
            "real_mcp": True,
        }

    def call_mcp_tool(
        self,
        tool_name: str,
        arguments: Optional[Dict[str, Any]] = None,
        *,
        operation_id: Optional[str] = None,
    ) -> RealMcpResult:
        arguments = arguments or {}
        if tool_name not in ALLOWED_TOOLS:
            return RealMcpResult(
                success=False,
                server_id=SERVER_ID,
                operation_id=operation_id or f"hkexnews.{tool_name}",
                arguments=redact_obj(arguments),
                error=f"Tool not in read-only allowlist: {tool_name}",
                error_type="unavailable",
                protocol_success=False,
                execution_success=False,
            )
        if tool_name == "download_hkexnews_announcement":
            if not self.allow_download:
                return RealMcpResult(
                    success=False,
                    server_id=SERVER_ID,
                    operation_id=operation_id or f"hkexnews.{tool_name}",
                    arguments=redact_obj(arguments),
                    error=(
                        "download_hkexnews_announcement disabled unless "
                        "allow_download=True (smoke tests only; no bulk download)."
                    ),
                    error_type="unavailable",
                    protocol_success=True,
                    execution_success=False,
                )
            if self._download_count >= 1:
                return RealMcpResult(
                    success=False,
                    server_id=SERVER_ID,
                    operation_id=operation_id or f"hkexnews.{tool_name}",
                    arguments=redact_obj(arguments),
                    error="Download budget exhausted (max 1 announcement per adapter session).",
                    error_type="unavailable",
                    protocol_success=True,
                    execution_success=False,
                )

        try:
            payload = run_coro(self._client.call_tool(tool_name, arguments))
        except RealMcpError as exc:
            return RealMcpResult(
                success=False,
                server_id=SERVER_ID,
                operation_id=operation_id or f"hkexnews.{tool_name}",
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
                operation_id=operation_id or f"hkexnews.{tool_name}",
                arguments=redact_obj(arguments),
                error=f"{type(exc).__name__}: {exc}",
                error_type="execution_error",
                protocol_success=False,
                execution_success=False,
            )

        if tool_name == "download_hkexnews_announcement" and not payload.get("is_error"):
            self._download_count += 1

        is_error = bool(payload.get("is_error"))
        # Some hkexnews tools return JSON status=error without MCP isError.
        normalized = self._extract_normalized(payload)
        if isinstance(normalized, dict) and normalized.get("status") == "error":
            is_error = True
        evidence = []
        if tool_name in {"keyword_search", "get_announcement_pages"} and not is_error:
            evidence.append(
                {
                    "type": "hkexnews_page_evidence",
                    "tool": tool_name,
                    "payload": redact_obj(normalized),
                }
            )
        return RealMcpResult(
            success=not is_error,
            server_id=SERVER_ID,
            operation_id=operation_id or f"hkexnews.{tool_name}",
            arguments=redact_obj(arguments),
            raw=redact_obj(payload),
            normalized=normalized,
            evidence=evidence,
            error=None if not is_error else self._error_text(payload, normalized),
            error_type=None if not is_error else "underlying_tool_error",
            latency_ms=payload.get("latency_ms"),
            real_mcp=True,
            transport="stdio",
            server_version=payload.get("server_version") or self._client.server_version,
            protocol_success=bool(payload.get("protocol_success")),
            execution_success=not is_error,
        )

    def _extract_normalized(self, payload: Dict[str, Any]) -> Any:
        for block in payload.get("content") or []:
            if isinstance(block, dict) and "json" in block:
                return block["json"]
            if isinstance(block, dict) and "text" in block:
                return {"text": block["text"]}
        return payload.get("structured_content")

    def _error_text(self, payload: Dict[str, Any], normalized: Any) -> str:
        if isinstance(normalized, dict):
            err = normalized.get("error") or normalized.get("message")
            if err:
                return str(err)[:2000]
        for block in payload.get("content") or []:
            if isinstance(block, dict) and block.get("text"):
                return str(block["text"])[:2000]
        return "HKEXNews MCP tool failed"
