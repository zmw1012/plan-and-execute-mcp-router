"""Generic read-only MCP stdio client with reusable sessions and cleanup."""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, Sequence

from hk_equity_mcp.real_mcp.redaction import redact_obj, redact_text

logger = logging.getLogger(__name__)


class RealMcpError(RuntimeError):
    def __init__(self, message: str, *, error_type: str = "mcp_error"):
        super().__init__(message)
        self.error_type = error_type


def _require_mcp():
    try:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
    except ImportError as exc:  # pragma: no cover - env dependent
        raise RealMcpError(
            "The 'mcp' package is required for Real MCP adapters. "
            "Install optional extra real_mcp (mcp>=1.7,<3) in the runtime venv.",
            error_type="dependency_missing",
        ) from exc
    return ClientSession, StdioServerParameters, stdio_client


def resolve_command(command: str) -> str:
    """Resolve executable without collapsing venv python via Path.resolve()."""
    if os.path.sep in command or command.startswith("."):
        path = Path(command)
        if not path.is_absolute():
            path = Path.cwd() / path
        if not path.exists():
            raise RealMcpError(
                f"MCP command not found: {command}",
                error_type="command_not_found",
            )
        return str(path)
    found = shutil.which(command)
    if not found:
        raise RealMcpError(
            f"MCP command not on PATH: {command}",
            error_type="command_not_found",
        )
    return found


def normalize_tool_result(result: Any) -> Dict[str, Any]:
    is_error = bool(
        getattr(result, "is_error", False) or getattr(result, "isError", False)
    )
    content: List[Any] = []
    for block in getattr(result, "content", []) or []:
        if getattr(block, "type", None) == "text":
            text = getattr(block, "text", "") or ""
            try:
                import json

                content.append({"json": json.loads(text)})
            except Exception:
                content.append({"text": redact_text(text[:20000])})
        else:
            content.append(
                {
                    "type": getattr(block, "type", None),
                    "repr": redact_text(repr(block)[:500]),
                }
            )
    structured = getattr(result, "structured_content", None) or getattr(
        result, "structuredContent", None
    )
    return {
        "is_error": is_error,
        "content": content,
        "structured_content": redact_obj(structured) if structured else None,
        "protocol_success": True,
    }


class StdioMcpClient:
    """Async one-shot stdio helper (used by tests / ephemeral calls)."""

    def __init__(
        self,
        *,
        server_id: str,
        command: str,
        args: Optional[Sequence[str]] = None,
        env: Optional[Dict[str, str]] = None,
        cwd: Optional[str] = None,
        timeout_sec: float = 90.0,
    ):
        self.server_id = server_id
        self.command = command
        self.args = list(args or [])
        self.env = env
        self.cwd = cwd
        self.timeout_sec = timeout_sec
        self.server_name: Optional[str] = None
        self.server_version: Optional[str] = None
        self.protocol_version: Optional[str] = None

    def _server_params(self):
        _, StdioServerParameters, _ = _require_mcp()
        cmd = resolve_command(self.command)
        env = os.environ.copy()
        if self.env:
            env.update({k: str(v) for k, v in self.env.items() if v is not None})
        return StdioServerParameters(
            command=cmd,
            args=self.args,
            env=env,
            cwd=self.cwd,
        )

    @asynccontextmanager
    async def session(self) -> AsyncIterator[Any]:
        ClientSession, _, stdio_client = _require_mcp()
        params = self._server_params()
        async with stdio_client(params) as streams:
            read, write = streams[0], streams[1]
            async with ClientSession(read, write) as session:
                init = await asyncio.wait_for(
                    session.initialize(), timeout=self.timeout_sec
                )
                si = getattr(init, "server_info", None) or getattr(
                    init, "serverInfo", None
                )
                self.server_name = getattr(si, "name", None) if si else None
                self.server_version = getattr(si, "version", None) if si else None
                self.protocol_version = getattr(
                    init, "protocol_version", None
                ) or getattr(init, "protocolVersion", None)
                yield session

    async def list_tools(self) -> List[Dict[str, Any]]:
        async with self.session() as session:
            tools = await asyncio.wait_for(
                session.list_tools(), timeout=self.timeout_sec
            )
            out = []
            for t in getattr(tools, "tools", []) or []:
                schema = getattr(t, "input_schema", None) or getattr(
                    t, "inputSchema", None
                )
                out.append(
                    {
                        "name": t.name,
                        "description": (t.description or "")[:500],
                        "inputSchema": schema,
                    }
                )
            return out

    async def call_tool(
        self, tool_name: str, arguments: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        arguments = arguments or {}
        started = time.perf_counter()
        async with self.session() as session:
            result = await asyncio.wait_for(
                session.call_tool(tool_name, arguments),
                timeout=self.timeout_sec,
            )
        latency_ms = (time.perf_counter() - started) * 1000.0
        payload = normalize_tool_result(result)
        payload["latency_ms"] = latency_ms
        payload["server_id"] = self.server_id
        payload["server_name"] = self.server_name
        payload["server_version"] = self.server_version
        payload["tool"] = tool_name
        payload["arguments"] = redact_obj(arguments)
        return payload


class PersistentStdioMcpClient:
    """Long-lived stdio MCP process reused across tool calls; always closable."""

    def __init__(
        self,
        *,
        server_id: str,
        command: str,
        args: Optional[Sequence[str]] = None,
        env: Optional[Dict[str, str]] = None,
        cwd: Optional[str] = None,
        timeout_sec: float = 90.0,
    ):
        self.server_id = server_id
        self.command = command
        self.args = list(args or [])
        self.env = env
        self.cwd = cwd
        self.timeout_sec = timeout_sec
        self.server_name: Optional[str] = None
        self.server_version: Optional[str] = None
        self.protocol_version: Optional[str] = None

        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._session: Any = None
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._start_error: Optional[BaseException] = None
        self._closed = False
        self._lock = threading.RLock()

    @property
    def is_open(self) -> bool:
        return self._session is not None and not self._closed

    def start(self) -> None:
        with self._lock:
            if self._closed:
                raise RealMcpError(
                    f"Persistent MCP client already closed: {self.server_id}",
                    error_type="client_closed",
                )
            if self.is_open:
                return
            self._ready.clear()
            self._stop.clear()
            self._start_error = None
            self._thread = threading.Thread(
                target=self._thread_main,
                name=f"mcp-stdio-{self.server_id}",
                daemon=True,
            )
            self._thread.start()
            if not self._ready.wait(timeout=self.timeout_sec):
                self.close()
                raise RealMcpError(
                    f"Timed out starting MCP server {self.server_id}",
                    error_type="startup_timeout",
                )
            if self._start_error is not None:
                err = self._start_error
                self.close()
                if isinstance(err, RealMcpError):
                    raise err
                raise RealMcpError(
                    f"Failed starting MCP server {self.server_id}: {err}",
                    error_type="startup_error",
                ) from err

    def _thread_main(self) -> None:
        try:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._loop.run_until_complete(self._run_session())
        except BaseException as exc:  # noqa: BLE001
            self._start_error = exc
            self._ready.set()
        finally:
            try:
                if self._loop is not None:
                    self._loop.close()
            except Exception:  # noqa: BLE001
                pass
            self._loop = None
            self._session = None

    async def _run_session(self) -> None:
        ClientSession, StdioServerParameters, stdio_client = _require_mcp()
        cmd = resolve_command(self.command)
        env = os.environ.copy()
        if self.env:
            env.update({k: str(v) for k, v in self.env.items() if v is not None})
        params = StdioServerParameters(
            command=cmd,
            args=self.args,
            env=env,
            cwd=self.cwd,
        )
        async with stdio_client(params) as streams:
            read, write = streams[0], streams[1]
            async with ClientSession(read, write) as session:
                init = await asyncio.wait_for(
                    session.initialize(), timeout=self.timeout_sec
                )
                si = getattr(init, "server_info", None) or getattr(
                    init, "serverInfo", None
                )
                self.server_name = getattr(si, "name", None) if si else None
                self.server_version = getattr(si, "version", None) if si else None
                self.protocol_version = getattr(
                    init, "protocol_version", None
                ) or getattr(init, "protocolVersion", None)
                self._session = session
                self._ready.set()
                # Keep process alive until close() signals stop.
                while not self._stop.is_set():
                    await asyncio.sleep(0.05)

    def _ensure_started(self) -> None:
        if not self.is_open:
            self.start()

    def call_tool(
        self, tool_name: str, arguments: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        arguments = arguments or {}
        with self._lock:
            self._ensure_started()
            if self._loop is None or self._session is None:
                raise RealMcpError(
                    f"MCP session not available for {self.server_id}",
                    error_type="session_unavailable",
                )
            started = time.perf_counter()

            async def _call():
                return await asyncio.wait_for(
                    self._session.call_tool(tool_name, arguments),
                    timeout=self.timeout_sec,
                )

            fut = asyncio.run_coroutine_threadsafe(_call(), self._loop)
            try:
                result = fut.result(timeout=self.timeout_sec + 5)
            except Exception as exc:  # noqa: BLE001
                raise RealMcpError(
                    f"MCP call failed ({self.server_id}/{tool_name}): {exc}",
                    error_type="execution_error",
                ) from exc
            latency_ms = (time.perf_counter() - started) * 1000.0
            payload = normalize_tool_result(result)
            payload["latency_ms"] = latency_ms
            payload["server_id"] = self.server_id
            payload["server_name"] = self.server_name
            payload["server_version"] = self.server_version
            payload["tool"] = tool_name
            payload["arguments"] = redact_obj(arguments)
            logger.info(
                "real_mcp call server=%s tool=%s success=%s latency_ms=%.1f",
                self.server_id,
                tool_name,
                not payload.get("is_error"),
                latency_ms,
            )
            return payload

    def list_tools(self) -> List[Dict[str, Any]]:
        with self._lock:
            self._ensure_started()
            if self._loop is None or self._session is None:
                raise RealMcpError(
                    f"MCP session not available for {self.server_id}",
                    error_type="session_unavailable",
                )

            async def _list():
                tools = await asyncio.wait_for(
                    self._session.list_tools(), timeout=self.timeout_sec
                )
                out = []
                for t in getattr(tools, "tools", []) or []:
                    schema = getattr(t, "input_schema", None) or getattr(
                        t, "inputSchema", None
                    )
                    out.append(
                        {
                            "name": t.name,
                            "description": (t.description or "")[:500],
                            "inputSchema": schema,
                        }
                    )
                return out

            fut = asyncio.run_coroutine_threadsafe(_list(), self._loop)
            return fut.result(timeout=self.timeout_sec + 5)

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._stop.set()
            thread = self._thread
            self._thread = None
            self._session = None
        if thread is not None and thread.is_alive():
            thread.join(timeout=self.timeout_sec)
        self._ready.set()

    def __enter__(self) -> "PersistentStdioMcpClient":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def run_coro(coro):
    """Run an async coroutine from sync code (fresh loop)."""
    return asyncio.run(coro)
