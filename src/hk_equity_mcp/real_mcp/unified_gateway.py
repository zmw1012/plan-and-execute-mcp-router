"""Unified Real-MCP gateway for HK Financial + Market + News operations.

All real invokes go through this gateway. Unknown / blocked operations fail
closed — never fall back to mock FinancialServer / local MCP servers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional, Protocol

from hk_equity_mcp.real_mcp.adapters.ah_disclosure import AhDisclosureAdapter
from hk_equity_mcp.real_mcp.adapters.aktools import AktoolsAdapter
from hk_equity_mcp.real_mcp.adapters.hkexnews import HkexnewsAdapter
from hk_equity_mcp.real_mcp.gateway import RealMcpGateway
from hk_equity_mcp.real_mcp.market_gateway import (
    BLOCKED_MARKET_OPS,
    MarketRealMcpGateway,
)
from hk_equity_mcp.real_mcp.types import RealMcpResult
from hk_equity_mcp.real_mcp.unified_catalog import (
    UnifiedCatalogBundle,
    build_unified_catalog,
)


class MockTransport(Protocol):
    def invoke(
        self, operation_id: str, arguments: Optional[Dict[str, Any]] = None
    ) -> RealMcpResult:
        ...


class ScriptedMockTransport:
    """Deterministic offline transport for --mock / unit tests (no network)."""

    def __init__(
        self,
        handler: Optional[
            Callable[[str, Dict[str, Any]], RealMcpResult]
        ] = None,
    ):
        self.handler = handler
        self.calls: list[Dict[str, Any]] = []

    def invoke(
        self, operation_id: str, arguments: Optional[Dict[str, Any]] = None
    ) -> RealMcpResult:
        arguments = dict(arguments or {})
        self.calls.append({"operation_id": operation_id, "arguments": arguments})
        if self.handler is not None:
            return self.handler(operation_id, arguments)
        return RealMcpResult(
            success=True,
            server_id="mock_transport",
            operation_id=operation_id,
            arguments=arguments,
            raw={"mock": True, "operation_id": operation_id},
            normalized={"mock": True, "operation_id": operation_id, "ok": True},
            evidence=[
                {
                    "source": "mock_transport",
                    "operation_id": operation_id,
                    "note": "offline mock result",
                }
            ],
            protocol_success=True,
            execution_success=True,
            real_mcp=False,
            transport="mock",
        )


class UnifiedRealMcpGateway:
    """Single RealInvokeBackend over Financial / Market / News adapters."""

    def __init__(
        self,
        *,
        catalog_bundle: Optional[UnifiedCatalogBundle] = None,
        root: Optional[Path] = None,
        aktools: Optional[AktoolsAdapter] = None,
        hkexnews: Optional[HkexnewsAdapter] = None,
        ah_disclosure: Optional[AhDisclosureAdapter] = None,
        allow_hkex_download: bool = False,
        persistent: bool = True,
        mock_transport: Optional[MockTransport] = None,
        open_adapters: bool = True,
    ):
        self.bundle = catalog_bundle or build_unified_catalog(root=root)
        self.registry = self.bundle.registry
        self.raw_ops = self.bundle.raw_ops
        self.records = self.bundle.records
        self.mock_transport = mock_transport
        self._opened = False
        self.invoke_count = 0
        self._open_adapters = open_adapters

        self.aktools = aktools
        self.hkexnews = hkexnews
        self.ah_disclosure = ah_disclosure
        self._financial: Optional[RealMcpGateway] = None
        self._market: Optional[MarketRealMcpGateway] = None

        if mock_transport is None:
            self.aktools = self.aktools or AktoolsAdapter(persistent=persistent)
            self.hkexnews = self.hkexnews or HkexnewsAdapter(
                allow_download=allow_hkex_download, persistent=persistent
            )
            self.ah_disclosure = self.ah_disclosure or AhDisclosureAdapter(
                persistent=persistent
            )
            # Reuse domain gateways for specialized execution/normalization,
            # but share adapter instances and the unified runtime registry view.
            fin_path = self.bundle.source_catalogs["financial"]
            mkt_path = self.bundle.source_catalogs["market"]
            self._financial = RealMcpGateway(
                catalog_path=fin_path,
                aktools=self.aktools,
                hkexnews=self.hkexnews,
                allow_hkex_download=allow_hkex_download,
                persistent=persistent,
            )
            self._market = MarketRealMcpGateway(
                catalog_path=mkt_path,
                aktools=self.aktools,
                persistent=persistent,
            )

    def open(self) -> "UnifiedRealMcpGateway":
        if self.mock_transport is not None:
            self._opened = True
            return self
        if not self._open_adapters:
            self._opened = True
            return self
        assert self.aktools is not None
        assert self.hkexnews is not None
        assert self.ah_disclosure is not None
        self.aktools.open()
        self.hkexnews.open()
        self.ah_disclosure.open()
        self._opened = True
        return self

    def begin_task(self) -> None:
        """Reset per-task adapter budgets before each workflow run.

        HKEX download budget is max 1 announcement per task — not shared across
        earlier tasks or earlier runs of the same persistent adapter.
        """
        if self.hkexnews is not None and hasattr(self.hkexnews, "reset_download_budget"):
            self.hkexnews.reset_download_budget()

    def close(self) -> None:
        if self.mock_transport is not None:
            self._opened = False
            return
        errors = []
        for adapter in (self.ah_disclosure, self.hkexnews, self.aktools):
            if adapter is None:
                continue
            try:
                adapter.close()
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{type(exc).__name__}: {exc}")
        self._opened = False
        if errors:
            raise RuntimeError("Unified gateway close errors: " + "; ".join(errors))

    def __enter__(self) -> "UnifiedRealMcpGateway":
        return self.open()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def invoke(
        self, operation_id: str, arguments: Optional[Dict[str, Any]] = None
    ) -> RealMcpResult:
        arguments = dict(arguments or {})
        self.invoke_count += 1

        if operation_id in BLOCKED_MARKET_OPS:
            return self._fail(
                operation_id,
                "unknown",
                arguments,
                f"Blocked FTShare/deferred Market operation: {operation_id}",
                error_type="blocked_operation",
            )

        rec = self.records.get(operation_id)
        if rec is not None and rec.is_blocked:
            return self._fail(
                operation_id,
                rec.server_id,
                arguments,
                f"Blocked/rejected operation not executable: {operation_id}",
                error_type="blocked_operation",
            )

        if operation_id not in self.registry or operation_id not in self.raw_ops:
            return self._fail(
                operation_id,
                "unknown",
                arguments,
                f"Unknown Real-MCP operation_id (fail-closed): {operation_id}",
                error_type="unknown_operation",
            )

        if self.mock_transport is not None:
            return self.mock_transport.invoke(operation_id, arguments)

        raw = self.raw_ops[operation_id]
        server_id = str(raw.get("server_id") or rec.server_id if rec else "unknown")
        dispatch = rec.dispatch_domain if rec else "financial"
        execution = dict(raw.get("execution") or {})
        kind = execution.get("kind")

        # Market-specialized path (cutoff / adjustment normalization).
        if dispatch == "market" and self._market is not None:
            if operation_id in self._market.registry:
                return self._market.invoke(operation_id, arguments)

        # Shared Financial gateway covers overlapping hkexnews / aktools ops.
        if self._financial is not None and operation_id in self._financial.registry:
            return self._financial.invoke(operation_id, arguments)

        # News-only / ah_disclosure / remaining aktools via direct dispatch.
        if kind == "aktools_get_data":
            assert self.aktools is not None
            interface = execution.get("interface") or raw.get(
                "underlying_aktools_interface"
            ) or raw.get("underlying_interface")
            if not interface:
                return self._fail(
                    operation_id, server_id, arguments, "Missing aktools interface"
                )
            param_keys = list(execution.get("param_keys") or [])
            params = {
                k: arguments[k]
                for k in param_keys
                if k in arguments and arguments[k] is not None
            }
            offset = int(arguments.get("offset", 0) or 0)
            max_rows = int(
                arguments.get("max_rows", arguments.get("limit", 20)) or 20
            )
            return self.aktools.get_data(
                interface=str(interface),
                params=params,
                offset=offset,
                max_rows=max_rows,
                operation_id=operation_id,
            )

        if kind == "mcp_tool":
            tool = execution.get("tool") or raw.get("exposed_mcp_tool")
            arg_map = dict(execution.get("arg_map") or {})
            call_args: Dict[str, Any] = {}
            for dest, src in arg_map.items():
                if src in arguments and arguments[src] is not None:
                    call_args[dest] = arguments[src]
            if not arg_map:
                call_args = {
                    k: v for k, v in arguments.items() if v is not None
                }
            if server_id == "aktools":
                assert self.aktools is not None
                result = self.aktools.call_mcp_tool(str(tool), call_args)
                result.operation_id = operation_id
                return result
            if server_id == "mcp_hkexnews":
                assert self.hkexnews is not None
                return self.hkexnews.call_mcp_tool(
                    str(tool), call_args, operation_id=operation_id
                )
            if server_id == "ah_disclosure":
                assert self.ah_disclosure is not None
                return self.ah_disclosure.call_mcp_tool(
                    str(tool), call_args, operation_id=operation_id
                )
            return self._fail(
                operation_id,
                server_id,
                arguments,
                f"Unsupported server_id={server_id}",
            )

        return self._fail(
            operation_id,
            server_id,
            arguments,
            f"Unsupported or missing execution.kind={kind!r} (fail-closed)",
            error_type="unavailable",
        )

    @staticmethod
    def _fail(
        operation_id: str,
        server_id: Any,
        arguments: Dict[str, Any],
        message: str,
        *,
        error_type: str = "unavailable",
    ) -> RealMcpResult:
        return RealMcpResult(
            success=False,
            server_id=str(server_id or "unknown"),
            operation_id=operation_id,
            arguments=arguments,
            error=message,
            error_type=error_type,
            protocol_success=False,
            execution_success=False,
            real_mcp=True,
        )
