"""Unified Real MCP gateway used by Operation-Visible invoke_api."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from hk_equity_mcp.operation_resolver.schemas import OperationSpec
from hk_equity_mcp.real_mcp.adapters.aktools import AktoolsAdapter
from hk_equity_mcp.real_mcp.adapters.hkexnews import HkexnewsAdapter
from hk_equity_mcp.real_mcp.catalog import load_real_mcp_catalog, load_real_mcp_registry
from hk_equity_mcp.real_mcp.types import RealMcpResult


class RealMcpGateway:
    """Route curated operation_ids to real MCP adapters (no mock FinancialServer)."""

    def __init__(
        self,
        catalog_path: Optional[Path] = None,
        aktools: Optional[AktoolsAdapter] = None,
        hkexnews: Optional[HkexnewsAdapter] = None,
        allow_hkex_download: bool = False,
    ):
        self.catalog_path = catalog_path
        self.catalog = load_real_mcp_catalog(catalog_path)
        self.registry: Dict[str, OperationSpec] = load_real_mcp_registry(catalog_path)
        self._raw_ops = {
            op["operation_id"]: op for op in self.catalog.get("operations", [])
        }
        self.aktools = aktools or AktoolsAdapter()
        self.hkexnews = hkexnews or HkexnewsAdapter(allow_download=allow_hkex_download)

    def invoke(self, operation_id: str, arguments: Optional[Dict[str, Any]] = None) -> RealMcpResult:
        arguments = dict(arguments or {})
        op = self.registry.get(operation_id)
        raw = self._raw_ops.get(operation_id)
        if op is None or raw is None:
            return RealMcpResult(
                success=False,
                server_id="unknown",
                operation_id=operation_id,
                arguments=arguments,
                error=f"Unknown Real-MCP operation_id: {operation_id}",
                error_type="unknown_operation",
                protocol_success=False,
                execution_success=False,
            )

        execution = dict(raw.get("execution") or {})
        kind = execution.get("kind")
        server_id = raw.get("server_id")

        # Apply defaults from catalog for missing optional args.
        for p in op.optional_params():
            if p.name not in arguments and p.default is not None:
                arguments[p.name] = p.default

        if kind == "aktools_get_data":
            interface = execution.get("interface") or raw.get(
                "underlying_aktools_interface"
            )
            if not interface:
                return self._unavailable(operation_id, server_id, arguments, "Missing interface")
            param_keys = list(execution.get("param_keys") or [])
            params = {k: arguments[k] for k in param_keys if k in arguments and arguments[k] is not None}
            # Drop empty optional indicator alias noise.
            params = {k: v for k, v in params.items() if v != ""}
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
            if server_id == "aktools":
                result = self.aktools.call_mcp_tool(str(tool), call_args)
                result.operation_id = operation_id
                return result
            if server_id == "mcp_hkexnews":
                result = self.hkexnews.call_mcp_tool(
                    str(tool), call_args, operation_id=operation_id
                )
                return result
            return self._unavailable(
                operation_id, server_id, arguments, f"Unsupported server_id={server_id}"
            )

        return self._unavailable(
            operation_id, server_id, arguments, f"Unsupported execution.kind={kind}"
        )

    def _unavailable(
        self,
        operation_id: str,
        server_id: Any,
        arguments: Dict[str, Any],
        message: str,
    ) -> RealMcpResult:
        return RealMcpResult(
            success=False,
            server_id=str(server_id or "unknown"),
            operation_id=operation_id,
            arguments=arguments,
            error=message,
            error_type="unavailable",
            protocol_success=False,
            execution_success=False,
        )
