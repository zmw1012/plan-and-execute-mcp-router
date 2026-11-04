"""Market Real-MCP gateway: AKTools-only, cutoff post-filter, fail-closed."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

from hk_equity_mcp.operation_resolver.schemas import OperationSpec
from hk_equity_mcp.real_mcp.adapters.aktools import AktoolsAdapter
from hk_equity_mcp.real_mcp.catalog import load_real_mcp_catalog, load_real_mcp_registry
from hk_equity_mcp.real_mcp.market_normalize import (
    CURRENCY_DEFAULT,
    CURRENCY_PROVENANCE,
    MARKET_CUTOFF,
    MARKET_TIMEZONE,
    extract_rows,
    filter_cutoff,
    market_result_schema,
    normalize_bar,
    parse_session_date,
    previous_valid_session,
    row_date,
    sort_by_date,
    to_ymd,
)
from hk_equity_mcp.real_mcp.types import RealMcpResult

DEFAULT_MARKET_CATALOG = (
    Path(__file__).resolve().parents[3]
    / "configs"
    / "operations"
    / "market_real_mcp_v1.yaml"
)

BLOCKED_MARKET_OPS = frozenset(
    {
        "ftshare.hk_hist",
        "ftshare.stock_hk_daily",
        "aktools.stock_hk_hist_min_em",
    }
)


class MarketRealMcpGateway:
    """Deterministic Market Real-MCP invoke path (no mock FinancialServer)."""

    def __init__(
        self,
        catalog_path: Optional[Path] = None,
        aktools: Optional[AktoolsAdapter] = None,
        *,
        persistent: bool = True,
        cutoff: date = MARKET_CUTOFF,
    ):
        self.catalog_path = Path(catalog_path) if catalog_path else DEFAULT_MARKET_CATALOG
        self.catalog = load_real_mcp_catalog(self.catalog_path)
        self.registry: Dict[str, OperationSpec] = load_real_mcp_registry(
            self.catalog_path
        )
        self._raw_ops = {
            op["operation_id"]: op for op in self.catalog.get("operations", [])
        }
        for blocked in BLOCKED_MARKET_OPS:
            if blocked in self.registry or blocked in self._raw_ops:
                raise ValueError(
                    f"Blocked/deferred operation must not be in active Market "
                    f"registry: {blocked}"
                )
        self.aktools = aktools or AktoolsAdapter(persistent=persistent)
        self.cutoff = cutoff
        self._opened = False

    def open(self) -> "MarketRealMcpGateway":
        self.aktools.open()
        self._opened = True
        return self

    def close(self) -> None:
        self.aktools.close()
        self._opened = False

    def __enter__(self) -> "MarketRealMcpGateway":
        return self.open()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def invoke(
        self, operation_id: str, arguments: Optional[Dict[str, Any]] = None
    ) -> RealMcpResult:
        arguments = dict(arguments or {})
        if operation_id in BLOCKED_MARKET_OPS:
            return self._fail(
                operation_id,
                "unknown",
                arguments,
                f"Blocked/deferred Market operation: {operation_id}",
                error_type="blocked_operation",
            )

        op = self.registry.get(operation_id)
        raw = self._raw_ops.get(operation_id)
        if op is None or raw is None:
            return self._fail(
                operation_id,
                "unknown",
                arguments,
                f"Unknown Market Real-MCP operation_id: {operation_id}",
                error_type="unknown_operation",
            )

        execution = dict(raw.get("execution") or {})
        kind = execution.get("kind")
        server_id = str(raw.get("server_id") or "aktools")

        for p in op.optional_params():
            if p.name not in arguments and p.default is not None:
                arguments[p.name] = p.default
        for k, v in dict(execution.get("defaults") or {}).items():
            arguments.setdefault(k, v)

        # Enforce explicit adjust for hist; never auto-mix.
        if operation_id == "aktools.stock_hk_hist":
            if "adjust" not in arguments or arguments["adjust"] is None:
                arguments["adjust"] = ""
            if arguments["adjust"] not in ("", "qfq", "hfq"):
                return self._fail(
                    operation_id,
                    server_id,
                    arguments,
                    f"Invalid adjust mode: {arguments['adjust']!r}",
                    error_type="validation_error",
                )

        if kind == "aktools_get_data":
            return self._invoke_get_data(operation_id, raw, execution, arguments)
        if kind == "mcp_tool":
            tool = execution.get("tool") or raw.get("exposed_mcp_tool")
            arg_map = dict(execution.get("arg_map") or {})
            call_args: Dict[str, Any] = {}
            for dest, src in arg_map.items():
                if src in arguments and arguments[src] is not None:
                    call_args[dest] = arguments[src]
            result = self.aktools.call_mcp_tool(str(tool), call_args)
            result.operation_id = operation_id
            schema = market_result_schema(
                success=result.success,
                source_server=server_id,
                source_operation=operation_id,
                provider=raw.get("provider"),
                symbol=arguments.get("symbol") or arguments.get("name"),
                requested_date_or_range=None,
                effective_session_date=None,
                currency_or_unit=None,
                currency_provenance=None,
                adjustment_mode=None,
                data=result.normalized,
                provenance={
                    "exposed_mcp_tool": tool,
                    "raw_arguments": dict(call_args),
                    "server_version": result.server_version,
                    "real_mcp": True,
                    "mock_fallback": False,
                },
                error=result.error,
            )
            result.normalized = schema
            return result

        return self._fail(
            operation_id,
            server_id,
            arguments,
            f"Unsupported execution.kind={kind}",
            error_type="unavailable",
        )

    def _invoke_get_data(
        self,
        operation_id: str,
        raw: Dict[str, Any],
        execution: Dict[str, Any],
        arguments: Dict[str, Any],
    ) -> RealMcpResult:
        server_id = str(raw.get("server_id") or "aktools")
        interface = execution.get("interface") or raw.get(
            "underlying_aktools_interface"
        )
        if not interface:
            return self._fail(
                operation_id, server_id, arguments, "Missing interface"
            )

        param_keys = list(execution.get("param_keys") or [])
        params = {
            k: arguments[k]
            for k in param_keys
            if k in arguments and arguments[k] is not None
        }
        # Keep empty-string adjust for stock_hk_hist.
        if operation_id == "aktools.stock_hk_hist" and "adjust" in arguments:
            params["adjust"] = arguments["adjust"]

        offset = int(
            arguments.get("offset", execution.get("offset", 0)) or 0
        )
        max_rows = int(
            arguments.get(
                "max_rows",
                execution.get("max_rows", arguments.get("limit", 20)),
            )
            or 20
        )
        # End-date must not exceed cutoff for hist-style windows.
        if "end_date" in params:
            end = parse_session_date(params["end_date"])
            if end is not None and end > self.cutoff:
                return self._fail(
                    operation_id,
                    server_id,
                    arguments,
                    f"end_date {end} exceeds Market cutoff {self.cutoff}",
                    error_type="cutoff_violation",
                )

        result = self.aktools.get_data(
            interface=str(interface),
            params=params,
            offset=offset,
            max_rows=max_rows,
            operation_id=operation_id,
        )
        if not result.success:
            return result

        rows = extract_rows(result.normalized)
        post_filter = execution.get("post_filter")
        if post_filter == "cutoff_inclusive":
            rows = filter_cutoff(rows, cutoff=self.cutoff)
            # Guard: never return post-cutoff dates.
            for row in rows:
                d = row_date(row)
                if d is not None and d > self.cutoff:
                    return self._fail(
                        operation_id,
                        server_id,
                        arguments,
                        f"post-filter leaked date {d} > cutoff",
                        error_type="cutoff_violation",
                    )

        rows = sort_by_date(rows)
        bars = [normalize_bar(r) for r in rows]

        requested = arguments.get("target_date") or arguments.get("requested_date")
        requested_range = None
        if "start_date" in arguments or "end_date" in arguments:
            requested_range = {
                "start_date": arguments.get("start_date"),
                "end_date": arguments.get("end_date"),
            }
        requested_date_or_range = requested or requested_range

        effective: Optional[str] = None
        prev_triggered: Optional[bool] = None
        selected_data: Any = bars

        if requested and execution.get("previous_valid_day"):
            req = parse_session_date(requested)
            if req is None:
                return self._fail(
                    operation_id,
                    server_id,
                    arguments,
                    f"Invalid target_date: {requested}",
                    error_type="validation_error",
                )
            # No selecting future relative to cutoff / requested.
            if req > self.cutoff:
                req = self.cutoff
            hit = previous_valid_session(rows, req, cutoff=self.cutoff)
            if hit is None:
                return self._fail(
                    operation_id,
                    server_id,
                    arguments,
                    f"No session <= {req} and <= cutoff",
                    error_type="no_session",
                )
            bar = normalize_bar(hit)
            effective = bar["date"]
            prev_triggered = effective != to_ymd(req)
            # Forbid forward-fill: effective must be <= requested.
            if parse_session_date(effective) and parse_session_date(effective) > req:
                return self._fail(
                    operation_id,
                    server_id,
                    arguments,
                    "forward-fill forbidden",
                    error_type="forward_fill_forbidden",
                )
            selected_data = bar
        elif len(bars) == 1:
            effective = bars[0]["date"]
            selected_data = bars[0]
        elif bars:
            effective = bars[-1]["date"]

        adjust_mode = arguments.get("adjust")
        if operation_id == "aktools.stock_hk_hist":
            adjust_mode = arguments.get("adjust", "")

        is_price_like = bool(raw.get("formal_gold_price_source")) or operation_id in {
            "aktools.stock_hk_daily",
            "aktools.stock_hk_index_daily_em",
            "aktools.stock_hk_index_daily_sina",
            "aktools.stock_hk_hist",
        }
        currency = CURRENCY_DEFAULT if is_price_like else None
        currency_prov = CURRENCY_PROVENANCE if is_price_like else None
        if operation_id.startswith("aktools.stock_hk_index"):
            currency = "index_points"
            currency_prov = "index_unit"

        schema = market_result_schema(
            success=True,
            source_server=server_id,
            source_operation=operation_id,
            provider=raw.get("provider"),
            symbol=arguments.get("symbol"),
            requested_date_or_range=requested_date_or_range,
            effective_session_date=effective,
            timezone=MARKET_TIMEZONE,
            currency_or_unit=currency,
            currency_provenance=currency_prov,
            adjustment_mode=adjust_mode if operation_id == "aktools.stock_hk_hist" else None,
            data=selected_data,
            provenance={
                "exposed_mcp_tool": "get_data",
                "underlying_aktools_interface": interface,
                "raw_invocation_arguments": {
                    "name": interface,
                    "params": params,
                    "offset": offset,
                    "max_rows": max_rows,
                },
                "row_count_after_filter": len(bars),
                "cutoff": to_ymd(self.cutoff),
                "server_version": result.server_version,
                "real_mcp": True,
                "mock_fallback": False,
                "tool_trade_date_hist_sina_is_hkex_official_calendar": False,
            },
            previous_valid_day_triggered=prev_triggered,
            error=None,
        )
        result.normalized = schema
        result.evidence = list(result.evidence or []) + [
            {
                "type": "market_post_filter",
                "cutoff": to_ymd(self.cutoff),
                "previous_valid_day_triggered": prev_triggered,
                "effective_session_date": effective,
            }
        ]
        return result

    def _fail(
        self,
        operation_id: str,
        server_id: str,
        arguments: Dict[str, Any],
        message: str,
        *,
        error_type: str = "unavailable",
    ) -> RealMcpResult:
        schema = market_result_schema(
            success=False,
            source_server=str(server_id or "unknown"),
            source_operation=operation_id,
            provider=None,
            symbol=arguments.get("symbol"),
            requested_date_or_range=arguments.get("target_date")
            or {
                "start_date": arguments.get("start_date"),
                "end_date": arguments.get("end_date"),
            },
            effective_session_date=None,
            currency_or_unit=None,
            currency_provenance=None,
            adjustment_mode=arguments.get("adjust"),
            data=None,
            provenance={"real_mcp": True, "mock_fallback": False},
            error=message,
        )
        return RealMcpResult(
            success=False,
            server_id=str(server_id or "unknown"),
            operation_id=operation_id,
            arguments=arguments,
            normalized=schema,
            error=message,
            error_type=error_type,
            protocol_success=False,
            execution_success=False,
            real_mcp=True,
        )


def require_market_real_mcp_ready(catalog_path: Optional[Path] = None) -> Path:
    """Fail-closed readiness for Market catalog (AKTools only; no mock fallback)."""
    from hk_equity_mcp.real_mcp.adapters.aktools import default_aktools_command
    from hk_equity_mcp.real_mcp.client import RealMcpError, resolve_command
    from hk_equity_mcp.real_mcp.readiness import mcp_sdk_available

    errors: List[str] = []
    cfg = Path(catalog_path) if catalog_path else DEFAULT_MARKET_CATALOG
    if not cfg.exists():
        errors.append(f"catalog missing: {cfg}")
    else:
        try:
            catalog = load_real_mcp_catalog(cfg)
            ops = catalog.get("operations") or []
            if len(ops) < 1:
                errors.append("catalog has zero operations")
            for op in ops:
                oid = op.get("operation_id")
                if oid in BLOCKED_MARKET_OPS:
                    errors.append(f"blocked op in active catalog: {oid}")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"catalog unreadable: {exc}")

    if not mcp_sdk_available():
        errors.append("Python package 'mcp' is not importable")

    try:
        resolve_command(default_aktools_command())
    except RealMcpError as exc:
        errors.append(f"aktools-mcp: {exc}")

    if errors:
        raise RuntimeError(
            "Market Real-MCP fail-closed: "
            + "; ".join(errors)
            + ". Refusing mock/local FinancialServer fallback."
        )
    return cfg
