"""Fail-closed readiness checks for Real-MCP execution backend."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from hk_equity_mcp.real_mcp.adapters.aktools import default_aktools_command
from hk_equity_mcp.real_mcp.adapters.hkexnews import default_hkexnews_command
from hk_equity_mcp.real_mcp.catalog import DEFAULT_CATALOG_PATH, load_real_mcp_catalog
from hk_equity_mcp.real_mcp.client import RealMcpError, resolve_command


def mcp_sdk_available() -> bool:
    try:
        import mcp  # noqa: F401
        from mcp import ClientSession  # noqa: F401
        from mcp.client.stdio import stdio_client  # noqa: F401

        return True
    except Exception:
        return False


def require_real_mcp_ready(catalog_path: Optional[Path] = None) -> Path:
    """Raise RuntimeError if Real-MCP cannot run; never silently fall back."""
    errors: List[str] = []
    cfg_path = Path(catalog_path) if catalog_path else DEFAULT_CATALOG_PATH
    if not cfg_path.exists():
        errors.append(f"catalog missing: {cfg_path}")
    else:
        try:
            catalog = load_real_mcp_catalog(cfg_path)
            ops = catalog.get("operations") or []
            if len(ops) < 1:
                errors.append("catalog has zero operations")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"catalog unreadable: {exc}")

    if not mcp_sdk_available():
        errors.append(
            "Python package 'mcp' is not importable in this interpreter "
            "(install optional extra real_mcp)"
        )

    for label, command in (
        ("aktools-mcp", default_aktools_command()),
        ("agentladle-mcp-hkexnews", default_hkexnews_command()),
    ):
        try:
            resolve_command(command)
        except RealMcpError as exc:
            errors.append(f"{label}: {exc}")

    if errors:
        raise RuntimeError(
            "Real-MCP execution backend fail-closed: "
            + "; ".join(errors)
            + ". Refusing to fall back to controlled_local / FinancialServer."
        )
    return cfg_path
