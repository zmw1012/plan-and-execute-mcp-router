#!/usr/bin/env python3
"""Entry point for MCP-Zero × ETOM runners."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hk_equity_mcp.mcp_zero_etom.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
