#!/usr/bin/env python3
"""Run Real-MCP smokes without paid LLM (writes gitignored summary only)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "real_mcp_smoke"
OUT.mkdir(parents=True, exist_ok=True)


def main() -> int:
    code = pytest.main(
        [
            str(ROOT / "tests" / "test_real_mcp_smoke.py"),
            str(ROOT / "tests" / "test_real_mcp_catalog.py"),
            "-q",
            "--tb=short",
        ]
    )
    summary = {
        "pytest_exit_code": code,
        "note": "No paid LLM. Live smokes skip if MCP binaries missing.",
        "catalog": "configs/operations/financial_real_mcp.yaml",
    }
    (OUT / "smoke_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return int(code)


if __name__ == "__main__":
    sys.exit(main())
