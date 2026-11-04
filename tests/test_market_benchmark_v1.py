"""Offline structural tests for Market Dev15 / Held-out20 (no live MCP / no LLM)."""

from __future__ import annotations

import json
import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEV = ROOT / "benchmarks" / "questions" / "market_development_15_v1.jsonl"
HELD = ROOT / "benchmarks" / "heldout" / "market_test_20_v1.jsonl"
FREEZE_DOC = ROOT / "docs" / "market_real_mcp_v1_freeze.md"
AUDIT_DOC = ROOT / "docs" / "market_real_mcp_provenance_audit_v1.md"


def _rows(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_files_exist():
    assert DEV.exists()
    assert HELD.exists()
    assert FREEZE_DOC.exists()
    assert AUDIT_DOC.exists()
    assert (ROOT / "docs" / "market_development_15_v1_design.md").exists()
    assert (ROOT / "docs" / "market_test_20_v1_design.md").exists()


def test_validator_script_ok():
    ns = runpy.run_path(str(ROOT / "scripts" / "validate_market_benchmark_v1.py"))
    assert ns["main"]() == 0


def test_no_blocked_ops_in_gold():
    blocked = {
        "ftshare.hk_hist",
        "ftshare.stock_hk_daily",
        "aktools.stock_hk_hist_min_em",
    }
    for row in _rows(DEV) + _rows(HELD):
        for op in row.get("gold_operations") or []:
            assert op not in blocked


def test_l5_and_locks():
    for row in _rows(HELD):
        assert row["locked"] is True
        assert row["dataset_split"] == "heldout"
    l5 = [r for r in _rows(DEV) + _rows(HELD) if r["etom_level"] == "L5"]
    assert len(l5) == 6
    for row in l5:
        assert row.get("confirmed_infeasible") is True
        assert row.get("gold_operations") in ([], None)
