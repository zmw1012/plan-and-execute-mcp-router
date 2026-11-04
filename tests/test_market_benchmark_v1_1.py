"""Offline tests for Market Dev15/Held-out20 v1.1 taxonomy correction (no LLM)."""

from __future__ import annotations

import json
import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEV = ROOT / "benchmarks" / "questions" / "market_development_15_v1_1.jsonl"
HELD = ROOT / "benchmarks" / "heldout" / "market_test_20_v1_1.jsonl"
DEV_V1 = ROOT / "benchmarks" / "questions" / "market_development_15_v1.jsonl"
HELD_V1 = ROOT / "benchmarks" / "heldout" / "market_test_20_v1.jsonl"


def _rows(path: Path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_docs_and_files_exist():
    assert DEV.exists() and HELD.exists()
    assert (ROOT / "docs" / "market_real_mcp_v1_1_freeze.md").exists()
    assert (ROOT / "docs" / "market_development_15_v1_1_design.md").exists()
    assert (ROOT / "docs" / "market_test_20_v1_1_design.md").exists()
    assert (ROOT / "docs" / "market_taxonomy_audit_v1_1.md").exists()
    assert (ROOT / "docs" / "financial_etom_taxonomy_readonly_audit.md").exists()


def test_validator_ok():
    ns = runpy.run_path(str(ROOT / "scripts" / "validate_market_benchmark_v1_1.py"))
    assert ns["main"]() == 0


def test_numerical_gold_unchanged_vs_v1():
    for a, b in ((DEV_V1, DEV), (HELD_V1, HELD)):
        for x, y in zip(_rows(a), _rows(b)):
            assert x["id"] == y["id"]
            assert x["reference_answer"] == y["reference_answer"]
            xv = json.loads(json.dumps(x.get("gold_values")))
            yv = json.loads(json.dumps(y.get("gold_values")))
            for g in (xv or []) + (yv or []):
                if isinstance(g, dict):
                    g.pop("adjustment_mode", None)
            assert xv == yv


def test_no_forced_etom_l4_and_l3_semantics():
    for row in _rows(DEV) + _rows(HELD):
        assert row.get("etom_level") != "L4"
        if row.get("etom_level") == "L3":
            assert row["server_count"] == 1
            assert row["operation_count"] >= 2
            assert row["has_output_input_dependency"] is True
        if row.get("etom_level") == "L5" and row.get("etom_alignment") == "exact":
            assert row["robustness_subtype"] == "capability_gap"
            assert row.get("gold_operations") == []


def test_adjustment_mode_schema():
    for row in _rows(DEV) + _rows(HELD):
        assert "adjustment_mode_reason" in row
        if row.get("robustness_subtype") in {
            "capability_gap",
            "temporal_policy",
            "unavailable_provider",
        }:
            assert row["adjustment_mode"] is None
        elif row.get("task_type") == "identity":
            assert row["adjustment_mode"] is None
        elif row.get("company") == "HSI":
            assert row["adjustment_mode"] == "not_applicable_index_level"
        else:
            assert row["adjustment_mode"] is not None
        assert "± ±" not in row["question"]
