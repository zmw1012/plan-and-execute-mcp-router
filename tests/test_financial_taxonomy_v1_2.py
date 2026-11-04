"""Offline tests for Financial taxonomy v1.2 (no LLM / no live MCP)."""

from __future__ import annotations

import json
import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEV = ROOT / "benchmarks" / "questions" / "financial_real_mcp_coverage_v1_2.jsonl"
HELD = ROOT / "benchmarks" / "heldout" / "financial_test_30_v1_2.jsonl"
DEV_V1 = ROOT / "benchmarks" / "questions" / "financial_real_mcp_coverage_v1.jsonl"
HELD_V11 = ROOT / "benchmarks" / "heldout" / "financial_test_30_v1_1.jsonl"


def _rows(path: Path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_docs_exist():
    assert (ROOT / "docs" / "financial_taxonomy_audit_v1_2.md").exists()
    assert (ROOT / "docs" / "financial_development_v1_2_design.md").exists()
    assert (ROOT / "docs" / "financial_heldout_v1_2_design.md").exists()
    assert DEV.exists() and HELD.exists()


def test_validator_ok():
    ns = runpy.run_path(str(ROOT / "scripts" / "validate_financial_taxonomy_v1_2.py"))
    assert ns["main"]() == 0


def test_numerical_gold_equality():
    for old, new in ((DEV_V1, DEV), (HELD_V11, HELD)):
        for a, b in zip(_rows(old), _rows(new)):
            assert a["id"] == b["id"]
            assert a["question"] == b["question"]
            assert a["reference_answer"] == b["reference_answer"]
            assert a.get("gold_values") == b.get("gold_values")
            assert a.get("gold_status") == b.get("gold_status")


def test_run_history_labels():
    for row in _rows(DEV):
        assert row["prior_file_status"] == "taxonomy_corrected_after_nonformal_pilot"
        assert row["supersession_status"] == "active"
    for row in _rows(HELD):
        assert row["prior_file_status"] == "superseded_before_any_agent_run"
        assert row["supersession_status"] == "active"
        assert row.get("locked") is True


def test_exact_l3_l4_semantics():
    for row in _rows(DEV) + _rows(HELD):
        if row.get("etom_alignment") != "exact":
            continue
        if row.get("etom_level") == "L3":
            assert row["server_count"] == 1
            assert row["operation_count"] >= 2
            assert row["has_output_input_dependency"] is True
            assert len(row.get("dependency_graph") or []) >= 1
        if row.get("etom_level") == "L4":
            assert row["server_count"] >= 2
            assert row["operation_count"] >= 2
            assert row["has_output_input_dependency"] is True
            edges = row.get("dependency_graph") or []
            assert any(
                str(e.get("from", "")).split(".")[0]
                != str(e.get("to", "")).split(".")[0]
                for e in edges
            )
        if row.get("etom_level") == "L5":
            assert row["robustness_subtype"] == "capability_gap"
            assert row.get("gold_operations") == []


def test_true_l4_ids():
    ids = {
        r["id"]
        for r in _rows(DEV) + _rows(HELD)
        if r.get("etom_level") == "L4" and r.get("etom_alignment") == "exact"
    }
    assert ids == {"fin_ht_017", "fin_ht_018"}
