"""Offline tests for Financial taxonomy v1.3 (no LLM / no paid API)."""

from __future__ import annotations

import json
import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEV = ROOT / "benchmarks" / "questions" / "financial_real_mcp_coverage_v1_3.jsonl"
HELD = ROOT / "benchmarks" / "heldout" / "financial_test_30_v1_3.jsonl"
DEV_V12 = ROOT / "benchmarks" / "questions" / "financial_real_mcp_coverage_v1_2.jsonl"
HELD_V12 = ROOT / "benchmarks" / "heldout" / "financial_test_30_v1_2.jsonl"
SMOKE = ROOT / "docs" / "financial_hkexnews_dependency_smoke_v1"


def _rows(path: Path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_docs_and_smoke_exist():
    assert (ROOT / "docs" / "financial_taxonomy_audit_v1_3.md").exists()
    assert (SMOKE / "observed_dependency_edges.json").exists()
    assert (SMOKE / "tool_input_schemas.json").exists()
    assert DEV.exists() and HELD.exists()


def test_validator_ok():
    ns = runpy.run_path(str(ROOT / "scripts" / "validate_financial_taxonomy_v1_3.py"))
    assert ns["main"]() == 0


def test_numerical_gold_equality_vs_v1_2():
    for old, new in ((DEV_V12, DEV), (HELD_V12, HELD)):
        for a, b in zip(_rows(old), _rows(new)):
            assert a["id"] == b["id"]
            assert a["question"] == b["question"]
            assert a["reference_answer"] == b["reference_answer"]
            assert a.get("gold_values") == b.get("gold_values")
            assert a.get("gold_status") == b.get("gold_status")


def test_supersession_and_run_history():
    for row in _rows(DEV):
        assert row["supersession_status"] == "active"
        assert row["prior_file_status"] == "superseded_before_any_agent_run"
        assert "nonformal" in row.get("development_exposure_history", "")
        assert row["taxonomy_version"] == "financial_taxonomy_v1_3"
    for row in _rows(HELD):
        assert row["supersession_status"] == "active"
        assert row["prior_file_status"] == "superseded_before_any_agent_run"
        assert row["heldout_agent_run_status"] == "never_agent_run"
        assert row.get("locked") is True


def test_exact_l3_field_edges():
    expected = {
        "fin_cf_003",
        "fin_cf_010",
        "fin_cf_011",
        "fin_cf_013",
        "fin_cf_016",
        "fin_cf_020",
        "fin_ht_011",
        "fin_ht_012",
        "fin_ht_013",
        "fin_ht_014",
        "fin_ht_015",
        "fin_ht_016",
    }
    got = set()
    for row in _rows(DEV) + _rows(HELD):
        if row.get("etom_level") == "L3" and row.get("etom_alignment") == "exact":
            got.add(row["id"])
            assert row["server_count"] == 1
            assert row["operation_count"] >= 2
            assert row["has_output_input_dependency"] is True
            ops = set(row.get("gold_operations") or [])
            assert "hkexnews.download_hkexnews_announcement" in ops
            assert "hkexnews.parse_hkexnews_announcement" in ops
            for e in row["dependency_graph"]:
                assert e.get("live_verified_dependency") is True
                for k in (
                    "producer_operation",
                    "producer_output_field",
                    "consumer_operation",
                    "consumer_input_field",
                ):
                    assert e.get(k)
    assert got == expected


def test_no_exact_l4():
    ids = {
        r["id"]
        for r in _rows(DEV) + _rows(HELD)
        if r.get("etom_level") == "L4" and r.get("etom_alignment") == "exact"
    }
    assert ids == set()
    for qid in ("fin_ht_017", "fin_ht_018"):
        row = next(r for r in _rows(HELD) if r["id"] == qid)
        assert row["etom_level"] == "L4"
        assert row["etom_alignment"] == "adapted"
        assert row["has_output_input_dependency"] is False
        assert row.get("dependency_graph") == []


def test_v1_2_marked_superseded():
    for path in (DEV_V12, HELD_V12):
        for row in _rows(path):
            assert row["supersession_status"] == "superseded_before_any_agent_run"
            assert "v1_3" in row.get("superseded_by", "")
