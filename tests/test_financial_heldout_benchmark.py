"""Deterministic tests for Financial held-out Test 30 v1.

No LLM, embedding, MCP live calls, or experiment execution.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
HELDOUT = ROOT / "benchmarks" / "heldout" / "financial_test_30_v1.jsonl"
DEV = ROOT / "benchmarks" / "questions" / "financial_mcp_free_30_v1.jsonl"
DEV_COV = ROOT / "benchmarks" / "questions" / "financial_real_mcp_coverage_v1.jsonl"
CATALOG = ROOT / "configs" / "operations" / "financial_real_mcp.yaml"
VALIDATOR = ROOT / "scripts" / "validate_financial_heldout_test_v1.py"
DESIGN = ROOT / "docs" / "financial_test_30_v1_design.md"
COVERAGE = ROOT / "docs" / "financial_test_30_v1_coverage.md"
GAPS = ROOT / "docs" / "financial_test_30_v1_mcp_gap_proposals.md"


def _load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_heldout_artifacts_exist():
    for path in (HELDOUT, DESIGN, COVERAGE, GAPS, VALIDATOR):
        assert path.is_file(), f"missing artifact: {path}"


def test_heldout_distributions_and_lock_metadata():
    rows = _load_jsonl(HELDOUT)
    assert len(rows) == 30
    assert len({r["id"] for r in rows}) == 30
    assert Counter(r["etom_level"] for r in rows) == {
        "L1": 6,
        "L2": 6,
        "L3": 6,
        "L4": 6,
        "L5": 6,
    }
    assert Counter(r["company"] for r in rows) == {
        "Tencent": 10,
        "HSBC": 10,
        "Alibaba": 10,
    }
    assert Counter(r["answerability_class"] for r in rows) == {
        "structured_answerable": 10,
        "evidence_answerable": 14,
        "intentionally_infeasible": 6,
    }
    for r in rows:
        assert r["dataset_split"] == "heldout_test"
        assert r["benchmark_role"] == "final_evaluation_only"
        assert r["locked"] is True
        assert r["domain"] == "financial"
        assert r["id"] == r["question_id"]
        assert r["formal_test_eligible"] is True


def test_gold_operations_exist_in_frozen_catalog():
    rows = _load_jsonl(HELDOUT)
    catalog = yaml.safe_load(CATALOG.read_text(encoding="utf-8"))
    op_ids = {op["operation_id"] for op in catalog["operations"]}
    assert len(op_ids) == 21
    for r in rows:
        for op in r.get("gold_operations") or []:
            assert op in op_ids, f"{r['id']} references unknown op {op}"


def test_l1_l4_have_evidence_path_and_l5_have_refusal_behavior():
    rows = _load_jsonl(HELDOUT)
    for r in rows:
        if r["etom_level"] in {"L1", "L2", "L3", "L4"}:
            assert r["answerability_class"] in {
                "structured_answerable",
                "evidence_answerable",
            }
            assert r.get("source_document")
            assert r.get("gold_operations")
            assert r.get("reference_answer")
            if r["etom_level"] == "L4":
                assert len(set(r.get("gold_server_ids") or [])) >= 2
        else:
            assert r["answerability_class"] == "intentionally_infeasible"
            assert r.get("expected_behavior") or r.get("clarification_or_refusal")
            assert r.get("gold_status") in {"unavailable", "clarification_needed"}


def test_no_exact_or_near_duplicate_of_development():
    # Delegate to the deterministic validator script.
    proc = subprocess.run(
        [sys.executable, str(VALIDATOR)],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "PASS" in proc.stdout


def test_development_set_untouched_line_count():
    # Guardrail: Development JSONL still exactly 30 lines / unique ids.
    dev = _load_jsonl(DEV)
    assert len(dev) == 30
    assert len({r["id"] for r in dev}) == 30
    cov = _load_jsonl(DEV_COV)
    assert len(cov) == 30
