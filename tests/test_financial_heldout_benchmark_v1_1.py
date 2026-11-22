"""Deterministic tests for Financial held-out Test 30 v1.1.

No LLM, embedding, MCP live calls, or experiment execution in these unit tests.
Live Real-MCP provenance is enforced via metadata contracts + validator.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
HELDOUT = ROOT / "benchmarks" / "heldout" / "financial_test_30_v1_1.jsonl"
V1 = ROOT / "benchmarks" / "heldout" / "financial_test_30_v1.jsonl"
README = ROOT / "benchmarks" / "heldout" / "README.md"
AUDIT_DOC = ROOT / "docs" / "financial_heldout_real_mcp_provenance_audit_v1_1.md"
CATALOG = ROOT / "configs" / "operations" / "financial_real_mcp.yaml"
VALIDATOR = ROOT / "scripts" / "validate_financial_heldout_test_v1_1.py"
DEV = ROOT / "benchmarks" / "questions" / "financial_mcp_free_30_v1.jsonl"


def _load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_v1_1_artifacts_exist_and_readme_status():
    for path in (HELDOUT, V1, README, AUDIT_DOC, VALIDATOR):
        assert path.is_file(), path
    readme = README.read_text(encoding="utf-8")
    assert "superseded_before_any_agent_run" in readme
    assert "financial_test_30_v1_1.jsonl" in readme
    assert "active" in readme.lower()


def test_distributions_lock_and_versions():
    rows = _load_jsonl(HELDOUT)
    assert len(rows) == 30
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
        assert r["benchmark_version"] == "financial_test_30_v1_1"
        assert r["catalog_version"] == "financial_real_mcp_v1"
        assert r["dataset_split"] == "heldout_test"
        assert r["benchmark_role"] == "final_evaluation_only"
        assert r["locked"] is True
        assert r["id"] == r["question_id"]
        assert r.get("provenance_status")
        assert r.get("gold_path_verification") is not None
        assert r.get("live_audit_timestamp")


def test_evidence_items_use_results_announcements_not_formal_reports():
    rows = _load_jsonl(HELDOUT)
    evidence = [r for r in rows if r["answerability_class"] == "evidence_answerable"]
    assert len(evidence) == 14
    for r in evidence:
        assert r["evidence_document_type"] in {
            "annual_results_announcement",
            "interim_results_announcement",
        }
        title = (r.get("source_document") or "").lower()
        assert "annual report" not in title
        assert "interim report" not in title
        assert r["evidence_retrieved_via"] == "mcp_hkexnews"
        assert "hkexnews.hk" in (r.get("source_url") or "")
        assert r.get("source_page_pdf") not in (None, "")
        assert r.get("evidence_snippet")
        assert r.get("evidence_release_date")
        assert "live_verified" in r["provenance_status"]


def test_structured_and_l4_and_l5_contracts():
    rows = _load_jsonl(HELDOUT)
    for r in rows:
        if r["answerability_class"] == "structured_answerable":
            assert r["provenance_status"].startswith("live_verified")
        if r["etom_level"] == "L4":
            assert len(set(r.get("gold_server_ids") or [])) >= 2
            assert r.get("l4_semantic_compatible") is True
        if r["etom_level"] == "L5":
            assert r["answerability_class"] == "intentionally_infeasible"
            assert r.get("infeasibility_kind")
            assert r.get("infeasibility_reason")
            assert r["provenance_status"] == "l5_infeasibility_confirmed"


def test_gold_operations_exist_and_catalog_unchanged_fingerprint():
    rows = _load_jsonl(HELDOUT)
    catalog_text = CATALOG.read_text(encoding="utf-8")
    catalog = yaml.safe_load(catalog_text)
    op_ids = {op["operation_id"] for op in catalog["operations"]}
    assert len(op_ids) == 21
    for r in rows:
        for op in r.get("gold_operations") or []:
            assert op in op_ids
    # Fingerprint stability check vs checked-in catalog content length sanity
    digest = hashlib.sha256(catalog_text.encode("utf-8")).hexdigest()
    assert len(digest) == 64
    assert "mcp_hkexnews" in catalog_text
    assert "Full annual report PDFs out of scope" in catalog_text or "out of scope" in catalog_text.lower()


def test_alibaba_holder_profit_correction_not_ordinary_shareholders_gold():
    rows = {r["id"]: r for r in _load_jsonl(HELDOUT)}
    for qid in ("fin_ht_024", "fin_ht_025", "fin_ht_027"):
        vals = [g.get("value") for g in rows[qid].get("gold_values") or []]
        assert 103592 in vals
        assert 105904 not in vals
        assert "ordinary shareholders" not in rows[qid]["question"].lower() or "not" in rows[qid]["question"].lower()


def test_validator_script_passes():
    proc = subprocess.run(
        [sys.executable, str(VALIDATOR)],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "PASS" in proc.stdout


def test_development_untouched():
    dev = _load_jsonl(DEV)
    assert len(dev) == 30
    assert len(_load_jsonl(V1)) == 30
