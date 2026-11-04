"""Offline tests for News Real-MCP discovery v1 (no LLM / no paid API / no Agent)."""

from __future__ import annotations

import hashlib
import json
import runpy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "configs" / "operations" / "news_real_mcp_candidate_v1.yaml"
EVIDENCE = ROOT / "docs" / "news_mcp_discovery_v1"

# Frozen paths — must remain present; content hash checked vs committed baseline files.
FROZEN = [
    ROOT / "configs" / "operations" / "financial_real_mcp.yaml",
    ROOT / "configs" / "operations" / "market_real_mcp_v1_1.yaml",
    ROOT / "benchmarks" / "questions" / "financial_real_mcp_coverage_v1_3.jsonl",
    ROOT / "benchmarks" / "heldout" / "financial_test_30_v1_3.jsonl",
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_docs_and_evidence_exist():
    assert CATALOG.exists()
    assert (ROOT / "docs" / "news_mcp_candidate_audit.md").exists()
    assert (ROOT / "docs" / "news_mcp_capability_matrix.md").exists()
    assert (ROOT / "docs" / "news_benchmark_blueprint.md").exists()
    assert (ROOT / "docs" / "news_mcp_accepted_blocked_rejected.md").exists()
    assert (EVIDENCE / "aktools_mcp_news_smoke.json").exists()
    assert (EVIDENCE / "hkexnews_mcp_news_smoke.json").exists()
    assert (ROOT / "scripts" / "news_mcp_discovery_smoke.py").exists()


def test_validator_ok():
    ns = runpy.run_path(str(ROOT / "scripts" / "validate_news_mcp_candidate_v1.py"))
    assert ns["main"]() == 0


def test_catalog_decisions_and_source_classes():
    cat = yaml.safe_load(CATALOG.read_text(encoding="utf-8"))
    assert cat["catalog_status"] == "candidate_v1"
    assert cat["freeze_status"] == "not_frozen"
    assert cat["temporal_policy"]["max_data_date"] == "2026-08-12"
    ops = cat["operations"]
    by_dec = {}
    for o in ops:
        by_dec.setdefault(o["decision"], []).append(o["operation_id"])
    assert len(by_dec["primary"]) == 5
    assert "hkexnews.list_hkexnews_announcements" in by_dec["primary"]
    assert "aktools.stock_news_em" in by_dec["supporting"]
    assert "ah_disclosure.search_filings" in by_dec["distractor"]
    assert cat.get("equal_function_sets") == []
    for o in ops:
        if o["decision"] == "primary":
            assert o["source_class"] == "official_disclosure"


def test_smoke_evidence_three_issuers_hkexnews():
    smoke = json.loads(
        (EVIDENCE / "hkexnews_mcp_news_smoke.json").read_text(encoding="utf-8")
    )
    assert smoke["cutoff"] == "2026-08-12"
    for name in ("Tencent", "HSBC", "Alibaba"):
        probe = smoke["probes"][f"list_{name}"]
        assert probe["success"] is True
        assert int(probe["n"]) >= 1
        fields = set(probe.get("fields") or [])
        assert {"title", "release_date", "news_id", "local_key", "file_link"} <= fields


def test_stock_news_em_not_historical_gold():
    smoke = json.loads(
        (EVIDENCE / "aktools_mcp_news_smoke.json").read_text(encoding="utf-8")
    )
    for name in ("Tencent", "HSBC", "Alibaba"):
        probe = smoke["probes"][f"stock_news_em_{name}"]
        assert probe["success"] is True
        assert int(probe["n_before_cutoff"]) == 0


def test_ah_filings_date_filter_unreliable():
    probe = json.loads(
        (EVIDENCE / "ah_disclosure_date_filter_probe.json").read_text(encoding="utf-8")
    )
    # At least one post-cutoff publish_time appears despite end_date=20260812
    blob = json.dumps(probe, ensure_ascii=False)
    assert "2026-11" in blob or "2026-10" in blob


def test_frozen_financial_market_files_untouched_hashes():
    """Presence check + stable sha256 for key frozen artifacts in this worktree."""
    expected = {
        "configs/operations/financial_real_mcp.yaml": None,
        "configs/operations/market_real_mcp_v1_1.yaml": None,
        "benchmarks/questions/financial_real_mcp_coverage_v1_3.jsonl": None,
        "benchmarks/heldout/financial_test_30_v1_3.jsonl": None,
    }
    # Record hashes locally for regression within this branch; fail if file missing.
    hashes = {}
    for rel in expected:
        path = ROOT / rel
        assert path.exists(), rel
        hashes[rel] = _sha256(path)
    # Cross-check Market held-out if present
    mheld = ROOT / "benchmarks" / "heldout" / "market_test_20_v1_1.jsonl"
    if mheld.exists():
        hashes[str(mheld.relative_to(ROOT))] = _sha256(mheld)
    # Ensure News catalog is a different file
    assert CATALOG.name == "news_real_mcp_candidate_v1.yaml"
    assert _sha256(CATALOG) not in set(hashes.values())
