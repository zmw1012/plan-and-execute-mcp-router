"""Offline tests for Cross-domain benchmark v1 (no Agent/LLM/paid API)."""

from __future__ import annotations

import json
import runpy
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs" / "operations" / "cross_domain_real_mcp_v1.yaml"
DEV = ROOT / "benchmarks" / "questions" / "cross_domain_development_10_v1.jsonl"
HELD = ROOT / "benchmarks" / "heldout" / "cross_domain_test_15_v1.jsonl"


def _rows(path: Path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_artifacts_exist():
    assert MANIFEST.exists()
    assert DEV.exists() and HELD.exists()
    assert (ROOT / "docs" / "cross_domain_benchmark_v1_design.md").exists()
    assert (ROOT / "docs" / "cross_domain_tool_dependency_audit_v1.md").exists()
    assert (ROOT / "docs" / "cross_domain_mcp_gap_proposals_v1.md").exists()
    assert (ROOT / "docs" / "cross_domain_discovery_v1" / "market_cross_domain_smoke.json").exists()


def test_validator_ok():
    ns = runpy.run_path(str(ROOT / "scripts" / "validate_cross_domain_benchmark_v1.py"))
    assert ns["main"]() == 0


def test_manifest_references_frozen_catalogs():
    man = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    paths = {r["path"] for r in man["referenced_catalogs"]}
    assert "configs/operations/financial_real_mcp.yaml" in paths
    assert "configs/operations/market_real_mcp_v1_1.yaml" in paths
    assert "configs/operations/news_real_mcp_v1.yaml" in paths
    assert man["counts"]["dedup_servers"] >= 3
    assert man["counts"]["namespaced_operations"] >= 40


def test_distributions():
    dev, held = _rows(DEV), _rows(HELD)
    assert len(dev) == 10 and len(held) == 15
    assert Counter(r["domain_combo"] for r in dev) == {
        "financial+market": 3,
        "financial+news": 3,
        "market+news": 2,
        "financial+market+news": 2,
    }
    assert Counter(r["domain_combo"] for r in held) == {
        "financial+market": 4,
        "financial+news": 4,
        "market+news": 4,
        "financial+market+news": 3,
    }
    exact = [r for r in dev + held if r.get("exact_etom_alignment") is True]
    assert exact
    assert all(r.get("exact_etom_level") == "L4" for r in exact)
    assert all(r.get("dependency_graph") for r in exact)
    for r in exact:
        assert any(
            e.get("producer_output_field") == "release_date"
            and e.get("consumer_input_field") in {"start_date", "end_date"}
            for e in r["dependency_graph"]
        )


def test_heldout_lock():
    for r in _rows(HELD):
        assert r["locked"] is True
        assert r["heldout_agent_run_status"] == "never_agent_run"
        assert r["id"].startswith("xd_ht_")


def test_frozen_domain_files_untouched_presence():
    for rel in (
        "configs/operations/financial_real_mcp.yaml",
        "configs/operations/market_real_mcp_v1_1.yaml",
        "configs/operations/news_real_mcp_v1.yaml",
        "benchmarks/questions/news_development_15_v1.jsonl",
        "benchmarks/heldout/news_test_20_v1.jsonl",
    ):
        assert (ROOT / rel).exists()
