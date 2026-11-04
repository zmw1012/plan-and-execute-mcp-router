"""Offline tests for News Real-MCP benchmark v1 (no Agent / LLM / paid API)."""

from __future__ import annotations

import json
import runpy
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "configs" / "operations" / "news_real_mcp_v1.yaml"
DEV = ROOT / "benchmarks" / "questions" / "news_development_15_v1.jsonl"
HELD = ROOT / "benchmarks" / "heldout" / "news_test_20_v1.jsonl"


def _rows(path: Path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _bucket(r):
    if r.get("etom_alignment") == "exact" and r.get("etom_level") == "L5":
        return "exact_L5"
    if r.get("etom_alignment") == "adapted":
        return "adapted_robustness"
    if r.get("project_reasoning"):
        return "project_reasoning"
    if r.get("etom_alignment") == "exact" and r.get("etom_level") == "L3":
        return "exact_L3"
    if r.get("etom_alignment") == "exact" and r.get("etom_level") == "L1":
        return "exact_L1"
    return "other"


def test_artifacts_exist():
    assert CATALOG.exists()
    assert DEV.exists() and HELD.exists()
    assert (ROOT / "docs" / "news_real_mcp_v1_freeze.md").exists()
    assert (ROOT / "docs" / "news_development_15_v1_design.md").exists()
    assert (ROOT / "docs" / "news_test_20_v1_design.md").exists()


def test_validator_ok():
    ns = runpy.run_path(str(ROOT / "scripts" / "validate_news_benchmark_v1.py"))
    assert ns["main"]() == 0


def test_catalog_freeze_and_media_ban():
    cat = yaml.safe_load(CATALOG.read_text(encoding="utf-8"))
    assert cat["freeze_status"] == "frozen"
    assert cat["catalog_status"] == "frozen_v1"
    media = next(o for o in cat["operations"] if o["operation_id"] == "aktools.stock_news_em")
    assert media["formal_gold_eligible"] is False
    assert media["exposed_to_agent"] is True
    for o in cat["operations"]:
        if o["decision"] in {"blocked", "rejected"}:
            assert o.get("runtime_registry") is False
            assert o.get("exposed_to_agent") is False


def test_distributions():
    dev, held = _rows(DEV), _rows(HELD)
    assert len(dev) == 15 and len(held) == 20
    assert Counter(r["company"] for r in dev) == {"Tencent": 5, "HSBC": 5, "Alibaba": 5}
    assert Counter(r["company"] for r in held) == {"Tencent": 7, "HSBC": 7, "Alibaba": 6}
    assert Counter(_bucket(r) for r in dev) == {
        "exact_L1": 5,
        "exact_L3": 4,
        "exact_L5": 2,
        "adapted_robustness": 2,
        "project_reasoning": 2,
    }
    assert Counter(_bucket(r) for r in held) == {
        "exact_L1": 6,
        "exact_L3": 5,
        "exact_L5": 3,
        "adapted_robustness": 3,
        "project_reasoning": 3,
    }
    assert not any(
        r.get("etom_alignment") == "exact" and r.get("etom_level") in {"L2", "L4"}
        for r in dev + held
    )


def test_exact_l3_field_edges():
    for r in _rows(DEV) + _rows(HELD):
        if not (r.get("etom_level") == "L3" and r.get("etom_alignment") == "exact"):
            continue
        assert r["has_output_input_dependency"] is True
        assert len(r["dependency_graph"]) >= 1
        assert all(e.get("live_verified_dependency") for e in r["dependency_graph"])
        ops = set(r["gold_operations"])
        assert "hkexnews.keyword_search" in ops
        assert "hkexnews.get_announcement_pages" in ops


def test_heldout_lock_and_never_run():
    for r in _rows(HELD):
        assert r["locked"] is True
        assert r["heldout_agent_run_status"] == "never_agent_run"
        assert r["benchmark_role"] == "final_evaluation"
        assert r["id"].startswith("news_ht_")
