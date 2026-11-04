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
BLUEPRINT = ROOT / "docs" / "news_benchmark_blueprint.md"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_docs_and_evidence_exist():
    assert CATALOG.exists()
    assert (ROOT / "docs" / "news_mcp_candidate_audit.md").exists()
    assert (ROOT / "docs" / "news_mcp_capability_matrix.md").exists()
    assert BLUEPRINT.exists()
    assert (ROOT / "docs" / "news_mcp_accepted_blocked_rejected.md").exists()
    assert (EVIDENCE / "aktools_mcp_news_smoke.json").exists()
    assert (EVIDENCE / "hkexnews_dependency_chain_summary.json").exists()
    assert (EVIDENCE / "observed_dependency_edges.json").exists()
    assert (ROOT / "scripts" / "news_hkexnews_dependency_smoke.py").exists()


def test_validator_ok():
    ns = runpy.run_path(str(ROOT / "scripts" / "validate_news_mcp_candidate_v1.py"))
    assert ns["main"]() == 0


def test_catalog_media_and_equal_sets():
    cat = yaml.safe_load(CATALOG.read_text(encoding="utf-8"))
    assert cat["equal_function_sets"] == []
    media = next(o for o in cat["operations"] if o["operation_id"] == "aktools.stock_news_em")
    assert media["formal_gold_eligible"] is False
    assert media["heldout_evidence_eligible"] is False
    assert media["decision"] == "distractor"
    assert media["role"] == "supporting_runtime_only"
    assert "formal_numeric_or_text_gold" in media["forbidden_uses"]
    assert "latest_before_cutoff_gold" in media["forbidden_uses"]


def test_blueprint_exact_l2_is_zero():
    text = BLUEPRINT.read_text(encoding="utf-8")
    assert "Exact L2 = 0" in text or "Exact L2 | **0**" in text
    assert "equal_function_sets" in text
    # no non-zero Exact L2 assignment
    assert "Exact L2 = 2" not in text
    assert "Exact L2 = 3" not in text
    assert "Dev Exact L2=2" not in text
    assert "list ∥ media" not in text
    assert "Official vs media source discrimination" not in text or "not Exact L2" in text


def test_dependency_chain_three_issuers_keywords_list():
    summary = json.loads(
        (EVIDENCE / "hkexnews_dependency_chain_summary.json").read_text(encoding="utf-8")
    )
    assert summary["all_success"] is True
    assert summary["cutoff"] == "2026-08-12"
    for name in ("Tencent", "HSBC", "Alibaba"):
        assert summary["issuers"][name]["success"] is True
        chain = json.loads(
            (
                EVIDENCE / f"hkexnews_dependency_chain_{name.lower()}.json"
            ).read_text(encoding="utf-8")
        )
        assert chain["success"] is True
        rd = str(chain["chosen"]["release_date"])[:10]
        assert rd <= "2026-08-12"
        kw = chain["steps"]["keyword_search"]
        assert kw["success"] is True
        assert isinstance(kw["args"]["keywords"], list)
        assert kw["n_matches"] >= 1
        pages = chain["steps"]["get_announcement_pages"]
        assert pages["success"] is True
        assert "start_page" in pages["args"]
        # corporate-event style: not revenue/EPS packaging
        title = str(chain["chosen"]["title"]).upper()
        assert "EPS" not in title


def test_stock_news_em_not_historical_gold():
    smoke = json.loads(
        (EVIDENCE / "aktools_mcp_news_smoke.json").read_text(encoding="utf-8")
    )
    for name in ("Tencent", "HSBC", "Alibaba"):
        probe = smoke["probes"][f"stock_news_em_{name}"]
        assert probe["success"] is True
        assert int(probe["n_before_cutoff"]) == 0


def test_frozen_financial_market_files_present():
    for rel in (
        "configs/operations/financial_real_mcp.yaml",
        "configs/operations/market_real_mcp_v1_1.yaml",
        "benchmarks/questions/financial_real_mcp_coverage_v1_3.jsonl",
        "benchmarks/heldout/financial_test_30_v1_3.jsonl",
    ):
        path = ROOT / rel
        assert path.exists(), rel
        assert _sha256(path)
    assert CATALOG.name == "news_real_mcp_candidate_v1.yaml"
