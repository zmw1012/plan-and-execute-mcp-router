"""Compatibility tier reranking + hard-filter tests."""

from __future__ import annotations

from pathlib import Path

from hk_equity_mcp.apv_fin.metadata import load_and_validate_metadata
from hk_equity_mcp.apv_fin.reranker import RetrievalContext, rerank_candidates
from hk_equity_mcp.real_mcp.unified_catalog import build_unified_catalog

ROOT = Path(__file__).resolve().parents[1]


def test_metadata_covers_all_runtime_ops_and_blocks():
    bundle = build_unified_catalog(root=ROOT)
    meta = load_and_validate_metadata(catalog=bundle, root=ROOT)
    assert len(meta) == bundle.stats.runtime_exposed_unique == 39
    for blocked in bundle.stats.blocked_operation_ids:
        assert blocked not in bundle.registry
        assert blocked not in meta or meta[blocked]["availability"] != "runtime"


def test_tier_abc_ordering_and_blocked_filter():
    bundle = build_unified_catalog(root=ROOT)
    meta = load_and_validate_metadata(catalog=bundle, root=ROOT)
    candidates = [
        {
            "operation_id": "aktools.news_cctv",
            "score": 0.99,
        },  # cn_mainland → hard filter for HK
        {
            "operation_id": "aktools.stock_hk_hist",
            "score": 0.50,
        },  # Tier A-ish market hist
        {
            "operation_id": "aktools.stock_financial_hk_analysis_indicator_em",
            "score": 0.80,
        },
        {
            "operation_id": "aktools.stock_hk_famous_spot_em",
            "score": 0.70,
        },  # spot_latest vs point_in_time → hard filter
        {
            "operation_id": "totally.unknown.op",
            "score": 1.0,
        },
        {
            "operation_id": "ftshare.ft_v1_report_announcements_list",
            "score": 0.95,
        },
    ]
    ctx = RetrievalContext(
        domain_hints=["market"],
        market_scope="hk",
        entity_scope_hints=["hk_equity"],
        identifier_types=["hk_code_5digit"],
        temporal_needs=["point_in_time", "date_range"],
        source_type_prefs=["exchange_feed"],
        metric_basis_hints=["price_unadjusted"],
        cutoff_date="2026-08-12",
        disallow_post_cutoff_spot=True,
    )
    out = rerank_candidates(
        candidates,
        meta,
        ctx,
        runtime_operation_ids=bundle.stats.runtime_operation_ids,
    )
    filtered_ids = {r["operation_id"] for r in out["hard_filtered"]}
    assert "totally.unknown.op" in filtered_ids
    assert "ftshare.ft_v1_report_announcements_list" in filtered_ids or any(
        "blocked" in str(r.get("hard_filter_reason"))
        or "not_in_runtime" in str(r.get("hard_filter_reason"))
        for r in out["hard_filtered"]
        if r["operation_id"] == "ftshare.ft_v1_report_announcements_list"
    )
    assert "aktools.news_cctv" in filtered_ids
    assert "aktools.stock_hk_famous_spot_em" in filtered_ids

    reranked = out["reranked"]
    assert reranked, "expected some kept candidates"
    # Tier order: A before B before C
    tiers = [r["compatibility_tier"] for r in reranked]
    assert tiers == sorted(tiers, key=lambda t: {"A": 0, "B": 1, "C": 2}[t])
    # Within same tier, higher base score first
    by_tier = {}
    for r in reranked:
        by_tier.setdefault(r["compatibility_tier"], []).append(r["base_score"])
    for scores in by_tier.values():
        assert scores == sorted(scores, reverse=True)
    # hist should outrank / appear as market-compatible
    ids = [r["operation_id"] for r in reranked]
    assert "aktools.stock_hk_hist" in ids
    assert reranked[0]["final_rank"] == 1
    assert "original_rank" in reranked[0]
    assert "compatibility_reasons" in reranked[0]


def test_company_generic_unseen_issuer_query_does_not_whitelist_fail():
    bundle = build_unified_catalog(root=ROOT)
    meta = load_and_validate_metadata(catalog=bundle, root=ROOT)
    # Unseen issuer context: still hk_listed_issuer generic scopes.
    ctx = RetrievalContext(
        domain_hints=["financial"],
        market_scope="hk",
        entity_scope_hints=["hk_listed_issuer"],
        identifier_types=["hk_code_5digit"],
        temporal_needs=["fiscal_period"],
        source_type_prefs=["structured_vendor", "official_disclosure"],
    )
    candidates = [
        {
            "operation_id": "aktools.stock_financial_hk_analysis_indicator_em",
            "score": 0.9,
        },
        {"operation_id": "hkexnews.list_hkexnews_announcements", "score": 0.8},
    ]
    out = rerank_candidates(
        candidates,
        meta,
        ctx,
        runtime_operation_ids=bundle.stats.runtime_operation_ids,
    )
    assert len(out["reranked"]) == 2
    assert all(r["compatibility_tier"] in {"A", "B"} for r in out["reranked"])
    blob = str(meta).lower()
    for banned in ("tencent", "hsbc", "alibaba", "pop mart", "xiaomi"):
        assert banned not in blob
