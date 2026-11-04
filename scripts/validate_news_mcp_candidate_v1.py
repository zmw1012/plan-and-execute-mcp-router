#!/usr/bin/env python3
"""Offline validator for News Real-MCP candidate catalog (no LLM / no network)."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import yaml

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "configs" / "operations" / "news_real_mcp_candidate_v1.yaml"
EVIDENCE = ROOT / "docs" / "news_mcp_discovery_v1"
REQUIRED_DOCS = [
    ROOT / "docs" / "news_mcp_candidate_audit.md",
    ROOT / "docs" / "news_mcp_capability_matrix.md",
    ROOT / "docs" / "news_benchmark_blueprint.md",
    ROOT / "docs" / "news_mcp_accepted_blocked_rejected.md",
]
FROZEN_MUST_EXIST_UNCHANGED = [
    ROOT / "configs" / "operations" / "financial_real_mcp.yaml",
    ROOT / "configs" / "operations" / "market_real_mcp_v1_1.yaml",
    ROOT / "benchmarks" / "questions" / "financial_real_mcp_coverage_v1_3.jsonl",
    ROOT / "benchmarks" / "heldout" / "financial_test_30_v1_3.jsonl",
    ROOT / "benchmarks" / "questions" / "market_development_15_v1_1.jsonl",
    ROOT / "benchmarks" / "heldout" / "market_test_20_v1_1.jsonl",
]


def main() -> int:
    errors: List[str] = []
    if not CATALOG.exists():
        print(f"missing catalog {CATALOG}")
        return 1
    cat: Dict[str, Any] = yaml.safe_load(CATALOG.read_text(encoding="utf-8"))
    if cat.get("catalog_status") != "candidate_v1":
        errors.append("catalog_status must be candidate_v1")
    if cat.get("freeze_status") != "not_frozen":
        errors.append("freeze_status must be not_frozen")
    if cat.get("domain") != "news":
        errors.append("domain must be news")
    tp = cat.get("temporal_policy") or {}
    if tp.get("max_data_date") != "2026-08-12":
        errors.append("temporal cutoff must be 2026-08-12")
    if tp.get("timezone") != "Asia/Hong_Kong":
        errors.append("timezone must be Asia/Hong_Kong")

    ops = cat.get("operations") or []
    if len(ops) < 10:
        errors.append(f"too few operations: {len(ops)}")
    decisions = Counter(str(o.get("decision")) for o in ops)
    for need in ("primary", "supporting", "distractor", "blocked", "rejected"):
        if decisions.get(need, 0) < 1:
            errors.append(f"missing decision class {need}")

    ids = [o.get("operation_id") for o in ops]
    if len(ids) != len(set(ids)):
        errors.append("duplicate operation_id")

    source_ok = {
        "official_disclosure",
        "company_ir",
        "financial_media",
        "aggregator",
        "metadata_only",
        "distractor",
        "blocked",
        "rejected",
    }
    for o in ops:
        sc = o.get("source_class")
        if sc not in source_ok:
            errors.append(f"{o.get('operation_id')} bad source_class={sc}")
        if o.get("decision") == "primary" and sc != "official_disclosure":
            errors.append(f"primary must be official_disclosure: {o.get('operation_id')}")

    # Primary chain presence
    for req in (
        "hkexnews.list_hkexnews_announcements",
        "hkexnews.download_hkexnews_announcement",
        "hkexnews.parse_hkexnews_announcement",
        "hkexnews.keyword_search",
        "hkexnews.get_announcement_pages",
    ):
        if req not in ids:
            errors.append(f"missing primary op {req}")

    if cat.get("equal_function_sets") not in ([], None):
        # Allow empty only for v1 discovery; if non-empty, each set needs ≥2 ops
        for s in cat.get("equal_function_sets") or []:
            if len(s.get("operations") or []) < 2:
                errors.append(f"equal set too small: {s}")

    for doc in REQUIRED_DOCS:
        if not doc.exists():
            errors.append(f"missing doc {doc.relative_to(ROOT)}")
    for name in (
        "aktools_mcp_news_smoke.json",
        "hkexnews_mcp_news_smoke.json",
        "ah_disclosure_date_filter_probe.json",
    ):
        if not (EVIDENCE / name).exists():
            errors.append(f"missing evidence {name}")

    for path in FROZEN_MUST_EXIST_UNCHANGED:
        if not path.exists():
            errors.append(f"expected frozen path missing: {path.relative_to(ROOT)}")

    # Ensure catalog is not writing into frozen financial/market filenames
    if CATALOG.name in {
        "financial_real_mcp.yaml",
        "market_real_mcp_v1_1.yaml",
        "market_real_mcp_v1.yaml",
    }:
        errors.append("News catalog must not overwrite Market/Financial filenames")

    if errors:
        print("VALIDATION FAILED:")
        for e in errors:
            print(" -", e)
        return 1

    print("VALIDATION OK")
    print("decisions=", dict(decisions))
    print("n_operations=", len(ops))
    print("servers=", list((cat.get("servers") or {}).keys()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
