#!/usr/bin/env python3
"""Offline validator for News Real-MCP candidate catalog + blueprint constraints."""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import yaml

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "configs" / "operations" / "news_real_mcp_candidate_v1.yaml"
BLUEPRINT = ROOT / "docs" / "news_benchmark_blueprint.md"
EVIDENCE = ROOT / "docs" / "news_mcp_discovery_v1"
REQUIRED_DOCS = [
    ROOT / "docs" / "news_mcp_candidate_audit.md",
    ROOT / "docs" / "news_mcp_capability_matrix.md",
    BLUEPRINT,
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
FLOATING_MEDIA_OPS = {"aktools.stock_news_em"}


def _parse_exact_l2_counts(text: str) -> List[int]:
    """Extract Exact L2 planned counts from blueprint markdown tables."""
    counts: List[int] = []
    for line in text.splitlines():
        if re.search(r"Exact\s+L2", line, flags=re.I):
            # table row like: | Exact L2 | **0** | ... or | Exact L2 | 0 |
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) >= 2 and re.search(r"Exact\s+L2", cells[0], flags=re.I):
                m = re.search(r"(\d+)", cells[1].replace("*", ""))
                if m:
                    counts.append(int(m.group(1)))
    return counts


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
    by_id = {o["operation_id"]: o for o in ops if o.get("operation_id")}

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

    for req in (
        "hkexnews.list_hkexnews_announcements",
        "hkexnews.download_hkexnews_announcement",
        "hkexnews.parse_hkexnews_announcement",
        "hkexnews.keyword_search",
        "hkexnews.get_announcement_pages",
    ):
        if req not in ids:
            errors.append(f"missing primary op {req}")

    # equal_function_sets empty => Exact L2 planning must be 0
    efs = cat.get("equal_function_sets")
    if efs not in ([], None):
        for s in efs or []:
            if len(s.get("operations") or []) < 2:
                errors.append(f"equal set too small: {s}")
    else:
        if not BLUEPRINT.exists():
            errors.append("missing blueprint for Exact L2 check")
        else:
            bp = BLUEPRINT.read_text(encoding="utf-8")
            l2_counts = _parse_exact_l2_counts(bp)
            if not l2_counts:
                errors.append("blueprint missing Exact L2 planning rows")
            for c in l2_counts:
                if c != 0:
                    errors.append(
                        f"equal_function_sets=[] forbids Exact L2 planning count={c}"
                    )
            # ban legacy Exact L2 planning phrases that imply non-zero without table
            if re.search(r"Exact L2\s*=\s*[1-9]", bp):
                errors.append("blueprint still assigns non-zero Exact L2")

    # Floating media must not be formal gold
    for oid in FLOATING_MEDIA_OPS:
        o = by_id.get(oid)
        if not o:
            errors.append(f"missing floating media op {oid}")
            continue
        if o.get("formal_gold_eligible") is not False:
            errors.append(f"{oid} must set formal_gold_eligible=false")
        if o.get("heldout_evidence_eligible") is not False:
            errors.append(f"{oid} must set heldout_evidence_eligible=false")
        if o.get("decision") not in {"distractor", "rejected", "blocked"}:
            errors.append(f"{oid} decision must be distractor/rejected/blocked for formal gold ban")
        forbidden = set(o.get("forbidden_uses") or [])
        for req_forbid in (
            "formal_numeric_or_text_gold",
            "heldout_answer_evidence",
            "latest_before_cutoff_gold",
        ):
            if req_forbid not in forbidden:
                errors.append(f"{oid} missing forbidden_uses entry {req_forbid}")

    # Dependency smoke evidence must show three-issuer success with keywords:list
    summary_path = EVIDENCE / "hkexnews_dependency_chain_summary.json"
    if not summary_path.exists():
        errors.append("missing hkexnews_dependency_chain_summary.json")
    else:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if not summary.get("all_success"):
            errors.append("dependency chain summary all_success is false")
        for name in ("Tencent", "HSBC", "Alibaba"):
            iss = (summary.get("issuers") or {}).get(name) or {}
            if not iss.get("success"):
                errors.append(f"dependency chain failed for {name}")
            chain_path = EVIDENCE / f"hkexnews_dependency_chain_{name.lower()}.json"
            if not chain_path.exists():
                errors.append(f"missing {chain_path.name}")
                continue
            chain = json.loads(chain_path.read_text(encoding="utf-8"))
            kw = (chain.get("steps") or {}).get("keyword_search") or {}
            args = kw.get("args") or {}
            if not kw.get("success"):
                errors.append(f"{name} keyword_search success=false (do not hand-edit)")
            if not isinstance(args.get("keywords"), list):
                errors.append(f"{name} keyword_search must use keywords:list")
            pages = (chain.get("steps") or {}).get("get_announcement_pages") or {}
            if not pages.get("success"):
                errors.append(f"{name} get_announcement_pages success=false")
            rd = str((chain.get("chosen") or {}).get("release_date") or "")[:10]
            if not rd or rd > "2026-08-12":
                errors.append(f"{name} chosen release_date not <= cutoff: {rd}")

    for doc in REQUIRED_DOCS:
        if not doc.exists():
            errors.append(f"missing doc {doc.relative_to(ROOT)}")
    for name in (
        "aktools_mcp_news_smoke.json",
        "hkexnews_mcp_news_smoke.json",
        "ah_disclosure_date_filter_probe.json",
        "observed_dependency_edges.json",
    ):
        if not (EVIDENCE / name).exists():
            errors.append(f"missing evidence {name}")

    for path in FROZEN_MUST_EXIST_UNCHANGED:
        if not path.exists():
            errors.append(f"expected frozen path missing: {path.relative_to(ROOT)}")

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
    print("equal_function_sets=", cat.get("equal_function_sets"))
    print("servers=", list((cat.get("servers") or {}).keys()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
