#!/usr/bin/env python3
"""Offline smoke for gap-discovery artifacts (no network, no LLM)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "mcp_discovery_v2"
DOCS = ROOT / "docs"

REQUIRED = [
    OUT / "aktools_gap_search.json",
    OUT / "aktools_gap_describe.json",
    OUT / "aktools_gap_get_data.json",
    OUT / "ftshare_audit.json",
    OUT / "ah_disclosure_tools_list.json",
    OUT / "hkexnews_tools_list.json",
    OUT / "reconciliation_summary.json",
    DOCS / "financial_mcp_gap_discovery.md",
    DOCS / "financial_mcp_candidate_audit.md",
]


def main() -> int:
    missing = [str(p.relative_to(ROOT)) for p in REQUIRED if not p.exists()]
    if missing:
        print("FAIL missing:", *missing, sep="\n  ")
        return 1

    ft = json.loads((OUT / "ftshare_audit.json").read_text(encoding="utf-8"))
    ah = json.loads((OUT / "ah_disclosure_tools_list.json").read_text(encoding="utf-8"))
    hk = json.loads((OUT / "hkexnews_tools_list.json").read_text(encoding="utf-8"))
    recon = json.loads((OUT / "reconciliation_summary.json").read_text(encoding="utf-8"))

    ft_count = (ft.get("tools_list_no_key") or {}).get("tool_count")
    assert ft_count == 250, ft_count
    assert ah.get("tool_count") == 40, ah.get("tool_count")
    assert hk.get("tool_count") == 6, hk.get("tool_count")
    assert "tencent_fy2025_revenue" in recon
    assert "hsbc_fy2025" in recon

    gap_doc = (DOCS / "financial_mcp_gap_discovery.md").read_text(encoding="utf-8")
    audit = (DOCS / "financial_mcp_candidate_audit.md").read_text(encoding="utf-8")
    assert "financial_mcp_gap_discovery.md" in audit
    assert "Recommended final Financial catalog" in gap_doc

    print(
        "OK gap-discovery smoke:",
        f"FTShare tools/list={ft_count},",
        f"ah-disclosure={ah.get('tool_count')},",
        f"hkexnews={hk.get('tool_count')},",
        f"reconcile_keys={len(recon)}",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
