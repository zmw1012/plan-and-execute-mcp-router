#!/usr/bin/env python3
"""Fully offline Formal-100 final audit (read-only; no LLM/Embedding/MCP)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from hk_equity_mcp.pe_fin_replan_etom.formal100_final_audit_v1 import (  # noqa: E402
    AUDIT_VERSION,
    run_audit,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=f"{AUDIT_VERSION}: read-only Formal-100 audit"
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        required=True,
        help="Directory containing Formal-100 raw artifacts (predictions.jsonl, ...)",
    )
    parser.add_argument(
        "--audit-dir",
        type=Path,
        required=True,
        help="Audit output directory (e.g. outputs/.../final_audit_v1)",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=ROOT,
        help="Repository root (default: inferred from script location)",
    )
    parser.add_argument(
        "--docs-dir",
        type=Path,
        default=None,
        help="Optional docs mirror directory for report-ready summaries",
    )
    args = parser.parse_args(argv)

    result = run_audit(
        raw_dir=args.raw_dir,
        audit_dir=args.audit_dir,
        repo_root=args.repo_root,
        docs_dir=args.docs_dir,
    )
    print(f"AUDIT={AUDIT_VERSION}")
    print(f"ok={result['ok']}")
    print(f"audit_dir={result['audit_dir']}")
    headline = result["final_report"]["headline"]
    print(
        "all_task_route_em={:.4f} failures_as_zero_em={:.4f} completed_only_diag_em={:.4f}".format(
            headline["all_task_route_exact_match"],
            headline["failures_as_zero_exact_match"],
            headline["completed_only_exact_match_diagnostic"],
        )
    )
    print(f"raw_unchanged={result['verify']['all_unchanged']}")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
