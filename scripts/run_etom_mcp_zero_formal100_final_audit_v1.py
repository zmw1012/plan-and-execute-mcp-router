#!/usr/bin/env python3
"""Offline Formal-100 final audit for MCP-Zero × ETOM.

Never re-runs tasks. Never calls LLM/Embedding/MCP APIs.
Never modifies raw predictions or run_metadata.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hk_equity_mcp.mcp_zero_etom.final_audit_v1 import run_final_audit  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT,
        help="Worktree root",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=ROOT / "outputs" / "etom_external_formal100_mcp_zero_v1",
        help="Raw Formal-100 run directory",
    )
    parser.add_argument(
        "--audit-dir",
        type=Path,
        default=None,
        help="Derived audit output (default: <run-dir>/final_audit_v1)",
    )
    parser.add_argument(
        "--docs-mirror-dir",
        type=Path,
        default=None,
        help="Committable docs mirror",
    )
    args = parser.parse_args()
    result = run_final_audit(
        root=args.root,
        run_dir=args.run_dir,
        audit_dir=args.audit_dir,
        docs_mirror_dir=args.docs_mirror_dir,
    )
    summary = result["summary"]
    print("FINAL-AUDIT-V1 OK")
    print(
        f"strict={summary['strict_overall']['successes']}/"
        f"{summary['strict_overall']['denominator']} "
        f"completed_only={summary['completed_only_overall']['successes']}/"
        f"{summary['completed_only_overall']['denominator']} "
        f"raw_sha_unchanged={summary['raw_sha_unchanged']}"
    )
    print(f"audit_dir={result['audit_dir']}")
    print(f"docs_mirror_dir={result['docs_mirror_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
