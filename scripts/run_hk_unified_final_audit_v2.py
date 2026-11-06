#!/usr/bin/env python3
"""Offline final audit for HK Unified Operation-Visible exploratory-155.

Writes only to <run-dir>/final_audit_v2/.
Does NOT call LLM/Embedding/MCP or modify raw predictions.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hk_equity_mcp.experiment.unified_final_audit import run_final_audit  # noqa: E402


def _git_head(repo: Path) -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=str(repo), text=True
            )
            .strip()
        )
    except Exception:
        return "UNKNOWN"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--run-dir",
        default="outputs/hk_unified_exploratory155_operation_visible_v1",
        help="Completed experiment directory containing predictions.jsonl",
    )
    p.add_argument(
        "--baseline-git-commit",
        default="39bd31b651deb1d064942ee5c6f5e9068946ac15",
        help="Baseline implementation commit to record in provenance",
    )
    p.add_argument(
        "--out-dir",
        default=None,
        help="Optional override for audit output directory",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = (ROOT / run_dir).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else None
    result = run_final_audit(
        run_dir,
        baseline_git_commit=args.baseline_git_commit,
        audit_outdir=out_dir,
    )
    result["auditor_worktree_head"] = _git_head(ROOT)
    print("FINAL AUDIT V2 OK")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
