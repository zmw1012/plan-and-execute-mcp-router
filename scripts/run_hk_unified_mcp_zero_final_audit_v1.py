#!/usr/bin/env python3
"""Offline final audit for HK-155 MCP-Zero exploratory run.

Writes to <run-dir>/final_audit_mcp_zero_v1/ and docs mirror.
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

from hk_equity_mcp.experiment.mcp_zero_final_audit import (  # noqa: E402
    run_mcp_zero_final_audit,
)


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
        default="outputs/hk_unified_mcp_zero_exploratory155_v2",
        help="Completed MCP-Zero experiment directory",
    )
    p.add_argument(
        "--out-dir",
        default=None,
        help="Optional override for audit output directory",
    )
    p.add_argument(
        "--docs-mirror",
        default="docs/hk_unified_mcp_zero_final_audit_v1",
        help="Docs mirror directory (relative to repo root)",
    )
    p.add_argument(
        "--no-docs-mirror",
        action="store_true",
        help="Skip writing docs mirror",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = (ROOT / run_dir).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else None
    docs = None
    if not args.no_docs_mirror:
        docs = Path(args.docs_mirror)
        if not docs.is_absolute():
            docs = (ROOT / docs).resolve()
    result = run_mcp_zero_final_audit(
        run_dir,
        root=ROOT,
        audit_outdir=out_dir,
        docs_mirror_dir=docs,
    )
    result["auditor_worktree_head"] = _git_head(ROOT)
    print("MCP-ZERO FINAL AUDIT V1 OK")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
