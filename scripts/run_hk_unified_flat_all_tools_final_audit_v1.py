#!/usr/bin/env python3
"""Offline final audit for HK-155 Flat All Tools exploratory live run.

Writes to <run-dir>/final_audit_flat_v1/ and docs/hk_unified_flat_all_tools_final_audit_v1/.
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

from hk_equity_mcp.experiment.unified_flat_final_audit import (  # noqa: E402
    run_flat_final_audit,
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
        default="outputs/hk_unified_flat_all_tools_155_v1",
        help="Completed Flat All Tools experiment directory",
    )
    p.add_argument(
        "--out-dir",
        default=None,
        help="Optional override for audit output directory",
    )
    p.add_argument(
        "--docs-mirror",
        default=str(ROOT / "docs" / "hk_unified_flat_all_tools_final_audit_v1"),
        help="Tracked docs mirror directory (set empty to skip)",
    )
    p.add_argument(
        "--human-grades",
        action="append",
        default=None,
        help=(
            "Path to human_grades.json (repeatable). Default: discover under "
            "audit out-dir / docs mirror."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = (ROOT / run_dir).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else None
    docs_mirror = None
    if args.docs_mirror:
        docs_mirror = Path(args.docs_mirror)
        if not docs_mirror.is_absolute():
            docs_mirror = (ROOT / docs_mirror).resolve()
    human_grades_paths = None
    if args.human_grades:
        human_grades_paths = []
        for raw in args.human_grades:
            p = Path(raw)
            if not p.is_absolute():
                p = (ROOT / p).resolve()
            human_grades_paths.append(p)

    result = run_flat_final_audit(
        run_dir,
        audit_outdir=out_dir,
        docs_mirror=docs_mirror,
        human_grades_paths=human_grades_paths,
    )
    result["auditor_worktree_head"] = _git_head(ROOT)
    print("FLAT FINAL AUDIT V1 OK")
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
