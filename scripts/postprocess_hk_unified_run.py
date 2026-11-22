#!/usr/bin/env python3
"""Offline re-evaluate an existing unified HK run directory.

Reads predictions.jsonl and regenerates:
  per_task_metrics.csv, manual_grading.csv, cost_summary.json,
  retrieval_analysis.json, error_summary.json, evaluation_metadata.json

Does NOT:
  - call Agent / LLM / Embedding / MCP
  - modify predictions.jsonl
  - overwrite run_metadata identity fields
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hk_equity_mcp.experiment.unified_postprocess import postprocess_run_dir  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Offline postprocess unified HK run")
    p.add_argument(
        "--run-dir",
        required=True,
        help="Existing run output directory containing predictions.jsonl",
    )
    p.add_argument(
        "--skip-manual-grading",
        action="store_true",
        help="Do not rewrite manual_grading.csv",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = (ROOT / run_dir).resolve()
    result = postprocess_run_dir(
        run_dir, write_manual_grading=not args.skip_manual_grading
    )
    print("POSTPROCESS OK")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
