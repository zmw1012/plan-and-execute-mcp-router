#!/usr/bin/env python3
"""Convenience wrapper: Operation-Visible + Real-MCP execution backend.

Reuses FinancialExperimentRunner — does not duplicate experiment logic.
Defaults:
  --condition operation_visible
  --execution-backend real_mcp
  --questions benchmarks/questions/financial_mcp_free_30_v1.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hk_equity_mcp.experiment.runner import (  # noqa: E402
    FinancialExperimentRunner,
    load_experiment_config,
    parse_ids,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Real-MCP Operation-Visible financial experiment wrapper"
    )
    p.add_argument("--questions", default=None)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--ids", default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--allow-dirty", action="store_true")
    p.add_argument("--env-file", default=None)
    p.add_argument("--embedding-env-file", default=None)
    p.add_argument(
        "--real-mcp-catalog",
        default=str(ROOT / "configs" / "operations" / "financial_real_mcp.yaml"),
    )
    p.add_argument("--allow-hkex-download", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_experiment_config(ROOT / "configs" / "financial_experiment.yaml")
    questions = args.questions or cfg.get(
        "default_questions", "benchmarks/questions/financial_mcp_free_30_v1.jsonl"
    )
    questions_path = Path(questions)
    if not questions_path.is_absolute():
        questions_path = (ROOT / questions_path).resolve()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (ROOT / output_dir).resolve()
    catalog_path = Path(args.real_mcp_catalog)
    if not catalog_path.is_absolute():
        catalog_path = (ROOT / catalog_path).resolve()

    runner = FinancialExperimentRunner(
        condition="operation_visible",
        execution_backend="real_mcp",
        questions_path=questions_path,
        output_dir=output_dir,
        limit=args.limit,
        ids=parse_ids(args.ids),
        resume=args.resume,
        dry_run=args.dry_run,
        allow_dirty=args.allow_dirty,
        env_file=args.env_file,
        embedding_env_file=args.embedding_env_file,
        config=cfg,
        root=ROOT,
        json_mode=True,
        real_mcp_catalog=catalog_path,
        allow_hkex_download=args.allow_hkex_download,
    )
    runner.run()


if __name__ == "__main__":
    main()
