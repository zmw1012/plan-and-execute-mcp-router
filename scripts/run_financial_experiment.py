#!/usr/bin/env python3
"""Unified financial experiment entrypoint.

Conditions: mcp_free | flat | operation_visible

Generation profile: ENV_FILE=.env.aliyun (shell env wins).
Embedding profile (operation_visible only): EMBEDDING_ENV_FILE=.env.embedding-relay.

Never prints API keys. Do not commit .env* files.
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
from hk_equity_mcp.experiment.schema import CONDITIONS  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MCP-Free / Flat / Operation-Visible financial experiments"
    )
    parser.add_argument(
        "--condition",
        required=True,
        choices=list(CONDITIONS),
        help="Experiment condition",
    )
    parser.add_argument(
        "--questions",
        default=None,
        help="Path to questions JSONL (default from configs/financial_experiment.yaml)",
    )
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of questions")
    parser.add_argument(
        "--ids",
        default=None,
        help="Comma-separated question ids in smoke/subset mode "
        "(keeps original benchmark order; all ids must exist)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip question ids already present in predictions.jsonl",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config/env/registry without calling any network API",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Allow formal paid runs with a dirty git tree (default: refuse)",
    )
    parser.add_argument(
        "--env-file",
        default=None,
        help="Generation env profile path (default ENV_FILE or .env.aliyun)",
    )
    parser.add_argument(
        "--embedding-env-file",
        default=None,
        help="Embedding env profile path (default EMBEDDING_ENV_FILE or .env.embedding-relay)",
    )
    parser.add_argument(
        "--execution-backend",
        default=None,
        choices=["none", "controlled_local", "real_mcp"],
        help=(
            "Tool execution backend. mcp_free→none; "
            "operation_visible defaults to controlled_local; "
            "real_mcp loads configs/operations/financial_real_mcp.yaml and "
            "RealMcpGateway (fail-closed; never silent local fallback)."
        ),
    )
    parser.add_argument(
        "--real-mcp-catalog",
        default=None,
        help="Path to frozen Real-MCP catalog YAML "
        "(default configs/operations/financial_real_mcp.yaml)",
    )
    parser.add_argument(
        "--allow-hkex-download",
        action="store_true",
        help="Permit mcp-hkexnews download tool (still max 1 per gateway session)",
    )
    return parser.parse_args()


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
    catalog = args.real_mcp_catalog
    catalog_path = Path(catalog) if catalog else None
    if catalog_path is not None and not catalog_path.is_absolute():
        catalog_path = (ROOT / catalog_path).resolve()

    runner = FinancialExperimentRunner(
        condition=args.condition,
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
        execution_backend=args.execution_backend,
        real_mcp_catalog=catalog_path,
        allow_hkex_download=args.allow_hkex_download,
    )
    runner.run()


if __name__ == "__main__":
    main()
