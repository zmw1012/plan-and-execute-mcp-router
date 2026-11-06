#!/usr/bin/env python3
"""Unified HK MCP-Zero Real-MCP baseline runner.

Defaults:
  --split development
  MCP-Zero hierarchical active tool retrieval + UnifiedRealMcpGateway
  42 unique / 39 runtime-exposed operations

Locks:
  --split heldout requires --confirm-heldout-final
  --split exploratory_all requires --confirm-retire-heldout

Offline:
  --dry-run / --mock never call paid LLM, embedding, or MCP network APIs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hk_equity_mcp.experiment.runner import parse_ids  # noqa: E402
from hk_equity_mcp.experiment.unified_mcp_zero_runner import (  # noqa: E402
    UnifiedHkMcpZeroRunner,
    load_mcp_zero_config,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Unified HK MCP-Zero Real-MCP experiment"
    )
    p.add_argument(
        "--split",
        default="development",
        choices=["development", "heldout", "exploratory_all"],
        help="Dataset split (default: development)",
    )
    p.add_argument(
        "--confirm-heldout-final",
        action="store_true",
        help="Required together with --split heldout",
    )
    p.add_argument(
        "--confirm-retire-heldout",
        action="store_true",
        help="Required together with --split exploratory_all",
    )
    p.add_argument("--smoke", action="store_true", help="Run locked Dev smoke-8")
    p.add_argument("--ids", default=None, help="Comma-separated task ids")
    p.add_argument("--max-tasks", type=int, default=None)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--mock",
        action="store_true",
        help="Offline mock LLM + hash embedding + mock MCP transport",
    )
    p.add_argument("--allow-dirty", action="store_true")
    p.add_argument("--env-file", default=None)
    p.add_argument("--embedding-env-file", default=None)
    p.add_argument("--allow-hkex-download", action="store_true")
    p.add_argument(
        "--config",
        default=str(ROOT / "configs" / "mcp_zero_hk_v1.yaml"),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_mcp_zero_config(Path(args.config), root=ROOT)
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (ROOT / output_dir).resolve()

    runner = UnifiedHkMcpZeroRunner(
        output_dir=output_dir,
        split=args.split,
        smoke=args.smoke,
        ids=parse_ids(args.ids),
        max_tasks=args.max_tasks,
        resume=args.resume,
        dry_run=args.dry_run,
        mock=args.mock,
        allow_dirty=args.allow_dirty,
        confirm_heldout_final=args.confirm_heldout_final,
        confirm_retire_heldout=args.confirm_retire_heldout,
        env_file=args.env_file,
        embedding_env_file=args.embedding_env_file,
        config=cfg,
        root=ROOT,
        allow_hkex_download=args.allow_hkex_download,
        json_mode=True,
    )
    runner.run()


if __name__ == "__main__":
    main()
