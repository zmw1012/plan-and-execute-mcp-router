#!/usr/bin/env python3
"""Unified HK MCP-Free (closed-book) baseline runner.

Defaults:
  condition=mcp_free
  execution_backend=none

Locks:
  --split exploratory_all requires --confirm-retire-heldout

Offline-safe: --dry-run / --mock (no paid APIs).
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
from hk_equity_mcp.experiment.unified_mcp_free_runner import (  # noqa: E402
    DEFAULT_OUTPUT_DIR,
    UnifiedHkMcpFreeRunner,
    load_mcp_free_config,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Unified HK MCP-Free closed-book experiment"
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
        help="Required together with --split heldout (legacy)",
    )
    p.add_argument(
        "--confirm-retire-heldout",
        action="store_true",
        help="Required together with --split exploratory_all",
    )
    p.add_argument("--smoke", action="store_true", help="Run locked Dev smoke-8")
    p.add_argument("--ids", default=None, help="Comma-separated task ids")
    p.add_argument("--max-tasks", type=int, default=None)
    p.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--mock",
        action="store_true",
        help="Offline mock LLM only (no Embedding / MCP / network)",
    )
    p.add_argument("--allow-dirty", action="store_true")
    p.add_argument("--env-file", default=None)
    p.add_argument(
        "--config",
        default=str(ROOT / "configs" / "hk_unified_mcp_free_v1.yaml"),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_mcp_free_config(Path(args.config), root=ROOT)
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (ROOT / output_dir).resolve()

    runner = UnifiedHkMcpFreeRunner(
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
        config=cfg,
        root=ROOT,
        json_mode=True,
    )
    runner.run()


if __name__ == "__main__":
    main()
