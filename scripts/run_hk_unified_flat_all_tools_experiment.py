#!/usr/bin/env python3
"""Unified HK Flat All Tools Real-MCP baseline runner.

Defaults:
  --split development
  Flat All Tools + UnifiedRealMcpGateway + 39 runtime operations exposed

Locks:
  --split exploratory_all requires --confirm-retire-heldout

Offline-safe:
  --dry-run / --mock (no paid LLM / Embedding / Real-MCP network)

Use sibling discovery python for real runs, e.g.:
  ../hk-equity-mcp-agent/.venv-aktools-discovery/bin/python
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
from hk_equity_mcp.experiment.unified_flat_runner import (  # noqa: E402
    UnifiedHkFlatAllToolsRunner,
    load_flat_config,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Unified HK Flat All Tools Real-MCP experiment"
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
        help=(
            "Required together with --split exploratory_all. After the first real "
            "Agent run, previously reserved items cease to be unseen final-test."
        ),
    )
    p.add_argument("--smoke", action="store_true", help="Run locked Dev smoke-8")
    p.add_argument("--ids", default=None, help="Comma-separated task ids")
    p.add_argument(
        "--max-tasks",
        type=int,
        default=None,
        help="Cumulative prefix of locked run order / ids (not resume identity)",
    )
    p.add_argument("--output-dir", required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--mock",
        action="store_true",
        help="Offline mock LLM + mock MCP transport (no embedding)",
    )
    p.add_argument("--allow-dirty", action="store_true")
    p.add_argument("--env-file", default=None)
    p.add_argument("--allow-hkex-download", action="store_true")
    p.add_argument(
        "--config",
        default=str(ROOT / "configs" / "hk_unified_flat_all_tools_v1.yaml"),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_flat_config(Path(args.config), root=ROOT)
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (ROOT / output_dir).resolve()

    runner = UnifiedHkFlatAllToolsRunner(
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
        allow_hkex_download=args.allow_hkex_download,
        json_mode=True,
    )
    runner.run()


if __name__ == "__main__":
    main()
