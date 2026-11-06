"""CLI for MCP-Zero × ETOM offline/live runners."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

from hk_equity_mcp.mcp_zero_etom.runner import EtomMcpZeroRunner


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="MCP-Zero × ETOM routing-only runner")
    p.add_argument("--split", choices=["pilot10", "formal100"], required=True)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--mock", action="store_true")
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--index-status", action="store_true")
    mode.add_argument("--build-index-only", action="store_true")
    mode.add_argument("--live", action="store_true", help="Disabled in offline construction round")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--max-tasks", type=int, default=None)
    p.add_argument("--ids", nargs="*", default=None, help="Debug-only task id filter")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--allow-dirty", action="store_true")
    p.add_argument("--require-complete-index", action="store_true")
    p.add_argument("--embedding-cache-dir", type=Path, default=None)
    p.add_argument("--etom-source-dir", type=str, default=None)
    p.add_argument("--etom-fixture-dir", type=str, default=None)
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--root", type=Path, default=None)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.root) if args.root else Path(__file__).resolve().parents[3]
    if args.live:
        raise SystemExit(
            "Live mode is disabled in this offline construction round. "
            "Use --mock / --dry-run / --preflight."
        )
    if args.dry_run:
        mode = "dry-run"
    elif args.mock:
        mode = "mock"
    elif args.preflight:
        mode = "preflight"
    elif args.index_status:
        mode = "index-status"
    elif args.build_index_only:
        mode = "mock"  # offline deterministic build only
    else:
        mode = "dry-run"
    split = args.split
    default_out = root / "outputs" / (
        f"etom_external_{'formal100' if split=='formal100' else 'pilot10'}_mcp_zero_v1"
        + ("_mock" if mode == "mock" else "_offline")
    )
    output_dir = Path(args.output_dir) if args.output_dir else default_out
    runner = EtomMcpZeroRunner(
        root,
        output_dir,
        split=split,
        mode=mode,
        resume=bool(args.resume),
        allow_dirty=bool(args.allow_dirty),
        max_tasks=args.max_tasks,
        ids=args.ids,
        config_path=args.config,
        etom_source_dir=args.etom_source_dir,
        fixture_dir=args.etom_fixture_dir,
        embedding_cache_dir=args.embedding_cache_dir,
        require_complete_index=bool(args.require_complete_index),
        index_status_only=bool(args.index_status),
        build_index_only=bool(args.build_index_only),
    )
    runner.execute()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
