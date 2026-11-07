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
    mode.add_argument(
        "--build-index-only",
        action="store_true",
        help="Build/resume shared hierarchical embedding index with the real embedding API",
    )
    mode.add_argument(
        "--live",
        action="store_true",
        help="Paid live routing-only run (requires complete index + dual env profiles)",
    )
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument(
        "--max-tasks",
        type=int,
        default=None,
        help="Cumulative locked-order prefix size (not part of resume identity)",
    )
    p.add_argument("--ids", nargs="*", default=None, help="Debug-only task id filter")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--allow-dirty", action="store_true")
    p.add_argument(
        "--require-complete-index",
        action="store_true",
        help="Refuse to run/rebuild when the shared index is incomplete (forced on --live)",
    )
    p.add_argument("--embedding-cache-dir", type=Path, default=None)
    p.add_argument("--etom-source-dir", type=str, default=None)
    p.add_argument(
        "--etom-fixture-dir",
        type=str,
        default=None,
        help="Offline fixture only (mock/dry-run/preflight/index-status)",
    )
    p.add_argument(
        "--env-file",
        type=str,
        default=None,
        help="Generation env profile (LLM_*). Never prints API keys.",
    )
    p.add_argument(
        "--embedding-env-file",
        type=str,
        default=None,
        help="Embedding env profile (EMBEDDING_*). Never prints API keys.",
    )
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--root", type=Path, default=None)
    return p


def _resolve_mode(args: argparse.Namespace) -> str:
    if args.dry_run:
        return "dry-run"
    if args.mock:
        return "mock"
    if args.preflight:
        return "preflight"
    if args.index_status:
        return "index-status"
    if args.build_index_only:
        return "build-index"
    if args.live:
        return "live"
    raise SystemExit("Exactly one mode flag is required.")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.root) if args.root else Path(__file__).resolve().parents[3]
    mode = _resolve_mode(args)
    split = args.split
    if mode == "live":
        default_out = root / "outputs" / (
            f"etom_external_{'formal100' if split == 'formal100' else 'pilot10'}_mcp_zero_v1"
        )
    elif mode == "mock":
        default_out = root / "outputs" / (
            f"etom_external_{'formal100' if split == 'formal100' else 'pilot10'}_mcp_zero_v1_mock"
        )
    elif mode == "build-index":
        default_out = root / "outputs" / "_mcp_zero_etom_index_ops"
    else:
        default_out = root / "outputs" / (
            f"etom_external_{'formal100' if split == 'formal100' else 'pilot10'}_mcp_zero_v1_offline"
        )
    output_dir = Path(args.output_dir) if args.output_dir else default_out
    require_complete = bool(args.require_complete_index) or mode == "live"
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
        require_complete_index=require_complete,
        index_status_only=mode == "index-status",
        build_index_only=mode == "build-index",
        env_file=args.env_file,
        embedding_env_file=args.embedding_env_file,
    )
    runner.execute()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
