#!/usr/bin/env python3
"""CLI for PE-Fin RePlan ETOM-100 (offline mock/dry-run + future live gate)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hk_equity_mcp.experiment.unified_pe_fin_replan_etom_runner import (  # noqa: E402
    EtomPeFinReplanRunnerError,
    UnifiedPeFinReplanEtomRunner,
)
from hk_equity_mcp.pe_fin_replan_etom.operation_embedding_index import (  # noqa: E402
    LiveEmbeddingGatedError,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PE-Fin RePlan ETOM experiment runner")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--index-status", action="store_true")
    parser.add_argument("--build-index-only", action="store_true")
    parser.add_argument("--formal100", action="store_true", default=False)
    parser.add_argument("--require-complete-index", action="store_true")
    parser.add_argument("--env-file", default=None)
    parser.add_argument("--embedding-env-file", default=None)
    parser.add_argument("--embedding-cache-dir", default=None)
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "outputs" / "pe_fin_replan_etom_formal100_v1"),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-tasks", type=int, default=100)
    parser.add_argument(
        "--mock-embeddings",
        action="store_true",
        help="Allow hash embedding index build (offline only).",
    )
    args = parser.parse_args(argv)

    mode = "dry-run"
    if args.mock:
        mode = "mock"
    elif args.dry_run or args.index_status:
        mode = "dry-run"
    elif args.build_index_only and args.mock_embeddings:
        mode = "mock"
    elif not (args.index_status or args.build_index_only or args.dry_run or args.mock):
        # bare --formal100 implies future live path (gated inside runner)
        mode = "live"

    runner = UnifiedPeFinReplanEtomRunner(
        ROOT,
        Path(args.output_dir),
        mode=mode,
        formal100=bool(args.formal100 or True),
        resume=args.resume,
        env_file=args.env_file,
        embedding_env_file=args.embedding_env_file,
        embedding_cache_dir=args.embedding_cache_dir,
        require_complete_index=args.require_complete_index,
        index_status_only=args.index_status,
        build_index_only=args.build_index_only,
        mock_embeddings=args.mock_embeddings or args.mock,
        max_tasks=args.max_tasks,
    )
    try:
        result = runner.execute()
    except (LiveEmbeddingGatedError, EtomPeFinReplanRunnerError) as exc:
        # Fail closed with a readable message (offline scaffolding gates live paths).
        print(f"GATED: {exc}", file=sys.stderr)
        return 3
    print(json_dumps(result))
    return 0 if result.get("ok", True) is not False and runner.run_status != "aborted" else 2


def json_dumps(obj) -> str:
    import json

    return json.dumps(obj, ensure_ascii=False, indent=2, default=str)


if __name__ == "__main__":
    raise SystemExit(main())
