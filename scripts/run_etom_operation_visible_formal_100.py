#!/usr/bin/env python3
"""Run the Operation-Visible ETOM-100 formal evaluation.

Reads benchmarks/etom/formal_100_v1.lock.json only. Never resamples.
Live mode requires a complete catalog embedding index.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hk_equity_mcp.etom.embeddings import (  # noqa: E402
    CatalogIndexError,
    IncompleteCatalogIndexError,
    ModelMismatchError,
    scrub_secret,
)
from hk_equity_mcp.etom.formal_runner import EtomFormal100Runner, FormalPilotError  # noqa: E402
from hk_equity_mcp.etom.sampler import SamplerError  # noqa: E402
from hk_equity_mcp.etom.source import EtomSourceError  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Operation-Visible ETOM-100 formal routing-only evaluation"
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/etom_external_formal100_operation_visible_v1",
    )
    parser.add_argument("--max-tasks", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--index-status", action="store_true")
    parser.add_argument("--require-complete-index", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--env-file", default=None)
    parser.add_argument("--embedding-env-file", default=None)
    args = parser.parse_args()

    selected = int(args.dry_run) + int(args.mock) + int(args.live) + int(args.index_status)
    if args.index_status:
        if args.dry_run or args.live:
            print("Do not combine --index-status with --dry-run or --live.", file=sys.stderr)
            return 2
        mode = "mock" if args.mock else "live"
    elif selected != 1:
        print(
            "Choose exactly one of --dry-run, --mock, --live, or --index-status.",
            file=sys.stderr,
        )
        return 2
    else:
        mode = "dry-run" if args.dry_run else "mock" if args.mock else "live"

    output = Path(args.output_dir)
    if not output.is_absolute():
        output = (ROOT / output).resolve()
    try:
        summary = EtomFormal100Runner(
            ROOT,
            output,
            mode=mode,
            resume=args.resume,
            allow_dirty=args.allow_dirty,
            env_file=args.env_file,
            embedding_env_file=args.embedding_env_file,
            max_tasks=args.max_tasks,
            require_complete_index=bool(args.require_complete_index) or mode == "live",
            index_status_only=args.index_status,
        ).execute()
    except (
        FormalPilotError,
        SamplerError,
        EtomSourceError,
        ModelMismatchError,
        IncompleteCatalogIndexError,
        CatalogIndexError,
    ) as exc:
        print(scrub_secret(str(exc)), file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print(
            "Stopped. Completed formal tasks are on disk; rerun with --resume.",
            file=sys.stderr,
        )
        return 130
    if isinstance(summary, dict):
        print(
            "status={status} completed={completed} failed={failed} real_tool_executions={tools}".format(
                status=summary.get("run_status") or summary.get("mode"),
                completed=summary.get("completed", summary.get("max_tasks")),
                failed=summary.get("failed", 0),
                tools=summary.get("real_tool_executions", 0),
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
