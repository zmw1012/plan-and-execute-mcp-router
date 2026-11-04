#!/usr/bin/env python3
"""Run the 10-task Operation-Visible ETOM routing pilot.

Index build and agent run are separate stages. The key values in the env
files are never printed. max_tasks must be 10.
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
from hk_equity_mcp.etom.runner import EtomPilotRunner, PilotLimitError  # noqa: E402
from hk_equity_mcp.etom.source import EtomSourceError  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Operation-Visible adapted for ETOM routing-only evaluation (10-task cost pilot)"
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-tasks", type=int, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--mock-llm", action="store_true")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--build-index-only", action="store_true")
    parser.add_argument("--index-status", action="store_true")
    parser.add_argument(
        "--require-complete-index",
        action="store_true",
        help="Refuse agent run unless the final catalog index is complete.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--env-file", default=None)
    parser.add_argument("--embedding-env-file", default=None)
    args = parser.parse_args()

    stage_flags = int(args.dry_run) + int(args.live) + int(args.build_index_only) + int(args.index_status)
    if args.mock_llm and (args.dry_run or args.live):
        print("Do not combine --mock-llm with --dry-run or --live.", file=sys.stderr)
        return 2
    if args.build_index_only or args.index_status:
        if stage_flags != 1:
            print(
                "Choose exactly one stage: --dry-run, --live, --build-index-only, or --index-status.",
                file=sys.stderr,
            )
            return 2
        mode = "mock" if args.mock_llm else "live"
    else:
        if int(args.dry_run) + int(args.mock_llm) + int(args.live) != 1:
            print(
                "Choose exactly one of --dry-run, --mock-llm, --live, "
                "--build-index-only, or --index-status.",
                file=sys.stderr,
            )
            return 2
        mode = "dry-run" if args.dry_run else "mock" if args.mock_llm else "live"

    require_complete = bool(args.require_complete_index) or mode == "live"
    if args.build_index_only or args.index_status:
        require_complete = False

    output = Path(args.output_dir)
    if not output.is_absolute():
        output = (ROOT / output).resolve()
    try:
        summary = EtomPilotRunner(
            ROOT,
            output,
            mode=mode,
            resume=args.resume,
            allow_dirty=args.allow_dirty,
            env_file=args.env_file,
            embedding_env_file=args.embedding_env_file,
            max_tasks=args.max_tasks,
            build_index_only=args.build_index_only,
            require_complete_index=require_complete,
            index_status_only=args.index_status,
        ).execute()
    except (
        PilotLimitError,
        EtomSourceError,
        ModelMismatchError,
        IncompleteCatalogIndexError,
        CatalogIndexError,
    ) as exc:
        print(scrub_secret(str(exc)), file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        if args.build_index_only:
            print(
                "Stopped during index build. Successful batches are checkpointed; "
                "rerun --build-index-only to resume.",
                file=sys.stderr,
            )
        else:
            print(
                "Stopped. Completed tasks are on disk; rerun with --resume.",
                file=sys.stderr,
            )
        return 130
    if isinstance(summary, dict):
        if args.index_status:
            print(
                "status=index-status completed={completed}/{total} final_complete={final}".format(
                    completed=summary.get("completed"),
                    total=summary.get("total"),
                    final=summary.get("final_complete"),
                )
            )
        elif args.build_index_only:
            print(
                "status=build-index completed={completed}/{total} cache_reused={reused}".format(
                    completed=summary.get("completed_count"),
                    total=summary.get("total_count"),
                    reused=summary.get("cache_reused"),
                )
            )
        else:
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
