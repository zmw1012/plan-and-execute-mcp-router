#!/usr/bin/env python3
"""Run the 10-task Operation-Visible ETOM routing pilot.

Modes are mutually exclusive: --dry-run, --mock-llm, --live.
max_tasks must be 10. The key values in the env files are never printed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hk_equity_mcp.etom.embeddings import ModelMismatchError, scrub_secret  # noqa: E402
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
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--env-file", default=None)
    parser.add_argument("--embedding-env-file", default=None)
    args = parser.parse_args()
    selected = int(args.dry_run) + int(args.mock_llm) + int(args.live)
    if selected != 1:
        print("Choose exactly one of --dry-run, --mock-llm, or --live.", file=sys.stderr)
        return 2
    mode = "dry-run" if args.dry_run else "mock" if args.mock_llm else "live"
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
        ).execute()
    except (PilotLimitError, EtomSourceError, ModelMismatchError) as exc:
        print(scrub_secret(str(exc)), file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Stopped. Completed tasks are on disk; rerun with --resume.", file=sys.stderr)
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
