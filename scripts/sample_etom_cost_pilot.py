#!/usr/bin/env python3
"""Write or check the locked 10-task ETOM cost-pilot sample."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hk_equity_mcp.etom.loader import load_official_tasks  # noqa: E402
from hk_equity_mcp.etom.sampler import (  # noqa: E402
    assert_lock_matches,
    sample_cost_pilot,
    write_lock,
)
from hk_equity_mcp.etom.source import lock_path  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Lock the ETOM 10-task cost pilot")
    parser.add_argument("--write", action="store_true", help="Create the lock if it is absent")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing lock")
    args = parser.parse_args()
    tasks = load_official_tasks(ROOT)
    selected = sample_cost_pilot(tasks)
    path = lock_path(ROOT)
    if args.write:
        if path.exists() and not args.force:
            raise SystemExit(f"Lock already exists: {path}. Refusing to resample.")
        write_lock(ROOT, selected)
        print(f"wrote {path}")
    assert_lock_matches(ROOT, tasks)
    print("lock_matches_sampler=true")
    for task in selected:
        print(
            f"{task.task_id}\tL{task.level}\tservers={','.join(sorted(task.server_names)) or '-'}"
            f"\tsequential={task.single_server_sequential}"
            f"\tcross={task.cross_server_dependency}"
            f"\tinfeasible={task.infeasible}"
        )


if __name__ == "__main__":
    main()
