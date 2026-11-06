#!/usr/bin/env python3
"""Write or verify the frozen ETOM-100 external formal sample lock."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hk_equity_mcp.etom.formal_sampler import (  # noqa: E402
    assert_formal_100_lock_matches,
    build_formal_100_lock,
    summarize_formal_sample,
    write_formal_100_lock,
)
from hk_equity_mcp.etom.loader import load_official_tasks  # noqa: E402
from hk_equity_mcp.etom.source import formal_100_lock_path, verify_source  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze or verify ETOM formal-100 lock")
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--report-json", default=None)
    args = parser.parse_args()
    verify_source(ROOT)
    tasks = load_official_tasks(ROOT, verify=False)
    path = formal_100_lock_path(ROOT)
    if args.write:
        if path.exists() and not args.force:
            print(f"Lock already exists: {path}. Refusing to resample.", file=sys.stderr)
            return 2
        payload = build_formal_100_lock(ROOT, tasks)
        write_formal_100_lock(ROOT, payload)
        print(f"wrote {path}")
        print(f"sample_sha256={payload['sample_sha256']}")
    payload = assert_formal_100_lock_matches(ROOT, tasks)
    summary = summarize_formal_sample(tasks, payload)
    print("lock_matches_sampler=true")
    print(f"sample_size={summary['sample_size']}")
    print(f"level_counts={summary['level_counts']}")
    print(f"pilot_overlap={summary['pilot_overlap']}")
    print(f"unique_servers={summary['unique_servers']}")
    if args.report_json:
        Path(args.report_json).write_text(
            json.dumps({"lock": payload, "summary": summary}, ensure_ascii=False, indent=2)
            + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
