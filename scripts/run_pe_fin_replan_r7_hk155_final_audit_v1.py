#!/usr/bin/env python3
"""Offline final audit for PE-Fin RePlan-7 HK-155 (zero network; no raw mutation)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hk_equity_mcp.experiment.pe_fin_replan_r7_hk155_final_audit.runner import (  # noqa: E402
    run_final_audit,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--run-dir",
        default=str(ROOT / "outputs" / "pe_fin_replan_r7_hk155_live_v1"),
    )
    p.add_argument("--root", default=str(ROOT))
    args = p.parse_args()
    summary = run_final_audit(
        root=Path(args.root),
        run_dir=Path(args.run_dir),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    if not summary.get("raw_hashes_unchanged"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
