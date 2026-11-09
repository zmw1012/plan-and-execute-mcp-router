#!/usr/bin/env python3
"""Offline final audit for APV-Fin Diagnostic-12 live v1.4.1.

Read-only: no LLM / embedding / MCP / paid API.
Does not modify raw predictions or APV-Fin method code paths beyond writing audit outputs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hk_equity_mcp.experiment.apv_fin_diagnostic12_final_audit_v1 import (  # noqa: E402
    run_final_audit,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--run-dir",
        default="outputs/apv_fin_diagnostic12_live_v1_4_1",
        help="Completed Diagnostic-12 run directory",
    )
    p.add_argument(
        "--out-dir",
        default=None,
        help="Optional override for <run-dir>/final_audit_v1",
    )
    p.add_argument(
        "--docs-dir",
        default=None,
        help="Optional override for docs/apv_fin_diagnostic12_final_audit_v1",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = (ROOT / run_dir).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else None
    docs_dir = Path(args.docs_dir).resolve() if args.docs_dir else None
    result = run_final_audit(
        root=ROOT,
        run_dir=run_dir,
        out_dir=out_dir,
        docs_dir=docs_dir,
    )
    print("APV-FIN DIAGNOSTIC-12 FINAL AUDIT V1 OK")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
