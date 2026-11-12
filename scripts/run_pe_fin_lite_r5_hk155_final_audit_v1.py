#!/usr/bin/env python3
"""CLI: read-only PE-Fin Lite R5 HK-155 final audit (zero-network)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hk_equity_mcp.experiment.pe_fin_lite_r5_hk155_final_audit.runner import (  # noqa: E402
    run_final_audit,
)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--run-dir",
        default=str(ROOT / "outputs" / "pe_fin_lite_r5_hk155_live_v1"),
    )
    p.add_argument("--root", default=str(ROOT))
    p.add_argument(
        "--output-dir",
        default=None,
        help="Default: <run-dir>/final_audit_r5_v1",
    )
    p.add_argument(
        "--docs-mirror",
        default=str(ROOT / "docs" / "pe_fin_lite_r5_hk155_final_audit_v1"),
    )
    args = p.parse_args(argv)
    result = run_final_audit(
        run_dir=Path(args.run_dir),
        root=Path(args.root),
        output_dir=Path(args.output_dir) if args.output_dir else None,
        docs_mirror=Path(args.docs_mirror) if args.docs_mirror else None,
    )
    print("PE-FIN-LITE R5 HK-155 FINAL AUDIT OK")
    print(f"out_dir={result['out_dir']}")
    print(f"docs_dir={result['docs_dir']}")
    print(f"engineering_valid={result['engineering_valid']}")
    print(f"manual_pending_n={result['manual_pending_n']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
