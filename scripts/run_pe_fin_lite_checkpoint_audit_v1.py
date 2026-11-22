#!/usr/bin/env python3
"""Offline zero-network PE-Fin Lite End-to-End Checkpoint Audit v1.

Read-only on raw Diagnostic-12 (or fixture) run artifacts. Never calls
LLM / Embedding / MCP. Never mutates predictions or run_metadata.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from hk_equity_mcp.experiment.pe_fin_lite_checkpoint_audit.runner import (  # noqa: E402
    run_checkpoint_audit,
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--run-dir",
        required=True,
        help="Diagnostic-12 live/mock/fixture output directory containing predictions.jsonl",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help="Audit output directory (default: <run-dir>/final_checkpoint_audit_v1)",
    )
    p.add_argument(
        "--docs-mirror",
        default=None,
        help="Optional docs/ mirror directory to copy audit artifacts into",
    )
    p.add_argument("--expected-n", type=int, default=12)
    p.add_argument(
        "--allow-missing-predictions",
        action="store_true",
        help="Allow empty predictions (fixture scaffolding only)",
    )
    args = p.parse_args()
    result = run_checkpoint_audit(
        run_dir=Path(args.run_dir),
        output_dir=Path(args.output_dir) if args.output_dir else None,
        expected_n=int(args.expected_n),
        docs_mirror=Path(args.docs_mirror) if args.docs_mirror else None,
        allow_missing_predictions=bool(args.allow_missing_predictions),
    )
    print("PE-FIN-LITE CHECKPOINT AUDIT v1 OK")
    print(f"output_dir={result['output_dir']}")
    print(f"engineering_pass={result['engineering_pass']}")
    print(f"raw_hashes_unchanged={result['raw_hashes_unchanged']}")
    print("explicitly_not_formal_accuracy=True")
    print("zero_network=True")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
