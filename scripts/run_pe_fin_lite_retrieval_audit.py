#!/usr/bin/env python3
"""PE-Fin Lite HK-155 Retrieval Audit CLI (exploratory/descriptive).

Offline-safe by default (--mock). Does not modify PE-Fin Lite Agent runtime.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hk_equity_mcp.experiment.pe_fin_lite_retrieval_audit import (  # noqa: E402
    DEFAULT_OUTPUT_DIR,
)
from hk_equity_mcp.experiment.pe_fin_lite_retrieval_audit.runner import (  # noqa: E402
    RetrievalAuditRunner,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=f"Default: {DEFAULT_OUTPUT_DIR}",
    )
    p.add_argument("--question-level", action="store_true")
    p.add_argument("--planner-only", action="store_true")
    p.add_argument("--rank-from-planner-cache", action="store_true")
    p.add_argument("--catalog-coverage", action="store_true")
    p.add_argument("--select-formula", action="store_true")
    p.add_argument("--evaluate-frozen-formula", action="store_true")
    p.add_argument("--report-all", action="store_true")
    p.add_argument(
        "--mock",
        action="store_true",
        default=True,
        help="Use offline hash embeddings / mock planner (default True)",
    )
    p.add_argument("--no-mock", action="store_true", help="Disable mock mode")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--allow-network",
        action="store_true",
        help="Forbidden this round; runner will refuse",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    mock = bool(args.mock) and not bool(args.no_mock)
    modes = [
        args.question_level,
        args.planner_only,
        args.rank_from_planner_cache,
        args.catalog_coverage,
        args.select_formula,
        args.evaluate_frozen_formula,
        args.report_all,
    ]
    if not any(modes):
        print("Select at least one mode flag.", file=sys.stderr)
        return 2

    runner = RetrievalAuditRunner(
        root=args.root,
        output_dir=args.output_dir,
        mock=mock,
        dry_run=bool(args.dry_run),
        allow_network=bool(args.allow_network),
    )
    results = {}
    if args.question_level:
        results["question_level"] = runner.run_question_level()
    if args.planner_only:
        results["planner_only"] = runner.run_planner_only()
    if args.rank_from_planner_cache:
        results["rank_from_planner_cache"] = runner.run_rank_from_planner_cache()
    if args.catalog_coverage:
        results["catalog_coverage"] = runner.run_catalog_coverage()
    if args.select_formula:
        results["select_formula"] = runner.run_select_formula()
    if args.evaluate_frozen_formula:
        results["evaluate_frozen_formula"] = runner.run_evaluate_frozen_formula()
    if args.report_all:
        results["report_all"] = runner.run_report_all()

    print("PE-FIN-LITE RETRIEVAL AUDIT OK")
    print(f"output_dir={runner.output_dir}")
    print(f"modes={list(results)}")
    print("result_role=exploratory_descriptive")
    print("no_unseen_heldout_claim=True")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
