#!/usr/bin/env python3
"""PE-Fin Lite HK-155 Retrieval Audit CLI (exploratory/descriptive).

Modes (explicit; no silent --no-mock switch):
  --mock      hash embeddings + mock planner
  --dry-run   identity/config checks only (zero network)
  --live      real generation/embedding via env files (fail-closed)

Offline ranking/report stages are zero-network and must not pass --live.
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
from hk_equity_mcp.experiment.pe_fin_lite_retrieval_audit.live_path import (  # noqa: E402
    LivePathError,
    resolve_mode,
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

    # Stages
    p.add_argument("--index-status", action="store_true")
    p.add_argument("--build-index-only", action="store_true")
    p.add_argument("--embed-question-queries", action="store_true")
    p.add_argument("--embed-planner-step-queries", action="store_true")
    p.add_argument("--question-level", action="store_true")
    p.add_argument("--planner-only", action="store_true")
    p.add_argument("--rank-from-planner-cache", action="store_true")
    p.add_argument("--catalog-coverage", action="store_true")
    p.add_argument("--select-formula", action="store_true")
    p.add_argument("--evaluate-frozen-formula", action="store_true")
    p.add_argument("--report-all", action="store_true")
    p.add_argument(
        "--formal-lock",
        action="store_true",
        help="With --select-formula: write formal lock only if real embedding backend",
    )

    # Modes
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--mock",
        action="store_true",
        help="Hash embeddings + mock planner (offline)",
    )
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Identity/config checks only; zero network",
    )
    mode.add_argument(
        "--live",
        action="store_true",
        help="Real generation/embedding via env files (fail-closed)",
    )
    p.add_argument(
        "--no-mock",
        action="store_true",
        help="Deprecated/refused; use --live explicitly",
    )

    # Env / cache
    p.add_argument(
        "--env-file",
        type=str,
        default=None,
        help="Generation env (e.g. .env.relay-v41)",
    )
    p.add_argument(
        "--embedding-env-file",
        type=str,
        default=None,
        help="Embedding env (e.g. .env.embedding-relay)",
    )
    p.add_argument(
        "--embedding-cache-dir",
        type=str,
        default=None,
        help="Catalog embedding cache dir (must end with pe_fin_lite_operation_embeddings_v1)",
    )
    p.add_argument(
        "--require-complete-index",
        action="store_true",
        help="Fail if catalog embedding index incomplete",
    )
    return p


def _classify_stages(args: argparse.Namespace) -> dict:
    live_stages = [
        args.build_index_only,
        args.embed_question_queries,
        args.embed_planner_step_queries,
        args.planner_only and args.live,
    ]
    offline_stages = [
        args.question_level,
        args.rank_from_planner_cache,
        args.select_formula,
        args.evaluate_frozen_formula,
        args.catalog_coverage,
        args.report_all,
        args.index_status,
        args.planner_only and (args.mock or args.dry_run),
    ]
    any_stage = any(
        [
            args.index_status,
            args.build_index_only,
            args.embed_question_queries,
            args.embed_planner_step_queries,
            args.question_level,
            args.planner_only,
            args.rank_from_planner_cache,
            args.catalog_coverage,
            args.select_formula,
            args.evaluate_frozen_formula,
            args.report_all,
        ]
    )
    return {
        "any": any_stage,
        "needs_live": any(live_stages),
        "needs_offline": any(offline_stages)
        and not (args.mock or args.dry_run)
        and not args.live,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    stages = _classify_stages(args)
    if not stages["any"]:
        print("Select at least one stage flag.", file=sys.stderr)
        return 2

    try:
        if args.no_mock:
            raise LivePathError(
                "--no-mock is deprecated and refused. Use explicit --live, "
                "--mock, or --dry-run."
            )

        # Default: offline ranking stages without mode → offline_rank;
        # live stages without --live → fail; else mock when --mock or default tooling.
        offline_rank = False
        mock = bool(args.mock)
        dry_run = bool(args.dry_run)
        live = bool(args.live)

        if stages["needs_live"] and not live and not mock and not dry_run:
            raise LivePathError(
                "Live stages require explicit --live (and env files). "
                "Refusing silent non-mock fallback."
            )

        if (
            not mock
            and not dry_run
            and not live
            and (
                args.question_level
                or args.rank_from_planner_cache
                or args.select_formula
                or args.evaluate_frozen_formula
                or args.catalog_coverage
                or args.report_all
                or args.index_status
            )
            and not (
                args.build_index_only
                or args.embed_question_queries
                or args.embed_planner_step_queries
                or args.planner_only
            )
        ):
            offline_rank = True

        if not mock and not dry_run and not live and not offline_rank:
            # Ambiguous non-mock without live for planner-only etc.
            if args.planner_only:
                raise LivePathError(
                    "planner-only requires --mock, --dry-run, or --live"
                )
            raise LivePathError(
                "Ambiguous mode. Pass --mock, --dry-run, or --live."
            )

        resolved = resolve_mode(
            mock=mock,
            dry_run=dry_run,
            live=live,
            no_mock=False,
            offline_rank=offline_rank,
        )

        runner = RetrievalAuditRunner(
            root=args.root,
            output_dir=args.output_dir,
            mode=resolved,
            env_file=args.env_file,
            embedding_env_file=args.embedding_env_file,
            embedding_cache_dir=args.embedding_cache_dir,
            require_complete_index=bool(args.require_complete_index),
        )

        results = {}
        if args.index_status:
            results["index_status"] = runner.get_index_status()
        if args.build_index_only:
            results["build_index_only"] = runner.build_index_only()
        if args.embed_question_queries:
            results["embed_question_queries"] = runner.embed_question_queries_stage()
        if args.planner_only:
            results["planner_only"] = runner.run_planner_only()
        if args.embed_planner_step_queries:
            results["embed_planner_step_queries"] = (
                runner.embed_planner_step_queries_stage()
            )
        if args.question_level:
            results["question_level"] = runner.run_question_level()
        if args.rank_from_planner_cache:
            results["rank_from_planner_cache"] = runner.run_rank_from_planner_cache()
        if args.catalog_coverage:
            results["catalog_coverage"] = runner.run_catalog_coverage()
        if args.select_formula:
            results["select_formula"] = runner.run_select_formula(
                formal=bool(args.formal_lock)
            )
        if args.evaluate_frozen_formula:
            results["evaluate_frozen_formula"] = runner.run_evaluate_frozen_formula()
        if args.report_all:
            results["report_all"] = runner.run_report_all()
    except LivePathError as exc:
        print(f"PE-FIN-LITE RETRIEVAL AUDIT FAIL-CLOSED: {exc}", file=sys.stderr)
        return 2

    print("PE-FIN-LITE RETRIEVAL AUDIT OK")
    print(f"output_dir={runner.output_dir}")
    print(f"audit_mode={resolved.mode.value}")
    print(f"modes={list(results)}")
    print("result_role=exploratory_descriptive")
    print("no_unseen_heldout_claim=True")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
