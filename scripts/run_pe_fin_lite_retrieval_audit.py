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
    p.add_argument(
        "--planner-diagnostic10",
        action="store_true",
        help=(
            "Run frozen Planner protocol Diagnostic-10 with --planner-only only "
            "(engineering verification; not accuracy / not formal)"
        ),
    )
    p.add_argument("--rank-from-planner-cache", action="store_true")
    p.add_argument(
        "--planner-cache-audit",
        action="store_true",
        help="Zero-network Planner-only gate/audit after --planner-only",
    )
    p.add_argument(
        "--retry-failed-planner-cache",
        action="store_true",
        help=(
            "Live: derive ok=false task IDs from planner_cache, freeze attempt "
            "snapshot, retry only those tasks once, atomic-promote iff 155/155 ok. "
            "Requires --live + --env-file. No task-id hardcode."
        ),
    )
    p.add_argument(
        "--global-signal-ablation",
        action="store_true",
        help=(
            "Zero-network R4_global_coverage / R5_global_lexical ablations "
            "from complete planner + step-query + catalog caches."
        ),
    )
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
        help=(
            "Embedding env (e.g. .env.embedding-relay). Required for real "
            "index-status (--require-complete-index) and recommended for all "
            "offline ranking so catalog identity uses host/model/dim "
            "(identity-only; zero network). Without it, index-status is "
            "inventory-only and will not select host=null mock/hash cache."
        ),
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
    p.add_argument(
        "--allow-incomplete-planner-cache",
        action="store_true",
        help=(
            "Diagnostic only: allow step embedding/ranking when Planner has "
            "failures. Formal formula lock remains forbidden."
        ),
    )
    return p


def _classify_stages(args: argparse.Namespace) -> dict:
    live_stages = [
        args.build_index_only,
        args.embed_question_queries,
        args.embed_planner_step_queries,
        args.planner_only and args.live,
        args.retry_failed_planner_cache and args.live,
    ]
    offline_stages = [
        args.question_level,
        args.rank_from_planner_cache,
        args.planner_cache_audit,
        args.global_signal_ablation,
        args.select_formula,
        args.evaluate_frozen_formula,
        args.catalog_coverage,
        args.report_all,
        args.index_status,
        args.planner_only and (args.mock or args.dry_run),
        args.retry_failed_planner_cache and args.dry_run,
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
            args.planner_cache_audit,
            args.retry_failed_planner_cache,
            args.global_signal_ablation,
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
    if not stages["any"] and not args.planner_diagnostic10:
        print("Select at least one stage flag.", file=sys.stderr)
        return 2

    try:
        if args.no_mock:
            raise LivePathError(
                "--no-mock is deprecated and refused. Use explicit --live, "
                "--mock, or --dry-run."
            )

        if args.planner_diagnostic10:
            if not args.planner_only:
                raise LivePathError(
                    "--planner-diagnostic10 must be combined with --planner-only"
                )
            other = [
                args.index_status,
                args.build_index_only,
                args.embed_question_queries,
                args.embed_planner_step_queries,
                args.question_level,
                args.rank_from_planner_cache,
                args.planner_cache_audit,
                args.retry_failed_planner_cache,
                args.catalog_coverage,
                args.select_formula,
                args.evaluate_frozen_formula,
                args.report_all,
            ]
            if any(other):
                raise LivePathError(
                    "--planner-diagnostic10 only supports --planner-only "
                    "(no ranking / embeddings / formula / report)"
                )

        if args.retry_failed_planner_cache:
            other = [
                args.index_status,
                args.build_index_only,
                args.embed_question_queries,
                args.embed_planner_step_queries,
                args.question_level,
                args.planner_only,
                args.rank_from_planner_cache,
                args.planner_cache_audit,
                args.catalog_coverage,
                args.select_formula,
                args.evaluate_frozen_formula,
                args.report_all,
            ]
            if any(other):
                raise LivePathError(
                    "--retry-failed-planner-cache must run alone "
                    "(with --live --env-file [--embedding-env-file] "
                    "--embedding-cache-dir --output-dir)"
                )
            if args.mock:
                raise LivePathError(
                    "--retry-failed-planner-cache refuses --mock on CLI; "
                    "use --live for real retry (unit tests cover mock path)"
                )
            if not args.live and not args.dry_run:
                raise LivePathError(
                    "--retry-failed-planner-cache requires --live "
                    "(or --dry-run for failed-ID / snapshot check only)"
                )
            if args.live and not args.env_file:
                raise LivePathError(
                    "--retry-failed-planner-cache --live requires --env-file"
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
                or args.planner_cache_audit
                or args.global_signal_ablation
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
                or args.retry_failed_planner_cache
            )
        ):
            offline_rank = True

        if not mock and not dry_run and not live and not offline_rank:
            # Ambiguous non-mock without live for planner-only etc.
            if args.planner_only:
                raise LivePathError(
                    "planner-only requires --mock, --dry-run, or --live"
                )
            if args.retry_failed_planner_cache:
                raise LivePathError(
                    "retry-failed-planner-cache requires --live or --dry-run"
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
            allow_incomplete_planner_cache=bool(
                args.allow_incomplete_planner_cache
            ),
            planner_diagnostic10=bool(args.planner_diagnostic10),
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
        if args.retry_failed_planner_cache:
            results["retry_failed_planner_cache"] = (
                runner.run_retry_failed_planner_cache()
            )
        if args.embed_planner_step_queries:
            results["embed_planner_step_queries"] = (
                runner.embed_planner_step_queries_stage()
            )
        if args.question_level:
            results["question_level"] = runner.run_question_level()
        if args.rank_from_planner_cache:
            results["rank_from_planner_cache"] = runner.run_rank_from_planner_cache()
        if args.planner_cache_audit:
            results["planner_cache_audit"] = runner.run_planner_cache_audit()
        if args.global_signal_ablation:
            results["global_signal_ablation"] = runner.run_global_signal_ablation()
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
    if args.planner_diagnostic10:
        print("role=engineering_protocol_diagnostic_only")
        print("explicitly_not_formal_accuracy=True")
        diag = (results.get("planner_only") or {}).get("planner_protocol_diagnostic")
        if diag is not None:
            print(f"diagnostic10_ok={diag.get('ok')}")
            print(f"diagnostic10_failed={diag.get('failed')}")
            print(f"diagnostic10_engineering_pass={diag.get('engineering_pass')}")
    print("result_role=exploratory_descriptive")
    print("no_unseen_heldout_claim=True")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
