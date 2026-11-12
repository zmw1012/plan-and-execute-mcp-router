#!/usr/bin/env python3
"""PE-Fin Lite proposed-method runner (separated from APV-Fin / OV baseline).

Defaults stay offline-safe when --dry-run / --mock are set.
Paid LLM/Embedding/MCP only when explicitly run without those flags.

Index tooling (no generation LLM / no MCP tools):
  --build-index-only
  --index-status
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

from hk_equity_mcp.experiment.runner import parse_ids  # noqa: E402
from hk_equity_mcp.experiment.unified_pe_fin_lite_runner import (  # noqa: E402
    UnifiedHkPeFinLiteRunner,
    load_pe_fin_lite_config,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Unified HK PE-Fin Lite Real-MCP experiment")
    p.add_argument(
        "--split",
        default="development",
        choices=["development", "heldout", "exploratory_all"],
    )
    p.add_argument("--confirm-heldout-final", action="store_true")
    p.add_argument("--confirm-retire-heldout", action="store_true")
    p.add_argument("--smoke", action="store_true", help="Run locked Dev smoke-8")
    p.add_argument(
        "--diagnostic12",
        action="store_true",
        help="Run Diagnostic-12 pointer subset (same IDs/order as APV-Fin)",
    )
    p.add_argument(
        "--smoke5",
        action="store_true",
        help="Run engineering Smoke-5 (not a formal accuracy result)",
    )
    p.add_argument(
        "--diagnostic20-eea",
        action="store_true",
        help=(
            "Run PE-Fin Lite v1.2-EEA Development Diagnostic-20 "
            "(post-hoc exploratory; NOT formal accuracy)"
        ),
    )
    p.add_argument(
        "--diagnostic20-replan",
        action="store_true",
        help=(
            "Run PE-Fin RePlan on frozen Diagnostic-20 "
            "(post-hoc exploratory; NOT formal accuracy; separate from EEA-only)"
        ),
    )
    p.add_argument(
        "--replan-smoke5",
        action="store_true",
        help=(
            "Run PE-Fin RePlan live-engineering Smoke-5 "
            "(NOT accuracy; cannot replace Diagnostic-20 / HK-155)"
        ),
    )
    p.add_argument(
        "--eea",
        action="store_true",
        help="Enable PE-Fin Lite v1.2-EEA execution/evidence alignment (post-hoc)",
    )
    p.add_argument(
        "--replan",
        action="store_true",
        help=(
            "Enable PE-Fin RePlan (Planner→Execute-one→Replanner loop; "
            "post-hoc exploratory; implies EEA)"
        ),
    )
    p.add_argument("--ids", default=None, help="Comma-separated task ids")
    p.add_argument("--max-tasks", type=int, default=None)
    p.add_argument(
        "--output-dir",
        default=None,
        help="Required for experiment runs; optional for --index-status/--build-index-only",
    )
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--mock",
        action="store_true",
        help="Offline mock LLM + hash embedding + mock MCP transport",
    )
    p.add_argument("--allow-dirty", action="store_true")
    p.add_argument("--env-file", default=None)
    p.add_argument("--embedding-env-file", default=None)
    p.add_argument("--allow-hkex-download", action="store_true")
    p.add_argument(
        "--config",
        default=str(ROOT / "configs" / "hk_unified_pe_fin_lite_v1.yaml"),
    )
    p.add_argument(
        "--build-index-only",
        action="store_true",
        help="Build/resume shared operation embedding index only (no LLM/MCP/agent)",
    )
    p.add_argument(
        "--index-status",
        action="store_true",
        help="Print local embedding index status (zero network)",
    )
    p.add_argument(
        "--require-complete-index",
        action="store_true",
        help="Fail closed unless shared catalog embedding index is complete",
    )
    p.add_argument(
        "--embedding-cache-dir",
        default=None,
        help="Override shared PE-Fin Lite embedding cache dir (must end with namespace)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_pe_fin_lite_config(Path(args.config), root=ROOT)

    if args.diagnostic12 and args.smoke5:
        raise SystemExit("Cannot combine --diagnostic12 and --smoke5")
    if args.diagnostic20_eea and (args.diagnostic12 or args.smoke5):
        raise SystemExit("Cannot combine --diagnostic20-eea with diagnostic12/smoke5")
    if args.diagnostic20_replan and (args.diagnostic12 or args.smoke5):
        raise SystemExit(
            "Cannot combine --diagnostic20-replan with diagnostic12/smoke5"
        )
    if args.replan_smoke5 and (
        args.diagnostic12 or args.smoke5 or args.diagnostic20_eea
    ):
        raise SystemExit(
            "Cannot combine --replan-smoke5 with diagnostic12/smoke5/diagnostic20-eea"
        )
    if args.diagnostic20_eea and args.diagnostic20_replan:
        raise SystemExit(
            "Cannot combine --diagnostic20-eea with --diagnostic20-replan"
        )
    if args.diagnostic20_replan and args.replan_smoke5:
        raise SystemExit(
            "Cannot combine --diagnostic20-replan with --replan-smoke5"
        )

    if args.output_dir:
        output_dir = Path(args.output_dir)
        if not output_dir.is_absolute():
            output_dir = (ROOT / output_dir).resolve()
    else:
        if not (args.build_index_only or args.index_status):
            raise SystemExit(
                "--output-dir is required unless --build-index-only/--index-status"
            )
        output_dir = ROOT / "outputs" / "_pe_fin_lite_index_tooling_placeholder"

    emb_cache = Path(args.embedding_cache_dir) if args.embedding_cache_dir else None
    runner = UnifiedHkPeFinLiteRunner(
        output_dir=output_dir,
        split=args.split,
        smoke=args.smoke,
        diagnostic12=args.diagnostic12,
        smoke5=args.smoke5,
        diagnostic20_eea=args.diagnostic20_eea,
        diagnostic20_replan=args.diagnostic20_replan,
        replan_smoke5=args.replan_smoke5,
        eea_enabled=bool(
            args.eea
            or args.diagnostic20_eea
            or args.replan
            or args.diagnostic20_replan
            or args.replan_smoke5
        ),
        replan_enabled=bool(
            args.replan or args.diagnostic20_replan or args.replan_smoke5
        ),
        ids=parse_ids(args.ids),
        max_tasks=args.max_tasks,
        resume=args.resume,
        dry_run=args.dry_run,
        mock=args.mock,
        allow_dirty=args.allow_dirty,
        confirm_heldout_final=args.confirm_heldout_final,
        confirm_retire_heldout=args.confirm_retire_heldout,
        env_file=args.env_file,
        embedding_env_file=args.embedding_env_file,
        config=cfg,
        root=ROOT,
        allow_hkex_download=args.allow_hkex_download,
        json_mode=True,
        require_complete_index=args.require_complete_index,
        embedding_cache_dir=emb_cache,
        allow_auto_build_index=False,
    )

    if args.index_status:
        status = runner.get_index_status()
        print(json.dumps(status, ensure_ascii=False, indent=2, sort_keys=True))
        return

    if args.build_index_only:
        result = runner.build_index_only()
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        if not result.get("complete"):
            raise SystemExit(2)
        return

    if not args.dry_run and not args.mock and not args.require_complete_index:
        if args.smoke5 or args.diagnostic12 or args.replan_smoke5:
            raise SystemExit(
                "Real Smoke-5 / Diagnostic-12 / RePlan Smoke-5 requires "
                "--require-complete-index (build with --build-index-only first)."
            )

    runner.run()


if __name__ == "__main__":
    main()
