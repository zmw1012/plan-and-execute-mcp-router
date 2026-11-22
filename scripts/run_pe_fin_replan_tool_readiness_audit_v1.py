#!/usr/bin/env python3
"""PE-Fin RePlan tool readiness audit v1.

Modes:
  --contract-audit      zero-network; covers all runtime operations + provider profile
  --critical-live-audit future live only (refused in offline default path)
  --market-sina-probe   future live Sina probe only (refused offline)

This script never calls generation LLM or embedding APIs in contract mode.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from hk_equity_mcp.pe_fin_replan.tool_readiness import (  # noqa: E402
    build_future_sina_probe_command,
    build_future_smoke5_v4_command,
    run_contract_audit,
    run_critical_live_audit,
    run_market_sina_probe,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--contract-audit", action="store_true")
    mode.add_argument("--critical-live-audit", action="store_true")
    mode.add_argument("--market-sina-probe", action="store_true")
    p.add_argument(
        "--output-dir",
        default=None,
        help="Audit output directory",
    )
    p.add_argument(
        "--market-provider-profile",
        default=str(
            ROOT
            / "configs/operations/hk_runtime_provider_availability_sina_market_v1.yaml"
        ),
    )
    p.add_argument(
        "--env-file",
        default=None,
        help="Only for future live probe modes (unused by contract-audit)",
    )
    p.add_argument(
        "--print-future-commands",
        action="store_true",
        help="Print deferred live probe + Smoke-5 v4 commands and exit",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.print_future_commands:
        print(build_future_sina_probe_command(ROOT))
        print()
        print(build_future_smoke5_v4_command(ROOT))
        return 0

    if args.contract_audit:
        out = (
            Path(args.output_dir)
            if args.output_dir
            else ROOT / "outputs/pe_fin_replan_tool_readiness_v1/contract_audit"
        )
        if not out.is_absolute():
            out = (ROOT / out).resolve()
        summary = run_contract_audit(
            root=ROOT,
            output_dir=out,
            profile_path=Path(args.market_provider_profile),
        )
        print("PE-FIN-REPLAN TOOL READINESS CONTRACT AUDIT OK")
        print(f"status={summary['status']}")
        print(f"runtime_operation_count={summary['runtime_operation_count']}")
        print(f"disabled_count={len(summary['disabled_operation_ids'])}")
        print(f"network_called={summary['network_called']}")
        print(f"live_pass_forged={summary['live_pass_forged']}")
        print(f"output_dir={out}")
        return 0 if summary["status"] == "pass" else 2

    if args.critical_live_audit:
        out = (
            Path(args.output_dir)
            if args.output_dir
            else ROOT / "outputs/pe_fin_replan_tool_readiness_v1/critical_live_audit"
        )
        run_critical_live_audit(root=ROOT, output_dir=out)
        return 1

    if args.market_sina_probe:
        out = (
            Path(args.output_dir)
            if args.output_dir
            else ROOT / "outputs/pe_fin_replan_tool_readiness_v1/market_sina_probe_v3"
        )
        if not out.is_absolute():
            out = (ROOT / out).resolve()
        summary = run_market_sina_probe(
            root=ROOT,
            output_dir=out,
            profile_path=Path(args.market_provider_profile),
            env_file=args.env_file,
        )
        print("PE-FIN-REPLAN MARKET SINA PROBE")
        print(f"status={summary['status']}")
        print(f"network_called={summary['network_called']}")
        print(f"gateway_success={summary['gateway_success']}")
        print(f"normalization_success={summary['normalization_success']}")
        print(f"failure_class={summary.get('failure_class')}")
        print(f"output_dir={out}")
        return 0 if summary["status"] == "pass" else 2

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
