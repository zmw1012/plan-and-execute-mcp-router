#!/usr/bin/env python3
"""No-LLM Market Real-MCP discovery smoke (AKTools).

Writes under outputs/mcp_discovery_market_v1/. Does not call LLMs or embeddings.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from hk_equity_mcp.real_mcp.adapters.aktools import AktoolsAdapter

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "mcp_discovery_market_v1"
CUTOFF = "2026-08-12"
SYMBOLS = {"Tencent": "00700", "HSBC": "00005", "Alibaba": "09988"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def rows_of(result) -> list:
    norm = result.normalized
    if isinstance(norm, dict):
        return list(norm.get("rows") or [])
    return []


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    report: dict = {"timestamp": utc_now(), "cutoff": CUTOFF, "probes": {}}
    ad = AktoolsAdapter(timeout_sec=240)
    ad.open()
    try:
        listed = ad.list_tools()
        report["aktools_tools"] = [t["name"] for t in listed.get("tools") or []]
        report["server_version"] = listed.get("server_version")

        # Primary hist for three issuers on cutoff
        for name, sym in SYMBOLS.items():
            t0 = time.time()
            res = ad.get_data(
                interface="stock_hk_hist",
                params={
                    "symbol": sym,
                    "period": "daily",
                    "start_date": "20260812",
                    "end_date": "20260812",
                    "adjust": "",
                },
                offset=0,
                max_rows=5,
                operation_id="aktools.stock_hk_hist",
            )
            rows = rows_of(res)
            report["probes"][f"hist_{name}"] = {
                "success": res.success,
                "error": res.error,
                "latency_ms": int((time.time() - t0) * 1000),
                "row": rows[0] if rows else None,
            }

        # HSI via negative offset
        t0 = time.time()
        res = ad.get_data(
            interface="stock_hk_index_daily_sina",
            params={"symbol": "HSI"},
            offset=-120,
            max_rows=120,
            operation_id="aktools.stock_hk_index_daily_sina",
        )
        rows = rows_of(res)
        on = [r for r in rows if str(r.get("date", ""))[:10] == CUTOFF]
        report["probes"]["hsi_sina"] = {
            "success": res.success,
            "error": res.error,
            "latency_ms": int((time.time() - t0) * 1000),
            "on_cutoff": on[:1],
            "n_rows": len(rows),
        }

        # Invalid symbol failure shape
        res = ad.get_data(
            interface="stock_hk_hist",
            params={
                "symbol": "999999",
                "period": "daily",
                "start_date": "20260801",
                "end_date": "20260812",
                "adjust": "",
            },
            offset=0,
            max_rows=3,
            operation_id="aktools.stock_hk_hist",
        )
        report["probes"]["hist_invalid"] = {
            "success": res.success,
            "error": res.error,
        }
    finally:
        ad.close()

    path = args.out / "smoke_report.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"wrote": str(path), "tools": report.get("aktools_tools"), "ok_hist": all(report["probes"][f"hist_{n}"]["success"] for n in SYMBOLS), "hsi": bool(report["probes"]["hsi_sina"].get("on_cutoff"))}, ensure_ascii=False))
    # Fail closed if primary probes fail
    if not all(report["probes"][f"hist_{n}"]["success"] for n in SYMBOLS):
        return 2
    if not report["probes"]["hsi_sina"].get("on_cutoff"):
        return 3
    if report["probes"]["hist_invalid"].get("success") is True:
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
