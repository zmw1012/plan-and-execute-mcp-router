#!/usr/bin/env python3
"""No-LLM News Real-MCP discovery smoke (AKTools + mcp-hkexnews + ah-disclosure).

Writes under outputs/mcp_discovery_news_v1/. No LLM / embedding / paid API.
Temporal policy: max gold date 2026-08-12 Asia/Hong_Kong; no floating latest as gold.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT.parent / "hk-equity-mcp-agent"
OUT = ROOT / "outputs" / "mcp_discovery_news_v1"
CUTOFF = "2026-08-12"
WINDOW_START = "2026-07-01"
SYMBOLS = {"Tencent": "00700", "HSBC": "00005", "Alibaba": "09988"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_cmds() -> None:
    ak = PARENT / ".venv-aktools-discovery" / "bin" / "aktools-mcp"
    hk = PARENT / ".venv-hkexnews-discovery" / "bin" / "agentladle-mcp-hkexnews"
    ah = PARENT / ".venv-ahdisclosure-discovery" / "bin" / "ah-disclosure-mcp"
    if ak.exists():
        os.environ.setdefault("AKTOOLS_MCP_COMMAND", str(ak))
    if hk.exists():
        os.environ.setdefault("HKEXNEWS_MCP_COMMAND", str(hk))
    if ah.exists():
        os.environ.setdefault("AH_DISCLOSURE_MCP_COMMAND", str(ah))
    vendor = PARENT / "vendor" / "ah-disclosure-kit" / "src"
    if vendor.exists():
        os.environ.setdefault(
            "PYTHONPATH",
            f"{vendor}{os.pathsep}{os.environ.get('PYTHONPATH', '')}",
        )


def rows_of(result) -> List[Dict[str, Any]]:
    norm = result.normalized
    if isinstance(norm, dict):
        return list(norm.get("rows") or [])
    return []


def dump(path: Path, obj: Any) -> None:
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def smoke_aktools(out: Path) -> Dict[str, Any]:
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    from hk_equity_mcp.real_mcp.adapters.aktools import AktoolsAdapter

    report: Dict[str, Any] = {
        "timestamp": utc_now(),
        "server": "aktools",
        "cutoff": CUTOFF,
        "probes": {},
    }
    ad = AktoolsAdapter(timeout_sec=240)
    ad.open()
    try:
        listed = ad.list_tools()
        report["raw_mcp_tools"] = [t["name"] for t in listed.get("tools") or []]
        report["server_version"] = listed.get("server_version")

        for kw in ["news", "公告", "资讯", "notice", "disclosure", "gsrl"]:
            r = ad.call_mcp_tool("search_interfaces", {"keyword": kw, "limit": 40})
            items = []
            if isinstance(r.normalized, dict):
                items = r.normalized.get("items") or r.normalized.get("results") or []
                if not items and "text" in r.normalized:
                    items = [r.normalized["text"][:500]]
            report["probes"][f"search_{kw}"] = {
                "success": r.success,
                "error": r.error,
                "n": len(items) if isinstance(items, list) else None,
                "sample": (items[:15] if isinstance(items, list) else items),
            }

        r = ad.call_mcp_tool("describe_interface", {"name": "stock_news_em"})
        report["probes"]["describe_stock_news_em"] = {
            "success": r.success,
            "error": r.error,
            "data": r.normalized,
        }

        for name, sym in SYMBOLS.items():
            t0 = time.time()
            r = ad.get_data(
                interface="stock_news_em",
                params={"symbol": sym},
                offset=0,
                max_rows=100,
                operation_id="aktools.stock_news_em",
            )
            rows = rows_of(r)
            before = [
                row
                for row in rows
                if str(row.get("发布时间") or "")[:10] <= CUTOFF
            ]
            report["probes"][f"stock_news_em_{name}"] = {
                "success": r.success,
                "error": r.error,
                "latency_ms": int((time.time() - t0) * 1000),
                "n_rows": len(rows),
                "n_before_cutoff": len(before),
                "columns": list(rows[0].keys()) if rows else [],
                "sample": rows[:2],
                "latest_before_cutoff": before[0] if before else None,
                "note": (
                    "East Money keyword search; pageSize hardcoded ~10; "
                    "no start/end date params — floating recent feed"
                ),
            }

        # Company-name keyword (media aggregator behavior)
        for name, kw in [
            ("Tencent", "腾讯控股"),
            ("HSBC", "汇丰控股"),
            ("Alibaba", "阿里巴巴"),
        ]:
            t0 = time.time()
            r = ad.get_data(
                interface="stock_news_em",
                params={"symbol": kw},
                offset=0,
                max_rows=20,
                operation_id="aktools.stock_news_em",
            )
            rows = rows_of(r)
            before = [
                row
                for row in rows
                if str(row.get("发布时间") or "")[:10] <= CUTOFF
            ]
            report["probes"][f"stock_news_em_name_{name}"] = {
                "success": r.success,
                "error": r.error,
                "latency_ms": int((time.time() - t0) * 1000),
                "n_rows": len(rows),
                "n_before_cutoff": len(before),
                "sample": rows[:1],
            }

        for iface, params in [
            ("stock_gsrl_gsdt_em", {"date": "20260812"}),
            (
                "stock_individual_notice_report",
                {
                    "security": "00700",
                    "symbol": "全部",
                    "begin_date": "20260701",
                    "end_date": "20260812",
                },
            ),
            ("stock_notice_report", {"symbol": "全部", "date": "20260812"}),
            ("stock_news_main_cx", {}),
            ("stock_info_global_em", {}),
            ("news_cctv", {"date": "20260812"}),
        ]:
            t0 = time.time()
            r = ad.get_data(
                interface=iface,
                params=params,
                offset=0,
                max_rows=15,
                operation_id=f"aktools.{iface}",
            )
            rows = rows_of(r)
            report["probes"][iface] = {
                "success": r.success,
                "error": r.error,
                "latency_ms": int((time.time() - t0) * 1000),
                "n_rows": len(rows),
                "columns": list(rows[0].keys()) if rows else [],
                "sample": rows[:2],
            }
    finally:
        ad.close()
    dump(out / "aktools_mcp_news_smoke.json", report)
    return report


def smoke_hkexnews(out: Path) -> Dict[str, Any]:
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    from hk_equity_mcp.real_mcp.adapters.hkexnews import HkexnewsAdapter

    cache = out / "hkexnews_cache"
    cache.mkdir(parents=True, exist_ok=True)
    report: Dict[str, Any] = {
        "timestamp": utc_now(),
        "server": "mcp_hkexnews",
        "cutoff": CUTOFF,
        "window": [WINDOW_START, CUTOFF],
        "probes": {},
    }
    ad = HkexnewsAdapter(
        allow_download=True, cache_dir=cache, timeout_sec=180, persistent=True
    )
    ad.open()
    try:
        listed = ad.list_tools() if hasattr(ad, "list_tools") else None
        if listed:
            report["raw_mcp_tools"] = [
                t.get("name") for t in (listed.get("tools") or [])
            ]

        for name, sym in SYMBOLS.items():
            t0 = time.time()
            r = ad.call_mcp_tool(
                "list_hkexnews_announcements",
                {
                    "stock_code": sym,
                    "start_date": WINDOW_START,
                    "end_date": CUTOFF,
                    "limit": 50,
                },
                operation_id="hkexnews.list_hkexnews_announcements",
            )
            data = r.normalized if isinstance(r.normalized, dict) else {}
            anns = []
            if isinstance(data, dict):
                anns = (
                    data.get("announcements")
                    or (data.get("data") or {}).get("announcements")
                    or []
                )
            # Classify titles roughly
            titles = [str(a.get("title") or "") for a in anns[:20]]
            report["probes"][f"list_{name}"] = {
                "success": r.success,
                "error": r.error,
                "latency_ms": int((time.time() - t0) * 1000),
                "n": len(anns),
                "sample": anns[:3],
                "title_sample": titles[:8],
                "fields": list(anns[0].keys()) if anns else [],
            }

        # One download+parse+keyword chain (HSBC, non-results-friendly window item if any)
        hsbc = report["probes"].get("list_HSBC") or {}
        sample = (hsbc.get("sample") or [None])[0]
        if sample and hsbc.get("success"):
            local_key = sample.get("local_key")
            t0 = time.time()
            dl = ad.call_mcp_tool(
                "download_hkexnews_announcement",
                {
                    "stock_code": sample.get("stock_code") or "00005",
                    "news_id": sample.get("news_id"),
                    "release_date": sample.get("release_date"),
                    "title_keyword": (sample.get("title") or "")[:40],
                    "local_key": local_key,
                },
                operation_id="hkexnews.download_hkexnews_announcement",
            )
            report["probes"]["download_HSBC_sample"] = {
                "success": dl.success,
                "error": dl.error,
                "latency_ms": int((time.time() - t0) * 1000),
                "normalized_keys": list((dl.normalized or {}).keys())
                if isinstance(dl.normalized, dict)
                else type(dl.normalized).__name__,
                "title": sample.get("title"),
            }
            if dl.success:
                lk = None
                if isinstance(dl.normalized, dict):
                    lk = dl.normalized.get("local_key") or local_key
                else:
                    lk = local_key
                pr = ad.call_mcp_tool(
                    "parse_hkexnews_announcement",
                    {"local_key": lk},
                    operation_id="hkexnews.parse_hkexnews_announcement",
                )
                report["probes"]["parse_HSBC_sample"] = {
                    "success": pr.success,
                    "error": pr.error,
                }
                if pr.success:
                    kw = ad.call_mcp_tool(
                        "keyword_search",
                        {"local_key": lk, "keywords": ["HSBC"], "max_hits": 5},
                        operation_id="hkexnews.keyword_search",
                    )
                    hits = []
                    if isinstance(kw.normalized, dict):
                        hits = kw.normalized.get("matches") or kw.normalized.get(
                            "hits"
                        ) or []
                    report["probes"]["keyword_HSBC_sample"] = {
                        "success": kw.success,
                        "error": kw.error,
                        "n_hits": len(hits) if isinstance(hits, list) else None,
                        "hit_sample": hits[:2] if isinstance(hits, list) else hits,
                    }

        # Fail-closed: future window after cutoff should be empty or rejected for gold
        r = ad.call_mcp_tool(
            "list_hkexnews_announcements",
            {
                "stock_code": "00700",
                "start_date": "2026-08-13",
                "end_date": "2026-08-20",
                "limit": 20,
            },
            operation_id="hkexnews.list_hkexnews_announcements",
        )
        data = r.normalized if isinstance(r.normalized, dict) else {}
        anns = []
        if isinstance(data, dict):
            anns = (
                data.get("announcements")
                or (data.get("data") or {}).get("announcements")
                or []
            )
        report["probes"]["list_Tencent_post_cutoff_window"] = {
            "success": r.success,
            "error": r.error,
            "n": len(anns),
            "note": "Post-cutoff window must not be used as gold; record for fail-closed policy",
        }

        # Empty / unlikely: nonsense stock
        r = ad.call_mcp_tool(
            "list_hkexnews_announcements",
            {
                "stock_code": "99999",
                "start_date": WINDOW_START,
                "end_date": CUTOFF,
                "limit": 10,
            },
            operation_id="hkexnews.list_hkexnews_announcements",
        )
        report["probes"]["list_invalid_stock"] = {
            "success": r.success,
            "error": r.error,
            "normalized": r.normalized
            if not isinstance(r.normalized, dict)
            else {
                k: r.normalized.get(k)
                for k in list(r.normalized)[:8]
            },
        }
    finally:
        ad.close()
    dump(out / "hkexnews_mcp_news_smoke.json", report)
    return report


def smoke_ah_disclosure(out: Path) -> Dict[str, Any]:
    """Optional third server: ah-disclosure-kit (stdio)."""
    import asyncio
    import sys

    cmd = os.environ.get("AH_DISCLOSURE_MCP_COMMAND")
    report: Dict[str, Any] = {
        "timestamp": utc_now(),
        "server": "ah_disclosure",
        "cutoff": CUTOFF,
        "probes": {},
        "command": cmd,
    }
    if not cmd or not Path(cmd).exists():
        report["skipped"] = "ah-disclosure-mcp binary not found"
        dump(out / "ah_disclosure_news_smoke.json", report)
        return report

    try:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
    except Exception as exc:  # noqa: BLE001
        report["skipped"] = f"mcp SDK unavailable: {exc}"
        dump(out / "ah_disclosure_news_smoke.json", report)
        return report

    async def _run() -> Dict[str, Any]:
        env = os.environ.copy()
        params = StdioServerParameters(command=cmd, args=[], env=env)
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()
                report["raw_mcp_tools"] = [t.name for t in tools.tools]
                # Prefer filings / annual search for official disclosure discovery
                for tool_name, args in [
                    (
                        "search_filings",
                        {
                            "stock_code": "00700",
                            "start_date": WINDOW_START,
                            "end_date": CUTOFF,
                        },
                    ),
                    (
                        "search_annual_report",
                        {"stock_code": "00700", "year": 2025},
                    ),
                    (
                        "search_filings",
                        {
                            "stock_code": "00005",
                            "start_date": WINDOW_START,
                            "end_date": CUTOFF,
                        },
                    ),
                    (
                        "search_filings",
                        {
                            "stock_code": "09988",
                            "start_date": WINDOW_START,
                            "end_date": CUTOFF,
                        },
                    ),
                ]:
                    t0 = time.time()
                    try:
                        res = await session.call_tool(tool_name, args)
                        content = []
                        for block in res.content or []:
                            if hasattr(block, "text"):
                                content.append(block.text[:1500])
                            elif hasattr(block, "data"):
                                content.append(str(block.data)[:1500])
                        report["probes"][f"{tool_name}_{args.get('stock_code')}"] = {
                            "success": not bool(getattr(res, "isError", False)),
                            "latency_ms": int((time.time() - t0) * 1000),
                            "content_sample": content[:2],
                            "args": args,
                        }
                    except Exception as e:  # noqa: BLE001
                        report["probes"][f"{tool_name}_{args.get('stock_code')}"] = {
                            "success": False,
                            "error": str(e)[:800],
                            "args": args,
                        }
        return report

    try:
        asyncio.run(_run())
    except Exception as exc:  # noqa: BLE001
        report["error"] = str(exc)[:1000]
    dump(out / "ah_disclosure_news_smoke.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument(
        "--servers",
        default="aktools,hkexnews,ah_disclosure",
        help="Comma list: aktools,hkexnews,ah_disclosure",
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    ensure_cmds()

    summary: Dict[str, Any] = {
        "timestamp": utc_now(),
        "cutoff": CUTOFF,
        "timezone": "Asia/Hong_Kong",
        "symbols": SYMBOLS,
        "servers": {},
    }
    wanted = {s.strip() for s in args.servers.split(",") if s.strip()}
    if "aktools" in wanted:
        summary["servers"]["aktools"] = {
            "ok": True,
            "path": str(args.out / "aktools_mcp_news_smoke.json"),
        }
        smoke_aktools(args.out)
    if "hkexnews" in wanted:
        summary["servers"]["hkexnews"] = {
            "ok": True,
            "path": str(args.out / "hkexnews_mcp_news_smoke.json"),
        }
        smoke_hkexnews(args.out)
    if "ah_disclosure" in wanted:
        summary["servers"]["ah_disclosure"] = {
            "ok": True,
            "path": str(args.out / "ah_disclosure_news_smoke.json"),
        }
        smoke_ah_disclosure(args.out)

    dump(args.out / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
