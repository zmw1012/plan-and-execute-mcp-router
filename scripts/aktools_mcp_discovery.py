#!/usr/bin/env python3
"""Real AKTools MCP discovery via stdio (no LLM; no direct akshare import as MCP).

Uses the official aktools-mcp entrypoint and MCP client protocol:
initialize → tools/list → tools/call(search_interfaces|describe_interface|get_data).
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "mcp_discovery"
OUT.mkdir(parents=True, exist_ok=True)

AKTOOLS_MCP = str(
    (ROOT / ".venv-aktools-discovery" / "bin" / "aktools-mcp").resolve()
)

SEARCH_QUERIES = [
    "Hong Kong financial report",
    "港股 财务报表",
    "港股 财务指标",
    "income statement",
    "balance sheet",
    "cash flow",
    "EPS",
    "company profile",
    "dividend",
    "valuation",
    "港股 财务",
    "香港 财务",
]

FOCUS_INTERFACES = [
    "stock_financial_hk_report_em",
    "stock_financial_hk_analysis_indicator_em",
    "stock_hk_financial_indicator_em",
    "stock_hk_security_profile_em",
    "stock_hk_company_profile_em",
    "stock_hk_dividend_payout_em",
    "stock_hk_growth_comparison_em",
    "stock_hk_valuation_comparison_em",
]

# Minimal get_data probes (max_rows small). Symbols: Tencent 00700, HSBC 00005, Alibaba 09988.
GET_DATA_PROBES = [
    {
        "label": "tencent_report",
        "interface": "stock_financial_hk_report_em",
        "params": {"stock": "00700", "symbol": "财务报表", "indicator": "年度"},
        "max_rows": 20,
        "offset": -20,
    },
    {
        "label": "tencent_report_interim",
        "interface": "stock_financial_hk_report_em",
        "params": {"stock": "00700", "symbol": "财务报表", "indicator": "报告期"},
        "max_rows": 20,
        "offset": -20,
    },
    {
        "label": "tencent_analysis",
        "interface": "stock_financial_hk_analysis_indicator_em",
        "params": {"symbol": "00700", "indicator": "年度"},
        "max_rows": 20,
        "offset": -20,
    },
    {
        "label": "hsbc_report",
        "interface": "stock_financial_hk_report_em",
        "params": {"stock": "00005", "symbol": "财务报表", "indicator": "年度"},
        "max_rows": 20,
        "offset": -20,
    },
    {
        "label": "hsbc_analysis",
        "interface": "stock_financial_hk_analysis_indicator_em",
        "params": {"symbol": "00005", "indicator": "年度"},
        "max_rows": 20,
        "offset": -20,
    },
    {
        "label": "alibaba_report",
        "interface": "stock_financial_hk_report_em",
        "params": {"stock": "09988", "symbol": "财务报表", "indicator": "年度"},
        "max_rows": 20,
        "offset": -20,
    },
    {
        "label": "alibaba_analysis",
        "interface": "stock_financial_hk_analysis_indicator_em",
        "params": {"symbol": "09988", "indicator": "年度"},
        "max_rows": 20,
        "offset": -20,
    },
    {
        "label": "tencent_profile",
        "interface": "stock_hk_company_profile_em",
        "params": {"symbol": "00700"},
        "max_rows": 5,
    },
    {
        "label": "tencent_security_profile",
        "interface": "stock_hk_security_profile_em",
        "params": {"symbol": "00700"},
        "max_rows": 5,
    },
    {
        "label": "tencent_dividend",
        "interface": "stock_hk_dividend_payout_em",
        "params": {"symbol": "00700"},
        "max_rows": 10,
        "offset": -10,
    },
]


def _tool_result_to_obj(result: Any) -> Any:
    """Normalize CallToolResult content into JSON-serializable Python objects."""
    payload: Dict[str, Any] = {
        "isError": bool(
            getattr(result, "is_error", getattr(result, "isError", False))
        ),
        "content": [],
        "structuredContent": getattr(
            result,
            "structured_content",
            getattr(result, "structuredContent", None),
        ),
    }
    for block in getattr(result, "content", []) or []:
        btype = getattr(block, "type", None)
        if btype == "text":
            text = getattr(block, "text", "")
            try:
                payload["content"].append({"type": "text", "json": json.loads(text)})
            except Exception:
                payload["content"].append({"type": "text", "text": text[:5000]})
        else:
            payload["content"].append({"type": str(btype), "repr": str(block)[:2000]})
    return payload


async def call_tool(session: ClientSession, name: str, arguments: Dict[str, Any]) -> Any:
    result = await session.call_tool(name, arguments=arguments)
    return _tool_result_to_obj(result)


async def main() -> int:
    if not Path(AKTOOLS_MCP).exists():
        print(f"Missing aktools-mcp at {AKTOOLS_MCP}", file=sys.stderr)
        return 2

    server = StdioServerParameters(
        command=AKTOOLS_MCP,
        args=[],
        env=None,
    )
    report: Dict[str, Any] = {
        "transport": "stdio",
        "command": AKTOOLS_MCP,
        "registry_name": "io.github.akfamily/aktools",
    }

    async with stdio_client(server) as (read, write):
        async with ClientSession(read, write) as session:
            init = await session.initialize()
            server_info = getattr(init, "server_info", None)
            report["initialize"] = {
                "protocolVersion": getattr(init, "protocol_version", None),
                "serverInfo": {
                    "name": getattr(server_info, "name", None),
                    "version": getattr(server_info, "version", None),
                    "website_url": getattr(server_info, "website_url", None),
                },
                "capabilities": str(getattr(init, "capabilities", None)),
            }

            tools = await session.list_tools()
            tool_rows = []
            for t in tools.tools:
                tool_rows.append(
                    {
                        "name": t.name,
                        "description": (t.description or "")[:1000],
                        "inputSchema": t.input_schema,
                        "annotations": {
                            "read_only_hint": getattr(
                                getattr(t, "annotations", None), "read_only_hint", None
                            ),
                            "destructive_hint": getattr(
                                getattr(t, "annotations", None), "destructive_hint", None
                            ),
                            "idempotent_hint": getattr(
                                getattr(t, "annotations", None), "idempotent_hint", None
                            ),
                        },
                    }
                )
            report["tools_list"] = tool_rows
            (OUT / "aktools_tools_list.json").write_text(
                json.dumps(tool_rows, ensure_ascii=False, indent=2), encoding="utf-8"
            )

            # searches
            search_results: Dict[str, Any] = {}
            for q in SEARCH_QUERIES:
                print(f"search_interfaces: {q}", flush=True)
                search_results[q] = await call_tool(
                    session, "search_interfaces", {"keyword": q}
                )
            (OUT / "aktools_search_results.json").write_text(
                json.dumps(search_results, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            report["search_queries"] = list(SEARCH_QUERIES)

            # describe focus interfaces (+ any HK financial names found in search)
            names_from_search: List[str] = []
            for payload in search_results.values():
                for block in payload.get("content") or []:
                    data = block.get("json")
                    if isinstance(data, list):
                        for row in data:
                            if isinstance(row, dict) and row.get("name"):
                                names_from_search.append(str(row["name"]))
                    elif isinstance(data, dict):
                        items = data.get("interfaces") or data.get("results") or data.get("items")
                        if isinstance(items, list):
                            for row in items:
                                if isinstance(row, dict) and (
                                    row.get("name") or row.get("interface")
                                ):
                                    names_from_search.append(
                                        str(row.get("name") or row.get("interface"))
                                    )

            describe_targets = list(
                dict.fromkeys(FOCUS_INTERFACES + [n for n in names_from_search if "hk" in n.lower() or "港" in n])
            )
            describes: Dict[str, Any] = {}
            for name in describe_targets:
                print(f"describe_interface: {name}", flush=True)
                describes[name] = await call_tool(
                    session, "describe_interface", {"name": name}
                )
            (OUT / "aktools_describe_results.json").write_text(
                json.dumps(describes, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            report["describe_targets"] = describe_targets

            # get_data probes — only for interfaces that describe succeeded
            get_data_results: Dict[str, Any] = {}
            for probe in GET_DATA_PROBES:
                iface = probe["interface"]
                print(f"get_data: {probe['label']} -> {iface}", flush=True)
                args: Dict[str, Any] = {
                    "name": iface,
                    "params": probe["params"],
                    "max_rows": probe.get("max_rows", 20),
                }
                if "offset" in probe:
                    args["offset"] = probe["offset"]
                # Some MCP schemas use interface= instead of name=
                try:
                    get_data_results[probe["label"]] = await call_tool(
                        session, "get_data", args
                    )
                except Exception as exc:  # noqa: BLE001
                    # Retry alternate argument shape
                    alt = {
                        "interface": iface,
                        "params": probe["params"],
                        "max_rows": probe.get("max_rows", 20),
                    }
                    if "offset" in probe:
                        alt["offset"] = probe["offset"]
                    try:
                        get_data_results[probe["label"]] = await call_tool(
                            session, "get_data", alt
                        )
                    except Exception as exc2:  # noqa: BLE001
                        get_data_results[probe["label"]] = {
                            "isError": True,
                            "error": f"{type(exc).__name__}/{type(exc2).__name__}: {exc2}",
                        }
            (OUT / "aktools_get_data_probes.json").write_text(
                json.dumps(get_data_results, ensure_ascii=False, indent=2), encoding="utf-8"
            )

    (OUT / "aktools_discovery_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report["initialize"], ensure_ascii=False, indent=2))
    print("tools:", [t["name"] for t in report["tools_list"]])
    print(f"wrote outputs under {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
