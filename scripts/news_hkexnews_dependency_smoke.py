#!/usr/bin/env python3
"""Free direct-MCP HKEXnews dependency smoke for News discovery (no LLM).

Chain per issuer (fresh adapter session — download budget is 1/session):
  list → download → parse → keyword_search(keywords=[...]) → get_announcement_pages

Uses fixed announcements with release_date <= 2026-08-12. No floating latest.
Corporate-event keywords only (not revenue/EPS extraction).
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT.parent / "hk-equity-mcp-agent"
OUT = ROOT / "outputs" / "mcp_discovery_news_v1" / "hkexnews_dependency_chain"
DOCS = ROOT / "docs" / "news_mcp_discovery_v1"
CUTOFF = "2026-08-12"
WINDOW_START = "2026-07-01"

# Prefer corporate-event disclosures; avoid packaging results numerics as News gold.
PICKS = {
    "Tencent": {
        "stock_code": "00700",
        "title_ Prefer": [
            "Monthly Return of Equity Issuer",
            "Next Day Disclosure Return",
        ],
        "keywords": ["Monthly", "Return", "Issuer"],
        "avoid_title_substrings": ["RESULTS FOR THE"],
    },
    "HSBC": {
        "stock_code": "00005",
        "title_ Prefer": [
            "Transaction in own shares",
            "Next Day Disclosure Return",
        ],
        "keywords": ["Transaction", "shares", "HSBC"],
        "avoid_title_substrings": ["RESULTS"],
    },
    "Alibaba": {
        "stock_code": "09988",
        "title_prefer": [
            "DATE OF BOARD MEETING",
            "NOTICE OF ANNUAL GENERAL MEETING",
            "RECORD DATE",
        ],
        "keywords": ["Board", "Meeting", "Alibaba"],
        "avoid_title_substrings": ["RESULTS"],
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_cmd() -> None:
    hk = PARENT / ".venv-hkexnews-discovery" / "bin" / "agentladle-mcp-hkexnews"
    if hk.exists():
        os.environ["HKEXNEWS_MCP_COMMAND"] = str(hk.resolve())


def dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def normalize(result) -> Any:
    return result.normalized


def announcements_from(result) -> List[Dict[str, Any]]:
    data = normalize(result)
    if not isinstance(data, dict):
        return []
    anns = data.get("announcements")
    if anns is None and isinstance(data.get("data"), dict):
        anns = data["data"].get("announcements")
    return list(anns or [])


def pick_announcement(
    anns: List[Dict[str, Any]], cfg: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    avoid = [s.upper() for s in cfg.get("avoid_title_substrings") or []]
    prefer = cfg.get("title_prefer") or cfg.get("title_ Prefer") or []
    # Fix typo key support
    prefer = cfg.get("title_prefer") or []
    if not prefer and "title_ Prefer" in cfg:
        prefer = cfg["title_ Prefer"]

    eligible = []
    for a in anns:
        title = str(a.get("title") or "")
        rd = str(a.get("release_date") or "")[:10]
        if not rd or rd > CUTOFF:
            continue
        if any(x in title.upper() for x in avoid):
            continue
        eligible.append(a)

    for pref in prefer:
        for a in eligible:
            if pref.lower() in str(a.get("title") or "").lower():
                return a
    return eligible[0] if eligible else None


def edge(
    producer: str,
    pout: str,
    consumer: str,
    cin: str,
    *,
    example: Any = None,
    note: str = "",
) -> Dict[str, Any]:
    d: Dict[str, Any] = {
        "producer_operation": producer,
        "producer_output_field": pout,
        "consumer_operation": consumer,
        "consumer_input_field": cin,
        "live_verified_dependency": True,
    }
    if example is not None:
        d["example_value"] = example
    if note:
        d["note"] = note
    return d


def run_issuer(name: str, cfg: Dict[str, Any], cache: Path) -> Dict[str, Any]:
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    from hk_equity_mcp.real_mcp.adapters.hkexnews import HkexnewsAdapter

    stock = cfg["stock_code"]
    report: Dict[str, Any] = {
        "issuer": name,
        "stock_code": stock,
        "cutoff": CUTOFF,
        "window": [WINDOW_START, CUTOFF],
        "steps": {},
        "dependency_edges": [],
        "success": False,
    }
    ad = HkexnewsAdapter(
        allow_download=True, cache_dir=cache, timeout_sec=240, persistent=True
    )
    ad.open()
    try:
        t0 = time.time()
        listed = ad.call_mcp_tool(
            "list_hkexnews_announcements",
            {
                "stock_code": stock,
                "start_date": WINDOW_START,
                "end_date": CUTOFF,
                "limit": 50,
            },
            operation_id="hkexnews.list_hkexnews_announcements",
        )
        anns = announcements_from(listed)
        report["steps"]["list"] = {
            "success": listed.success,
            "error": listed.error,
            "latency_ms": int((time.time() - t0) * 1000),
            "n": len(anns),
            "raw_status": (normalize(listed) or {}).get("status")
            if isinstance(normalize(listed), dict)
            else None,
        }
        if not listed.success or not anns:
            report["steps"]["list"]["sample_titles"] = [
                str(a.get("title")) for a in anns[:5]
            ]
            return report

        chosen = pick_announcement(anns, cfg)
        if not chosen:
            report["steps"]["pick"] = {
                "success": False,
                "error": "no eligible corporate-event announcement <= cutoff",
                "titles": [str(a.get("title")) for a in anns[:10]],
            }
            return report

        report["chosen"] = {
            "title": chosen.get("title"),
            "release_date": chosen.get("release_date"),
            "release_datetime": chosen.get("release_datetime"),
            "news_id": chosen.get("news_id"),
            "local_key": chosen.get("local_key"),
            "file_link": chosen.get("file_link"),
            "category_label": chosen.get("category_label"),
        }
        if str(chosen.get("release_date") or "")[:10] > CUTOFF:
            report["steps"]["pick"] = {
                "success": False,
                "error": "chosen release_date after cutoff",
                "chosen": report["chosen"],
            }
            return report

        # download
        t0 = time.time()
        title = str(chosen.get("title") or "")
        # Short title_keyword is required; long truncations fail matching.
        short_kw = (cfg.get("download_title_keyword") or "").strip()
        if not short_kw:
            short_kw = " ".join(title.split()[:2])[:24]
        dl_args = {
            "stock_code": chosen.get("stock_code") or stock,
            "news_id": chosen.get("news_id"),
            "release_date": chosen.get("release_date"),
            "title_keyword": short_kw,
            "local_key": chosen.get("local_key"),
        }
        dl = ad.call_mcp_tool(
            "download_hkexnews_announcement",
            dl_args,
            operation_id="hkexnews.download_hkexnews_announcement",
        )
        dl_norm = normalize(dl) if isinstance(normalize(dl), dict) else {}
        local_key = (
            (dl_norm or {}).get("local_key")
            or ((dl_norm or {}).get("data") or {}).get("local_key")
            or chosen.get("local_key")
        )
        report["steps"]["download"] = {
            "success": dl.success,
            "error": dl.error,
            "latency_ms": int((time.time() - t0) * 1000),
            "args": dl_args,
            "local_key": local_key,
            "normalized_keys": list(dl_norm.keys()) if isinstance(dl_norm, dict) else [],
        }
        report["dependency_edges"].extend(
            [
                edge(
                    "hkexnews.list_hkexnews_announcements",
                    "local_key",
                    "hkexnews.download_hkexnews_announcement",
                    "local_key",
                    example=chosen.get("local_key"),
                ),
                edge(
                    "hkexnews.list_hkexnews_announcements",
                    "news_id",
                    "hkexnews.download_hkexnews_announcement",
                    "news_id",
                    example=chosen.get("news_id"),
                ),
                edge(
                    "hkexnews.list_hkexnews_announcements",
                    "release_date",
                    "hkexnews.download_hkexnews_announcement",
                    "release_date",
                    example=chosen.get("release_date"),
                ),
                edge(
                    "hkexnews.list_hkexnews_announcements",
                    "title",
                    "hkexnews.download_hkexnews_announcement",
                    "title_keyword",
                    example=short_kw,
                    note="list.title truncated into download.title_keyword",
                ),
            ]
        )
        if not dl.success or not local_key:
            return report

        # parse
        t0 = time.time()
        pr = ad.call_mcp_tool(
            "parse_hkexnews_announcement",
            {"local_key": local_key},
            operation_id="hkexnews.parse_hkexnews_announcement",
        )
        report["steps"]["parse"] = {
            "success": pr.success,
            "error": pr.error,
            "latency_ms": int((time.time() - t0) * 1000),
            "local_key": local_key,
        }
        report["dependency_edges"].append(
            edge(
                "hkexnews.download_hkexnews_announcement",
                "local_key",
                "hkexnews.parse_hkexnews_announcement",
                "local_key",
                example=local_key,
            )
        )
        if not pr.success:
            return report

        # keyword_search — MUST use keywords:list
        keywords = list(cfg["keywords"])
        t0 = time.time()
        kw = ad.call_mcp_tool(
            "keyword_search",
            {
                "local_key": local_key,
                "keywords": keywords,
                "max_results": 5,
            },
            operation_id="hkexnews.keyword_search",
        )
        kw_norm = normalize(kw) if isinstance(normalize(kw), dict) else {}
        data = kw_norm.get("data") if isinstance(kw_norm, dict) else {}
        matches = []
        if isinstance(data, dict):
            matches = list(data.get("matches") or [])
        elif isinstance(kw_norm, dict):
            matches = list(kw_norm.get("matches") or [])
        report["steps"]["keyword_search"] = {
            "success": kw.success,
            "error": kw.error,
            "latency_ms": int((time.time() - t0) * 1000),
            "args": {"local_key": local_key, "keywords": keywords, "max_results": 5},
            "n_matches": len(matches),
            "matches_sample": matches[:3],
            "raw_status": kw_norm.get("status") if isinstance(kw_norm, dict) else None,
        }
        report["dependency_edges"].append(
            edge(
                "hkexnews.parse_hkexnews_announcement",
                "local_key",
                "hkexnews.keyword_search",
                "local_key",
                example=local_key,
                note="keywords must be a list; singular keyword fails schema",
            )
        )
        if not kw.success or not matches:
            return report

        page_number = matches[0].get("page_number")
        report["dependency_edges"].append(
            edge(
                "hkexnews.keyword_search",
                "page_number",
                "hkexnews.get_announcement_pages",
                "start_page",
                example=page_number,
            )
        )

        # pages
        t0 = time.time()
        pages = ad.call_mcp_tool(
            "get_announcement_pages",
            {
                "local_key": local_key,
                "start_page": int(page_number),
                "page_count": 1,
            },
            operation_id="hkexnews.get_announcement_pages",
        )
        pg_norm = normalize(pages) if isinstance(normalize(pages), dict) else {}
        # slim page text for durable evidence
        slim = pg_norm
        if isinstance(pg_norm, dict):
            slim = {
                k: pg_norm.get(k)
                for k in list(pg_norm.keys())[:12]
            }
            data_pg = pg_norm.get("data")
            if isinstance(data_pg, dict):
                pages_list = data_pg.get("pages") or data_pg.get("page_texts") or []
                slim_pages = []
                for p in (pages_list or [])[:2]:
                    if isinstance(p, dict):
                        slim_pages.append(
                            {
                                kk: (
                                    (str(vv)[:400] + "…")
                                    if isinstance(vv, str) and len(vv) > 400
                                    else vv
                                )
                                for kk, vv in p.items()
                            }
                        )
                    else:
                        slim_pages.append(str(p)[:400])
                slim["data_slim"] = {
                    "keys": list(data_pg.keys()),
                    "pages": slim_pages,
                }
        report["steps"]["get_announcement_pages"] = {
            "success": pages.success,
            "error": pages.error,
            "latency_ms": int((time.time() - t0) * 1000),
            "args": {
                "local_key": local_key,
                "start_page": int(page_number),
                "page_count": 1,
            },
            "normalized_slim": slim,
        }
        report["dependency_edges"].append(
            edge(
                "hkexnews.parse_hkexnews_announcement",
                "local_key",
                "hkexnews.get_announcement_pages",
                "local_key",
                example=local_key,
            )
        )
        report["success"] = bool(
            listed.success
            and dl.success
            and pr.success
            and kw.success
            and pages.success
            and matches
        )
        return report
    finally:
        ad.close()


def main() -> int:
    ensure_cmd()
    OUT.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)
    cache = OUT / "cache"
    cache.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Any] = {
        "timestamp": utc_now(),
        "cutoff": CUTOFF,
        "timezone": "Asia/Hong_Kong",
        "chain": [
            "list_hkexnews_announcements",
            "download_hkexnews_announcement",
            "parse_hkexnews_announcement",
            "keyword_search(keywords=list)",
            "get_announcement_pages(start_page=page_number)",
        ],
        "issuers": {},
        "all_success": False,
        "observed_dependency_edges": [],
    }

    # Fix PICKS keys - clean the typo
    picks = {
        "Tencent": {
            "stock_code": "00700",
            "title_prefer": [
                "Monthly Return of Equity Issuer",
                "Next Day Disclosure Return",
            ],
            "download_title_keyword": "Monthly Return",
            "keywords": ["Monthly", "Return", "Issuer"],
            "avoid_title_substrings": ["RESULTS FOR THE"],
        },
        "HSBC": {
            "stock_code": "00005",
            "title_prefer": [
                "Transaction in own shares",
                "Next Day Disclosure Return",
            ],
            "download_title_keyword": "Transaction in own shares",
            "keywords": ["Transaction", "shares", "HSBC"],
            "avoid_title_substrings": ["RESULTS"],
        },
        "Alibaba": {
            "stock_code": "09988",
            "title_prefer": [
                "DATE OF BOARD MEETING",
                "NOTICE OF ANNUAL GENERAL MEETING",
                "RECORD DATE",
            ],
            "download_title_keyword": "DATE OF BOARD",
            "keywords": ["Board", "Meeting", "Director"],
            "avoid_title_substrings": ["RESULTS"],
        },
    }

    for name, cfg in picks.items():
        print(f"=== {name} ===")
        # fresh cache subdir optional; adapter download budget is per session
        rep = run_issuer(name, cfg, cache)
        dump(OUT / f"chain_{name.lower()}.json", rep)
        dump(DOCS / f"hkexnews_dependency_chain_{name.lower()}.json", rep)
        summary["issuers"][name] = {
            "success": rep.get("success"),
            "chosen_title": (rep.get("chosen") or {}).get("title"),
            "release_date": (rep.get("chosen") or {}).get("release_date"),
            "steps": {
                k: {"success": v.get("success"), "error": v.get("error")}
                for k, v in (rep.get("steps") or {}).items()
                if isinstance(v, dict)
            },
            "n_edges": len(rep.get("dependency_edges") or []),
            "evidence": str(
                (DOCS / f"hkexnews_dependency_chain_{name.lower()}.json").relative_to(
                    ROOT
                )
            ),
        }
        if rep.get("success"):
            summary["observed_dependency_edges"].extend(rep.get("dependency_edges") or [])
        print(
            name,
            "success=",
            rep.get("success"),
            "title=",
            (rep.get("chosen") or {}).get("title"),
            "kw=",
            (rep.get("steps") or {}).get("keyword_search", {}).get("success"),
            "pages=",
            (rep.get("steps") or {}).get("get_announcement_pages", {}).get("success"),
            "err_kw=",
            (rep.get("steps") or {}).get("keyword_search", {}).get("error"),
        )

    summary["all_success"] = all(
        v.get("success") for v in summary["issuers"].values()
    )
    # dedupe edges by quadruple
    seen = set()
    uniq = []
    for e in summary["observed_dependency_edges"]:
        key = (
            e.get("producer_operation"),
            e.get("producer_output_field"),
            e.get("consumer_operation"),
            e.get("consumer_input_field"),
        )
        if key in seen:
            continue
        seen.add(key)
        uniq.append(e)
    summary["observed_dependency_edges"] = uniq

    dump(OUT / "summary.json", summary)
    dump(DOCS / "hkexnews_dependency_chain_summary.json", summary)
    dump(DOCS / "observed_dependency_edges.json", uniq)
    print(json.dumps(summary["issuers"], indent=2, ensure_ascii=False))
    print("all_success=", summary["all_success"])
    return 0 if summary["all_success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
