"""Fixed mock Hong Kong equity data used by market/news servers.

Financial multi-period snapshots live in data/mock/financial.json and are
loaded by hk_equity_mcp.mcp_servers.financial.data_store.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import json
import shutil

# Default as-of date for market/news mock snapshots
AS_OF_DATE = "2024-12-31"

# Daily close prices (mock), newest last
MOCK_PRICES: Dict[str, Dict[str, Any]] = {
    "0700.HK": {
        "currency": "HKD",
        "series": [
            {"date": "2024-12-27", "close": 410.0},
            {"date": "2024-12-30", "close": 415.2},
            {"date": "2024-12-31", "close": 418.0},
        ],
        "evidence": ["Mock HKEX daily close snapshot 2024-12-31"],
    },
    "0005.HK": {
        "currency": "HKD",
        "series": [
            {"date": "2024-12-27", "close": 72.5},
            {"date": "2024-12-30", "close": 73.1},
            {"date": "2024-12-31", "close": 73.8},
        ],
        "evidence": ["Mock HKEX daily close snapshot 2024-12-31"],
    },
    "9988.HK": {
        "currency": "HKD",
        "series": [
            {"date": "2024-12-27", "close": 82.0},
            {"date": "2024-12-30", "close": 83.4},
            {"date": "2024-12-31", "close": 84.1},
        ],
        "evidence": ["Mock HKEX daily close snapshot 2024-12-31"],
    },
}

MOCK_NEWS: Dict[str, Dict[str, Any]] = {
    "0700.HK": {
        "announcements": [
            {
                "date": "2024-12-20",
                "headline": "Tencent announces strategic investment in AI infrastructure",
                "summary": "Company plans additional cloud GPU capacity for 2025.",
            },
            {
                "date": "2024-11-15",
                "headline": "Interim dividend declaration",
                "summary": "Board declared an interim dividend for FY2024.",
            },
        ],
        "news_summary": (
            "Recent coverage focused on AI infrastructure investment and "
            "steady gaming / advertising momentum."
        ),
        "evidence": [
            "Mock HKEX announcement 2024-12-20",
            "Mock news wire digest 2024-12",
        ],
    },
    "0005.HK": {
        "announcements": [
            {
                "date": "2024-12-10",
                "headline": "HSBC updates capital distribution plan",
                "summary": "Bank reiterated progressive dividend and buyback capacity.",
            }
        ],
        "news_summary": (
            "Market discussion centered on capital returns and Asia lending outlook."
        ),
        "evidence": [
            "Mock HKEX announcement 2024-12-10",
            "Mock news wire digest 2024-12",
        ],
    },
    "9988.HK": {
        "announcements": [
            {
                "date": "2024-12-05",
                "headline": "Alibaba expands international e-commerce logistics",
                "summary": "New warehouse capacity planned in Southeast Asia.",
            }
        ],
        "news_summary": (
            "Headlines highlighted logistics expansion and cloud growth commentary."
        ),
        "evidence": [
            "Mock HKEX announcement 2024-12-05",
            "Mock news wire digest 2024-12",
        ],
    },
}


def normalize_ticker(ticker: str) -> str:
    """Normalize common HK ticker inputs to XXXX.HK form."""
    t = ticker.strip().upper()
    if t.endswith(".HK"):
        return t
    if t.isdigit():
        return f"{int(t):04d}.HK"
    return t


def dump_mock_json(target_dir: Path) -> None:
    """Write mock tables to data/mock for inspection."""
    target_dir.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[3]
    financial_src = root / "data" / "mock" / "financial.json"
    if financial_src.exists() and financial_src.resolve() != (target_dir / "financial.json").resolve():
        shutil.copyfile(financial_src, target_dir / "financial.json")
    (target_dir / "market.json").write_text(
        json.dumps(MOCK_PRICES, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (target_dir / "news.json").write_text(
        json.dumps(MOCK_NEWS, indent=2, ensure_ascii=False), encoding="utf-8"
    )
