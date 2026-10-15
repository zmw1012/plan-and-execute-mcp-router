"""Fixed mock Hong Kong equity data used by all mock servers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import json

# Default as-of date for the mock snapshot
AS_OF_DATE = "2024-12-31"

# Small fixed universe: Tencent, HSBC, Alibaba
MOCK_FINANCIALS: Dict[str, Dict[str, Any]] = {
    "0700.HK": {
        "name": "Tencent Holdings",
        "currency": "HKD",
        "metrics": {
            "revenue": {"value": 609015_000_000, "unit": "HKD", "period": "FY2024"},
            "net_profit": {"value": 194073_000_000, "unit": "HKD", "period": "FY2024"},
            "eps": {"value": 20.45, "unit": "HKD", "period": "FY2024"},
        },
        "filing_summary": (
            "FY2024 results showed continued growth in value-added services and "
            "online advertising, with improved operating margins."
        ),
        "evidence": [
            "Mock Annual Report FY2024, p.12 — Revenue HKD 609.0bn",
            "Mock Annual Report FY2024, p.15 — Net profit HKD 194.1bn",
        ],
    },
    "0005.HK": {
        "name": "HSBC Holdings",
        "currency": "USD",
        "metrics": {
            "revenue": {"value": 65_900_000_000, "unit": "USD", "period": "FY2024"},
            "net_profit": {"value": 22_900_000_000, "unit": "USD", "period": "FY2024"},
            "eps": {"value": 1.15, "unit": "USD", "period": "FY2024"},
        },
        "filing_summary": (
            "FY2024 banking results reflected higher net interest income and "
            "disciplined cost control across major regions."
        ),
        "evidence": [
            "Mock Annual Report FY2024, p.8 — Revenue USD 65.9bn",
            "Mock Annual Report FY2024, p.10 — Net profit USD 22.9bn",
        ],
    },
    "9988.HK": {
        "name": "Alibaba Group",
        "currency": "CNY",
        "metrics": {
            "revenue": {"value": 941_168_000_000, "unit": "CNY", "period": "FY2024"},
            "net_profit": {"value": 71_332_000_000, "unit": "CNY", "period": "FY2024"},
            "eps": {"value": 3.78, "unit": "CNY", "period": "FY2024"},
        },
        "filing_summary": (
            "FY2024 highlighted cloud and international commerce growth while "
            "domestic commerce remained the largest revenue contributor."
        ),
        "evidence": [
            "Mock Annual Report FY2024, p.20 — Revenue CNY 941.2bn",
            "Mock Annual Report FY2024, p.24 — Net profit CNY 71.3bn",
        ],
    },
}

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
    (target_dir / "financial.json").write_text(
        json.dumps(MOCK_FINANCIALS, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (target_dir / "market.json").write_text(
        json.dumps(MOCK_PRICES, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (target_dir / "news.json").write_text(
        json.dumps(MOCK_NEWS, indent=2, ensure_ascii=False), encoding="utf-8"
    )
