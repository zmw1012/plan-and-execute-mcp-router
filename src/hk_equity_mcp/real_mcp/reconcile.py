"""Offline reconciliation helpers (mock snapshot vs Real MCP payloads)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

ROOT = Path(__file__).resolve().parents[3]
MOCK_PATH = ROOT / "data" / "mock" / "financial.json"


def load_mock() -> Dict[str, Any]:
    return json.loads(MOCK_PATH.read_text(encoding="utf-8"))


def mock_metric(
    ticker: str,
    fiscal_year_label: str,
    *,
    metric: str = "revenue",
    metric_type: str = "reported",
) -> Optional[Dict[str, Any]]:
    company = load_mock()["companies"].get(ticker)
    if not company:
        return None
    for report in company.get("reports") or []:
        if report.get("fiscal_year_label") != fiscal_year_label:
            continue
        for m in report.get("metrics") or []:
            if m.get("metric") == metric and m.get("metric_type") == metric_type:
                return {
                    "value": m.get("value"),
                    "currency": report.get("currency"),
                    "metric": m,
                    "report": {
                        "fiscal_year_label": report.get("fiscal_year_label"),
                        "period_end": report.get("period_end"),
                    },
                }
    return None


def extract_analysis_row(normalized: Any, report_date_prefix: str) -> Optional[Dict[str, Any]]:
    if not isinstance(normalized, dict):
        return None
    rows = normalized.get("rows") or []
    for row in rows:
        rd = str(row.get("REPORT_DATE") or "")
        if rd.startswith(report_date_prefix):
            return row
    return rows[0] if rows else None


def reconcile_tencent_fy2025_revenue(normalized: Any) -> Tuple[bool, str]:
    gold = mock_metric("0700.HK", "FY2025", metric="revenue")
    if gold is None:
        return False, "mock gold missing"
    row = extract_analysis_row(normalized, "2025-12-31")
    if row is None:
        return False, "no MCP analysis row"
    value = row.get("OPERATE_INCOME")
    currency = row.get("CURRENCY")
    if value == gold["value"] and str(currency) in {"人民币", "CNY", "RMB"}:
        return True, "exact match OPERATE_INCOME + CNY currency label"
    return False, f"mismatch value={value} currency={currency} gold={gold['value']}"
