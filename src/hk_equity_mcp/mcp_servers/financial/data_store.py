"""Load and query multi-period financial report snapshots."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from hk_equity_mcp.common.mock_data import normalize_ticker

_DEFAULT_JSON = (
    Path(__file__).resolve().parents[4] / "data" / "mock" / "financial.json"
)


def default_financial_json_path() -> Path:
    return _DEFAULT_JSON


class FinancialDataStore:
    """In-memory index over data/mock/financial.json."""

    def __init__(self, json_path: Optional[Path] = None):
        self.json_path = Path(json_path) if json_path else default_financial_json_path()
        self._raw = self._load()
        self.companies: Dict[str, Dict[str, Any]] = self._raw.get("companies", {})
        meta = self._raw.get("meta", {})
        self.snapshot_created_date = meta.get("snapshot_created_date", "")
        self.latest_source_publication_date = meta.get(
            "latest_source_publication_date", ""
        )
        # Deprecated alias — do not treat period_end as snapshot creation date.
        self.as_of_snapshot = self.latest_source_publication_date
        self._validate_meta()

    def _validate_meta(self) -> None:
        pubs = [
            r.get("publication_date")
            for c in self.companies.values()
            for r in c.get("reports") or []
            if r.get("publication_date")
        ]
        if pubs and self.latest_source_publication_date:
            expected = max(pubs)
            if self.latest_source_publication_date != expected:
                raise ValueError(
                    "meta.latest_source_publication_date "
                    f"({self.latest_source_publication_date}) != max publication_date "
                    f"({expected})"
                )

    def _load(self) -> Dict[str, Any]:
        with self.json_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def get_company(self, ticker: str) -> Optional[Dict[str, Any]]:
        return self.companies.get(normalize_ticker(ticker))

    def list_reports(self, ticker: str) -> List[Dict[str, Any]]:
        company = self.get_company(ticker)
        if not company:
            return []
        return list(company.get("reports") or [])

    def latest_report(
        self, ticker: str, report_type: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        reports = self.list_reports(ticker)
        if report_type:
            reports = [r for r in reports if r.get("report_type") == report_type]
        if not reports:
            return None
        return max(reports, key=lambda r: r.get("period_end", ""))

    def resolve_report(
        self,
        ticker: str,
        period: Optional[str] = None,
        report_type: Optional[str] = None,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """Resolve a report.

        Returns (report, error_message).
        Default when period is omitted: latest annual (explicit policy).
        """
        company = self.get_company(ticker)
        if company is None:
            return None, f"Unknown ticker: {normalize_ticker(ticker)}"

        reports = list(company.get("reports") or [])
        if not reports:
            return None, f"No reports available for {normalize_ticker(ticker)}"

        period_n = period.strip() if period else None
        report_type_n = report_type.strip().lower() if report_type else None
        if report_type_n and report_type_n not in {"annual", "interim"}:
            return None, f"Unknown report_type: {report_type}. Use annual or interim."

        # Ambiguous calendar-year style query without report_type.
        if period_n and period_n.isdigit() and len(period_n) == 4 and not report_type_n:
            return None, (
                f"Ambiguous period '{period_n}': specify report_type "
                f"('annual' or 'interim') and/or a fiscal_year_label / period_end "
                f"(e.g. FY2025, 1H2026, 2025-12-31)."
            )

        candidates = reports
        if report_type_n:
            candidates = [r for r in candidates if r.get("report_type") == report_type_n]
            if not candidates:
                return None, (
                    f"Report type mismatch: no '{report_type_n}' report for "
                    f"{normalize_ticker(ticker)}"
                )

        if period_n:
            matched = [
                r
                for r in candidates
                if period_n
                in {
                    r.get("fiscal_year_label"),
                    r.get("period_end"),
                    r.get("period_start"),
                }
            ]
            if not matched:
                available = sorted(
                    {
                        f"{r.get('fiscal_year_label')}({r.get('period_end')}/{r.get('report_type')})"
                        for r in reports
                    }
                )
                return None, (
                    f"Period mismatch: requested {period_n}"
                    + (f" with report_type={report_type_n}" if report_type_n else "")
                    + f". Available: {', '.join(available)}"
                )
            if len(matched) > 1 and not report_type_n:
                labels = [
                    f"{r.get('fiscal_year_label')}/{r.get('report_type')}" for r in matched
                ]
                return None, (
                    f"Ambiguous period '{period_n}' matches multiple reports: {labels}. "
                    f"Specify report_type."
                )
            return matched[0], None

        # No period: default to latest annual unless interim explicitly requested.
        target_type = report_type_n or "annual"
        selected = self.latest_report(ticker, report_type=target_type)
        if selected is None:
            return None, (
                f"No {target_type} report available for {normalize_ticker(ticker)}"
            )
        return selected, None

    def find_metric(
        self,
        report: Dict[str, Any],
        metric: str,
        metric_type: str = "reported",
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        metric_key = metric.strip().lower()
        metric_type_n = (metric_type or "reported").strip().lower()
        metrics = list(report.get("metrics") or [])
        names = sorted({m.get("metric") for m in metrics if m.get("metric")})
        typed = [
            m
            for m in metrics
            if m.get("metric") == metric_key and m.get("metric_type") == metric_type_n
        ]
        if typed:
            return typed[0], None
        same_name = [m for m in metrics if m.get("metric") == metric_key]
        if same_name:
            available_types = sorted({m.get("metric_type") for m in same_name})
            return None, (
                f"Metric type mismatch for '{metric_key}': requested '{metric_type_n}', "
                f"available types: {available_types}"
            )
        return None, f"Unknown metric: {metric_key}. Available: {names}"


# Module-level singleton for servers/tests
_STORE: Optional[FinancialDataStore] = None


def get_store() -> FinancialDataStore:
    global _STORE
    if _STORE is None:
        _STORE = FinancialDataStore()
    return _STORE


def reload_store(json_path: Optional[Path] = None) -> FinancialDataStore:
    global _STORE
    _STORE = FinancialDataStore(json_path=json_path)
    return _STORE
