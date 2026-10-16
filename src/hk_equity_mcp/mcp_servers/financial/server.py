"""Mock Financial MCP Server with annual / interim period support."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from hk_equity_mcp.common.mock_data import normalize_ticker
from hk_equity_mcp.common.schema import make_response
from hk_equity_mcp.mcp_servers.financial.data_store import get_store

SOURCE = "mock_financial"


class FinancialServer:
    """Exposes get_financial_metric and get_filing_summary."""

    def __init__(self) -> None:
        self.store = get_store()

    def _as_of(self, report: Optional[Dict[str, Any]] = None) -> str:
        if report and report.get("period_end"):
            return str(report["period_end"])
        return self.store.as_of_snapshot or ""

    def _report_meta(self, company: Dict[str, Any], report: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "company_name": company.get("company_name"),
            "report_type": report.get("report_type"),
            "fiscal_year_label": report.get("fiscal_year_label"),
            "period_start": report.get("period_start"),
            "period_end": report.get("period_end"),
            "publication_date": report.get("publication_date"),
            "audit_status": report.get("audit_status"),
            "currency": report.get("currency"),
            "unit": report.get("unit"),
            "accounting_standard": report.get("accounting_standard"),
            "source_document": report.get("source_document"),
            "source_url": report.get("source_url"),
        }

    def get_financial_metric(
        self,
        ticker: str,
        metric: str,
        period: Optional[str] = None,
        report_type: Optional[str] = None,
        metric_type: str = "reported",
    ) -> Dict[str, Any]:
        ticker_n = normalize_ticker(ticker)
        company = self.store.get_company(ticker_n)
        if company is None:
            return make_response(
                success=False,
                source=SOURCE,
                ticker=ticker_n,
                as_of_date=self._as_of(),
                error=f"Unknown ticker: {ticker_n}",
            )

        report, err = self.store.resolve_report(
            ticker_n, period=period, report_type=report_type
        )
        if err or report is None:
            return make_response(
                success=False,
                source=SOURCE,
                ticker=ticker_n,
                as_of_date=self._as_of(),
                error=err or "Unable to resolve report",
            )

        metric_row, metric_err = self.store.find_metric(
            report, metric=metric, metric_type=metric_type
        )
        if metric_err or metric_row is None:
            return make_response(
                success=False,
                source=SOURCE,
                ticker=ticker_n,
                as_of_date=self._as_of(report),
                data={"selected_report": self._report_meta(company, report)},
                error=metric_err or "Metric not found",
            )

        evidence = [metric_row.get("evidence_text") or ""]
        evidence = [e for e in evidence if e]
        if report.get("source_url"):
            evidence.append(f"source_url: {report['source_url']}")

        return make_response(
            success=True,
            source=SOURCE,
            ticker=ticker_n,
            as_of_date=self._as_of(report),
            data={
                **self._report_meta(company, report),
                "metric": metric_row.get("metric"),
                "metric_type": metric_row.get("metric_type"),
                "value": metric_row.get("value"),
                "source_page": metric_row.get("source_page"),
                "verification_status": metric_row.get("verification_status"),
            },
            evidence=evidence,
        )

    def get_filing_summary(
        self,
        ticker: str,
        period: Optional[str] = None,
        report_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        ticker_n = normalize_ticker(ticker)
        company = self.store.get_company(ticker_n)
        if company is None:
            return make_response(
                success=False,
                source=SOURCE,
                ticker=ticker_n,
                as_of_date=self._as_of(),
                error=f"Unknown ticker: {ticker_n}",
            )

        report, err = self.store.resolve_report(
            ticker_n, period=period, report_type=report_type
        )
        if err or report is None:
            return make_response(
                success=False,
                source=SOURCE,
                ticker=ticker_n,
                as_of_date=self._as_of(),
                error=err or "Unable to resolve report",
            )

        key_metrics: List[Dict[str, Any]] = []
        evidence: List[str] = []
        summary_facts = list(report.get("summary_facts") or [])
        evidence.extend(summary_facts)
        for row in report.get("metrics") or []:
            if row.get("metric_type") != "reported":
                continue
            key_metrics.append(
                {
                    "metric": row.get("metric"),
                    "metric_type": row.get("metric_type"),
                    "value": row.get("value"),
                    "currency": report.get("currency"),
                    "unit": report.get("unit"),
                    "verification_status": row.get("verification_status"),
                }
            )
            if row.get("evidence_text"):
                evidence.append(row["evidence_text"])
        if report.get("source_url"):
            evidence.append(f"source_url: {report['source_url']}")
        structured = {
            "selected_report": self._report_meta(company, report),
            "key_reported_metrics": key_metrics,
            "currency": report.get("currency"),
            "unit": report.get("unit"),
            "period": {
                "fiscal_year_label": report.get("fiscal_year_label"),
                "period_start": report.get("period_start"),
                "period_end": report.get("period_end"),
                "report_type": report.get("report_type"),
            },
            "source": {
                "source_document": report.get("source_document"),
                "source_url": report.get("source_url"),
            },
            "summary_facts": summary_facts,
            # Plain-text join for simple string matching in smoke evals; factual only.
            "summary": " ".join(summary_facts),
        }

        return make_response(
            success=True,
            source=SOURCE,
            ticker=ticker_n,
            as_of_date=self._as_of(report),
            data=structured,
            evidence=evidence,
        )

    def call(self, tool_name: str, **kwargs: Any) -> Dict[str, Any]:
        tools = {
            "get_financial_metric": self.get_financial_metric,
            "get_filing_summary": self.get_filing_summary,
        }
        if tool_name not in tools:
            return make_response(
                success=False,
                source=SOURCE,
                ticker=normalize_ticker(str(kwargs.get("ticker", ""))),
                as_of_date=self._as_of(),
                error=f"Unknown financial tool: {tool_name}",
            )
        return tools[tool_name](**kwargs)
