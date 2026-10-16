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
        """Response as_of_date uses selected report period_end when available.

        Snapshot creation dates live in financial.json meta
        (snapshot_created_date / latest_source_publication_date), not here.
        """
        if report and report.get("period_end"):
            return str(report["period_end"])
        return self.store.latest_source_publication_date or ""

    def _report_meta(self, company: Dict[str, Any], report: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "company_name": company.get("company_name"),
            "report_type": report.get("report_type"),
            "document_type": report.get("document_type"),
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

    def _metric_fields(self, metric_row: Dict[str, Any]) -> Dict[str, Any]:
        page_pdf = metric_row.get("source_page_pdf")
        return {
            "metric": metric_row.get("metric"),
            "canonical_metric": metric_row.get("canonical_metric"),
            "metric_definition": metric_row.get("metric_definition"),
            "metric_type": metric_row.get("metric_type"),
            "adjustment_basis": metric_row.get("adjustment_basis"),
            "eps_basis": metric_row.get("eps_basis"),
            "value": metric_row.get("value"),
            "source_page_pdf": page_pdf,
            "source_page_printed": metric_row.get("source_page_printed"),
            # Backward-compatible alias for older callers/tests.
            "source_page": page_pdf,
            "verification_status": metric_row.get("verification_status"),
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
        if metric_row.get("source_page_pdf") is not None:
            evidence.append(f"source_page_pdf: {metric_row['source_page_pdf']}")
        if report.get("source_url"):
            evidence.append(f"source_url: {report['source_url']}")

        return make_response(
            success=True,
            source=SOURCE,
            ticker=ticker_n,
            as_of_date=self._as_of(report),
            data={
                **self._report_meta(company, report),
                **self._metric_fields(metric_row),
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
                    **self._metric_fields(row),
                    "currency": report.get("currency"),
                    "unit": report.get("unit"),
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
                "document_type": report.get("document_type"),
                "source_document": report.get("source_document"),
                "source_url": report.get("source_url"),
                "audit_status": report.get("audit_status"),
            },
            "summary_facts": summary_facts,
            # Plain-text join for gold-path smoke evals; factual only.
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
