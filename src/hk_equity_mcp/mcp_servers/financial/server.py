"""Mock Financial MCP Server."""

from __future__ import annotations

from typing import Any, Dict, Optional

from hk_equity_mcp.common.mock_data import AS_OF_DATE, MOCK_FINANCIALS, normalize_ticker
from hk_equity_mcp.common.schema import make_response

SOURCE = "mock_financial"


class FinancialServer:
    """Exposes get_financial_metric and get_filing_summary."""

    def get_financial_metric(
        self, ticker: str, metric: str, period: Optional[str] = None
    ) -> Dict[str, Any]:
        ticker_n = normalize_ticker(ticker)
        record = MOCK_FINANCIALS.get(ticker_n)
        if record is None:
            return make_response(
                success=False,
                source=SOURCE,
                ticker=ticker_n,
                as_of_date=AS_OF_DATE,
                error=f"Unknown ticker: {ticker_n}",
            )

        metric_key = metric.strip().lower()
        metric_data = record["metrics"].get(metric_key)
        if metric_data is None:
            return make_response(
                success=False,
                source=SOURCE,
                ticker=ticker_n,
                as_of_date=AS_OF_DATE,
                error=f"Unknown metric: {metric_key}",
            )

        if period and metric_data.get("period") != period:
            return make_response(
                success=False,
                source=SOURCE,
                ticker=ticker_n,
                as_of_date=AS_OF_DATE,
                error=f"Period mismatch: requested {period}, available {metric_data.get('period')}",
            )

        return make_response(
            success=True,
            source=SOURCE,
            ticker=ticker_n,
            as_of_date=AS_OF_DATE,
            data={
                "name": record["name"],
                "metric": metric_key,
                **metric_data,
            },
            evidence=list(record["evidence"]),
        )

    def get_filing_summary(
        self, ticker: str, period: Optional[str] = "FY2024"
    ) -> Dict[str, Any]:
        ticker_n = normalize_ticker(ticker)
        record = MOCK_FINANCIALS.get(ticker_n)
        if record is None:
            return make_response(
                success=False,
                source=SOURCE,
                ticker=ticker_n,
                as_of_date=AS_OF_DATE,
                error=f"Unknown ticker: {ticker_n}",
            )

        return make_response(
            success=True,
            source=SOURCE,
            ticker=ticker_n,
            as_of_date=AS_OF_DATE,
            data={
                "name": record["name"],
                "period": period,
                "summary": record["filing_summary"],
            },
            evidence=list(record["evidence"]),
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
                as_of_date=AS_OF_DATE,
                error=f"Unknown financial tool: {tool_name}",
            )
        return tools[tool_name](**kwargs)
