"""Mock Market MCP Server."""

from __future__ import annotations

from typing import Any, Dict, Optional

from hk_equity_mcp.common.mock_data import AS_OF_DATE, MOCK_PRICES, normalize_ticker
from hk_equity_mcp.common.schema import make_response

SOURCE = "mock_market"


class MarketServer:
    """Exposes get_price_series and calculate_return."""

    def get_price_series(
        self,
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        ticker_n = normalize_ticker(ticker)
        record = MOCK_PRICES.get(ticker_n)
        if record is None:
            return make_response(
                success=False,
                source=SOURCE,
                ticker=ticker_n,
                as_of_date=AS_OF_DATE,
                error=f"Unknown ticker: {ticker_n}",
            )

        series = list(record["series"])
        if start_date:
            series = [row for row in series if row["date"] >= start_date]
        if end_date:
            series = [row for row in series if row["date"] <= end_date]

        return make_response(
            success=True,
            source=SOURCE,
            ticker=ticker_n,
            as_of_date=AS_OF_DATE,
            data={"currency": record["currency"], "series": series},
            evidence=list(record["evidence"]),
        )

    def calculate_return(
        self, ticker: str, start_date: str, end_date: str
    ) -> Dict[str, Any]:
        ticker_n = normalize_ticker(ticker)
        record = MOCK_PRICES.get(ticker_n)
        if record is None:
            return make_response(
                success=False,
                source=SOURCE,
                ticker=ticker_n,
                as_of_date=AS_OF_DATE,
                error=f"Unknown ticker: {ticker_n}",
            )

        by_date = {row["date"]: row["close"] for row in record["series"]}
        if start_date not in by_date or end_date not in by_date:
            return make_response(
                success=False,
                source=SOURCE,
                ticker=ticker_n,
                as_of_date=AS_OF_DATE,
                error=f"Missing prices for {start_date} or {end_date}",
            )

        start_px = by_date[start_date]
        end_px = by_date[end_date]
        ret = (end_px / start_px) - 1.0

        return make_response(
            success=True,
            source=SOURCE,
            ticker=ticker_n,
            as_of_date=AS_OF_DATE,
            data={
                "start_date": start_date,
                "end_date": end_date,
                "start_price": start_px,
                "end_price": end_px,
                "return": round(ret, 6),
            },
            evidence=list(record["evidence"]),
        )

    def call(self, tool_name: str, **kwargs: Any) -> Dict[str, Any]:
        tools = {
            "get_price_series": self.get_price_series,
            "calculate_return": self.calculate_return,
        }
        if tool_name not in tools:
            return make_response(
                success=False,
                source=SOURCE,
                ticker=normalize_ticker(str(kwargs.get("ticker", ""))),
                as_of_date=AS_OF_DATE,
                error=f"Unknown market tool: {tool_name}",
            )
        return tools[tool_name](**kwargs)
