"""Mock News MCP Server."""

from __future__ import annotations

from typing import Any, Dict, Optional

from hk_equity_mcp.common.mock_data import AS_OF_DATE, MOCK_NEWS, normalize_ticker
from hk_equity_mcp.common.schema import make_response

SOURCE = "mock_news"


class NewsServer:
    """Exposes search_announcements and get_news_summary."""

    def search_announcements(
        self, ticker: str, query: Optional[str] = None, limit: int = 5
    ) -> Dict[str, Any]:
        ticker_n = normalize_ticker(ticker)
        record = MOCK_NEWS.get(ticker_n)
        if record is None:
            return make_response(
                success=False,
                source=SOURCE,
                ticker=ticker_n,
                as_of_date=AS_OF_DATE,
                error=f"Unknown ticker: {ticker_n}",
            )

        items = list(record["announcements"])
        if query:
            q = query.lower()
            items = [
                item
                for item in items
                if q in item["headline"].lower() or q in item["summary"].lower()
            ]

        return make_response(
            success=True,
            source=SOURCE,
            ticker=ticker_n,
            as_of_date=AS_OF_DATE,
            data={"announcements": items[: max(limit, 0)]},
            evidence=list(record["evidence"]),
        )

    def get_news_summary(self, ticker: str) -> Dict[str, Any]:
        ticker_n = normalize_ticker(ticker)
        record = MOCK_NEWS.get(ticker_n)
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
            data={"summary": record["news_summary"]},
            evidence=list(record["evidence"]),
        )

    def call(self, tool_name: str, **kwargs: Any) -> Dict[str, Any]:
        tools = {
            "search_announcements": self.search_announcements,
            "get_news_summary": self.get_news_summary,
        }
        if tool_name not in tools:
            return make_response(
                success=False,
                source=SOURCE,
                ticker=normalize_ticker(str(kwargs.get("ticker", ""))),
                as_of_date=AS_OF_DATE,
                error=f"Unknown news tool: {tool_name}",
            )
        return tools[tool_name](**kwargs)
