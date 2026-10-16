"""Unified agent / router framework with mock tool execution.

No real LLM is called. Modes reserve interfaces for later experiments.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import time

import yaml

from hk_equity_mcp.agent.modes import ExperimentMode, parse_mode
from hk_equity_mcp.mcp_servers.financial import FinancialServer
from hk_equity_mcp.mcp_servers.market import MarketServer
from hk_equity_mcp.mcp_servers.news import NewsServer

TOOL_DOMAIN = {
    "get_financial_metric": "financial",
    "get_filing_summary": "financial",
    "get_price_series": "market",
    "calculate_return": "market",
    "search_announcements": "news",
    "get_news_summary": "news",
}


class AgentRouter:
    """Placeholder agent that can execute known mock tools under three modes."""

    def __init__(self, config_path: Optional[str] = None, mode: Optional[str] = None):
        self.config = self._load_config(config_path)
        mode_value = mode or self.config.get("experiment_mode", "flat")
        self.mode = parse_mode(mode_value)
        self.financial = FinancialServer()
        self.market = MarketServer()
        self.news = NewsServer()
        self.servers = {
            "financial": self.financial,
            "market": self.market,
            "news": self.news,
        }

    @staticmethod
    def _load_config(config_path: Optional[str]) -> Dict[str, Any]:
        if config_path is None:
            default = Path(__file__).resolve().parents[3] / "configs" / "agent.yaml"
            config_path = str(default)
        path = Path(config_path)
        if not path.exists():
            return {"experiment_mode": "flat"}
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def list_tools(self) -> List[str]:
        return sorted(TOOL_DOMAIN.keys())

    def route_domain(self, question: str) -> str:
        """Very small keyword router used by routed modes (placeholder)."""
        text = question.lower()
        routing = self.config.get("routing", {})
        scores = {
            "financial": sum(
                1 for k in routing.get("financial_keywords", []) if k in text
            ),
            "market": sum(1 for k in routing.get("market_keywords", []) if k in text),
            "news": sum(1 for k in routing.get("news_keywords", []) if k in text),
        }
        best = max(scores, key=scores.get)
        return best if scores[best] > 0 else "financial"

    def call_tool(self, tool_name: str, **arguments: Any) -> Dict[str, Any]:
        domain = TOOL_DOMAIN.get(tool_name)
        if domain is None:
            return {
                "success": False,
                "source": "router",
                "ticker": str(arguments.get("ticker", "")),
                "as_of_date": "",
                "data": None,
                "evidence": [],
                "error": f"Unknown tool: {tool_name}",
            }
        return self.servers[domain].call(tool_name, **arguments)

    def run(
        self,
        question: str,
        tool_name: Optional[str] = None,
        arguments: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute one mock step.

        For now, callers may pass an explicit tool + arguments (used by eval).
        If omitted, a trivial keyword-based placeholder selection is used.
        """
        started = time.perf_counter()
        arguments = arguments or {}
        routed_domain = None

        if self.mode in (ExperimentMode.ROUTED, ExperimentMode.ROUTED_STRUCTURED):
            routed_domain = self.route_domain(question)

        if tool_name is None:
            # Placeholder selection — not an LLM.
            domain = routed_domain or self.route_domain(question)
            defaults = {
                "financial": ("get_filing_summary", {"ticker": "0700.HK"}),
                "market": (
                    "get_price_series",
                    {"ticker": "0700.HK", "start_date": "2024-12-27"},
                ),
                "news": ("get_news_summary", {"ticker": "0700.HK"}),
            }
            tool_name, arguments = defaults[domain]

        if (
            self.mode in (ExperimentMode.ROUTED, ExperimentMode.ROUTED_STRUCTURED)
            and TOOL_DOMAIN.get(tool_name) != routed_domain
            and routed_domain is not None
            and tool_name in TOOL_DOMAIN
        ):
            # Soft constraint placeholder for routed modes.
            routed_domain = TOOL_DOMAIN[tool_name]

        tool_result = self.call_tool(tool_name, **arguments)
        latency_ms = (time.perf_counter() - started) * 1000

        answer = None
        if tool_result.get("success"):
            data = tool_result.get("data")
            answer = str(data)

        return {
            "question": question,
            "mode": self.mode.value,
            "routed_domain": routed_domain,
            "selected_tool": tool_name,
            "arguments": arguments,
            "tool_result": tool_result,
            "answer": answer,
            "evidence": tool_result.get("evidence", []),
            "tokens": 0,  # no LLM yet
            "latency_ms": round(latency_ms, 3),
            "structured": self.mode == ExperimentMode.ROUTED_STRUCTURED,
        }


def main() -> None:
    """Minimal mock demo entrypoint."""
    router = AgentRouter()
    demo_questions = [
        ("What was Tencent FY2025 annual revenue?", "get_financial_metric", {
            "ticker": "0700.HK",
            "metric": "revenue",
            "period": "FY2025",
            "report_type": "annual",
        }),
        ("HSBC price return late Dec 2024?", "calculate_return", {
            "ticker": "0005.HK",
            "start_date": "2024-12-27",
            "end_date": "2024-12-31",
        }),
        ("Any Alibaba announcements about logistics?", "search_announcements", {
            "ticker": "9988.HK", "query": "logistics"
        }),
    ]
    for question, tool, args in demo_questions:
        result = router.run(question, tool_name=tool, arguments=args)
        print("=" * 60)
        print(f"Q: {question}")
        print(f"mode={result['mode']} tool={result['selected_tool']}")
        print(f"success={result['tool_result']['success']}")
        print(f"data={result['tool_result']['data']}")
        print(f"evidence={result['evidence']}")


if __name__ == "__main__":
    main()
