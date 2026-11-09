"""Task-level LLM call budget: single counter wrapping the generation client."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from hk_equity_mcp.agent.llm_client import LLMClient


class LlmCallBudgetExhausted(RuntimeError):
    """Raised before an LLM request when the per-task generation cap is reached."""

    # Canonical failure class for metrics/taxonomy (v1.1 controller).
    error_type = "resource_limit_failure"

    def __init__(self, *, used: int, limit: int, stage: str):
        self.used = used
        self.limit = limit
        self.stage = stage
        super().__init__(
            f"resource_limit_failure: llm_call_budget used={used} limit={limit} stage={stage}"
        )


class BudgetedLLMClient(LLMClient):
    """Wraps an LLMClient so every generation call shares one task-level counter.

    Embedding / MCP / catalog indexing are never routed through this client.
    """

    def __init__(self, inner: LLMClient, budget: "BudgetTrackerProtocol"):
        self._inner = inner
        self._budget = budget
        self.provider = getattr(inner, "provider", "unknown")
        self.model = getattr(inner, "model", "unknown")

    def complete(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        stage: Optional[str] = None,
    ) -> Dict[str, Any]:
        stage_name = stage or "unspecified"
        self._budget.check_llm_budget_or_raise(stage=stage_name)
        self._budget.record_llm_call(stage=stage_name)
        return self._inner.complete(
            messages, temperature=temperature, max_tokens=max_tokens
        )


class BudgetTrackerProtocol:
    """Structural protocol documented for type checkers; real impl is BudgetTracker."""

    def check_llm_budget_or_raise(self, *, stage: str) -> None:  # pragma: no cover
        raise NotImplementedError

    def record_llm_call(self, *, stage: str) -> None:  # pragma: no cover
        raise NotImplementedError
