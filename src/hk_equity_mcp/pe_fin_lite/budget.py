"""PE-Fin Lite budgets — LLM + MCP only; fail-closed, no Monitor counters."""

from __future__ import annotations

from typing import Any, Dict, Optional

from hk_equity_mcp.apv_fin.llm_budget import BudgetedLLMClient, LlmCallBudgetExhausted


class McpBudgetExhausted(RuntimeError):
    error_type = "resource_limit_failure"

    def __init__(self, *, used: int, limit: int, stage: str):
        self.used = used
        self.limit = limit
        self.stage = stage
        super().__init__(
            f"resource_limit_failure: mcp_budget used={used} limit={limit} stage={stage}"
        )


class BudgetTracker:
    def __init__(self, limits: Dict[str, Any]):
        self.limits = {
            "max_plan_steps": int(limits.get("max_plan_steps", 6)),
            "max_total_mcp_invocations": int(limits.get("max_total_mcp_invocations", 8)),
            "max_total_llm_calls_per_task": int(
                limits.get("max_total_llm_calls_per_task", 8)
            ),
            "max_local_fallback_per_retrieve": int(
                limits.get("max_local_fallback_per_retrieve", 1)
            ),
            "max_final_repair_rounds": int(limits.get("max_final_repair_rounds", 1)),
            "retrieval_top_k": int(limits.get("retrieval_top_k", 5)),
            "describe_top_n": int(limits.get("describe_top_n", 3)),
        }
        self.llm_calls = 0
        self.mcp_invocations = 0
        self.calls_by_stage: Dict[str, int] = {}
        self.embedding_query_calls = 0
        self.local_fallback_count = 0
        self.stopping_reason: Optional[str] = None
        self.final_repair_rounds = 0
        self.planner_repair_count = 0
        self.binder_calls = 0
        self.primary_selected = 0
        self.fallback_used = 0

    @property
    def llm_call_budget(self) -> int:
        return int(self.limits["max_total_llm_calls_per_task"])

    @property
    def llm_calls_remaining(self) -> int:
        return max(0, self.llm_call_budget - self.llm_calls)

    @property
    def llm_budget_exhausted(self) -> bool:
        return self.llm_calls >= self.llm_call_budget

    def check_llm_budget_or_raise(self, *, stage: str) -> None:
        if self.llm_calls >= self.llm_call_budget:
            self.stopping_reason = "llm_call_budget_exhausted"
            raise LlmCallBudgetExhausted(
                used=self.llm_calls, limit=self.llm_call_budget, stage=stage
            )

    def record_llm_call(self, *, stage: str) -> None:
        self.llm_calls += 1
        key = str(stage or "unspecified")
        self.calls_by_stage[key] = int(self.calls_by_stage.get(key) or 0) + 1

    def check_mcp_budget_or_raise(self, *, stage: str) -> None:
        limit = int(self.limits["max_total_mcp_invocations"])
        if self.mcp_invocations >= limit:
            self.stopping_reason = "mcp_budget_exhausted"
            raise McpBudgetExhausted(used=self.mcp_invocations, limit=limit, stage=stage)

    def record_mcp_invoke(self, *, stage: str = "invoke") -> None:
        self.check_mcp_budget_or_raise(stage=stage)
        self.mcp_invocations += 1

    def llm_budget_fields(self) -> Dict[str, Any]:
        return {
            "llm_call_budget": self.llm_call_budget,
            "llm_calls_used": self.llm_calls,
            "llm_calls_remaining": self.llm_calls_remaining,
            "llm_budget_exhausted": self.llm_budget_exhausted,
            "calls_by_stage": dict(self.calls_by_stage),
        }

    def snapshot(self) -> Dict[str, Any]:
        return {
            "limits": dict(self.limits),
            "llm_calls": self.llm_calls,
            "mcp_invocations": self.mcp_invocations,
            "embedding_query_calls": self.embedding_query_calls,
            "local_fallback_count": self.local_fallback_count,
            "binder_calls": self.binder_calls,
            "primary_selected": self.primary_selected,
            "fallback_used": self.fallback_used,
            "final_repair_rounds": self.final_repair_rounds,
            "planner_repair_count": self.planner_repair_count,
            "stopping_reason": self.stopping_reason,
            "monitor_calls": 0,
            "global_replan_rounds": 0,
            **self.llm_budget_fields(),
        }


def structured_budget_failure(
    exc: Exception,
    *,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    error_type = getattr(exc, "error_type", None) or "resource_limit_failure"
    payload = {
        "status": "failed",
        "error": str(exc),
        "error_type": error_type,
        "stopping_reason": getattr(exc, "stage", None) or error_type,
        "final_answer": {
            "status": "unavailable",
            "final_answer": "",
            "reasoning_brief": "Task stopped due to resource budget exhaustion.",
        },
        "predicted_status": "unavailable",
    }
    if extra:
        payload.update(extra)
    return payload


__all__ = [
    "BudgetTracker",
    "BudgetedLLMClient",
    "LlmCallBudgetExhausted",
    "McpBudgetExhausted",
    "structured_budget_failure",
]
