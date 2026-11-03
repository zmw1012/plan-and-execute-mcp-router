"""Normalize agent final payloads into the shared grading fields."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from hk_equity_mcp.experiment.schema import ANSWER_STATUSES


def normalize_final_payload(
    final_answer: Any,
    *,
    fallback_status: str = "unavailable",
) -> Tuple[str, str]:
    """Return (predicted_status, final_answer_text).

    Accepts either the shared answer object or a plain string.
    """
    if isinstance(final_answer, dict):
        status = str(final_answer.get("status") or fallback_status).strip().lower()
        if status not in ANSWER_STATUSES:
            status = fallback_status if fallback_status in ANSWER_STATUSES else "unavailable"
        text = final_answer.get("final_answer")
        if text is None:
            text = final_answer.get("answer") or ""
        return status, str(text)
    if final_answer is None:
        return fallback_status if fallback_status in ANSWER_STATUSES else "unavailable", ""
    return (
        fallback_status if fallback_status in ANSWER_STATUSES else "answered",
        str(final_answer),
    )


def agent_completion_to_fallback(agent_status: Optional[str]) -> str:
    mapping = {
        "full": "answered",
        "partial": "clarification_needed",
        "failed": "unavailable",
    }
    return mapping.get(str(agent_status or ""), "unavailable")


def empty_resolver_calls() -> Dict[str, int]:
    return {"find_operations": 0, "describe_operation": 0}
