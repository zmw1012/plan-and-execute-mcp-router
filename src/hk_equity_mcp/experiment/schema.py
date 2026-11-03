"""Shared schemas for fair financial experiment conditions."""

from __future__ import annotations

CONDITIONS = ("mcp_free", "flat", "operation_visible")

ANSWER_STATUSES = ("answered", "clarification_needed", "unavailable")

# Shared final-answer JSON contract shown to the model in all conditions.
FINAL_ANSWER_CONTRACT = """Return the final answer as a JSON object with exactly:
- status: one of answered, clarification_needed, unavailable
- final_answer: concise answer; include currency, unit, accounting basis, and period when relevant
- reasoning_brief: at most two sentences

If the question is ambiguous, use clarification_needed.
If the requested information is not reliably available, use unavailable rather than guessing.
Do not invent exact figures. Do not include markdown fences around the JSON."""

PREDICTION_FIELDS = (
    "id",
    "condition",
    "question",
    "final_answer",
    "predicted_status",
    "trace",
    "llm_calls",
    "resolver_calls",
    "underlying_operation_calls",
    "tokens",
    "latency_ms",
    "provider_metadata",
    "error",
)

MANUAL_GRADING_FIELDS = (
    "id",
    "condition",
    "company",
    "task_type",
    "question",
    "gold_status",
    "reference_answer",
    "predicted_status",
    "predicted_answer",
    "correct_0_or_1",
    "error_type",
    "notes",
)
