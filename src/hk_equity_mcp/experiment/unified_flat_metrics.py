"""Flat All Tools metrics: tool selection applicable; retrieval metrics = N/A."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from hk_equity_mcp.experiment.unified_aggregates import build_grouped_aggregates
from hk_equity_mcp.experiment.unified_evaluator import (
    count_parse_repairs,
    evaluate_task,
    summarize_predictions,
)

RETRIEVAL_NA = "N/A"


def apply_flat_retrieval_na(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Mark retrieval-stage fields N/A without neutralizing tool-selection metrics.

    Important: do NOT coerce N/A into 1.0 / True.
    """
    out = dict(metrics)
    out["retrieved_candidates"] = RETRIEVAL_NA
    out["retrieval_metrics"] = RETRIEVAL_NA
    out["retrieval_calls"] = 0
    out["retrieval_recall_at_k"] = RETRIEVAL_NA
    out["retrieval_hit_at_k"] = RETRIEVAL_NA
    out["describe_calls"] = []
    out["embedding_calls"] = 0
    out["embedding_tokens"] = 0
    out["finder_calls"] = 0
    out["resolver_find_calls"] = 0
    out["resolver_describe_calls"] = 0
    out["retrieval_metrics_note"] = (
        "Flat All Tools: no embedding/retrieval/find/describe; "
        "retrieval metrics are N/A (not 1.0)."
    )
    return out


def evaluate_flat_task(
    *,
    item: Dict[str, Any],
    result: Dict[str, Any],
) -> Dict[str, Any]:
    """Reuse evaluator v1.2 tool/answer metrics; overlay retrieval N/A."""
    base = evaluate_task(item=item, result=result)
    out = apply_flat_retrieval_na(base)

    # Prefer agent-reported Flat bookkeeping when present.
    if result.get("selected_operations") is not None:
        out["selected_operations"] = list(result.get("selected_operations") or [])
    else:
        out["selected_operations"] = list(out.get("predicted_operations") or [])
    if result.get("invoked_operations") is not None:
        out["invoked_operations"] = list(result.get("invoked_operations") or [])
    else:
        out["invoked_operations"] = [
            r.get("operation_id")
            for r in (out.get("invoke_calls") or [])
            if r.get("operation_id")
        ]
    if result.get("successful_invoked_operations") is not None:
        out["successful_invoked_operations"] = list(
            result.get("successful_invoked_operations") or []
        )
    else:
        out["successful_invoked_operations"] = [
            r.get("operation_id")
            for r in (out.get("invoke_calls") or [])
            if r.get("success") and r.get("operation_id")
        ]
    out["invalid_selections"] = list(result.get("invalid_selections") or [])

    if result.get("parse_repair_count") is not None:
        out["parse_repair_count"] = int(result["parse_repair_count"])
    else:
        out["parse_repair_count"] = count_parse_repairs(result.get("trace") or [])
    out["protocol_repair_count"] = int(
        result.get("protocol_repair_count")
        if result.get("protocol_repair_count") is not None
        else out["parse_repair_count"]
    )

    out["embedding_calls"] = 0
    out["embedding_tokens"] = 0
    out["retrieval_calls"] = 0
    resolver = result.get("resolver_calls") or {
        "find_operations": 0,
        "describe_operation": 0,
    }
    out["resolver_calls"] = {
        "find_operations": int(resolver.get("find_operations") or 0),
        "describe_operation": int(resolver.get("describe_operation") or 0),
    }
    out["schema_bundle_overhead"] = result.get("schema_bundle_overhead") or {}
    return out


def summarize_flat_predictions(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    stock = summarize_predictions(list(rows))
    stock["retrieval_metrics"] = RETRIEVAL_NA
    stock["total_embedding_calls"] = sum(int(r.get("embedding_calls") or 0) for r in rows)
    stock["total_retrieval_calls"] = sum(int(r.get("retrieval_calls") or 0) for r in rows)
    stock["total_find_operations"] = sum(
        int((r.get("resolver_calls") or {}).get("find_operations") or 0) for r in rows
    )
    stock["total_describe_operation"] = sum(
        int((r.get("resolver_calls") or {}).get("describe_operation") or 0) for r in rows
    )
    stock["note_flat"] = (
        "Tool-selection metrics (precision/recall/exact_tool_set) apply. "
        "Retrieval metrics are N/A (not scored as 1.0)."
    )
    return stock


def build_flat_grouped_aggregates(
    rows: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    stock = build_grouped_aggregates(list(rows))
    return {
        "condition": "flat",
        "retrieval_metrics": RETRIEVAL_NA,
        "stock_grouped_aggregates": stock,
        "reporting": {
            "A_final_answer_accuracy": "see grading_summary / manual overall",
            "B_exact_tool_set_success": stock.get("overall", {}).get("exact_tool_set_rate")
            if isinstance(stock.get("overall"), dict)
            else None,
            "C_strict_overall_task_success": (
                "manual overall_task_success_0_or_1 (not auto-filled)"
            ),
        },
    }
