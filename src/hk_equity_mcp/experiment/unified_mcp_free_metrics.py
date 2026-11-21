"""MCP-Free metric overlays: tool fields are N/A; answer/refusal metrics retained."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence

from hk_equity_mcp.experiment.unified_aggregates import build_grouped_aggregates
from hk_equity_mcp.experiment.unified_evaluator import (
    count_parse_repairs,
    evaluate_task,
    refusal_correctness,
)
from hk_equity_mcp.experiment.unified_grading import grade_task

TOOL_NA = "N/A"


def apply_mcp_free_tool_na(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Replace tool-selection / tool-execution fields with N/A.

    Zero tool calls must NOT be scored as tool failure or vacuous tool success.
    """
    out = dict(metrics)
    out["predicted_operations"] = []
    out["retrieved_candidates"] = []
    out["describe_calls"] = []
    out["invoke_calls"] = []
    out["dependency_edges_predicted"] = []
    out["operation_precision"] = TOOL_NA
    out["operation_recall"] = TOOL_NA
    out["operation_f1"] = TOOL_NA
    out["exact_tool_set"] = TOOL_NA
    out["graph_dependency_accuracy"] = TOOL_NA
    out["wrong_tool_rate"] = TOOL_NA
    out["all_tool_invocations_successful"] = TOOL_NA
    out["tool_execution_success"] = TOOL_NA
    out["successful_invocation_count"] = 0
    out["failed_invocation_count"] = 0
    out["recovered_after_tool_error"] = TOOL_NA
    out["protocol_success_rate"] = TOOL_NA
    out["final_operation_success_by_operation_id"] = {}
    out["real_mcp_execution_count"] = 0
    out["tool_metrics_applicable"] = False
    out["tool_metrics_note"] = (
        "MCP-Free: no catalog/tools/embedding/MCP; tool metrics are N/A "
        "(zero calls are not tool failures)."
    )
    return out


def evaluate_mcp_free_task(
    *,
    item: Dict[str, Any],
    result: Dict[str, Any],
) -> Dict[str, Any]:
    """Reuse evaluator v1.2 answer/refusal logic; neutralize tool metrics."""
    # Run stock evaluator for refusal_correctness / parse_repair, then overlay N/A.
    base = evaluate_task(item=item, result=result)
    overlay = apply_mcp_free_tool_na(base)
    # Ensure refusal uses item fields even when gold_ops empty for MCP-free.
    if overlay.get("refusal_correctness") is None:
        overlay["refusal_correctness"] = refusal_correctness(
            answerability_class=item.get("answerability_class"),
            gold_status=item.get("gold_status"),
            predicted_status=result.get("predicted_status"),
            gold_operations=item.get("gold_operations") or [],
        )
    # Prefer agent-reported parse repairs when present.
    if result.get("parse_repair_count") is not None:
        overlay["parse_repair_count"] = int(result["parse_repair_count"])
    else:
        overlay["parse_repair_count"] = count_parse_repairs(result.get("trace") or [])
    overlay["embedding_calls"] = int(result.get("embedding_calls") or 0)
    overlay["embedding_tokens"] = int(result.get("embedding_tokens") or 0)
    overlay["underlying_operation_calls"] = int(
        result.get("underlying_operation_calls") or 0
    )
    return overlay


def _is_hallucinated_exact_value(row: Dict[str, Any]) -> bool:
    reason = str(row.get("auto_grade_reason") or row.get("reason") or "").lower()
    if "fabricat" in reason or "numeric_substitute" in reason:
        return True
    # Answered with concrete numbers that failed deterministic numeric grade.
    if (
        str(row.get("predicted_status") or "").lower() == "answered"
        and row.get("auto_grade_status") == "graded"
        and row.get("answer_correct") is False
        and any(
            tok in reason
            for tok in (
                "numeric",
                "mismatch",
                "labeled_multi",
                "value_mismatch",
                "scale",
            )
        )
    ):
        return True
    return False


def summarize_mcp_free_group(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {
            "task_count": 0,
            "answer_accuracy": None,
            "refusal_accuracy": None,
            "unavailable_rate": None,
            "hallucinated_exact_value_rate": None,
            "mean_llm_calls": None,
            "mean_total_tokens": None,
            "mean_latency_ms": None,
            "tool_metrics": TOOL_NA,
        }

    auto = [r for r in rows if r.get("auto_grade_status") == "graded"]
    auto_correct = [r for r in auto if r.get("answer_correct") is True]
    refusal_vals = [
        float(r["refusal_correctness"])
        for r in rows
        if isinstance(r.get("refusal_correctness"), (int, float))
    ]
    unavailable = sum(
        1
        for r in rows
        if str(r.get("predicted_status") or "").lower() == "unavailable"
    )
    halluc = sum(1 for r in rows if _is_hallucinated_exact_value(r))

    def _avg(vals: Sequence[float]) -> Optional[float]:
        return (sum(vals) / len(vals)) if vals else None

    return {
        "task_count": n,
        "auto_gradable_count": len(auto),
        "auto_graded_correct": len(auto_correct),
        "answer_accuracy": (len(auto_correct) / len(auto)) if auto else None,
        "refusal_accuracy": _avg(refusal_vals),
        "unavailable_rate": unavailable / n,
        "hallucinated_exact_value_rate": halluc / n,
        "mean_llm_calls": _avg(
            [float(r["llm_calls"]) for r in rows if isinstance(r.get("llm_calls"), (int, float))]
        ),
        "mean_total_tokens": _avg(
            [
                float((r.get("tokens") or {}).get("total_tokens") or r.get("total_tokens") or 0)
                for r in rows
            ]
        ),
        "mean_latency_ms": _avg(
            [float(r["latency_ms"]) for r in rows if isinstance(r.get("latency_ms"), (int, float))]
        ),
        "total_llm_calls": sum(int(r.get("llm_calls") or 0) for r in rows),
        "total_tokens": sum(
            int((r.get("tokens") or {}).get("total_tokens") or 0) for r in rows
        ),
        "tool_metrics": TOOL_NA,
        "tool_calls": 0,
        "resolver_calls": 0,
        "real_mcp_executions": 0,
        "embedding_calls": 0,
        "note": (
            "answer_accuracy is auto-gradable subset only; "
            "tool metrics are N/A for MCP-Free."
        ),
    }


def build_mcp_free_grouped_aggregates(
    rows: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Grouped MCP-Free metrics + retain stock aggregates with tool fields N/A."""
    stock = build_grouped_aggregates(rows)
    out: Dict[str, Any] = {
        "overall": summarize_mcp_free_group(rows),
        "stock_aggregates_note": (
            "stock_grouped_aggregates retained for schema parity; "
            "prefer mcp_free overall/domain/... for reporting."
        ),
        "stock_grouped_aggregates": stock,
    }

    for key in (
        "domain",
        "project_difficulty",
        "exact_etom_level",
        "answerability_class",
        "company",
        "source_split",
    ):
        buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in rows:
            val = row.get(key)
            name = "null" if val is None or val == "" else str(val)
            buckets[name].append(row)
        out[key] = {
            name: summarize_mcp_free_group(group_rows)
            for name, group_rows in sorted(buckets.items())
        }

    # D1–D5 alias: project_difficulty when present.
    if out.get("project_difficulty"):
        out["D1_D5"] = out["project_difficulty"]

    dev_rows = [
        r
        for r in rows
        if r.get("source_split") == "development"
        or r.get("previously_reserved") is False
    ]
    reserved_rows = [
        r
        for r in rows
        if r.get("source_split") == "previously_reserved"
        or r.get("previously_reserved") is True
    ]
    out["development_origin"] = summarize_mcp_free_group(dev_rows)
    out["previously_reserved_origin"] = summarize_mcp_free_group(reserved_rows)
    if rows and all(r.get("source_split") for r in rows):
        out["origin_count_check"] = {
            "development": len(dev_rows),
            "previously_reserved": len(reserved_rows),
            "sum": len(dev_rows) + len(reserved_rows),
            "overall": len(rows),
            "ok": len(dev_rows) + len(reserved_rows) == len(rows),
        }
    return out


def grade_and_evaluate_mcp_free(
    item: Dict[str, Any], result: Dict[str, Any]
) -> Dict[str, Any]:
    metrics = evaluate_mcp_free_task(item=item, result=result)
    grade = grade_task(item, {**result, **metrics, "task_id": item["task_id"]})
    return {**metrics, "grade": grade}
