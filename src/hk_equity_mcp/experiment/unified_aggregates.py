"""Grouped aggregate metrics for unified / exploratory runs."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence


GROUP_KEYS = (
    "overall",
    "domain",
    "domain_combination",
    "source_split",
    "project_difficulty",
    "exact_etom_level",
    "answerability_class",
    "company",
    "development_origin",
    "previously_reserved_origin",
)


def _avg(vals: Sequence[float]) -> Optional[float]:
    if not vals:
        return None
    return sum(vals) / len(vals)


def _rate(rows: Sequence[Dict[str, Any]], key: str) -> Optional[float]:
    if not rows:
        return None
    return sum(1 for r in rows if r.get(key)) / len(rows)


def summarize_group(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    completed = [
        r
        for r in rows
        if r.get("status") == "full" or (r.get("error") in (None, "", "null"))
    ]
    failed = [r for r in rows if r.get("error") not in (None, "", "null")]
    auto_grades = [
        r for r in rows if r.get("auto_grade_status") == "graded"
    ]
    auto_correct = [r for r in auto_grades if r.get("answer_correct") is True]
    exact_etom_rows = [
        r for r in rows if r.get("exact_etom_level") not in (None, "", "null")
    ]

    return {
        "task_count": len(rows),
        "completed_count": len(completed),
        "failed_count": len(failed),
        "answer_grading_coverage": (
            len(auto_grades) / len(rows) if rows else None
        ),
        "auto_gradable_count": len(auto_grades),
        "auto_graded_correct": len(auto_correct),
        "auto_graded_accuracy": (
            len(auto_correct) / len(auto_grades) if auto_grades else None
        ),
        "manual_required_count": sum(
            1 for r in rows if r.get("auto_grade_status") == "manual_required"
        ),
        "exact_tool_set_rate": _rate(rows, "exact_tool_set"),
        "mean_operation_precision": _avg(
            [float(r["operation_precision"]) for r in rows if isinstance(r.get("operation_precision"), (int, float))]
        ),
        "mean_operation_recall": _avg(
            [float(r["operation_recall"]) for r in rows if isinstance(r.get("operation_recall"), (int, float))]
        ),
        "mean_operation_f1": _avg(
            [float(r["operation_f1"]) for r in rows if isinstance(r.get("operation_f1"), (int, float))]
        ),
        "mean_dependency_accuracy": _avg(
            [
                float(r["graph_dependency_accuracy"])
                for r in rows
                if isinstance(r.get("graph_dependency_accuracy"), (int, float))
            ]
        ),
        "all_tool_invocations_successful_rate": _rate(
            rows, "all_tool_invocations_successful"
        ),
        "recovered_after_tool_error_rate": _rate(rows, "recovered_after_tool_error"),
        "refusal_accuracy": _avg(
            [
                float(r["refusal_correctness"])
                for r in rows
                if isinstance(r.get("refusal_correctness"), (int, float))
            ]
        ),
        "mean_wrong_tool_rate": _avg(
            [float(r["wrong_tool_rate"]) for r in rows if isinstance(r.get("wrong_tool_rate"), (int, float))]
        ),
        "mean_llm_calls": _avg(
            [float(r["llm_calls"]) for r in rows if isinstance(r.get("llm_calls"), (int, float))]
        ),
        "mean_total_tokens": _avg(
            [
                float((r.get("tokens") or {}).get("total_tokens") or r.get("total_tokens") or 0)
                for r in rows
                if ((r.get("tokens") or {}).get("total_tokens") is not None)
                or isinstance(r.get("total_tokens"), (int, float))
            ]
        ),
        "mean_real_mcp_execution_count": _avg(
            [
                float(r["real_mcp_execution_count"])
                for r in rows
                if isinstance(r.get("real_mcp_execution_count"), (int, float))
            ]
        ),
        "mean_latency_ms": _avg(
            [float(r["latency_ms"]) for r in rows if isinstance(r.get("latency_ms"), (int, float))]
        ),
        "exact_etom_applicable_count": len(exact_etom_rows),
        "note": (
            "exact_tool_set_rate is not answer accuracy; "
            "overall answer accuracy requires manual grading completion"
        ),
    }


def _group_value(row: Dict[str, Any], key: str) -> str:
    if key == "development_origin":
        return "development" if row.get("source_split") == "development" or (
            row.get("previously_reserved") is False
        ) else "_exclude"
    if key == "previously_reserved_origin":
        return "previously_reserved" if (
            row.get("source_split") == "previously_reserved"
            or row.get("previously_reserved") is True
        ) else "_exclude"
    val = row.get(key)
    if val is None or val == "":
        return "null"
    return str(val)


def build_grouped_aggregates(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"overall": summarize_group(rows)}
    for key in (
        "domain",
        "domain_combination",
        "source_split",
        "project_difficulty",
        "exact_etom_level",
        "answerability_class",
        "company",
    ):
        buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in rows:
            buckets[_group_value(row, key)].append(row)
        out[key] = {
            name: summarize_group(group_rows)
            for name, group_rows in sorted(buckets.items())
        }

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
    out["development_origin"] = summarize_group(dev_rows)
    out["previously_reserved_origin"] = summarize_group(reserved_rows)

    # Sanity: origin counts should sum to overall when source_split is present.
    if rows and all(r.get("source_split") for r in rows):
        out["origin_count_check"] = {
            "development": len(dev_rows),
            "previously_reserved": len(reserved_rows),
            "sum": len(dev_rows) + len(reserved_rows),
            "overall": len(rows),
            "ok": len(dev_rows) + len(reserved_rows) == len(rows),
        }
    return out
