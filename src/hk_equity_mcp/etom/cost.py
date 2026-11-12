"""Token, latency, and estimated USD accounting.

Prices are explicit assumptions. Cache-hit discounts are not applied.
"""

from __future__ import annotations

import math
import statistics
from typing import Any, Dict, List, Optional, Sequence


def usd(tokens: int, usd_per_million: float) -> float:
    return tokens / 1_000_000.0 * usd_per_million


def percentile(values: Sequence[float], fraction: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[int(position)])
    weight = position - lower
    return float(ordered[lower] * (1 - weight) + ordered[upper] * weight)


def summarize_costs(
    rows: Sequence[Dict[str, Any]],
    *,
    pricing: Dict[str, Any],
    catalog: Dict[str, Any],
    probe: Dict[str, Any],
) -> Dict[str, Any]:
    latest = {}
    for row in rows:
        latest[row["task_id"]] = row
    ordered = list(latest.values())
    completed = [row for row in ordered if row.get("status") == "completed"]
    failed = [row for row in ordered if row.get("status") != "completed"]
    by_level: Dict[str, Dict[str, Any]] = {}
    for level in range(1, 6):
        group = [row for row in ordered if int(row.get("level") or 0) == level]
        scored = [
            row
            for row in group
            if row.get("status") == "completed" and row.get("f1") is not None
        ]
        exact = [row for row in scored if row.get("exact_match") is True]
        by_level[f"L{level}"] = {
            "tasks": len(group),
            "completed": sum(1 for row in group if row.get("status") == "completed"),
            "failed": sum(1 for row in group if row.get("status") != "completed"),
            "exact_match_count": len(exact),
            "exact_match_rate": (len(exact) / len(scored)) if scored else None,
            "mean_precision": _mean(scored, "precision"),
            "mean_recall": _mean(scored, "recall"),
            "mean_f1": _mean(scored, "f1"),
        }
    variable_costs = [float(row.get("variable_cost_usd") or 0.0) for row in ordered]
    n = len(variable_costs)
    mean_variable = statistics.mean(variable_costs) if n else 0.0
    stdev_variable = statistics.stdev(variable_costs) if n >= 2 else 0.0
    llm_calls = sum(int(row.get("llm_calls") or 0) for row in ordered)
    input_tokens = sum(int(row.get("input_tokens") or 0) for row in ordered)
    output_tokens = sum(int(row.get("output_tokens") or 0) for row in ordered)
    query_embedding_calls = sum(int(row.get("query_embedding_calls") or 0) for row in ordered)
    query_embedding_tokens = sum(int(row.get("query_embedding_tokens") or 0) for row in ordered)
    latencies = [float(row["latency_ms"]) for row in ordered if row.get("latency_ms") is not None]
    one_time_tokens = int(catalog.get("tokens") or 0)
    one_time_usd = usd(one_time_tokens, float(pricing["embedding_usd_per_million"]))
    probe_usd = float(probe.get("usd") or 0.0)
    one_time_total = one_time_usd + probe_usd

    def project(task_count: int) -> Dict[str, float]:
        point = one_time_total + task_count * mean_variable
        buffer = task_count * stdev_variable
        return {
            "point_usd": round(point, 6),
            "buffer_usd": round(buffer, 6),
            "with_buffer_usd": round(point + buffer, 6),
        }

    scored = [row for row in completed if row.get("f1") is not None]
    exact = [row for row in scored if row.get("exact_match") is True]
    return {
        "tasks_total": len(ordered),
        "completed": len(completed),
        "failed": len(failed),
        "results_by_level": by_level,
        "exact_match_count": len(exact),
        "exact_match_rate": (len(exact) / len(scored)) if scored else None,
        "mean_precision": _mean(scored, "precision"),
        "mean_recall": _mean(scored, "recall"),
        "mean_graph_f1": _mean(scored, "f1"),
        "llm_calls": llm_calls + int(probe.get("llm_calls") or 0),
        "input_tokens": input_tokens + int(probe.get("input_tokens") or 0),
        "output_tokens": output_tokens + int(probe.get("output_tokens") or 0),
        "total_tokens": (
            input_tokens
            + output_tokens
            + int(probe.get("input_tokens") or 0)
            + int(probe.get("output_tokens") or 0)
        ),
        "embedding_calls": query_embedding_calls + int(catalog.get("calls") or 0) + int(probe.get("embedding_calls") or 0),
        "embedding_tokens": query_embedding_tokens + one_time_tokens + int(probe.get("embedding_tokens") or 0),
        "query_embedding_calls": query_embedding_calls,
        "query_embedding_tokens": query_embedding_tokens,
        "one_time_catalog_index": {
            "calls": int(catalog.get("calls") or 0),
            "retries": int(catalog.get("retries") or 0),
            "tokens": one_time_tokens,
            "token_source": catalog.get("token_source"),
            "usd": round(one_time_usd, 6),
            "cache_reused": bool(catalog.get("cache_reused")),
            "paid_on_this_run": not bool(catalog.get("cache_reused")),
        },
        "verification_probe": probe,
        "variable_per_task": {
            "tasks": n,
            "mean_usd": round(mean_variable, 6),
            "sample_stddev_usd": round(stdev_variable, 6),
            "total_usd": round(sum(variable_costs), 6),
        },
        "latency_ms": {
            "mean": _mean_numbers(latencies),
            "median": percentile(latencies, 0.5),
            "p95": percentile(latencies, 0.95),
        },
        "projected_cost_100_tasks": project(100),
        "projected_cost_200_tasks": project(200),
        "projection_uncertainty": {
            "sample_size": n,
            "per_task_stddev_usd": round(stdev_variable, 6),
            "buffer_policy": (
                "Point estimate = one-time catalog index + one verification probe "
                "+ N * mean completed-task variable cost. Buffer = N * sample "
                "standard deviation of per-task variable cost. Unit prices use "
                "peak cache-miss rates and do not apply cache-hit discounts. "
                "n=10, so the buffer is a spread indicator, not a confidence interval."
            ),
            "price_source": pricing.get("source"),
            "llm_input_usd_per_million": pricing.get("llm_input_usd_per_million"),
            "llm_output_usd_per_million": pricing.get("llm_output_usd_per_million"),
            "embedding_usd_per_million": pricing.get("embedding_usd_per_million"),
        },
        "retry_count": sum(int(row.get("http_retries") or 0) for row in ordered) + int(catalog.get("retries") or 0),
        "parse_repairs": sum(int(row.get("parse_repairs") or 0) for row in ordered),
        "timeouts": sum(1 for row in ordered if row.get("timeout")),
        "real_tool_executions": sum(int(row.get("real_tool_executions") or 0) for row in ordered),
        "this_run_estimated_usd": round(
            one_time_usd * (0 if catalog.get("cache_reused") else 1)
            + probe_usd
            + sum(variable_costs),
            6,
        ),
    }


def task_variable_cost(row: Dict[str, Any], pricing: Dict[str, Any]) -> float:
    llm = usd(int(row.get("input_tokens") or 0), float(pricing["llm_input_usd_per_million"]))
    llm += usd(int(row.get("output_tokens") or 0), float(pricing["llm_output_usd_per_million"]))
    embed = usd(
        int(row.get("query_embedding_tokens") or 0),
        float(pricing["embedding_usd_per_million"]),
    )
    return llm + embed


def _mean(rows: Sequence[Dict[str, Any]], key: str) -> Optional[float]:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    if not values:
        return None
    return statistics.mean(values)


def _mean_numbers(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return float(statistics.mean(values))
