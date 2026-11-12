"""Formal ETOM-100 reporting helpers. Routing-only; no MCP execution success."""

from __future__ import annotations

import statistics
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple


def gold_dependency_edges(ground_truth_tools: Any) -> Set[Tuple[int, int]]:
    """Return (dep_index, node_index) edges from the official gold graph."""
    if not isinstance(ground_truth_tools, list):
        return set()
    edges: Set[Tuple[int, int]] = set()
    for index, entry in enumerate(ground_truth_tools):
        tools = entry if isinstance(entry, list) else [entry]
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            for dep in tool.get("dependencies") or []:
                edges.add((int(dep), int(index)))
    return edges


def predicted_dependency_edges(predicted_nodes: Sequence[Dict[str, Any]]) -> Set[Tuple[int, int]]:
    edges: Set[Tuple[int, int]] = set()
    for index, node in enumerate(predicted_nodes or []):
        for dep in node.get("depends_on") or []:
            edges.add((int(dep), int(index)))
        for edge in node.get("dependency_edges") or []:
            if isinstance(edge, (list, tuple)) and len(edge) == 2:
                edges.add((int(edge[0]), int(edge[1])))
            elif isinstance(edge, dict) and "from" in edge and "to" in edge:
                edges.add((int(edge["from"]), int(edge["to"])))
    return edges


def dependency_accuracy(
    *,
    level: int,
    ground_truth_tools: Any,
    predicted_nodes: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    gold = gold_dependency_edges(ground_truth_tools)
    pred = predicted_dependency_edges(predicted_nodes)
    if level not in (3, 4):
        return {
            "dependency_edge_accuracy": None,
            "ordered_dependency_accuracy": None,
            "gold_dependency_edges": 0,
            "predicted_dependency_edges": len(pred),
            "matched_dependency_edges": 0,
        }
    if not gold and not pred:
        return {
            "dependency_edge_accuracy": 1.0,
            "ordered_dependency_accuracy": 1.0,
            "gold_dependency_edges": 0,
            "predicted_dependency_edges": 0,
            "matched_dependency_edges": 0,
        }
    matched = gold & pred
    edge_acc = (len(matched) / len(gold)) if gold else 0.0
    # Ordered dependency accuracy: fraction of gold edges whose relative order is present.
    ordered_ok = 0
    for src, dst in gold:
        if (src, dst) in pred and src < dst:
            ordered_ok += 1
        elif (src, dst) in pred:
            ordered_ok += 1
    ordered_acc = (ordered_ok / len(gold)) if gold else (1.0 if not pred else 0.0)
    return {
        "dependency_edge_accuracy": round(edge_acc, 6),
        "ordered_dependency_accuracy": round(ordered_acc, 6),
        "gold_dependency_edges": len(gold),
        "predicted_dependency_edges": len(pred),
        "matched_dependency_edges": len(matched),
    }


def _mean(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return float(statistics.mean(values))


def grouped_results(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    latest: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        latest[row["task_id"]] = row
    ordered = list(latest.values())
    by_level: Dict[str, Dict[str, Any]] = {}
    level_f1: List[float] = []
    level_em: List[float] = []
    for level in range(1, 6):
        group = [row for row in ordered if int(row.get("level") or 0) == level]
        scored = [
            row
            for row in group
            if row.get("status") == "completed" and row.get("f1") is not None
        ]
        exact = [row for row in scored if row.get("exact_match") is True]
        mean_f1 = _mean([float(row["f1"]) for row in scored])
        em_rate = (len(exact) / len(scored)) if scored else None
        if mean_f1 is not None:
            level_f1.append(mean_f1)
        if em_rate is not None:
            level_em.append(em_rate)
        by_level[f"L{level}"] = {
            "tasks": len(group),
            "completed": sum(1 for row in group if row.get("status") == "completed"),
            "failed": sum(1 for row in group if row.get("status") != "completed"),
            "exact_match_rate": em_rate,
            "mean_precision": _mean(
                [float(row["precision"]) for row in scored if row.get("precision") is not None]
            ),
            "mean_recall": _mean(
                [float(row["recall"]) for row in scored if row.get("recall") is not None]
            ),
            "mean_f1": mean_f1,
            "mean_dependency_edge_accuracy": _mean(
                [
                    float(row["dependency_edge_accuracy"])
                    for row in scored
                    if row.get("dependency_edge_accuracy") is not None
                ]
            ),
            "mean_ordered_dependency_accuracy": _mean(
                [
                    float(row["ordered_dependency_accuracy"])
                    for row in scored
                    if row.get("ordered_dependency_accuracy") is not None
                ]
            ),
            "infeasible_correct_rate": _mean(
                [
                    1.0 if row.get("exact_match") else 0.0
                    for row in scored
                    if int(row.get("level") or 0) == 5
                ]
            )
            if level == 5
            else None,
        }
    scored_all = [
        row
        for row in ordered
        if row.get("status") == "completed" and row.get("f1") is not None
    ]
    exact_all = [row for row in scored_all if row.get("exact_match") is True]
    return {
        "evaluation_mode": "routing_only",
        "real_mcp_tool_executed": False,
        "pooled_100": {
            "tasks": len(ordered),
            "completed": sum(1 for row in ordered if row.get("status") == "completed"),
            "exact_match_rate": (len(exact_all) / len(scored_all)) if scored_all else None,
            "mean_precision": _mean(
                [float(row["precision"]) for row in scored_all if row.get("precision") is not None]
            ),
            "mean_recall": _mean(
                [float(row["recall"]) for row in scored_all if row.get("recall") is not None]
            ),
            "mean_f1": _mean([float(row["f1"]) for row in scored_all]),
            "note": (
                "Pooled over the equal-level stratified formal sample. "
                "Not an ETOM natural-distribution population estimate."
            ),
        },
        "macro_average_across_levels": {
            "mean_f1": _mean(level_f1),
            "exact_match_rate": _mean(level_em),
            "note": "Unweighted mean of the five per-level means.",
        },
        "per_level": by_level,
        "server_coverage": {
            "unique_predicted_servers": len(
                {
                    node.get("server_name")
                    for row in ordered
                    for node in row.get("predicted_nodes") or []
                    if node.get("server_name")
                }
            )
        },
    }


def error_summary(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    latest: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        latest[row["task_id"]] = row
    ordered = list(latest.values())
    taxonomy = Counter()
    for row in ordered:
        if row.get("status") == "completed" and row.get("exact_match") is True:
            taxonomy["exact_match"] += 1
            continue
        if row.get("timeout"):
            taxonomy["timeout"] += 1
        elif row.get("status") != "completed":
            taxonomy[str(row.get("error") or row.get("status") or "failed")] += 1
        elif int(row.get("level") or 0) == 5:
            taxonomy["l5_incorrect_refusal"] += 1
        elif row.get("exact_match") is False:
            taxonomy["routing_mismatch"] += 1
        else:
            taxonomy["other"] += 1
    return {
        "tasks": len(ordered),
        "failure_taxonomy": dict(taxonomy),
        "parse_repairs": sum(int(row.get("parse_repairs") or 0) for row in ordered),
        "http_retries": sum(int(row.get("http_retries") or 0) for row in ordered),
        "real_mcp_tool_executed": False,
    }


def retrieval_analysis(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    latest: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        latest[row["task_id"]] = row
    ordered = list(latest.values())
    return {
        "tasks": len(ordered),
        "query_embedding_calls": sum(int(row.get("query_embedding_calls") or 0) for row in ordered),
        "query_embedding_tokens": sum(
            int(row.get("query_embedding_tokens") or 0) for row in ordered
        ),
        "mean_query_embedding_calls": _mean(
            [float(row.get("query_embedding_calls") or 0) for row in ordered]
        ),
        "note": (
            "Catalog index is one-time and shared. These counts are query-time "
            "embeddings only. Routing-only evaluation; no real MCP execution."
        ),
        "real_mcp_tool_executed": False,
    }
