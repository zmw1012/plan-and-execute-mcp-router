"""Official + MCP-Zero retrieval metrics for ETOM routing-only runs."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence

from hk_equity_mcp.etom.formal_metrics import (
    error_summary,
    grouped_results,
    retrieval_analysis,
)
from hk_equity_mcp.etom.loader import EtomTask
from hk_equity_mcp.mcp_zero_etom.l4_metrics import strict_level_rates


def _server_tool_pairs_from_gold(task: EtomTask) -> Dict[str, Any]:
    gold_servers = set(task.server_names or [])
    gold_ops = set(task.operation_ids or [])
    return {"gold_servers": gold_servers, "gold_ops": gold_ops}


def mcp_zero_retrieval_analysis(
    rows: Sequence[Dict[str, Any]],
    tasks_by_id: Mapping[str, EtomTask],
) -> Dict[str, Any]:
    base = retrieval_analysis(rows)
    latest: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        latest[row["task_id"]] = row
    server_hits = 0
    server_total = 0
    tool_hits = 0
    tool_total = 0
    gold_in_candidates = 0
    routed_gold_not_selected = 0
    selected_not_retrieved = 0
    for task_id, row in latest.items():
        task = tasks_by_id.get(task_id)
        if task is None:
            continue
        gold = _server_tool_pairs_from_gold(task)
        retrieved_servers = {
            s.get("server_id") or s.get("server_name")
            for s in (row.get("retrieved_servers") or [])
            if isinstance(s, dict)
        }
        # retrieved_servers may also be list of strings
        if row.get("retrieved_servers") and not retrieved_servers:
            retrieved_servers = set(row.get("retrieved_servers") or [])
        candidate_ops = set(row.get("candidate_operation_ids") or [])
        selected = set(row.get("selected_operations") or [])
        predicted_ops = {
            f"{n.get('server_name')}::{n.get('tool_name')}"
            for n in (row.get("predicted_nodes") or [])
            if n.get("server_name") and n.get("tool_name")
        }
        if gold["gold_servers"]:
            server_total += 1
            if gold["gold_servers"] & set(map(str, retrieved_servers)):
                server_hits += 1
        if gold["gold_ops"]:
            tool_total += 1
            if gold["gold_ops"] & candidate_ops:
                tool_hits += 1
                gold_in_candidates += 1
            if gold["gold_ops"] - predicted_ops:
                routed_gold_not_selected += 1
        if predicted_ops - candidate_ops:
            selected_not_retrieved += 1
        _ = selected  # reserved for diagnostics
    base.update(
        {
            "hierarchical": True,
            "server_recall": (server_hits / server_total) if server_total else None,
            "tool_recall": (tool_hits / tool_total) if tool_total else None,
            "gold_in_candidates_tasks": gold_in_candidates,
            "routed_gold_not_selected_tasks": routed_gold_not_selected,
            "selected_not_retrieved_tasks": selected_not_retrieved,
            "note": (
                "Post-hoc retrieval analysis from lock gold fields. Gold never "
                "enters the agent prompt. Routing-only; real_tool_executions=0."
            ),
        }
    )
    return base


def build_grouped_with_strict(
    rows: Sequence[Dict[str, Any]], *, expected_per_level: int = 20
) -> Dict[str, Any]:
    grouped = grouped_results(rows)
    latest: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        latest[row["task_id"]] = row
    ordered = list(latest.values())
    strict: Dict[str, Any] = {}
    for level in range(1, 6):
        group = [r for r in ordered if int(r.get("level") or 0) == level]
        exact_successes = sum(
            1
            for r in group
            if r.get("status") == "completed" and r.get("exact_match") is True
        )
        denom = expected_per_level
        completed = [r for r in group if r.get("status") == "completed"]
        completed_exact = sum(1 for r in completed if r.get("exact_match") is True)
        strict[f"L{level}"] = {
            "strict_denominator": denom,
            "strict_exact_route_rate": exact_successes / denom if denom else None,
            "completed_only_exact_route_rate": (
                (completed_exact / len(completed)) if completed else None
            ),
            "completion_rate": (len(completed) / denom) if denom else None,
            "failure_rate": ((denom - len(completed)) / denom) if denom else None,
            "failed_scored_zero": sum(1 for r in group if r.get("status") != "completed"),
        }
        if level == 4:
            for key in (
                "official_union_node_set_em",
                "efs_covering_node_set",
                "cross_server_dependency_correctness",
                "complete_route_success",
            ):
                strict[f"L4_{key}"] = strict_level_rates(
                    ordered, level=4, key=key, expected_n=expected_per_level
                )
    # overall exact route (strict over observed locked size or 100)
    n = len(ordered) or expected_per_level * 5
    overall_exact = sum(
        1
        for r in ordered
        if r.get("status") == "completed" and r.get("exact_match") is True
    )
    # micro F1: mean of per-task f1 with failures as 0
    f1_values = []
    for r in ordered:
        if r.get("status") == "completed" and r.get("f1") is not None:
            f1_values.append(float(r["f1"]))
        else:
            f1_values.append(0.0)
    micro_f1 = (sum(f1_values) / len(f1_values)) if f1_values else None
    grouped["strict"] = strict
    grouped["overall_exact_route"] = {
        "successes": overall_exact,
        "denominator": n,
        "rate": overall_exact / n if n else None,
    }
    grouped["micro_f1_strict"] = micro_f1
    grouped["macro_f1"] = (grouped.get("macro_average_across_levels") or {}).get(
        "mean_f1"
    )
    grouped["evaluation_mode"] = "routing_only"
    grouped["real_tool_executions"] = 0
    grouped["note"] = (
        "Route exact match is NOT final-answer accuracy. "
        "Failures remain in the strict denominator with score 0."
    )
    return grouped


def build_error_summary(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return error_summary(rows)


def build_protocol_summary(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    latest: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        latest[row["task_id"]] = row
    ordered = list(latest.values())
    return {
        "tasks": len(ordered),
        "parse_repairs": sum(int(r.get("parse_repairs") or 0) for r in ordered),
        "protocol_repairs": sum(
            int(r.get("protocol_repair_count") or 0) for r in ordered
        ),
        "max_llm_turns_failures": sum(
            1 for r in ordered if r.get("error_type") == "max_llm_turns"
        ),
        "task_protocol_success_rate": (
            sum(1 for r in ordered if r.get("task_protocol_success")) / len(ordered)
            if ordered
            else None
        ),
        "raw_content_preserved_tasks": sum(
            1
            for r in ordered
            if any(
                isinstance(e, dict) and e.get("raw_content") is not None
                for e in (r.get("trace") or [])
            )
            or r.get("raw_assistant_contents")
        ),
        "real_tool_executions_total": sum(
            int(r.get("real_tool_executions") or 0) for r in ordered
        ),
        "parser_protocol_version": "mcp_zero_protocol_v2",
    }
