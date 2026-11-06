"""L4 metrics recorded separately (must not be conflated).

1. official union node-set EM
2. EFS-covering node-set
3. cross-server dependency correctness
4. complete route success
Plus precision / recall / F1 from the audited evaluator.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from hk_equity_mcp.etom.evaluator import evaluate_prediction
from hk_equity_mcp.etom.formal_metrics import dependency_accuracy, gold_dependency_edges


def _normalize_tool(entry: Any) -> Optional[Tuple[str, str]]:
    if isinstance(entry, dict):
        server = str(entry.get("server_name") or entry.get("server") or "")
        tool = str(entry.get("tool_name") or "")
        if server and tool:
            return (server, tool)
    return None


def gold_union_nodes(ground_truth_tools: Any) -> Set[Tuple[str, str]]:
    nodes: Set[Tuple[str, str]] = set()
    if not isinstance(ground_truth_tools, list):
        return nodes
    for entry in ground_truth_tools:
        options = entry if isinstance(entry, list) else [entry]
        for opt in options:
            pair = _normalize_tool(opt)
            if pair:
                nodes.add(pair)
    return nodes


def gold_efs_groups(ground_truth_tools: Any) -> List[Set[Tuple[str, str]]]:
    groups: List[Set[Tuple[str, str]]] = []
    if not isinstance(ground_truth_tools, list):
        return groups
    for entry in ground_truth_tools:
        options = entry if isinstance(entry, list) else [entry]
        group: Set[Tuple[str, str]] = set()
        for opt in options:
            pair = _normalize_tool(opt)
            if pair:
                group.add(pair)
        if group:
            groups.append(group)
    return groups


def predicted_node_set(predicted_tools: Sequence[Dict[str, Any]]) -> Set[Tuple[str, str]]:
    out: Set[Tuple[str, str]] = set()
    for item in predicted_tools or []:
        pair = _normalize_tool(item)
        if pair:
            out.add(pair)
    return out


def efs_covering_node_set(
    ground_truth_tools: Any, predicted_tools: Sequence[Dict[str, Any]]
) -> bool:
    """True iff every gold EFS group has at least one predicted member."""
    pred = predicted_node_set(predicted_tools)
    groups = gold_efs_groups(ground_truth_tools)
    if not groups:
        return len(pred) == 0
    return all(bool(group & pred) for group in groups)


def official_union_node_set_em(
    ground_truth_tools: Any, predicted_tools: Sequence[Dict[str, Any]]
) -> bool:
    return gold_union_nodes(ground_truth_tools) == predicted_node_set(predicted_tools)


def cross_server_dependency_correct(
    *,
    ground_truth_tools: Any,
    predicted_nodes: Sequence[Dict[str, Any]],
) -> bool:
    dep = dependency_accuracy(
        level=4,
        ground_truth_tools=ground_truth_tools,
        predicted_nodes=predicted_nodes,
    )
    acc = dep.get("dependency_edge_accuracy")
    return acc is not None and float(acc) >= 1.0 - 1e-12


def complete_route_success(
    *,
    ground_truth_tools: Any,
    predicted_tools: Sequence[Dict[str, Any]],
    predicted_nodes: Sequence[Dict[str, Any]],
    status: str,
) -> bool:
    if status != "completed":
        return False
    eval_row = evaluate_prediction(4, ground_truth_tools, predicted_tools)
    return bool(eval_row.get("exact_match")) and cross_server_dependency_correct(
        ground_truth_tools=ground_truth_tools,
        predicted_nodes=predicted_nodes,
    )


def evaluate_l4_separate(
    *,
    ground_truth_tools: Any,
    predicted_tools: Sequence[Dict[str, Any]],
    predicted_nodes: Sequence[Dict[str, Any]],
    status: str,
) -> Dict[str, Any]:
    """Return the four L4 metrics separately plus P/R/F1."""
    if status != "completed":
        # Failures stay in the strict denominator with score 0.
        return {
            "official_union_node_set_em": False,
            "efs_covering_node_set": False,
            "cross_server_dependency_correctness": False,
            "complete_route_success": False,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "exact_match": False,
            "strict_score": 0.0,
            "gold_dependency_edges": len(gold_dependency_edges(ground_truth_tools)),
        }
    eval_row = evaluate_prediction(4, ground_truth_tools, predicted_tools)
    union_em = official_union_node_set_em(ground_truth_tools, predicted_tools)
    efs_cover = efs_covering_node_set(ground_truth_tools, predicted_tools)
    dep_ok = cross_server_dependency_correct(
        ground_truth_tools=ground_truth_tools,
        predicted_nodes=predicted_nodes,
    )
    complete = bool(eval_row.get("exact_match")) and dep_ok
    return {
        "official_union_node_set_em": union_em,
        "efs_covering_node_set": efs_cover,
        "cross_server_dependency_correctness": dep_ok,
        "complete_route_success": complete,
        "precision": float(eval_row.get("precision") or 0.0),
        "recall": float(eval_row.get("recall") or 0.0),
        "f1": float(eval_row.get("f1") or 0.0),
        "exact_match": bool(eval_row.get("exact_match")),
        "strict_score": 1.0 if complete else 0.0,
        "gold_dependency_edges": len(gold_dependency_edges(ground_truth_tools)),
        "eval_status": eval_row.get("status"),
    }


def strict_level_rates(
    rows: Sequence[Dict[str, Any]], *, level: int, key: str, expected_n: int = 20
) -> Dict[str, Any]:
    """Strict denominator = expected_n (failures score 0; never removed)."""
    group = [r for r in rows if int(r.get("level") or 0) == level]
    # Dedup by task_id last-write-wins
    latest: Dict[str, Dict[str, Any]] = {}
    for row in group:
        latest[row["task_id"]] = row
    ordered = list(latest.values())
    successes = 0
    for row in ordered:
        if row.get("status") != "completed":
            continue
        if row.get(key) is True:
            successes += 1
    denom = expected_n
    return {
        "level": level,
        "metric": key,
        "successes": successes,
        "strict_denominator": denom,
        "strict_rate": successes / denom if denom else None,
        "observed_tasks": len(ordered),
        "completed": sum(1 for r in ordered if r.get("status") == "completed"),
        "failed_scored_zero": sum(1 for r in ordered if r.get("status") != "completed"),
    }
