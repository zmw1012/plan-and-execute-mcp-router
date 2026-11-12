"""ETOM-compatible EM / Precision / Recall / F1.

The graph score follows eval/graph_evaluator.py and the level strategies in
eval/eval.py at the pinned commit. It scores a predicted tool route. It does
not score a natural-language answer.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple


def _normalize_ground_truth_tool(tool_candidate: Any) -> Optional[Dict[str, Any]]:
    if isinstance(tool_candidate, str):
        try:
            tool_candidate = json.loads(tool_candidate)
        except json.JSONDecodeError:
            return None
    if isinstance(tool_candidate, dict):
        server = tool_candidate.get("server_name", tool_candidate.get("server", ""))
        tool_name = tool_candidate.get("tool_name", "")
        dependencies = tool_candidate.get("dependencies", []) or []
        return {
            "server_name": server,
            "tool_name": tool_name,
            "dependencies": dependencies,
        }
    return None


def _format_simple_ground_truth(ground_truth_tools: Sequence[Any]) -> List[List[Dict[str, Any]]]:
    formatted: List[List[Dict[str, Any]]] = []
    for entry in ground_truth_tools:
        if isinstance(entry, list):
            group = []
            for option in entry:
                normalized = _normalize_ground_truth_tool(option)
                if normalized:
                    group.append(normalized)
            if group:
                formatted.append(group)
        else:
            normalized = _normalize_ground_truth_tool(entry)
            if normalized:
                formatted.append([normalized])
    return formatted


def _format_graph_ground_truth(
    ground_truth_tools: Sequence[Any], allow_efs: bool = False
) -> List[List[Dict[str, Any]]]:
    formatted: List[List[Dict[str, Any]]] = []
    for entry in ground_truth_tools:
        if allow_efs and isinstance(entry, list):
            group = []
            for option in entry:
                normalized = _normalize_ground_truth_tool(option)
                if normalized:
                    group.append(normalized)
            if group:
                formatted.append(group)
        else:
            normalized = _normalize_ground_truth_tool(entry)
            if normalized:
                formatted.append([normalized])
    return formatted


def _prediction_strings(prediction: Sequence[Any]) -> List[str]:
    raw: List[str] = []
    for item in prediction:
        if isinstance(item, str):
            raw.append(item)
        elif isinstance(item, dict):
            server = item.get("server_name", item.get("server", ""))
            tool_name = item.get("tool_name", "")
            raw.append(
                json.dumps(
                    {"server_name": server, "server": server, "tool_name": tool_name},
                    ensure_ascii=False,
                )
            )
        else:
            raw.append(json.dumps({"server_name": "", "tool_name": str(item)}))
    return raw


def _parse_prediction(raw_prediction_list: Sequence[str]) -> List[Tuple[str, str]]:
    predicted = []
    for step_str in raw_prediction_list:
        try:
            step = json.loads(step_str)
        except json.JSONDecodeError:
            continue
        server = step.get("server_name", step.get("server", ""))
        predicted.append((server, step.get("tool_name", "")))
    return predicted


def _parse_ground_truth(ground_truth_tools: Sequence[Sequence[Dict[str, Any]]]) -> Dict[int, Dict[str, Any]]:
    graph: Dict[int, Dict[str, Any]] = {}
    for step_idx, tool_step in enumerate(ground_truth_tools):
        if not tool_step:
            continue
        efs: Set[Tuple[str, str]] = set()
        dependencies: Set[int] = set()
        for tool in tool_step:
            server = tool.get("server_name", tool.get("server", ""))
            efs.add((server, tool.get("tool_name", "")))
            for dep_idx in tool.get("dependencies", []) or []:
                dependencies.add(dep_idx)
        graph[step_idx] = {"efs_tools": efs, "dependencies": dependencies}
    return graph


def _graph_scores(
    predicted: List[Tuple[str, str]], gt_graph: Dict[int, Dict[str, Any]]
) -> Dict[str, Any]:
    total_predictions = len(predicted)
    total_gt_nodes = len(gt_graph)
    pred_to_gt: Dict[int, int] = {}
    used_gt = set()
    for pred_pos, pred_tool in enumerate(predicted):
        for gt_node_id, node_info in gt_graph.items():
            if gt_node_id not in used_gt and pred_tool in node_info["efs_tools"]:
                pred_to_gt[pred_pos] = gt_node_id
                used_gt.add(gt_node_id)
                break
    gt_to_pred: Dict[int, int] = {}
    for pred_pos, gt_node_id in pred_to_gt.items():
        gt_to_pred.setdefault(gt_node_id, pred_pos)
    matched_gt = set(pred_to_gt.values())
    recall = len(matched_gt) / total_gt_nodes if total_gt_nodes else 0.0
    correctly_placed = 0
    for pred_pos, _pred_tool in enumerate(predicted):
        if pred_pos not in pred_to_gt:
            continue
        gt_node_id = pred_to_gt[pred_pos]
        required = gt_graph[gt_node_id]["dependencies"]
        ok = True
        for dep_node_id in required:
            if dep_node_id not in gt_to_pred or gt_to_pred[dep_node_id] >= pred_pos:
                ok = False
                break
        if ok:
            correctly_placed += 1
    precision = correctly_placed / total_predictions if total_predictions else 0.0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_match": f1 == 1.0,
        "tp": correctly_placed,
        "fp": total_predictions - correctly_placed,
        "fn": total_gt_nodes - len(matched_gt),
        "total_predictions": total_predictions,
        "total_gt_nodes": total_gt_nodes,
    }


def _evaluate_graph(
    formatted_gt: List[List[Dict[str, Any]]], raw_prediction: Sequence[str]
) -> Dict[str, Any]:
    if not raw_prediction and not formatted_gt:
        return {
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "exact_match": True,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "total_predictions": 0,
            "total_gt_nodes": 0,
        }
    if not raw_prediction:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "exact_match": False,
            "tp": 0,
            "fp": 0,
            "fn": len(formatted_gt),
            "total_predictions": 0,
            "total_gt_nodes": len(formatted_gt),
        }
    if not formatted_gt:
        return {
            "precision": 0.0,
            "recall": 1.0,
            "f1": 0.0,
            "exact_match": False,
            "tp": 0,
            "fp": len(raw_prediction),
            "fn": 0,
            "total_predictions": len(raw_prediction),
            "total_gt_nodes": 0,
        }
    return _graph_scores(_parse_prediction(raw_prediction), _parse_ground_truth(formatted_gt))


def _tool_info(raw_prediction: Sequence[str]) -> List[Dict[str, str]]:
    info = []
    for item in raw_prediction:
        try:
            payload = json.loads(item)
        except json.JSONDecodeError:
            info.append({"server": "parse_error", "tool_name": item})
            continue
        info.append(
            {
                "server": payload.get("server", "") or payload.get("server_name", ""),
                "tool_name": payload.get("tool_name", ""),
            }
        )
    return info


def _graph_level_metrics(
    level: int, ground_truth_tools: Sequence[Any], raw_prediction: Sequence[str]
) -> Dict[str, Any]:
    allow_efs = level == 4
    formatted = _format_graph_ground_truth(ground_truth_tools, allow_efs=allow_efs)
    flat = [tool for group in formatted for tool in group]
    if not raw_prediction:
        if not flat:
            return {
                "status": "conversational",
                "exact_match": True,
                "precision": 1.0,
                "recall": 1.0,
                "f1": 1.0,
                "graph_f1_score": 100.0,
                "graph_precision": 100.0,
                "graph_recall": 100.0,
            }
        return {
            "status": "tool_graph",
            "exact_match": False,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "graph_f1_score": 0.0,
            "graph_precision": 0.0,
            "graph_recall": 0.0,
        }
    scores = _evaluate_graph(formatted, raw_prediction)
    predicted_info = _tool_info(raw_prediction)
    pred_nodes = {
        (item["server"], item["tool_name"])
        for item in predicted_info
        if item["server"] and item["tool_name"]
    }
    gt_nodes = {
        (tool.get("server_name", ""), tool.get("tool_name", ""))
        for group in formatted
        for tool in group
        if tool.get("server_name") and tool.get("tool_name")
    }
    node_set_match = pred_nodes == gt_nodes
    is_gt_parallel = all(not tool.get("dependencies") for tool in flat) if flat else True
    parallel_aware_match = is_gt_parallel and node_set_match
    exact_match = bool(scores["exact_match"] or parallel_aware_match)
    return {
        "status": "tool_chain_efs" if level == 4 else "tool_graph",
        "exact_match": exact_match,
        "precision": scores["precision"],
        "recall": scores["recall"],
        "f1": scores["f1"],
        "graph_f1_score": scores["f1"] * 100,
        "graph_precision": scores["precision"] * 100,
        "graph_recall": scores["recall"] * 100,
        "graph_structure_match": scores["exact_match"],
        "node_set_match": node_set_match,
        "parallel_aware_match": parallel_aware_match,
        "tp": scores["tp"],
        "fp": scores["fp"],
        "fn": scores["fn"],
    }


def _level5_metrics(raw_prediction: Sequence[str]) -> Dict[str, Any]:
    if not raw_prediction:
        return {
            "status": "correct_no_tools",
            "exact_match": True,
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
        }
    if (
        len(raw_prediction) == 1
        and raw_prediction[0].strip().lower() == "no"
    ):
        return {
            "status": "correct_no_tools",
            "exact_match": True,
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
        }
    no_tools = 0
    actual = 0
    for item in raw_prediction:
        try:
            payload = json.loads(item)
        except json.JSONDecodeError:
            actual += 1
            continue
        server = str(payload.get("server") or payload.get("server_name") or "").lower()
        tool_name = str(payload.get("tool_name") or "").lower()
        if server == "no" and tool_name == "no":
            no_tools += 1
        else:
            actual += 1
    precision = no_tools / len(raw_prediction) if raw_prediction else 1.0
    recall = 1.0 if actual == 0 else 0.0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {
        "status": "correct_no_tools" if actual == 0 else "incorrect_tools_predicted",
        "exact_match": actual == 0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": no_tools,
        "fp": actual,
        "fn": 0,
    }


def evaluate_prediction(
    level: int, ground_truth_tools: Any, prediction: Sequence[Any]
) -> Dict[str, Any]:
    """Score one routed prediction with the pinned ETOM rules.

    ``prediction`` is an ordered list of ``{server_name, tool_name}`` dicts.
    An empty list is the infeasible / no-tool route.
    """
    if level not in (1, 2, 3, 4, 5):
        raise ValueError(f"Unsupported ETOM level: {level}")
    ground_truth_tools = ground_truth_tools or []
    raw = _prediction_strings(prediction)
    if level in (1, 2):
        if not raw:
            return {
                "level": level,
                "status": "empty_result_error",
                "exact_match": False,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "predicted_tools": [],
            }
        formatted = _format_simple_ground_truth(ground_truth_tools)
        scores = _evaluate_graph(formatted, raw)
        status = "tool_call"
        if len(raw) > 1:
            status = "multi_tool_error"
        return {
            "level": level,
            "status": status,
            "exact_match": scores["exact_match"],
            "precision": scores["precision"],
            "recall": scores["recall"],
            "f1": scores["f1"],
            "tp": scores["tp"],
            "fp": scores["fp"],
            "fn": scores["fn"],
            "predicted_tools": _tool_info(raw),
        }
    if level in (3, 4):
        metrics = _graph_level_metrics(level, ground_truth_tools, raw)
        metrics["level"] = level
        metrics["predicted_tools"] = _tool_info(raw)
        return metrics
    metrics = _level5_metrics(raw)
    metrics["level"] = level
    metrics["predicted_tools"] = _tool_info(raw) if raw else []
    return metrics
