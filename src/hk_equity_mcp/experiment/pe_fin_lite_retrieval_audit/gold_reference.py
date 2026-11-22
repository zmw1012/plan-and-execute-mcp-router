"""Evaluation-only gold / accepted-operation reference.

NEVER feed outputs of this module into Planner prompts, step queries,
operation documents, Binder, or Agent runtime predictions.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Set


NO_TOOL_ANSWERABILITY = frozenset(
    {
        "should_refuse_or_clarify",
        "intentionally_infeasible",
        "capability_gap",
    }
)

DISCLOSURE_OP_PREFIXES = (
    "hkexnews.",
    "ah_disclosure.",
)


def is_no_tool_task(task: Dict[str, Any]) -> bool:
    ac = str(task.get("answerability_class") or "").strip()
    return ac in NO_TOOL_ANSWERABILITY


def is_disclosure_operation(operation_id: str) -> bool:
    oid = str(operation_id or "")
    return any(oid.startswith(p) for p in DISCLOSURE_OP_PREFIXES)


def _as_str_list(seq: Any) -> List[str]:
    if not isinstance(seq, list):
        return []
    return [str(x) for x in seq if x is not None and str(x).strip()]


def acceptable_routes(task: Dict[str, Any]) -> List[List[str]]:
    """Return ordered acceptable operation routes (evaluation-only)."""
    sets = task.get("acceptable_tool_sets")
    routes: List[List[str]] = []
    if isinstance(sets, list) and sets:
        for item in sets:
            route = _as_str_list(item)
            if route:
                routes.append(route)
    gold = _as_str_list(task.get("gold_operations"))
    if gold and gold not in routes:
        routes.insert(0, gold)
    # Deduplicate while preserving order.
    seen = set()
    out: List[List[str]] = []
    for route in routes:
        key = tuple(route)
        if key in seen:
            continue
        seen.add(key)
        out.append(route)
    return out


def primary_route(task: Dict[str, Any]) -> List[str]:
    routes = acceptable_routes(task)
    if routes:
        return list(routes[0])
    return _as_str_list(task.get("gold_operations"))


def acceptable_ops_union(task: Dict[str, Any]) -> Set[str]:
    out: Set[str] = set()
    for route in acceptable_routes(task):
        out.update(route)
    return out


def equivalents_by_node(task: Dict[str, Any]) -> List[Set[str]]:
    """Per primary-route node, the set of acceptable equivalent operations.

    - Same-index ops across equal-length alternative routes are equivalents.
    - Singleton alternative routes are treated as full-route substitutes
      (any singleton op may satisfy a single-node requirement when all
      routes are length-1).
    """
    routes = acceptable_routes(task)
    if not routes:
        return []
    primary = routes[0]
    if all(len(r) == 1 for r in routes):
        # Any accepted singleton satisfies the (single) required node.
        return [{op for r in routes for op in r}]
    node_sets: List[Set[str]] = [{op} for op in primary]
    for route in routes[1:]:
        if len(route) == len(primary):
            for i, op in enumerate(route):
                node_sets[i].add(op)
            continue
        # Unequal-length alternative: attach each alt op to matching primary
        # node id or first same-server-family node.
        for op in route:
            placed = False
            for i, pop in enumerate(primary):
                if op == pop:
                    node_sets[i].add(op)
                    placed = True
                    break
            if placed:
                continue
            for i, pop in enumerate(primary):
                if pop.split(".", 1)[0] == op.split(".", 1)[0]:
                    node_sets[i].add(op)
                    break
    return node_sets


def domain_branches(task: Dict[str, Any]) -> Dict[str, Set[str]]:
    """Map domain branch → gold ops for cross-domain route coverage."""
    branches: Dict[str, Set[str]] = {}
    for op in primary_route(task):
        if op.startswith("hkexnews.") or op.startswith("ah_disclosure."):
            dom = "news"
        elif "financial" in op or "analysis_indicator" in op or "report_em" in op:
            dom = "financial"
        elif "hist" in op or "spot" in op or "index" in op or "market" in op:
            dom = "market"
        else:
            dom = "general"
        branches.setdefault(dom, set()).add(op)
    # Prefer explicit gold_domains when present.
    gold_domains = task.get("gold_domains") or task.get("domains")
    if isinstance(gold_domains, list) and gold_domains:
        # Keep ops grouped; empty branches still listed for coverage checks.
        for d in gold_domains:
            branches.setdefault(str(d), set())
    return branches


def is_disclosure_composite_task(task: Dict[str, Any]) -> bool:
    route = primary_route(task)
    if not route:
        return False
    disc = [op for op in route if is_disclosure_operation(op)]
    if len(disc) >= 2:
        return True
    # Explicit workflow-style gold chains.
    need = {
        "hkexnews.list_hkexnews_announcements",
        "hkexnews.download_hkexnews_announcement",
    }
    return need.issubset(set(route))


def build_evaluation_reference(task: Dict[str, Any]) -> Dict[str, Any]:
    """Compact evaluation reference card (must stay offline-eval only)."""
    route = primary_route(task)
    nodes = equivalents_by_node(task)
    disclosure = is_disclosure_composite_task(task)
    no_tool = is_no_tool_task(task)
    return {
        "task_id": task.get("task_id") or task.get("id"),
        "domain": task.get("domain"),
        "source_split": task.get("source_split"),
        "difficulty": task.get("project_difficulty_level") or task.get("difficulty"),
        "answerability_class": task.get("answerability_class"),
        "is_no_tool_task": no_tool,
        "is_disclosure_composite": disclosure,
        "include_in_tool_recall_denominator": (not no_tool) and (not disclosure),
        "primary_route": route,
        "acceptable_routes": acceptable_routes(task),
        "acceptable_ops_union": sorted(acceptable_ops_union(task)),
        "equivalents_by_node": [sorted(s) for s in nodes],
        "required_node_count": len(nodes) if nodes else len(route),
        "domain_branches": {k: sorted(v) for k, v in domain_branches(task).items()},
        "evaluation_only": True,
        "forbidden_in_runtime": True,
    }


def node_covered(equivalents: Set[str], ranked_ids: Sequence[str]) -> bool:
    ranked = set(str(x) for x in ranked_ids)
    return bool(equivalents & ranked)


def covered_node_count(
    equivalents_by_node_sets: Sequence[Set[str]],
    ranked_ids: Sequence[str],
) -> int:
    return sum(1 for eq in equivalents_by_node_sets if node_covered(eq, ranked_ids))
