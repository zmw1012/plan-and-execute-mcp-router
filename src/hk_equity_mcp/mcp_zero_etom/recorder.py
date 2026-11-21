"""Routing-only route recorder (no real MCP execution)."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Set


class EtomRouteRecorder:
    """Records predicted route nodes. ``real_tool_executions`` is always 0."""

    def __init__(
        self,
        official: Dict[str, Dict[str, Any]],
        retrieved_ids: Optional[Callable[[], Set[str]]] = None,
    ):
        self.official = official
        self._retrieved_ids = retrieved_ids
        self._retrieved_snapshot: Set[str] = set()
        self.predicted_nodes: List[Dict[str, Any]] = []
        self.real_tool_executions = 0

    def reset(self) -> None:
        self.predicted_nodes = []
        self.real_tool_executions = 0
        self._retrieved_snapshot = set()

    def set_retrieved(self, ids: Set[str]) -> None:
        self._retrieved_snapshot = set(ids)

    def retrieved(self) -> Set[str]:
        if self._retrieved_ids is not None:
            return set(self._retrieved_ids())
        return set(self._retrieved_snapshot)

    def invoke(
        self,
        operation_id: str,
        arguments: Optional[dict] = None,
        depends_on: Optional[list] = None,
    ) -> Dict[str, Any]:
        oid = str(operation_id)
        if oid not in self.official:
            return {
                "success": False,
                "executed": False,
                "business_result": None,
                "real_tool_executions": 0,
                "error": f"unknown operation_id {oid!r}",
                "error_type": "unknown_operation",
            }
        if oid not in self.retrieved():
            return {
                "success": False,
                "executed": False,
                "business_result": None,
                "real_tool_executions": 0,
                "error": f"operation_id {oid!r} was never retrieved",
                "error_type": "not_retrieved",
            }
        order = len(self.predicted_nodes)
        deps: List[int] = []
        for dep in depends_on or []:
            deps.append(int(dep))
        for dep in deps:
            if dep < 0 or dep >= order:
                return {
                    "success": False,
                    "executed": False,
                    "business_result": None,
                    "real_tool_executions": 0,
                    "error": f"depends_on index {dep} out of range for order={order}",
                    "error_type": "invalid_depends_on",
                }
        row = self.official[oid]
        node = {
            "order": order,
            "operation_id": oid,
            "server_name": row["server_name"],
            "tool_name": row["tool_name"],
            "depends_on": deps,
            "depends_on_orders": deps,
            "dependency_edges": [[dep, order] for dep in deps],
            "arguments": dict(arguments or {}),
        }
        self.predicted_nodes.append(node)
        return {
            "success": True,
            "executed": False,
            "business_result": None,
            "real_tool_executions": 0,
            "recorded_node": node,
            "note": "route node recorded; no real MCP tool execution",
        }

    @property
    def predicted_tools(self) -> List[Dict[str, str]]:
        return [
            {"server_name": n["server_name"], "tool_name": n["tool_name"]}
            for n in self.predicted_nodes
        ]
