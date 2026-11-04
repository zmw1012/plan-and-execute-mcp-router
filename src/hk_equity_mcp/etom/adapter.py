"""Operation-Visible meta-tools over the ETOM registry.

invoke_api records a predicted route node. It does not call an ETOM MCP tool
and does not invent a business payload.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional

from hk_equity_mcp.etom.convert import PSEUDO_OUTPUT_METADATA
from hk_equity_mcp.etom.embeddings import CatalogIndex
from hk_equity_mcp.operation_resolver.schemas import OperationSpec

META_TOOLS = ("find_operations", "describe_operation", "invoke_api")
FIND_ALIASES = {"find_operations", "find_operation"}


class EtomRoutingResolver:
    def __init__(
        self,
        registry: Dict[str, OperationSpec],
        official: Dict[str, Dict[str, Any]],
        index: CatalogIndex,
        embedder,
        *,
        lexical_weight: float = 0.12,
        top_k_default: int = 20,
        top_k_max: int = 20,
    ):
        self.registry = registry
        self.official = official
        self.index = index
        self.embedder = embedder
        self.lexical_weight = lexical_weight
        self.top_k_default = top_k_default
        self.top_k_max = top_k_max
        self.real_tool_executions = 0
        self.predicted_nodes: List[Dict[str, Any]] = []

    def reset_route(self) -> None:
        self.predicted_nodes = []

    def find_operations(self, query: str, top_k: int = 20) -> Dict[str, Any]:
        if not isinstance(query, str) or not query.strip():
            return {"success": False, "error": "query must be a non-empty string", "candidates": []}
        capped = self.top_k_default if top_k is None else min(int(top_k), self.top_k_max)
        if capped < 1:
            return {"success": False, "error": "top_k must be >= 1", "candidates": []}
        candidates = self.index.rank(
            query.strip(),
            self.registry,
            self.embedder,
            top_k=capped,
            lexical_weight=self.lexical_weight,
        )
        return {
            "success": True,
            "query": query.strip(),
            "top_k": capped,
            "catalog_size": len(self.registry),
            "candidates": candidates,
            "error": None,
        }

    def describe_operation(self, operation_id: str) -> Dict[str, Any]:
        official = self.official.get(operation_id)
        if official is None:
            return {
                "success": False,
                "operation_id": operation_id,
                "error": f"Unknown operation_id: {operation_id}",
            }
        return {
            "success": True,
            "operation_id": operation_id,
            "name": official["tool_name"],
            "server_name": official["server_name"],
            "description": official["description"],
            "input_schema": copy.deepcopy(official["input_schema"]),
            "pseudo_output_metadata": copy.deepcopy(
                official.get("pseudo_output_metadata") or PSEUDO_OUTPUT_METADATA
            ),
            "error": None,
        }

    def invoke_api(
        self,
        operation_id: str,
        arguments: Optional[Dict[str, Any]] = None,
        depends_on: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        if self.real_tool_executions != 0:
            raise RuntimeError("ETOM routing mode recorded an unexpected tool execution.")
        official = self.official.get(operation_id)
        if official is None:
            return {
                "success": False,
                "executed": False,
                "execution_mode": "etom_routing_only",
                "operation_id": operation_id,
                "business_result": None,
                "error": f"Unknown operation_id: {operation_id}",
            }
        order = len(self.predicted_nodes)
        depends_on_orders = self._normalize_depends_on(depends_on, order)
        edges = [[source, order] for source in depends_on_orders]
        node = {
            "order": order,
            "operation_id": operation_id,
            "server_name": official["server_name"],
            "tool_name": official["tool_name"],
            "depends_on_orders": depends_on_orders,
            "dependency_edges": edges,
        }
        self.predicted_nodes.append(node)
        recorded_arguments = dict(arguments or {})
        recorded_arguments.pop("depends_on", None)
        return {
            "success": True,
            "executed": False,
            "execution_mode": "etom_routing_only",
            "operation_id": operation_id,
            "server_name": official["server_name"],
            "name": official["tool_name"],
            "tool_name": official["tool_name"],
            "order": order,
            "depends_on_orders": depends_on_orders,
            "dependency_edges": edges,
            "recorded_arguments": recorded_arguments,
            "business_result": None,
            "pseudo_output_metadata": copy.deepcopy(
                official.get("pseudo_output_metadata") or PSEUDO_OUTPUT_METADATA
            ),
            "note": (
                "No ETOM MCP tool was executed and no business result was produced. "
                "This response only records a predicted route node."
            ),
            "error": None,
        }

    def call_meta(self, tool_name: str, **kwargs: Any) -> Dict[str, Any]:
        if tool_name in FIND_ALIASES:
            return self.find_operations(**kwargs)
        if tool_name == "describe_operation":
            return self.describe_operation(**kwargs)
        if tool_name == "invoke_api":
            return self.invoke_api(**kwargs)
        return {
            "success": False,
            "executed": False,
            "business_result": None,
            "error": (
                f"Refused tool '{tool_name}'. ETOM routing mode exposes only "
                f"{list(META_TOOLS)} and does not execute MCP tools."
            ),
        }

    def _normalize_depends_on(self, depends_on: Optional[List[Any]], order: int) -> List[int]:
        if not depends_on:
            return []
        resolved = []
        by_operation = {node["operation_id"]: node["order"] for node in self.predicted_nodes}
        by_tool = {
            (node["server_name"], node["tool_name"]): node["order"]
            for node in self.predicted_nodes
        }
        for item in depends_on:
            if isinstance(item, int) and 0 <= item < order:
                resolved.append(item)
            elif isinstance(item, str) and item in by_operation:
                resolved.append(by_operation[item])
            elif isinstance(item, dict):
                key = (
                    str(item.get("server_name") or item.get("server") or ""),
                    str(item.get("tool_name") or item.get("name") or ""),
                )
                if key in by_tool:
                    resolved.append(by_tool[key])
        return sorted(set(resolved))
