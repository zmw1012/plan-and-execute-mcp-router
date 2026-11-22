"""Deterministic conversion from the official ETOM registry to OperationSpec.

The raw registry file is never written. Pseudo-output schemas are not
synthesized: the pinned registry has no output schema.
"""

from __future__ import annotations

import copy
import json
import re
from typing import Any, Dict, List, Tuple

from hk_equity_mcp.operation_resolver.schemas import OperationSpec, ParameterSpec

PSEUDO_OUTPUT_METADATA: Dict[str, Any] = {
    "available": False,
    "official_field": None,
    "reason": (
        "Pinned mcp_registry.json stores tool_name, tool_description, and "
        "input_schema only. Official generate_pseudo_output_schema.py was not "
        "run, because it would synthesize schemas with an LLM."
    ),
}

_NAME_RE = re.compile(r"[A-Za-z0-9]+")
_TYPE_MAP = {
    "string": "string",
    "number": "number",
    "integer": "number",
    "boolean": "boolean",
    "object": "object",
    "array": "array",
}


def operation_id(server_name: str, tool_name: str) -> str:
    return f"{server_name}::{tool_name}"


def _json_type(schema: Dict[str, Any]) -> str:
    raw = schema.get("type", "string")
    if isinstance(raw, list):
        for item in raw:
            if item in _TYPE_MAP:
                return _TYPE_MAP[item]
        return "string"
    return _TYPE_MAP.get(str(raw), "string")


def parameters_from_input_schema(schema: Any) -> List[ParameterSpec]:
    if not isinstance(schema, dict):
        return []
    properties = schema.get("properties") or {}
    if not isinstance(properties, dict):
        return []
    required = set(schema.get("required") or [])
    params: List[ParameterSpec] = []
    for name, spec in properties.items():
        spec = spec if isinstance(spec, dict) else {}
        params.append(
            ParameterSpec(
                name=str(name),
                type=_json_type(spec),
                required=str(name) in required,
                description=str(spec.get("description") or ""),
                enum=list(spec["enum"]) if isinstance(spec.get("enum"), list) else None,
                default=spec.get("default"),
            )
        )
    return params


def action_terms(tool_name: str) -> List[str]:
    return _NAME_RE.findall(tool_name.replace("_", " "))


def convert_tool(server: Dict[str, Any], tool: Dict[str, Any]) -> Tuple[OperationSpec, Dict[str, Any]]:
    server_name = str(server["server_name"])
    tool_name = str(tool["tool_name"])
    description = str(tool.get("tool_description") or "")
    input_schema = copy.deepcopy(tool.get("input_schema") or {})
    categories = [str(item) for item in (server.get("categories") or [])]
    field_names = []
    if isinstance(input_schema, dict) and isinstance(input_schema.get("properties"), dict):
        field_names = [str(name) for name in input_schema["properties"].keys()]
    spec = OperationSpec(
        operation_id=operation_id(server_name, tool_name),
        service_id=server_name,
        name=tool_name,
        summary=description,
        description=description,
        action_terms=action_terms(tool_name),
        parameters=parameters_from_input_schema(input_schema),
        read_only=True,
        depends_on=[],
        constraints=[
            f"server_summary: {server.get('server_summary') or ''}",
            f"categories: {', '.join(categories)}",
        ],
        invocation_guidance="Input fields: " + ", ".join(field_names),
        underlying_tool=tool_name,
    )
    official = {
        "operation_id": spec.operation_id,
        "server_name": server_name,
        "server_description": server.get("server_description") or "",
        "server_summary": server.get("server_summary") or "",
        "categories": categories,
        "tool_name": tool_name,
        "description": description,
        "input_schema": input_schema,
        "pseudo_output_metadata": copy.deepcopy(PSEUDO_OUTPUT_METADATA),
    }
    return spec, official


def convert_registry(payload: Dict[str, Any]) -> Tuple[Dict[str, OperationSpec], Dict[str, Dict[str, Any]]]:
    specs: Dict[str, OperationSpec] = {}
    official: Dict[str, Dict[str, Any]] = {}
    for server in payload.get("servers") or []:
        for tool in server.get("tools") or []:
            spec, view = convert_tool(server, tool)
            if spec.operation_id in specs:
                raise ValueError(f"Duplicate ETOM operation_id {spec.operation_id}")
            specs[spec.operation_id] = spec
            official[spec.operation_id] = view
    return specs, official


def derived_registry_document(
    specs: Dict[str, OperationSpec], official: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """Serializable derived view. Callers write this outside the raw checkout."""
    operations = []
    for op_id in sorted(official):
        view = official[op_id]
        operations.append(
            {
                "operation_id": op_id,
                "server_name": view["server_name"],
                "tool_name": view["tool_name"],
                "description": view["description"],
                "input_schema": view["input_schema"],
                "pseudo_output_metadata": view["pseudo_output_metadata"],
                "index_document": specs[op_id].index_document(),
            }
        )
    return {
        "conversion": "etom_operation_visible_registry_v1",
        "fidelity": "Operation-Visible adapted for ETOM routing-only evaluation",
        "operation_count": len(operations),
        "operations": operations,
    }


def registry_fingerprint(official: Dict[str, Dict[str, Any]]) -> str:
    payload = json.dumps(
        [
            {
                "operation_id": op_id,
                "description": official[op_id]["description"],
                "input_schema": official[op_id]["input_schema"],
            }
            for op_id in sorted(official)
        ],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    import hashlib

    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
