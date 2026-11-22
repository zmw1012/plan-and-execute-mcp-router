"""Load frozen Financial Real-MCP catalog YAML into OperationSpec registry."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from hk_equity_mcp.operation_resolver.schemas import OperationSpec, ParameterSpec

DEFAULT_CATALOG_PATH = (
    Path(__file__).resolve().parents[3]
    / "configs"
    / "operations"
    / "financial_real_mcp.yaml"
)


def load_real_mcp_catalog(path: Optional[Path] = None) -> Dict[str, Any]:
    cfg_path = Path(path) if path else DEFAULT_CATALOG_PATH
    with cfg_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data.get("operations"), list):
        raise ValueError(f"Invalid catalog (operations missing): {cfg_path}")
    return data


def _param_from_yaml(raw: Dict[str, Any]) -> ParameterSpec:
    return ParameterSpec(
        name=str(raw["name"]),
        type=str(raw.get("type", "string")),
        required=bool(raw.get("required", False)),
        description=str(raw.get("description", "")),
        enum=raw.get("enum"),
        default=raw.get("default"),
    )


def operation_from_yaml(raw: Dict[str, Any]) -> OperationSpec:
    params = [_param_from_yaml(p) for p in (raw.get("parameters") or [])]
    risks = list(raw.get("known_risks") or [])
    constraints = list(raw.get("constraints") or [])
    if risks:
        constraints = constraints + [f"risk: {r}" for r in risks]
    role = raw.get("role") or ""
    if role:
        constraints = [f"role: {role}"] + constraints
    underlying = raw.get("exposed_mcp_tool") or raw.get("name") or ""
    return OperationSpec(
        operation_id=str(raw["operation_id"]),
        service_id=str(raw["server_id"]),
        name=str(raw.get("name") or raw["operation_id"]),
        summary=str(raw.get("summary") or ""),
        description=str(raw.get("description") or ""),
        action_terms=list(raw.get("action_terms") or []),
        parameters=params,
        read_only=bool(raw.get("read_only", True)),
        constraints=constraints,
        invocation_guidance=str(raw.get("invocation_guidance") or ""),
        underlying_tool=str(underlying),
        role=role,
        audit_status=str(raw.get("audit_status") or ""),
        known_risks=risks,
        exposed_mcp_tool=str(raw.get("exposed_mcp_tool") or ""),
        underlying_aktools_interface=raw.get("underlying_aktools_interface"),
        execution=dict(raw.get("execution") or {}),
    )


def load_real_mcp_registry(path: Optional[Path] = None) -> Dict[str, OperationSpec]:
    catalog = load_real_mcp_catalog(path)
    ops = [operation_from_yaml(raw) for raw in catalog["operations"]]
    registry = {op.operation_id: op for op in ops}
    if len(registry) != len(ops):
        raise ValueError("Duplicate operation_id in financial_real_mcp.yaml")
    return registry


def catalog_operation_records(path: Optional[Path] = None) -> List[Dict[str, Any]]:
    return list(load_real_mcp_catalog(path)["operations"])
