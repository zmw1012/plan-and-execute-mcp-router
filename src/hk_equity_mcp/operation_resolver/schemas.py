"""Schemas for Operation-Resolver operations and responses."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ParameterSpec:
    name: str
    type: str  # string | number | boolean | object | array
    required: bool = False
    description: str = ""
    enum: Optional[List[Any]] = None
    default: Any = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class OperationSpec:
    """One underlying executable operation in the local registry."""

    operation_id: str
    service_id: str
    name: str
    summary: str
    description: str
    action_terms: List[str]
    parameters: List[ParameterSpec]
    read_only: bool = True
    depends_on: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    invocation_guidance: str = ""
    # Underlying callable name on the domain server
    underlying_tool: str = ""

    def required_params(self) -> List[ParameterSpec]:
        return [p for p in self.parameters if p.required]

    def optional_params(self) -> List[ParameterSpec]:
        return [p for p in self.parameters if not p.required]

    def param_map(self) -> Dict[str, ParameterSpec]:
        return {p.name: p for p in self.parameters}

    def index_document(self) -> str:
        """Text used for embedding + lexical retrieval (paper §4.1 fields)."""
        req = ", ".join(f"{p.name}:{p.type}" for p in self.required_params())
        opt = ", ".join(f"{p.name}:{p.type}" for p in self.optional_params())
        actions = " ".join(self.action_terms)
        constraints = "; ".join(self.constraints)
        return (
            f"summary: {self.summary}\n"
            f"service: {self.service_id}\n"
            f"operation: {self.name}\n"
            f"action_terms: {actions}\n"
            f"required_parameters: {req}\n"
            f"optional_parameters: {opt}\n"
            f"constraints: {constraints}\n"
            f"description: {self.description}\n"
            f"invocation_guidance: {self.invocation_guidance}"
        )

    def contract(self) -> Dict[str, Any]:
        return {
            "operation_id": self.operation_id,
            "service_id": self.service_id,
            "name": self.name,
            "summary": self.summary,
            "description": self.description,
            "required_parameters": [p.to_dict() for p in self.required_params()],
            "optional_parameters": [p.to_dict() for p in self.optional_params()],
            "json_schema": self.json_schema(),
            "constraints": list(self.constraints),
            "invocation_guidance": self.invocation_guidance,
            "read_only": self.read_only,
            "depends_on": list(self.depends_on),
            "action_terms": list(self.action_terms),
        }

    def json_schema(self) -> Dict[str, Any]:
        props: Dict[str, Any] = {}
        required: List[str] = []
        for p in self.parameters:
            schema: Dict[str, Any] = {"type": p.type, "description": p.description}
            if p.enum is not None:
                schema["enum"] = list(p.enum)
            if p.default is not None:
                schema["default"] = p.default
            props[p.name] = schema
            if p.required:
                required.append(p.name)
        return {
            "type": "object",
            "properties": props,
            "required": required,
            "additionalProperties": False,
        }
