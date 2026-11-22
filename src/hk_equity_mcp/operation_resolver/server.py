"""Operation-Resolver MCP server: find / describe / invoke."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Sequence

from hk_equity_mcp.mcp_servers.financial import FinancialServer
from hk_equity_mcp.operation_resolver.registry import build_default_registry
from hk_equity_mcp.operation_resolver.retrieval import (
    DEFAULT_TOP_K,
    MAX_TOP_K,
    EmbeddingBackend,
    OperationRetriever,
)
from hk_equity_mcp.operation_resolver.schemas import OperationSpec, ParameterSpec

META_TOOLS = ("find_operations", "describe_operation", "invoke_api")


class RealInvokeBackend(Protocol):
    def invoke(self, operation_id: str, arguments: Optional[Dict[str, Any]] = None) -> Any:
        ...


class OperationResolverServer:
    """Deterministic operation resolution boundary (no internal LLM planner)."""

    def __init__(
        self,
        embedding_backend: EmbeddingBackend,
        financial_server: Optional[FinancialServer] = None,
        registry: Optional[Dict[str, OperationSpec]] = None,
        cache_dir: Optional[Path] = None,
        real_mcp_gateway: Optional[RealInvokeBackend] = None,
        *,
        preloaded_doc_vectors: Optional[Dict[str, list]] = None,
        ensure_docs: bool = True,
    ):
        self.registry = registry or build_default_registry()
        self.real_mcp_gateway = real_mcp_gateway
        # Local FinancialServer remains for Controlled Local-Tool Pilot tests only.
        # Real-MCP mode must pass real_mcp_gateway and must not use mock JSON as tool data.
        self.financial = None if real_mcp_gateway is not None else (
            financial_server or FinancialServer()
        )
        self.retriever = OperationRetriever(
            self.registry,
            embedding_backend,
            cache_dir=cache_dir,
            preloaded_doc_vectors=preloaded_doc_vectors,
            ensure_docs=ensure_docs,
        )
        self._servers = {}
        if self.financial is not None:
            self._servers["financial"] = self.financial

    # ---- public meta-tools -------------------------------------------------

    def find_operations(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        *,
        allowed_ids: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        if not isinstance(query, str) or not query.strip():
            return {
                "success": False,
                "error": "query must be a non-empty string",
                "candidates": [],
                "top_k": top_k,
            }
        capped = min(int(top_k), MAX_TOP_K) if top_k is not None else DEFAULT_TOP_K
        candidates = self.retriever.rank(
            query.strip(), top_k=capped, allowed_ids=allowed_ids
        )
        return {
            "success": True,
            "query": query.strip(),
            "top_k": capped,
            "max_top_k": MAX_TOP_K,
            "candidates": candidates,
            "allowed_pool_size": (
                None if allowed_ids is None else len({str(x) for x in allowed_ids})
            ),
            "error": None,
        }

    def describe_operation(self, operation_id: str) -> Dict[str, Any]:
        op = self.registry.get(operation_id)
        if op is None:
            return {
                "success": False,
                "operation_id": operation_id,
                "error": f"Unknown operation_id: {operation_id}",
                "contract": None,
            }
        return {
            "success": True,
            "operation_id": operation_id,
            "contract": op.contract(),
            "error": None,
        }

    def invoke_api(
        self, operation_id: str, arguments: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        arguments = arguments or {}
        op = self.registry.get(operation_id)
        if op is None:
            return {
                "success": False,
                "operation_id": operation_id,
                "error_type": "unknown_operation",
                "error": f"Unknown operation_id: {operation_id}",
                "result": None,
            }
        validation_error = self._validate_arguments(op, arguments)
        if validation_error is not None:
            return {
                "success": False,
                "operation_id": operation_id,
                "error_type": "validation_error",
                "error": validation_error,
                "result": None,
            }

        # Apply defaults for optional params with defaults.
        call_args = dict(arguments)
        for p in op.optional_params():
            if p.name not in call_args and p.default is not None:
                call_args[p.name] = p.default

        if self.real_mcp_gateway is not None:
            try:
                mcp_result = self.real_mcp_gateway.invoke(operation_id, call_args)
            except Exception as exc:  # noqa: BLE001
                return {
                    "success": False,
                    "operation_id": operation_id,
                    "error_type": "execution_error",
                    "error": f"Real MCP invoke failed: {type(exc).__name__}: {exc}",
                    "result": None,
                }
            payload = (
                mcp_result.to_dict()
                if hasattr(mcp_result, "to_dict")
                else dict(mcp_result)
            )
            return {
                "success": bool(payload.get("success")),
                "operation_id": operation_id,
                "arguments": call_args,
                "result": payload,
                "error_type": None
                if payload.get("success")
                else (payload.get("error_type") or "underlying_tool_error"),
                "error": payload.get("error"),
            }

        server = self._servers.get(op.service_id)
        if server is None:
            return {
                "success": False,
                "operation_id": operation_id,
                "error_type": "execution_error",
                "error": f"No underlying server for service_id={op.service_id}",
                "result": None,
            }
        try:
            result = server.call(op.underlying_tool, **call_args)
        except TypeError as exc:
            return {
                "success": False,
                "operation_id": operation_id,
                "error_type": "execution_error",
                "error": f"Underlying call TypeError: {exc}",
                "result": None,
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "success": False,
                "operation_id": operation_id,
                "error_type": "execution_error",
                "error": f"Underlying call failed: {type(exc).__name__}: {exc}",
                "result": None,
            }

        return {
            "success": bool(result.get("success")),
            "operation_id": operation_id,
            "arguments": call_args,
            "result": result,
            "error_type": None if result.get("success") else "underlying_tool_error",
            "error": result.get("error"),
        }

    def call_meta(self, tool_name: str, **kwargs: Any) -> Dict[str, Any]:
        if tool_name not in META_TOOLS:
            return {
                "success": False,
                "error": (
                    f"Unknown meta-tool '{tool_name}'. Operation-Visible mode exposes "
                    f"only {list(META_TOOLS)}; underlying financial tools are not "
                    "directly callable."
                ),
            }
        return getattr(self, tool_name)(**kwargs)

    def list_meta_tools(self) -> List[str]:
        return list(META_TOOLS)

    def list_underlying_operation_ids(self) -> List[str]:
        return sorted(self.registry.keys())

    # ---- validation --------------------------------------------------------

    def _validate_arguments(
        self, op: OperationSpec, arguments: Dict[str, Any]
    ) -> Optional[str]:
        if not isinstance(arguments, dict):
            return "arguments must be an object/dict"
        known = op.param_map()
        unknown = sorted(set(arguments) - set(known))
        if unknown:
            return f"Unknown arguments: {unknown}"
        missing = [p.name for p in op.required_params() if p.name not in arguments]
        if missing:
            return f"Missing required arguments: {missing}"
        for name, value in arguments.items():
            err = self._validate_value(known[name], value)
            if err:
                return err
        return None

    def _validate_value(self, spec: ParameterSpec, value: Any) -> Optional[str]:
        if value is None:
            return f"Argument '{spec.name}' cannot be null"
        expected = spec.type
        if expected == "string" and not isinstance(value, str):
            return f"Argument '{spec.name}' must be string"
        if expected == "number" and not isinstance(value, (int, float)):
            return f"Argument '{spec.name}' must be number"
        if expected == "boolean" and not isinstance(value, bool):
            return f"Argument '{spec.name}' must be boolean"
        if expected == "object" and not isinstance(value, dict):
            return f"Argument '{spec.name}' must be object"
        if expected == "array" and not isinstance(value, list):
            return f"Argument '{spec.name}' must be array"
        if spec.enum is not None and value not in spec.enum:
            return (
                f"Argument '{spec.name}' must be one of {spec.enum}; got {value!r}"
            )
        # Light period format check when present.
        if spec.name == "period" and isinstance(value, str):
            if value.isdigit() and len(value) == 4:
                # Ambiguity is enforced by the underlying tool when report_type missing;
                # still allow the string through if report_type is separately provided.
                pass
        return None
