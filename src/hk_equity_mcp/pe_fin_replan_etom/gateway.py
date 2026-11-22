"""ETOM routing gateway: wraps ``EtomRoutingResolver`` for the Executor.

* ``invoke`` records a predicted route node (an "MCP invocation" for budgeting);
  it never executes a real ETOM MCP tool and never fabricates business data.
* transport success/failure is separated from business result (always ``None``
  in routing-only mode).
* failed-signature support: blockable failures are recorded in a
  ``FailedInvocationGuard`` and the same operation+arguments is refused later;
  a refused call does NOT count as an MCP invocation and records no route node.
* predicted tools/edges are read from the nodes this gateway recorded - never
  from gold.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from hk_equity_mcp.etom.adapter import EtomRoutingResolver
from hk_equity_mcp.pe_fin_replan.failed_invocation import (
    FailedInvocationGuard,
    classify_blockable_failure,
)
from hk_equity_mcp.pe_fin_replan_etom.envelope import RawResultStore, normalize_invoke_result


class EtomGateway:
    """Primary ETOM routing gateway used by the Executor."""
    def __init__(
        self,
        resolver: EtomRoutingResolver,
        *,
        budget: Any,
        guard: Optional[FailedInvocationGuard] = None,
        raw_store: Optional[RawResultStore] = None,
    ):
        self.resolver = resolver
        self.budget = budget
        self.guard = guard if guard is not None else FailedInvocationGuard()
        self.raw_store = raw_store if raw_store is not None else RawResultStore()
        self.invocation_log: List[Dict[str, Any]] = []

    # -- lifecycle -----------------------------------------------------------
    def reset_route(self) -> None:
        self.resolver.reset_route()
        self.invocation_log = []

    # -- invoke ----------------------------------------------------------------
    def invoke(
        self,
        operation_id: str,
        arguments: Optional[Dict[str, Any]] = None,
        *,
        depends_on: Optional[Sequence[int]] = None,
        step_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        arguments = dict(arguments or {})
        blocked, reason = self.guard.is_blocked(operation_id, arguments)
        if blocked:
            self.guard.mark_blocked_attempt(
                operation_id=operation_id, arguments=arguments, reason=str(reason)
            )
            out = self.guard.blocked_response(operation_id=operation_id, reason=str(reason))
            out.update(
                {
                    "operation_id": operation_id,
                    "transport_ok": False,
                    "route_recorded": False,
                    "signature_blocked": True,
                    "executed": False,
                    "business_result": None,
                    "counted_as_mcp_invocation": False,
                }
            )
            self._log(step_id, operation_id, out)
            return out

        # Budget check happens before any route node is recorded.
        self.budget.record_mcp_invoke(stage="invoke")
        try:
            raw = self.resolver.invoke_api(
                operation_id=operation_id,
                arguments=arguments,
                depends_on=list(depends_on or []),
            )
        except Exception as exc:  # noqa: BLE001 - transport-level failure
            raw = {
                "success": False,
                "executed": False,
                "execution_mode": "etom_routing_only",
                "operation_id": operation_id,
                "business_result": None,
                "error": f"{type(exc).__name__}: {exc}",
                "error_type": "transport_failure",
            }
        transport_ok = bool(raw.get("success"))
        envelope = normalize_invoke_result(raw, store=self.raw_store)
        out = {
            **raw,
            "transport_ok": transport_ok,
            "route_recorded": transport_ok and raw.get("order") is not None,
            "signature_blocked": False,
            "business_result": raw.get("business_result"),
            "envelope": envelope,
            "counted_as_mcp_invocation": True,
        }
        if not transport_ok:
            cls = classify_blockable_failure(
                error=str(raw.get("error") or ""),
                error_type=str(raw.get("error_type") or ""),
                success=False,
            )
            if cls:
                self.guard.record_failure(
                    operation_id=operation_id,
                    arguments=arguments,
                    failure_class=cls,
                    error=str(raw.get("error") or ""),
                )
                out["blockable_failure_class"] = cls
        self._log(step_id, operation_id, out)
        return out

    def _log(self, step_id: Optional[str], operation_id: str, out: Dict[str, Any]) -> None:
        self.invocation_log.append(
            {
                "step_id": step_id,
                "operation_id": operation_id,
                "success": bool(out.get("success")),
                "route_recorded": bool(out.get("route_recorded")),
                "signature_blocked": bool(out.get("signature_blocked")),
                "order": out.get("order"),
                "error": out.get("error"),
            }
        )

    # -- recorded route (predictions) -----------------------------------------
    @property
    def predicted_nodes(self) -> List[Dict[str, Any]]:
        return [dict(n) for n in self.resolver.predicted_nodes]

    @property
    def predicted_tools(self) -> List[Dict[str, str]]:
        return [
            {"server_name": n["server_name"], "tool_name": n["tool_name"]}
            for n in self.resolver.predicted_nodes
        ]

    @property
    def dependency_edges(self) -> List[List[int]]:
        edges: List[List[int]] = []
        for node in self.resolver.predicted_nodes:
            edges.extend([list(e) for e in node.get("dependency_edges") or []])
        return edges

    @property
    def mcp_invocation_count(self) -> int:
        return int(self.budget.mcp_invocations)

    @property
    def real_tool_executions(self) -> int:
        return int(self.resolver.real_tool_executions)

    # Compatibility alias used by older agent wiring.
    @property
    def invocation_guard(self) -> FailedInvocationGuard:
        return self.guard


EtomReplanGateway = EtomGateway
