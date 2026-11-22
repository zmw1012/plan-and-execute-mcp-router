"""Deterministic offline LLM for PE-Fin RePlan-ETOM mock runs and tests.

Implements the same ``complete(messages, *, temperature, max_tokens, stage)`` API
as ``hk_equity_mcp.agent.llm_client.LLMClient``. It never touches the network,
never reads gold, and refuses any context that contains gold markers.

Policy ``route_first_candidate``:

* planner   -> one retrieve step whose objective is the (truncated) question;
* binder    -> first DESCRIBED candidate, required arguments filled with
               schema-typed placeholders (strings echo the question);
* reason    -> neutral summary;
* replanner -> keep the remaining plan until it is empty (or the round is final),
               then respond ``answered`` with a number-free routing note.

The mock result is an *engineering* artifact. It must never be reported as a live
or primary result.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

_GOLD_MARKERS = ("ground_truth", "GOLD_SENTINEL", "gold_operations", "accepted_tools")


def _payload_after_first_line(text: str) -> Dict[str, Any]:
    _head, _sep, rest = text.partition("\n")
    try:
        obj = json.loads(rest)
    except json.JSONDecodeError:
        return {}
    return obj if isinstance(obj, dict) else {}


def _placeholder(spec: Dict[str, Any], question: str) -> Any:
    if isinstance(spec.get("enum"), list) and spec["enum"]:
        return spec["enum"][0]
    kind = spec.get("type", "string")
    if isinstance(kind, list):
        kind = next((k for k in kind if k != "null"), "string")
    if kind == "integer":
        return 1
    if kind == "number":
        return 1
    if kind == "boolean":
        return True
    if kind == "array":
        return []
    if kind == "object":
        return {}
    return (question or "query")[:60] or "query"


class ScriptedEtomPeFinLLM:
    provider = "mock"
    model = "mock-pe-fin-replan-etom-llm"

    def __init__(self, policy: str = "route_first_candidate"):
        if policy not in {"route_first_candidate", "infeasible"}:
            raise ValueError(f"unknown mock policy {policy!r}")
        self.policy = policy
        self.calls: List[Dict[str, Any]] = []

    @staticmethod
    def _base_stage(stage: Optional[str]) -> str:
        name = str(stage or "")
        for suffix in ("_transport_retry", "_semantic_repair", "_repair"):
            if name.endswith(suffix):
                name = name[: -len(suffix)]
        return name

    def complete(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        stage: Optional[str] = None,
    ) -> Dict[str, Any]:
        full = "\n".join(str(m.get("content") or "") for m in messages)
        if any(marker in full for marker in _GOLD_MARKERS):
            raise RuntimeError("Mock LLM observed a gold marker in its context.")
        base = self._base_stage(stage)
        user = str(messages[-1].get("content") or "") if messages else ""
        self.calls.append({"stage": stage, "base_stage": base})
        if base == "planner":
            content = self._planner(user)
        elif base == "tool_binding":
            content = self._binder(user)
        elif base == "reason":
            content = json.dumps({"summary": "Evidence reviewed.", "claims": [], "gaps": []})
        elif base == "replanner":
            content = self._replanner(user)
        else:
            content = "{}"
        return {
            "content": content,
            "usage": {
                "input_tokens": max(1, len(full) // 4),
                "output_tokens": max(1, len(content) // 4),
                "total_tokens": max(1, (len(full) + len(content)) // 4),
            },
            "latency_ms": 0.1,
            "provider": self.provider,
            "model": self.model,
        }

    # -- stage policies -------------------------------------------------------
    def _planner(self, user: str) -> str:
        payload = _payload_after_first_line(user)
        question = str(payload.get("question") or "")
        if self.policy == "infeasible":
            step = {
                "id": "s1",
                "type": "reason",
                "domain": "reason",
                "objective": "explain that no available capability can satisfy the request",
                "required_outputs": [],
                "depends_on": [],
            }
        else:
            step = {
                "id": "s1",
                "type": "retrieve",
                "domain": "search",
                "objective": question[:400] or "perform the requested lookup",
                "required_outputs": ["result summary"],
                "depends_on": [],
                "scope_constraints": None,
                "compute_spec": None,
            }
        return json.dumps({"steps": [step]})

    def _binder(self, user: str) -> str:
        payload = _payload_after_first_line(user)
        question = str(payload.get("question") or "")
        cands = payload.get("described_candidates") or []
        if not cands:
            return json.dumps({"primary": {"operation_id": "", "arguments": {}}})
        first = cands[0]
        schema = first.get("json_schema") or {}
        props = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
        args = {
            name: _placeholder(props.get(name) or {}, question)
            for name in (schema.get("required") or [])
        }
        return json.dumps(
            {
                "primary": {"operation_id": first["operation_id"], "arguments": args},
                "fallback": None,
                "notes": None,
            }
        )

    def _replanner(self, user: str) -> str:
        payload = _payload_after_first_line(user)
        state = payload.get("state") or {}
        remaining = list(state.get("remaining_plan") or [])
        forced_final = bool(payload.get("forced_final_round"))
        if forced_final or not remaining:
            if self.policy == "infeasible":
                response = {
                    "completion": "insufficient_evidence",
                    "final_answer": "The request cannot be completed with the available tools.",
                    "reasoning_brief": "no suitable capability",
                }
            else:
                response = {
                    "completion": "answered",
                    "final_answer": "The request was routed to the selected tool.",
                    "reasoning_brief": "route recorded",
                }
            return json.dumps(
                {
                    "schema_version": "pe_fin_replan_etom_v1",
                    "action": "response",
                    "decision_reason": None,
                    "response": response,
                }
            )
        return json.dumps(
            {
                "schema_version": "pe_fin_replan_etom_v1",
                "action": "plan",
                "decision_reason": None,
                "remaining_steps": remaining,
            }
        )
