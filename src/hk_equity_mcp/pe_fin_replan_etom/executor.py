"""Executor for PE-Fin RePlan-ETOM: executes EXACTLY ONE step per call.

* retrieve: R5 retrieval -> described-only binder -> gateway invoke (route node)
* compute : safe calculator over verified operands (no LLM)
* reason  : strict-schema LLM summary, at most one repair

The functions here never call each other and never loop over steps, so the
graph has no Executor->Executor edge: the Agent calls one function, then the
Replanner.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from hk_equity_mcp.pe_fin_lite.budget import McpBudgetExhausted
from hk_equity_mcp.pe_fin_lite.calculator import (
    CalculatorError,
    compute_record,
    run_compute_spec,
)
from hk_equity_mcp.pe_fin_lite.step_result import make_step_result
from hk_equity_mcp.pe_fin_replan_etom.evidence import EtomEvidenceStore
from hk_equity_mcp.pe_fin_replan_etom.gateway import EtomGateway
from hk_equity_mcp.pe_fin_replan_etom.operation_documents import contract_from_official
from hk_equity_mcp.pe_fin_replan_etom.prompts import REASON_SYSTEM_PROMPT, render_reason_user
from hk_equity_mcp.pe_fin_replan_etom.schemas import (
    format_repair_user_message,
    parse_and_validate,
)
from hk_equity_mcp.pe_fin_replan_etom.tool_binding import run_tool_binder
from hk_equity_mcp.pe_fin_replan_etom.tool_retrieval import (
    ETOM_DESCRIBE_N,
    ETOM_RETRIEVAL_TOP_K,
    retrieve_candidates_for_step,
)


@dataclass
class ExecutorContext:
    llm: Any
    budget: Any
    gateway: EtomGateway
    registry: Dict[str, Any]
    official: Dict[str, Dict[str, Any]]
    vectors: Dict[str, Any]
    embedder: Any
    store: EtomEvidenceStore
    temperature: float = 0.0
    max_tokens: int = 2048
    top_k: int = ETOM_RETRIEVAL_TOP_K
    describe_n: int = ETOM_DESCRIBE_N
    metadata_by_id: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    runtime_operation_ids: Optional[List[str]] = None
    # step_id -> route orders it (transitively) produced; used for depends_on edges.
    route_orders_by_step: Dict[str, List[int]] = field(default_factory=dict)


def _merge_usage(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(a or {})
    for key, val in (b or {}).items():
        if isinstance(val, (int, float)) and isinstance(out.get(key), (int, float)):
            out[key] = out[key] + val
        else:
            out[key] = val
    return out


def _failed(
    step: Dict[str, Any],
    reason: str,
    *,
    failure_class: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    missing = [str(x) for x in (step.get("required_outputs") or [])] or ["step_result"]
    out = make_step_result(
        execution_ok=False,
        values={},
        provenance={},
        missing_outputs=missing,
        scope_match=True,
        status="failed",
        failure_reason=reason,
        mcp_transport_ok=None,
        tool_result_valid=False,
        extra=extra,
    )
    if failure_class:
        out["execution_failure_class"] = failure_class
    return out


def _depends_on_orders(step: Dict[str, Any], ctx: ExecutorContext) -> List[int]:
    orders: List[int] = []
    for dep in step.get("depends_on") or []:
        for order in ctx.route_orders_by_step.get(str(dep), []):
            if order not in orders:
                orders.append(order)
    return sorted(orders)


def _record_grounded_inputs(
    *, step_id: str, arguments: Dict[str, Any], question: str, store: EtomEvidenceStore
) -> List[str]:
    keys = []
    for name, value in arguments.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, str) and value and value in question:
            store.add_grounded_input(step_id=step_id, name=name, value=value)
            keys.append(f"{step_id}.input.{name}")
        elif isinstance(value, (int, float)) and str(value) in question:
            store.add_grounded_input(step_id=step_id, name=name, value=value)
            keys.append(f"{step_id}.input.{name}")
    return keys


def execute_retrieve_step(
    *, step: Dict[str, Any], question: str, ctx: ExecutorContext
) -> Dict[str, Any]:
    sid = str(step.get("id"))
    usage: Dict[str, Any] = {}
    funnel: Dict[str, Any] = {"step_id": sid, "invoked_operations": [], "invoke_gateway_success": {}}

    retrieval = retrieve_candidates_for_step(
        step=step,
        registry=ctx.registry,
        vectors=ctx.vectors,
        embedder=ctx.embedder,
        budget=ctx.budget,
        metadata_by_id=ctx.metadata_by_id,
        runtime_operation_ids=ctx.runtime_operation_ids,
        top_k=ctx.top_k,
        describe_n=ctx.describe_n,
    )
    funnel.update(
        {
            "search_query": retrieval.get("search_query"),
            "retrieval_formula": retrieval.get("retrieval_formula"),
            "top_k": retrieval.get("top_k"),
            "describe_n": retrieval.get("describe_n"),
            "domain_hard_filter_applied": False,
            "topk_operation_ids": [r["operation_id"] for r in retrieval.get("topk") or []],
            "topk_scores": [r["score"] for r in retrieval.get("topk") or []],
            "described_candidates": [r["operation_id"] for r in retrieval.get("described") or []],
            "safety_filter": retrieval.get("safety_filter"),
        }
    )
    if not retrieval.get("ok"):
        result = _failed(step, str(retrieval.get("error") or "retrieval_failed"), failure_class="no_candidates")
        ctx.store.add_retrieve_result(step_id=sid, result=result)
        return {"step_result": result, "usage": usage, "funnel": funnel}

    described = retrieval["described"]
    contracts = {
        r["operation_id"]: contract_from_official(r["operation_id"], ctx.official[r["operation_id"]])
        for r in described
    }
    binder = run_tool_binder(
        llm=ctx.llm,
        question=question,
        step=step,
        described_candidates=described,
        described_contracts=contracts,
        evidence=ctx.store.compact_for_prompt(max_chars=4000),
        temperature=ctx.temperature,
        max_tokens=min(ctx.max_tokens, 1024),
    )
    usage = _merge_usage(usage, binder.get("usage") or {})
    ctx.budget.binder_calls += 1 + int(binder.get("repair_count") or 0)
    funnel["binder"] = {
        "ok": binder.get("ok"),
        "error": binder.get("error"),
        "repair_count": binder.get("repair_count"),
        "call_observability": binder.get("call_observability"),
    }
    if not binder.get("ok"):
        result = _failed(step, f"binder_failed:{binder.get('error')}", failure_class="binder_failure")
        ctx.store.add_retrieve_result(step_id=sid, result=result)
        return {"step_result": result, "usage": usage, "funnel": funnel}

    binding = binder["binding"]
    attempts = [binding["primary"]]
    if binding.get("fallback"):
        attempts.append(binding["fallback"])
    funnel["primary"] = dict(binding["primary"])
    funnel["fallback"] = dict(binding["fallback"]) if binding.get("fallback") else None
    depends = _depends_on_orders(step, ctx)

    chosen: Optional[Dict[str, Any]] = None
    last_failure: Optional[str] = None
    last_class: Optional[str] = None
    for idx, call in enumerate(attempts):
        oid = call["operation_id"]
        if idx == 1:
            ctx.budget.local_fallback_count += 1
            ctx.budget.fallback_used += 1
        else:
            ctx.budget.primary_selected += 1
        invoked = ctx.gateway.invoke(
            oid, call["arguments"], depends_on=depends, step_id=sid
        )
        funnel["invoke_gateway_success"][oid] = bool(invoked.get("transport_ok"))
        if invoked.get("signature_blocked"):
            last_failure = str(invoked.get("error") or "signature_blocked")
            last_class = "signature_blocked"
            funnel.setdefault("signature_blocked", []).append(oid)
            continue
        funnel["invoked_operations"].append(oid)
        if invoked.get("transport_ok"):
            chosen = {"call": call, "invoked": invoked}
            break
        last_failure = str(invoked.get("error") or "invoke_failed")
        last_class = str(invoked.get("blockable_failure_class") or "mcp_transport_or_protocol_failure")

    if chosen is None:
        result = _failed(step, last_failure or "invoke_failed", failure_class=last_class)
        result["mcp_transport_ok"] = False
        ctx.store.add_retrieve_result(step_id=sid, result=result)
        return {"step_result": result, "usage": usage, "funnel": funnel}

    call, invoked = chosen["call"], chosen["invoked"]
    grounded_keys = _record_grounded_inputs(
        step_id=sid, arguments=call["arguments"], question=question, store=ctx.store
    )
    envelope = invoked.get("envelope") or {}
    business_available = bool(envelope.get("business_result_available"))
    order = invoked.get("order")
    if order is not None:
        ctx.route_orders_by_step[sid] = [int(order)]
    funnel["route_order"] = order
    funnel["grounded_input_keys"] = grounded_keys
    provenance = {
        "operation_id": call["operation_id"],
        "server_name": invoked.get("server_name"),
        "source": "etom_routing_gateway",
        "raw_result_ref": envelope.get("raw_result_ref"),
        "content_sha256": envelope.get("content_sha256"),
    }
    if business_available:
        values = dict(envelope.get("flat_values") or {})
        ok = bool(values) and not envelope.get("empty")
        result = make_step_result(
            execution_ok=ok,
            values=values,
            provenance=provenance,
            missing_outputs=[] if ok else ["step_result"],
            scope_match=True,
            status="completed" if ok else "empty",
            failure_reason=None if ok else "empty_result",
            mcp_transport_ok=True,
            tool_result_valid=ok,
            extra={"routing_only": False},
        )
        if not ok:
            result["execution_failure_class"] = "empty_result"
    else:
        # Routing-only: the route node is the evidence. required_outputs cannot be
        # verified (no business data exists), which is recorded explicitly.
        result = make_step_result(
            execution_ok=True,
            values={},
            provenance=provenance,
            missing_outputs=[],
            scope_match=True,
            status="completed",
            mcp_transport_ok=True,
            tool_result_valid=True,
            extra={
                "routing_only": True,
                "required_outputs_verified": False,
                "business_result_available": False,
                "operation_id": call["operation_id"],
                "route_order": order,
            },
        )
    ctx.store.add_retrieve_result(step_id=sid, result=result)
    return {"step_result": result, "usage": usage, "funnel": funnel}


def execute_compute_step(*, step: Dict[str, Any], ctx: ExecutorContext) -> Dict[str, Any]:
    sid = str(step.get("id"))
    spec = step.get("compute_spec") or {}
    store = ctx.store
    resolved: Dict[str, float] = {}
    unresolved: List[str] = []
    for name, ref in (spec.get("operand_refs") or {}).items():
        value = store.resolve_operand(str(ref))
        if value is None:
            unresolved.append(f"{name}->{ref}")
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            try:
                value = float(str(value).replace(",", ""))
            except ValueError:
                unresolved.append(f"{name}->{ref}(non_numeric)")
                continue
        resolved[str(name)] = value
    if unresolved:
        record = compute_record(
            step_id=sid, spec=spec, error=f"operand_unresolved:{';'.join(unresolved)}"
        )
        store.add_compute_record(record)
        return {
            "step_result": _failed(
                step, record["error"], failure_class="compute_operand_unresolved"
            ),
            "usage": {},
        }
    try:
        evaluated = run_compute_spec(spec, resolved_operands=resolved)
        record = compute_record(step_id=sid, spec=spec, evaluated=evaluated)
    except CalculatorError as exc:
        record = compute_record(step_id=sid, spec=spec, error=f"calculator_error:{exc}")
        store.add_compute_record(record)
        return {
            "step_result": _failed(step, record["error"], failure_class="compute_error"),
            "usage": {},
        }
    store.add_compute_record(record)
    name = str(spec.get("output_name"))
    result = make_step_result(
        execution_ok=True,
        values={name: record["result"]},
        provenance={"source": "compute", "formula": record.get("formula")},
        missing_outputs=[],
        scope_match=True,
        status="completed",
        extra={"derived": True},
    )
    ctx.route_orders_by_step[sid] = _depends_on_orders(step, ctx)
    return {"step_result": result, "usage": {}}


def execute_reason_step(
    *, step: Dict[str, Any], question: str, ctx: ExecutorContext
) -> Dict[str, Any]:
    sid = str(step.get("id"))
    messages = [
        {"role": "system", "content": REASON_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": render_reason_user(
                question=question, step=step, evidence=ctx.store.compact_for_prompt(max_chars=4000)
            ),
        },
    ]
    completion = ctx.llm.complete(
        messages, temperature=ctx.temperature, max_tokens=ctx.max_tokens, stage="reason"
    )
    usage = dict(completion.get("usage") or {})
    content = completion.get("content")
    payload, _rc, error = parse_and_validate(str(content or ""), "reason")
    repair_count = 0
    if payload is None:
        repair = ctx.llm.complete(
            messages
            + [
                {"role": "assistant", "content": str(content or "")},
                {
                    "role": "user",
                    "content": format_repair_user_message(
                        "reason", error or "invalid", previous_content=str(content or "")
                    ),
                },
            ],
            temperature=ctx.temperature,
            max_tokens=ctx.max_tokens,
            stage="reason_repair",
        )
        repair_count = 1
        usage = _merge_usage(usage, repair.get("usage") or {})
        payload, _rc, error = parse_and_validate(str(repair.get("content") or ""), "reason")
    if payload is None:
        return {
            "step_result": _failed(step, f"reason_schema_invalid:{error}", failure_class="reason_failure"),
            "usage": usage,
            "repair_count": repair_count,
        }
    ctx.store.add_reason_artifact(
        {
            "step_id": sid,
            "summary": payload.get("summary"),
            "claims": list(payload.get("claims") or []),
            "gaps": list(payload.get("gaps") or []),
        }
    )
    result = make_step_result(
        execution_ok=True,
        values={},
        provenance={"source": "reason"},
        missing_outputs=[],
        scope_match=True,
        status="completed",
        extra={"reason_summary": payload.get("summary")},
    )
    ctx.route_orders_by_step[sid] = _depends_on_orders(step, ctx)
    return {"step_result": result, "usage": usage, "repair_count": repair_count}
