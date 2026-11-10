"""Step executor: retrieve (Top-K + binder + invoke + local fallback), compute, reason."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence, Set

from hk_equity_mcp.agent.llm_client import LLMClient
from hk_equity_mcp.pe_fin_lite.budget import BudgetTracker
from hk_equity_mcp.pe_fin_lite.calculator import CalculatorError, compute_record, run_compute_spec
from hk_equity_mcp.pe_fin_lite.disclosure_workflow import (
    is_disclosure_composite_step,
    run_disclosure_composite,
)
from hk_equity_mcp.pe_fin_lite.envelope_projection import project_invoke_to_required_outputs
from hk_equity_mcp.pe_fin_lite.evidence_store import EvidenceStore
from hk_equity_mcp.pe_fin_lite.prompts import (
    PE_FIN_LITE_SYSTEM_PROMPT,
    render_reason_user,
)
from hk_equity_mcp.pe_fin_lite.schemas import (
    extract_json_object,
    parse_and_validate,
)
from hk_equity_mcp.pe_fin_lite.step_result import (
    check_scope_match,
    finalize_retrieve_status,
    make_step_result,
    retrieve_step_succeeded,
)
from hk_equity_mcp.pe_fin_lite.tool_binding import (
    ToolBindingError,
    mock_binding_from_candidates,
    run_tool_binder,
)
from hk_equity_mcp.pe_fin_lite.tool_retrieval import (
    build_step_search_query,
    hard_filter_operations,
    select_describe_candidates,
    select_embedding_topk,
)
from hk_equity_mcp.operation_resolver.server import OperationResolverServer


def _redact_args(arguments: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in (arguments or {}).items():
        lk = str(k).lower()
        if any(tok in lk for tok in ("api_key", "token", "password", "secret")):
            out[k] = "***REDACTED***"
        else:
            out[k] = v
    return out


def execute_compute_step(
    *,
    step: Dict[str, Any],
    store: EvidenceStore,
) -> Dict[str, Any]:
    spec = step.get("compute_spec") or {}
    refs = dict(spec.get("operand_refs") or {})
    resolved = {}
    missing = []
    for name, ref in refs.items():
        val = store.resolve_operand(str(ref))
        if val is None:
            missing.append(str(ref))
        else:
            try:
                resolved[name] = float(val) if not isinstance(val, (int, float)) else val
            except (TypeError, ValueError):
                missing.append(str(ref))
    if missing:
        record = compute_record(
            step_id=str(step["id"]),
            spec=spec,
            error=f"skipped_due_to_dependency:missing_operands:{missing}",
        )
        return {
            "step_result": make_step_result(
                status="skipped_due_to_dependency",
                failure_reason=record["error"],
                missing_outputs=missing,
                execution_ok=False,
            ),
            "compute_record": record,
        }
    try:
        evaluated = run_compute_spec(spec, resolved_operands=resolved)
        record = compute_record(step_id=str(step["id"]), spec=spec, evaluated=evaluated)
        store.add_compute_record(record)
        return {
            "step_result": make_step_result(
                execution_ok=True,
                values={str(spec.get("output_name")): evaluated["result"]},
                provenance={"source": "compute", "formula": evaluated["expression"]},
                status="completed",
            ),
            "compute_record": record,
        }
    except CalculatorError as exc:
        record = compute_record(step_id=str(step["id"]), spec=spec, error=str(exc))
        return {
            "step_result": make_step_result(
                status="failed",
                failure_reason=str(exc),
                execution_ok=False,
            ),
            "compute_record": record,
        }


def execute_reason_step(
    *,
    step: Dict[str, Any],
    store: EvidenceStore,
    llm: LLMClient,
    question: str,
    temperature: float = 0.0,
    max_tokens: int = 1024,
) -> Dict[str, Any]:
    evidence = store.compact_for_prompt()
    user = render_reason_user(question=question, step=step, evidence=evidence)
    completion = llm.complete(
        [
            {"role": "system", "content": PE_FIN_LITE_SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        stage="reason",
    )
    content = completion.get("content")
    parsed = parse_and_validate(str(content or ""), "reason")
    payload = parsed[0]
    if payload is None:
        # Soft fallback: keep a short deterministic note without inventing facts.
        payload = {
            "summary": "Reason step could not parse model output; relying on verified evidence only.",
            "gaps": ["reason_parse_failed"],
            "claims": [],
        }
    artifact = {
        "step_id": str(step["id"]),
        "summary": payload.get("summary"),
        "claims": payload.get("claims") or [],
        "gaps": payload.get("gaps") or [],
    }
    store.add_reason_artifact(artifact)
    return {
        "step_result": make_step_result(
            execution_ok=True,
            values={"reason_summary": artifact["summary"]},
            provenance={"source": "reason"},
            status="completed",
        ),
        "reason_artifact": artifact,
        "usage": completion.get("usage") or {},
    }


def _invoke_with_budget(
    *,
    resolver: OperationResolverServer,
    budget: BudgetTracker,
    operation_id: str,
    arguments: Dict[str, Any],
) -> Dict[str, Any]:
    budget.record_mcp_invoke(stage="invoke")
    return resolver.invoke_api(operation_id, arguments)


def _envelope_observability(
    invoke_result: Dict[str, Any], projected: Dict[str, Any]
) -> Dict[str, Any]:
    """Compact raw-envelope observability for future audits (no full payload dump)."""
    env = projected.get("envelope") if isinstance(projected.get("envelope"), dict) else {}
    rows = env.get("rows") if isinstance(env.get("rows"), list) else None
    n_rows = len(rows) if rows is not None else env.get("n_rows") or env.get("returned_rows")
    return {
        "gateway_success": bool(invoke_result.get("success")),
        "invoke_error": invoke_result.get("error"),
        "normalization_error": projected.get("normalization_error"),
        "adapted_ok": projected.get("adapted_ok"),
        "pool_keys": list(projected.get("pool_keys") or [])[:40],
        "n_rows": n_rows,
        "missing_outputs": list(projected.get("missing_outputs") or []),
    }


def _evaluate_invoke(
    *,
    invoke_result: Dict[str, Any],
    required_outputs: Sequence[str],
    step: Dict[str, Any],
    operation_id: str,
    arguments: Dict[str, Any],
) -> Dict[str, Any]:
    transport_ok = bool(invoke_result.get("success"))
    if invoke_result.get("error_type") in {"execution_error", "unknown_operation"}:
        transport_ok = False
    try:
        projected = project_invoke_to_required_outputs(
            invoke_result=invoke_result,
            required_outputs=required_outputs,
            domain=str(step.get("domain") or "general"),
        )
    except Exception as exc:  # noqa: BLE001
        return finalize_retrieve_status(
            make_step_result(
                execution_ok=False,
                mcp_transport_ok=transport_ok,
                tool_result_valid=False,
                missing_outputs=list(required_outputs),
                scope_match=False,
                failure_reason=f"envelope_error:{exc}",
                provenance={
                    "operation_id": operation_id,
                    "source": "mcp",
                    "raw_result_ref": f"invoke:{operation_id}",
                    "raw_result_observability": {
                        "gateway_success": bool(invoke_result.get("success")),
                        "invoke_error": str(exc),
                    },
                },
                status="failed",
            )
        )

    values = dict(projected.get("values") or {})
    missing = list(projected.get("missing_outputs") or [])
    envelope = projected.get("envelope") or {}
    rows = envelope.get("rows") if isinstance(envelope, dict) else None
    has_payload = bool(values) or bool(rows) or bool(
        isinstance(envelope, dict) and envelope.get("textual_content")
    )
    tool_valid = bool(invoke_result.get("success")) and (
        has_payload or not required_outputs
    )
    if not invoke_result.get("success"):
        tool_valid = False
    if projected.get("normalization_error") and not has_payload:
        tool_valid = False

    scope_constraints = (
        step.get("scope_constraints")
        if isinstance(step.get("scope_constraints"), dict)
        else {}
    )
    scope = check_scope_match(
        values,
        entity=step.get("entity") or scope_constraints.get("entity"),
        period=step.get("period") or scope_constraints.get("period"),
        date=step.get("date")
        or scope_constraints.get("trade_date")
        or scope_constraints.get("from_date"),
        arguments=arguments,
    )
    obs = _envelope_observability(invoke_result, projected)
    return finalize_retrieve_status(
        make_step_result(
            values=values,
            provenance={
                "operation_id": operation_id,
                "source": "mcp",
                "raw_result_ref": f"invoke:{operation_id}",
                "arguments_redacted": _redact_args(arguments),
                "raw_result_observability": obs,
            },
            missing_outputs=missing,
            scope_match=scope,
            mcp_transport_ok=transport_ok,
            tool_result_valid=tool_valid,
            # Transport success with missing required outputs → unavailable, not completed.
            extra={
                "invoke_success": bool(invoke_result.get("success")),
                "invoke_error": invoke_result.get("error"),
                "projection_pool_keys": projected.get("pool_keys"),
                "raw_result_observability": obs,
            },
        )
    )


def execute_retrieve_step(
    *,
    step: Dict[str, Any],
    question: str,
    resolver: OperationResolverServer,
    metadata_by_id: Dict[str, Dict[str, Any]],
    runtime_operation_ids: Sequence[str],
    store: EvidenceStore,
    budget: BudgetTracker,
    llm: LLMClient,
    allow_hkex_download: bool = False,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    use_mock_binder: bool = False,
) -> Dict[str, Any]:
    funnel: Dict[str, Any] = {
        "step_id": step.get("id"),
        "search_query": None,
        "hard_filtered": None,
        "embedding_topk": None,
        "described_candidates": None,
        "binding": None,
        "primary": None,
        "fallback": None,
        "local_fallback": None,
        "invoked_operations": [],
    }

    from hk_equity_mcp.pe_fin_lite.argument_grounding import (
        apply_grounding_to_step,
        collect_step_grounding,
    )

    grounding = collect_step_grounding(question=question, step=step)
    step = apply_grounding_to_step(step, grounding)
    funnel["grounding"] = {
        "stock_code": grounding.get("stock_code"),
        "from_date": grounding.get("from_date"),
        "to_date": grounding.get("to_date"),
        "title_keyword": grounding.get("title_keyword"),
        "code_available": grounding.get("code_available"),
    }

    # Disclosure composite workflow (fixed chain; still MCP-budgeted).
    if is_disclosure_composite_step(step):
        def _invoke(oid: str, args: Dict[str, Any]) -> Dict[str, Any]:
            return _invoke_with_budget(
                resolver=resolver, budget=budget, operation_id=oid, arguments=args
            )

        def _describe(oid: str) -> Dict[str, Any]:
            return resolver.describe_operation(oid)

        composite = run_disclosure_composite(
            step=step,
            metadata_by_id=metadata_by_id,
            runtime_operation_ids=runtime_operation_ids,
            invoke_fn=_invoke,
            describe_fn=_describe,
            allow_hkex_download=allow_hkex_download,
            question=question,
            grounding=grounding,
        )
        result = finalize_retrieve_status(
            make_step_result(
                execution_ok=bool(composite.get("execution_ok")),
                values=dict(composite.get("values") or {}),
                provenance={
                    "operation_id": (composite.get("invoked_operations") or [None])[-1],
                    "source": "disclosure_composite",
                    "workflow_id": composite.get("workflow_id"),
                    "raw_result_ref": "disclosure_composite",
                },
                missing_outputs=list(composite.get("missing_outputs") or []),
                scope_match=bool(composite.get("scope_match", True)),
                mcp_transport_ok=composite.get("mcp_transport_ok"),
                tool_result_valid=composite.get("tool_result_valid"),
                status=str(composite.get("status") or "failed"),
                failure_reason=composite.get("error"),
                extra={"disclosure_trace": composite.get("trace")},
            )
        )
        funnel["disclosure_composite"] = composite
        funnel["invoked_operations"] = list(composite.get("invoked_operations") or [])
        store.add_retrieve_result(step_id=str(step["id"]), result=result)
        return {"step_result": result, "funnel": funnel}

    query = build_step_search_query(step)
    funnel["search_query"] = query
    filt = hard_filter_operations(
        registry_ids=list(resolver.registry.keys()),
        metadata_by_id=metadata_by_id,
        runtime_operation_ids=runtime_operation_ids,
        step_domain=str(step.get("domain") or "general"),
    )
    funnel["hard_filtered"] = filt
    allowed: Set[str] = set(filt["kept"])

    budget.embedding_query_calls += 1
    find_res = resolver.find_operations(query, top_k=int(budget.limits["retrieval_top_k"]))
    candidates = list((find_res or {}).get("candidates") or [])
    topk = select_embedding_topk(
        candidates,
        allowed_ids=allowed,
        top_k=int(budget.limits["retrieval_top_k"]),
    )
    funnel["embedding_topk"] = [
        {"operation_id": r.get("operation_id"), "score": r.get("score")} for r in topk
    ]

    describe_n = int(budget.limits["describe_top_n"])
    to_describe = select_describe_candidates(topk, top_n=describe_n)
    described_ids = []
    described_contracts: Dict[str, Dict[str, Any]] = {}
    for row in to_describe:
        oid = str(row.get("operation_id"))
        desc = resolver.describe_operation(oid)
        contract = (desc or {}).get("contract") or {}
        described_ids.append(oid)
        described_contracts[oid] = contract
    funnel["described_candidates"] = described_ids

    if not described_ids:
        result = make_step_result(
            status="unavailable",
            failure_reason="no_described_candidates",
            missing_outputs=list(step.get("required_outputs") or []),
            execution_ok=False,
            scope_match=False,
        )
        store.add_retrieve_result(step_id=str(step["id"]), result=result)
        return {"step_result": result, "funnel": funnel}

    if use_mock_binder:
        try:
            binding = mock_binding_from_candidates(to_describe, described_contracts)
            binder_out = {"ok": True, "binding": binding, "usage": {}, "repair_count": 0}
        except ToolBindingError as exc:
            binder_out = {"ok": False, "binding": None, "error": str(exc), "usage": {}}
    else:
        binder_out = run_tool_binder(
            llm=llm,
            question=question,
            step=step,
            described_candidates=to_describe,
            described_contracts=described_contracts,
            evidence=store.compact_for_prompt(),
            temperature=temperature,
            max_tokens=max_tokens,
        )
    budget.binder_calls += 1
    funnel["binding"] = {
        "ok": binder_out.get("ok"),
        "error": binder_out.get("error"),
        "repair_count": binder_out.get("repair_count", 0),
    }
    if not binder_out.get("ok"):
        result = make_step_result(
            status="failed",
            failure_reason=f"binder_failed:{binder_out.get('error')}",
            missing_outputs=list(step.get("required_outputs") or []),
            execution_ok=False,
        )
        store.add_retrieve_result(step_id=str(step["id"]), result=result)
        return {"step_result": result, "funnel": funnel, "usage": binder_out.get("usage")}

    binding = binder_out["binding"]
    primary = binding["primary"]
    fallback = binding.get("fallback")
    funnel["primary"] = {
        "operation_id": primary["operation_id"],
        "arguments": _redact_args(primary["arguments"]),
    }
    if fallback:
        funnel["fallback"] = {
            "operation_id": fallback["operation_id"],
            "arguments": _redact_args(fallback["arguments"]),
        }

    required_outputs = list(step.get("required_outputs") or [])
    budget.primary_selected += 1
    primary_invoke = _invoke_with_budget(
        resolver=resolver,
        budget=budget,
        operation_id=primary["operation_id"],
        arguments=primary["arguments"],
    )
    funnel["invoked_operations"].append(primary["operation_id"])
    result = _evaluate_invoke(
        invoke_result=primary_invoke,
        required_outputs=required_outputs,
        step=step,
        operation_id=primary["operation_id"],
        arguments=primary["arguments"],
    )

    # Local fallback at most once.
    need_fallback = (not retrieve_step_succeeded(result)) and fallback is not None
    fallback_reason = None
    if need_fallback:
        if result.get("mcp_transport_ok") is False:
            fallback_reason = "mcp_transport_or_protocol_failure"
        elif result.get("tool_result_valid") is False:
            fallback_reason = "invalid_or_empty_result"
        elif result.get("missing_outputs"):
            fallback_reason = "required_outputs_missing"
        elif not result.get("scope_match"):
            fallback_reason = "scope_mismatch"
        else:
            fallback_reason = "retrieve_incomplete"

    if need_fallback and fallback_reason:
        if budget.local_fallback_count >= int(
            budget.limits["max_local_fallback_per_retrieve"]
        ):
            funnel["local_fallback"] = {"attempted": False, "reason": "budget_exhausted"}
        else:
            budget.local_fallback_count += 1
            budget.fallback_used += 1
            fb_invoke = _invoke_with_budget(
                resolver=resolver,
                budget=budget,
                operation_id=fallback["operation_id"],
                arguments=fallback["arguments"],
            )
            funnel["invoked_operations"].append(fallback["operation_id"])
            fb_result = _evaluate_invoke(
                invoke_result=fb_invoke,
                required_outputs=required_outputs,
                step=step,
                operation_id=fallback["operation_id"],
                arguments=fallback["arguments"],
            )
            funnel["local_fallback"] = {
                "attempted": True,
                "reason": fallback_reason,
                "operation_id": fallback["operation_id"],
                "success": retrieve_step_succeeded(fb_result),
            }
            result = fb_result

    store.add_retrieve_result(step_id=str(step["id"]), result=result)
    return {
        "step_result": result,
        "funnel": funnel,
        "usage": binder_out.get("usage") or {},
        "successful_invoked": [
            oid
            for oid in funnel["invoked_operations"]
            if retrieve_step_succeeded(result)
            and oid == (result.get("provenance") or {}).get("operation_id")
        ],
    }
