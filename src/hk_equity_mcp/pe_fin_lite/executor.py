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
from hk_equity_mcp.pe_fin_lite.retrieval_formula import (
    DESCRIBE_N,
    DOMAIN_HARD_FILTER,
    RETRIEVAL_FORMULA_NAME,
    RETRIEVAL_FORMULA_VERSION,
    RETRIEVAL_TOP_K,
    retrieval_formula_hash,
)
from hk_equity_mcp.pe_fin_lite.tool_retrieval import (
    build_step_search_query,
    safety_filter_operations,
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
    eea_enabled: bool = False,
) -> Dict[str, Any]:
    spec = step.get("compute_spec") or {}
    if eea_enabled:
        from hk_equity_mcp.pe_fin_lite.compute_binding_eea import bind_compute_operands

        bound = bind_compute_operands(spec=spec, store=store)
        if not bound.get("ok"):
            fc = str(bound.get("failure_class") or "operand_binding_failed")
            status = (
                "skipped_due_to_dependency"
                if fc == "upstream_evidence_missing"
                else "failed"
            )
            record = compute_record(
                step_id=str(step["id"]),
                spec=spec,
                error=f"{fc}:{bound.get('errors')}",
            )
            record["compute_operand_bindings"] = bound.get("bindings")
            record["failure_class"] = fc
            store.add_compute_record(record)
            return {
                "step_result": make_step_result(
                    status=status,
                    failure_reason=record["error"],
                    missing_outputs=[
                        str(b.get("operand_ref"))
                        for b in (bound.get("bindings") or [])
                        if not b.get("ok")
                    ],
                    execution_ok=False,
                    extra={
                        "compute_operand_bindings": bound.get("bindings"),
                        "failure_class": fc,
                    },
                ),
                "compute_record": record,
            }
        evaluated = bound["evaluated"]
        record = compute_record(step_id=str(step["id"]), spec=spec, evaluated=evaluated)
        record["compute_operand_bindings"] = bound.get("bindings")
        store.add_compute_record(record)
        return {
            "step_result": make_step_result(
                execution_ok=True,
                values={str(spec.get("output_name")): evaluated["result"]},
                provenance={"source": "compute", "formula": evaluated["expression"]},
                status="completed",
                extra={"compute_operand_bindings": bound.get("bindings")},
            ),
            "compute_record": record,
        }

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
        store.add_compute_record(record)
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
        store.add_compute_record(record)
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
    """Reason with at most one JSON repair.

    State machine (company-/task-generic):
    1) Direct parse success → reason_parse_failed=false, runtime_recovered=false,
       repair_count=0, status=completed.
    2) Initial fail + valid repair → reason_parse_failed=false,
       runtime_recovered=true, repair_count=1, status=completed; keep
       initial_parse_error for audit.
    3) Initial fail + invalid repair → reason_parse_failed=true,
       runtime_recovered=false, repair_count=1, status=failed. Soft payload may
       exist for fail-closed continuation, but Reason stage must not count pass.
    """
    from hk_equity_mcp.pe_fin_lite.schemas import format_repair_user_message

    evidence = store.compact_for_prompt()
    user = render_reason_user(question=question, step=step, evidence=evidence)
    messages = [
        {"role": "system", "content": PE_FIN_LITE_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
    usage: Dict[str, Any] = {}
    repair_count = 0
    reason_parse_failed = False
    runtime_recovered = False
    initial_parse_error: Optional[str] = None
    content = None
    payload: Optional[Dict[str, Any]] = None
    try:
        completion = llm.complete(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stage="reason",
        )
        usage = dict(completion.get("usage") or {})
        content = completion.get("content")
        parsed = parse_and_validate(
            str(content) if content is not None else "", "reason"
        )
        payload = parsed[0]
        if payload is None:
            initial_parse_error = parsed[2] or "invalid reason JSON"
            repair_msg = format_repair_user_message(
                "reason",
                initial_parse_error,
                previous_content=str(content or ""),
            )
            repair = llm.complete(
                messages
                + [
                    {"role": "assistant", "content": str(content or "")},
                    {"role": "user", "content": repair_msg},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                stage="reason_repair",
            )
            repair_count = 1
            for key, val in (repair.get("usage") or {}).items():
                if isinstance(val, (int, float)) and isinstance(usage.get(key), (int, float)):
                    usage[key] = usage[key] + val
                else:
                    usage[key] = val
            content = repair.get("content")
            parsed = parse_and_validate(
                str(content) if content is not None else "", "reason"
            )
            payload = parsed[0]
            if payload is not None:
                # Case 2: repair restored a schema-valid reason artifact.
                reason_parse_failed = False
                runtime_recovered = True
            else:
                # Case 3: repair still invalid — stage fail; not runtime_recovered.
                reason_parse_failed = True
                runtime_recovered = False
                payload = {
                    "summary": (
                        "Reason step could not parse model output; "
                        "relying on verified evidence only."
                    ),
                    "gaps": ["reason_parse_failed"],
                    "claims": [],
                }
        # else Case 1: direct success — flags remain false, repair_count=0.
    except Exception as exc:  # noqa: BLE001 — budget/network isolation
        reason_parse_failed = True
        runtime_recovered = False
        if initial_parse_error is None:
            initial_parse_error = f"reason_runtime_error:{type(exc).__name__}"
        payload = {
            "summary": (
                "Reason step unavailable; relying on verified evidence only."
            ),
            "gaps": [f"reason_runtime_error:{type(exc).__name__}"],
            "claims": [],
        }
    assert payload is not None
    artifact = {
        "step_id": str(step["id"]),
        "summary": payload.get("summary"),
        "claims": payload.get("claims") or [],
        "gaps": payload.get("gaps") or [],
        "reason_parse_failed": reason_parse_failed,
        "runtime_recovered": runtime_recovered,
        "repair_count": repair_count,
        "initial_parse_error": initial_parse_error,
    }
    store.add_reason_artifact(artifact)
    # Stage outcome fails only when final parse is still invalid.
    stage_ok = not reason_parse_failed
    return {
        "step_result": make_step_result(
            execution_ok=stage_ok,
            values={"reason_summary": artifact["summary"]},
            provenance={"source": "reason"},
            status="completed" if stage_ok else "failed",
            failure_reason="reason_parse_failed" if reason_parse_failed else None,
            extra={
                "reason_parse_failed": reason_parse_failed,
                "runtime_recovered": runtime_recovered,
                "repair_count": repair_count,
                "initial_parse_error": initial_parse_error,
            },
        ),
        "reason_artifact": artifact,
        "usage": usage,
        "reason_parse_failed": reason_parse_failed,
        "runtime_recovered": runtime_recovered,
        "repair_count": repair_count,
        "initial_parse_error": initial_parse_error,
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


def _build_accepted_field_map(
    *,
    projected_values: Dict[str, Any],
    required_outputs: Sequence[str],
    step: Dict[str, Any],
    arguments: Dict[str, Any],
    operation_id: str,
    question_grounding: Optional[Dict[str, Any]] = None,
    eea_enabled: bool = False,
    question: str = "",
) -> Dict[str, Any]:
    from hk_equity_mcp.pe_fin_lite.evidence_provenance import (
        apply_deterministic_derivations,
        grounded_context_from_arguments,
        observed_field_map,
        required_outputs_satisfaction,
        values_from_field_map,
    )

    pool = dict(projected_values or {})
    from hk_equity_mcp.pe_fin_lite.benchmark_cutoff_enforcement import (
        reject_post_cutoff_evidence_values,
    )
    from hk_equity_mcp.pe_fin_lite.temporal_policy import DEFAULT_DATASET_CUTOFF

    cutoff = str(
        (question_grounding or {}).get("benchmark_cutoff") or DEFAULT_DATASET_CUTOFF
    )
    pool, cutoff_reject = reject_post_cutoff_evidence_values(pool, cutoff=cutoff)
    if cutoff_reject:
        # Post-cutoff disclosure payload must not enter Evidence Store.
        pool = {
            k: v
            for k, v in pool.items()
            if k
            not in {
                "release_date",
                "title",
                "news_id",
                "local_key",
                "file_link",
                "source_url",
                "snippet",
                "page_number",
            }
        }
    if eea_enabled:
        from hk_equity_mcp.pe_fin_lite.evidence_acceptance_eea import (
            apply_aliases_to_pool,
            forbid_question_target_as_observed,
            make_evidence_field_record,
            required_outputs_satisfaction_eea,
        )

        pool = apply_aliases_to_pool(pool)
        # Drop answer-bearing values that are verbatim question target numbers.
        for key in list(pool.keys()):
            if forbid_question_target_as_observed(
                field=str(key), value=pool.get(key), question=question
            ):
                pool.pop(key, None)

    observed = observed_field_map(
        pool,
        source="mcp",
        ref=f"invoke:{operation_id}",
    )
    grounded = grounded_context_from_arguments(
        arguments=arguments,
        step=step,
        question_grounding=question_grounding,
    )
    # Fill only missing context fields from grounded_input (never overwrite observed).
    merged = dict(observed)
    for key, entry in grounded.items():
        if key not in merged or merged[key].get("value") in (None,):
            # Allow grounded adjustment="" even when absent.
            merged[key] = entry
    merged = apply_deterministic_derivations(
        merged, arguments=arguments, step=step
    )
    if eea_enabled:
        sat = required_outputs_satisfaction_eea(merged, required_outputs)
        field_records = [
            make_evidence_field_record(
                canonical_name=k,
                raw_name=k,
                value=(v or {}).get("value"),
                provenance_class=str((v or {}).get("provenance_class") or ""),
                operation_id=operation_id,
            )
            for k, v in merged.items()
        ]
    else:
        sat = required_outputs_satisfaction(merged, required_outputs)
        field_records = []
    return {
        "field_map": merged,
        "values": values_from_field_map(merged),
        "satisfaction": sat,
        "missing_outputs": list(sat.get("missing") or []),
        "missing_context": list(sat.get("missing_context") or []) if eea_enabled else [],
        "missing_optional": list(sat.get("missing_optional") or []) if eea_enabled else [],
        "evidence_field_records": field_records,
    }


def _evaluate_invoke(
    *,
    invoke_result: Dict[str, Any],
    required_outputs: Sequence[str],
    step: Dict[str, Any],
    operation_id: str,
    arguments: Dict[str, Any],
    question_grounding: Optional[Dict[str, Any]] = None,
    eea_enabled: bool = False,
    question: str = "",
) -> Dict[str, Any]:
    from hk_equity_mcp.pe_fin_lite.evidence_provenance import scope_fingerprint

    gateway_success = bool(invoke_result.get("success"))
    if invoke_result.get("error_type") in {"execution_error", "unknown_operation"}:
        gateway_success = False
    transport_ok = gateway_success
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
                extra={
                    "gateway_success": gateway_success,
                    "normalization_success": False,
                    "required_outputs_satisfied": False,
                    "step_completed": False,
                },
            )
        )

    accepted = _build_accepted_field_map(
        projected_values=dict(projected.get("values") or {}),
        required_outputs=required_outputs,
        step=step,
        arguments=arguments,
        operation_id=operation_id,
        question_grounding=question_grounding,
        eea_enabled=eea_enabled,
        question=question,
    )
    values = dict(accepted["values"])
    missing = list(accepted["missing_outputs"])
    field_map = accepted["field_map"]
    envelope = projected.get("envelope") or {}
    rows = envelope.get("rows") if isinstance(envelope, dict) else None
    has_payload = bool(values) or bool(rows) or bool(
        isinstance(envelope, dict) and envelope.get("textual_content")
    )
    normalization_success = not bool(projected.get("normalization_error"))
    tool_valid = gateway_success and (has_payload or not required_outputs)
    if projected.get("normalization_error") and not has_payload:
        tool_valid = False
        normalization_success = False

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
    required_ok = len(missing) == 0
    result = finalize_retrieve_status(
        make_step_result(
            values=values,
            provenance={
                "operation_id": operation_id,
                "source": "mcp",
                "raw_result_ref": f"invoke:{operation_id}",
                "arguments_redacted": _redact_args(arguments),
                "raw_result_observability": obs,
                "field_provenance": field_map,
            },
            missing_outputs=missing,
            scope_match=scope,
            mcp_transport_ok=transport_ok,
            tool_result_valid=tool_valid,
            extra={
                "gateway_success": gateway_success,
                "normalization_success": normalization_success,
                "required_outputs_satisfied": required_ok,
                "scope_match": scope,
                "step_completed": False,  # set below after finalize
                "invoke_success": gateway_success,
                "invoke_error": invoke_result.get("error"),
                "projection_pool_keys": projected.get("pool_keys"),
                "raw_result_observability": obs,
                "field_map": field_map,
                "scope_fingerprint": scope_fingerprint(step, arguments),
                "satisfaction": accepted.get("satisfaction"),
                "evidence_field_records": accepted.get("evidence_field_records") or [],
                "missing_context": accepted.get("missing_context") or [],
                "missing_optional": accepted.get("missing_optional") or [],
                "answer_bearing_outputs_satisfied": bool(
                    (accepted.get("satisfaction") or {}).get("ok")
                )
                if eea_enabled
                else None,
                "context_outputs_satisfied": (
                    not bool(accepted.get("missing_context")) if eea_enabled else None
                ),
                "optional_outputs_missing": accepted.get("missing_optional") or [],
            },
        )
    )
    result["step_completed"] = retrieve_step_succeeded(result)
    result["gateway_success"] = gateway_success
    result["normalization_success"] = normalization_success
    result["required_outputs_satisfied"] = required_ok
    if eea_enabled:
        result["evidence_field_records"] = accepted.get("evidence_field_records") or []
        result["optional_outputs_missing"] = accepted.get("missing_optional") or []
        result["context_outputs_missing"] = accepted.get("missing_context") or []
        result["answer_bearing_outputs_satisfied"] = required_ok
    return result


def merge_partial_retrieve_results(
    primary: Dict[str, Any],
    fallback: Dict[str, Any],
    *,
    eea_enabled: bool = False,
) -> Dict[str, Any]:
    """Merge primary partial verified values with fallback fill-ins (same scope only)."""
    from hk_equity_mcp.pe_fin_lite.evidence_provenance import (
        merge_field_maps,
        required_outputs_satisfaction,
        scopes_compatible,
        values_from_field_map,
    )

    p_map = dict(primary.get("field_map") or (primary.get("provenance") or {}).get("field_provenance") or {})
    f_map = dict(fallback.get("field_map") or (fallback.get("provenance") or {}).get("field_provenance") or {})
    p_scope = primary.get("scope_fingerprint") or {}
    f_scope = fallback.get("scope_fingerprint") or {}
    if eea_enabled:
        from hk_equity_mcp.pe_fin_lite.evidence_acceptance_eea import (
            merge_primary_fallback_eea,
            required_outputs_satisfaction_eea,
        )

        merge_out = merge_primary_fallback_eea(
            p_map, f_map, primary_scope=p_scope, fallback_scope=f_scope
        )
        if not merge_out.get("ok"):
            out = dict(primary)
            out.update(
                {
                    "status": "unavailable",
                    "execution_ok": False,
                    "failure_reason": "partial_merge_rejected_eea",
                    "merge_conflicts": merge_out.get("conflicts") or [],
                    "field_map": merge_out.get("merged") or p_map,
                    "values": values_from_field_map(merge_out.get("merged") or p_map),
                    "step_completed": False,
                    "required_outputs_satisfied": False,
                    "primary_fallback_merge_status": merge_out.get(
                        "primary_fallback_merge_status"
                    ),
                }
            )
            return out
        merged_map = merge_out["merged"]
        conflicts: List[Any] = []
    elif p_scope and f_scope and not scopes_compatible(p_scope, f_scope):
        out = dict(fallback)
        out["status"] = "unavailable"
        out["execution_ok"] = False
        out["failure_reason"] = "partial_merge_scope_mismatch"
        out["merge_conflicts"] = [{"reason": "scope_mismatch", "primary": p_scope, "fallback": f_scope}]
        out["field_map"] = f_map
        out["step_completed"] = False
        # Preserve primary partials in merge_trace only.
        out["primary_partial_values"] = dict(primary.get("values") or {})
        out["primary_fallback_merge_status"] = "rejected_scope"
        return out
    else:
        merged_map, conflicts = merge_field_maps(p_map, f_map)

    if conflicts:
        out = dict(primary)
        out.update(
            {
                "status": "unavailable",
                "execution_ok": False,
                "failure_reason": "partial_merge_value_conflict",
                "merge_conflicts": conflicts,
                "field_map": merged_map,
                "values": values_from_field_map(merged_map),
                "step_completed": False,
                "required_outputs_satisfied": False,
            }
        )
        # Keep gateway success flags from either successful invoke.
        out["gateway_success"] = bool(
            primary.get("gateway_success") or fallback.get("gateway_success")
        )
        return out

    required = list(
        (primary.get("satisfaction") or {}).get("present")
        or []
    )
    # Prefer missing list from primary/fallback required_outputs union via satisfaction.
    req_outputs = list(
        dict.fromkeys(
            list(primary.get("missing_outputs") or [])
            + list((primary.get("values") or {}).keys())
            + list(fallback.get("missing_outputs") or [])
            + list((fallback.get("values") or {}).keys())
            + list(merged_map.keys())
        )
    )
    # Reconstruct required from primary missing+present if available.
    sat_primary = primary.get("satisfaction") or {}
    required_outputs = list(
        dict.fromkeys(
            list(sat_primary.get("present") or [])
            + list(sat_primary.get("missing") or [])
            + list((fallback.get("satisfaction") or {}).get("present") or [])
            + list((fallback.get("satisfaction") or {}).get("missing") or [])
        )
    ) or req_outputs
    if eea_enabled:
        from hk_equity_mcp.pe_fin_lite.evidence_acceptance_eea import (
            required_outputs_satisfaction_eea,
        )

        sat = required_outputs_satisfaction_eea(merged_map, required_outputs)
    else:
        sat = required_outputs_satisfaction(merged_map, required_outputs)
    values = values_from_field_map(merged_map)
    out = dict(primary)
    out["values"] = values
    out["field_map"] = merged_map
    out["missing_outputs"] = list(sat.get("missing") or [])
    out["required_outputs_satisfied"] = bool(sat.get("ok"))
    out["satisfaction"] = sat
    out["gateway_success"] = bool(
        primary.get("gateway_success") or fallback.get("gateway_success")
    )
    out["normalization_success"] = bool(
        primary.get("normalization_success") or fallback.get("normalization_success")
    )
    out["mcp_transport_ok"] = bool(
        primary.get("mcp_transport_ok") or fallback.get("mcp_transport_ok")
    )
    out["tool_result_valid"] = bool(
        primary.get("tool_result_valid") or fallback.get("tool_result_valid")
    )
    out["scope_match"] = bool(primary.get("scope_match", True) and fallback.get("scope_match", True))
    out["provenance"] = {
        **(primary.get("provenance") or {}),
        "field_provenance": merged_map,
        "merged_from": [
            (primary.get("provenance") or {}).get("operation_id"),
            (fallback.get("provenance") or {}).get("operation_id"),
        ],
    }
    out = finalize_retrieve_status(out)
    out["step_completed"] = retrieve_step_succeeded(out)
    out["merge_conflicts"] = []
    out["primary_fallback_merge_status"] = "merged"
    return out


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
    eea_enabled: bool = False,
    invocation_guard: Any = None,
    provider_profile: Optional[Dict[str, Any]] = None,
    provider_raw_ops: Optional[Dict[str, Dict[str, Any]]] = None,
    provider_records: Optional[Dict[str, Any]] = None,
    question_for_provider: Optional[str] = None,
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
        "signature_blocks": [],
        "provider_availability": None,
        "cutoff_enforcement": [],
    }
    grounding: Dict[str, Any] = {}

    def _guarded_invoke(oid: str, args: Dict[str, Any]) -> Dict[str, Any]:
        from hk_equity_mcp.pe_fin_lite.benchmark_cutoff_enforcement import (
            blocked_invoke_response,
            enforce_disclosure_invoke_arguments,
            is_disclosure_download_operation,
            is_disclosure_list_operation,
        )
        from hk_equity_mcp.pe_fin_lite.temporal_policy import DEFAULT_DATASET_CUTOFF

        call_args = dict(args or {})
        g_live = grounding if isinstance(grounding, dict) else {}
        if is_disclosure_list_operation(oid) or is_disclosure_download_operation(oid):
            call_args, cutoff_notes, block_reason = enforce_disclosure_invoke_arguments(
                oid,
                call_args,
                cutoff=str(g_live.get("benchmark_cutoff") or DEFAULT_DATASET_CUTOFF),
                grounding=g_live,
            )
            if cutoff_notes or block_reason:
                funnel.setdefault("cutoff_enforcement", []).append(
                    {
                        "operation_id": oid,
                        "notes": cutoff_notes,
                        "block_reason": block_reason,
                        "step_id": step.get("id"),
                        "arguments": {
                            k: call_args.get(k)
                            for k in (
                                "start_date",
                                "end_date",
                                "from_date",
                                "to_date",
                                "release_date",
                                "local_key",
                                "news_id",
                            )
                            if k in call_args
                        },
                    }
                )
            if block_reason:
                # Do not call MCP — provider would default to 1999..today.
                return blocked_invoke_response(
                    operation_id=oid, reason=str(block_reason)
                )

        if invocation_guard is not None:
            blocked, reason = invocation_guard.is_blocked(oid, call_args)
            if blocked:
                invocation_guard.mark_blocked_attempt(
                    operation_id=oid, arguments=call_args, reason=str(reason or "blocked")
                )
                funnel["signature_blocks"].append(
                    {
                        "operation_id": oid,
                        "reason": reason,
                        "step_id": step.get("id"),
                    }
                )
                return invocation_guard.blocked_response(
                    operation_id=oid, reason=str(reason or "blocked")
                )
        result = _invoke_with_budget(
            resolver=resolver, budget=budget, operation_id=oid, arguments=call_args
        )
        if invocation_guard is not None:
            invocation_guard.observe_invoke_result(
                operation_id=oid,
                arguments=call_args,
                invoke_result=result if isinstance(result, dict) else {},
            )
        return result

    from hk_equity_mcp.pe_fin_lite.argument_grounding import (
        apply_grounding_to_step,
        collect_step_grounding,
    )

    grounding = collect_step_grounding(question=question, step=step)
    from hk_equity_mcp.pe_fin_lite.disclosure_workflow import enrich_disclosure_grounding
    from hk_equity_mcp.pe_fin_lite.temporal_policy import DEFAULT_DATASET_CUTOFF

    grounding = enrich_disclosure_grounding(
        grounding,
        step=step,
        question=question or "",
        cutoff=DEFAULT_DATASET_CUTOFF,
    )
    step = apply_grounding_to_step(step, grounding)
    funnel["grounding"] = {
        "stock_code": grounding.get("stock_code"),
        "from_date": grounding.get("from_date"),
        "to_date": grounding.get("to_date"),
        "title_keyword": grounding.get("title_keyword"),
        "code_available": grounding.get("code_available"),
        "benchmark_cutoff": grounding.get("benchmark_cutoff") or DEFAULT_DATASET_CUTOFF,
    }

    # Disclosure composite workflow (fixed chain; still MCP-budgeted).
    if is_disclosure_composite_step(step):
        def _invoke(oid: str, args: Dict[str, Any]) -> Dict[str, Any]:
            return _guarded_invoke(oid, args)

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
            eea_enabled=eea_enabled,
        )
        gw_flags = dict(composite.get("gateway_success_by_operation") or {})
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
                extra={
                    "disclosure_trace": composite.get("trace"),
                    "gateway_success": bool(composite.get("gateway_success")),
                    "normalization_success": True,
                    "required_outputs_satisfied": not bool(composite.get("missing_outputs")),
                    "step_completed": bool(composite.get("execution_ok")),
                    "stopped_after_list": bool(composite.get("stopped_after_list")),
                },
            )
        )
        result["gateway_success"] = bool(composite.get("gateway_success"))
        result["step_completed"] = retrieve_step_succeeded(result)
        funnel["disclosure_composite"] = composite
        funnel["invoked_operations"] = list(composite.get("invoked_operations") or [])
        funnel["invoke_gateway_success"] = gw_flags
        store.add_retrieve_result(step_id=str(step["id"]), result=result)
        gateway_ok_ids = [oid for oid, ok in gw_flags.items() if ok]
        from hk_equity_mcp.pe_fin_lite.checkpoint_observability import (
            enrich_retrieve_funnel_passive,
        )

        funnel = enrich_retrieve_funnel_passive(funnel, step=step, step_result=result)
        return {
            "step_result": result,
            "funnel": funnel,
            "successful_invoked": gateway_ok_ids,
            "gateway_successful_operations": gateway_ok_ids,
        }

    query = build_step_search_query(step)
    funnel["search_query"] = query
    # R5: safety filter only (blocked / non-runtime). No domain hard filter.
    filt = safety_filter_operations(
        registry_ids=list(resolver.registry.keys()),
        metadata_by_id=metadata_by_id,
        runtime_operation_ids=runtime_operation_ids,
    )
    funnel["hard_filtered"] = filt
    funnel["domain_hard_filter"] = DOMAIN_HARD_FILTER
    funnel["retrieval_formula"] = {
        "name": RETRIEVAL_FORMULA_NAME,
        "version": RETRIEVAL_FORMULA_VERSION,
        "hash": retrieval_formula_hash(),
        "lexical_weight": 0.12,
        "top_k": int(budget.limits.get("retrieval_top_k", RETRIEVAL_TOP_K)),
        "describe_n": int(budget.limits.get("describe_top_n", DESCRIBE_N)),
        "domain_hard_filter": DOMAIN_HARD_FILTER,
    }
    allowed: Set[str] = set(filt["kept"])
    provider_trace: Optional[Dict[str, Any]] = None
    if provider_profile:
        from hk_equity_mcp.pe_fin_replan.provider_availability import (
            apply_provider_availability_filter,
        )

        provider_trace = apply_provider_availability_filter(
            operation_ids=sorted(allowed),
            metadata_by_id=metadata_by_id,
            profile=provider_profile,
            raw_ops=provider_raw_ops,
            records=provider_records,
        )
        allowed = set(provider_trace.get("operations_after_provider_filter") or [])
        funnel["provider_availability"] = dict(provider_trace)

    budget.embedding_query_calls += 1
    top_k = int(budget.limits.get("retrieval_top_k", RETRIEVAL_TOP_K))
    find_res = resolver.find_operations(
        query, top_k=top_k, allowed_ids=sorted(allowed)
    )
    candidates = list((find_res or {}).get("candidates") or [])
    topk = select_embedding_topk(
        candidates,
        allowed_ids=allowed,
        top_k=top_k,
    )
    funnel["embedding_topk"] = [
        {
            "operation_id": r.get("operation_id"),
            "score": r.get("score"),
            "cosine_similarity": r.get("cosine_similarity"),
            "lexical_score": r.get("lexical_score"),
        }
        for r in topk
    ]
    funnel["n_candidates_scored"] = len(allowed)

    describe_n = int(budget.limits.get("describe_top_n", DESCRIBE_N))
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
    # Passive ranks / not-described ids (observability only; no ranking change).
    funnel["embedding_topk_ranked"] = [
        {
            "rank": i + 1,
            "operation_id": r.get("operation_id"),
            "score": r.get("score"),
        }
        for i, r in enumerate(topk)
    ]
    funnel["rank4_5_not_described"] = [
        r.get("operation_id")
        for r in funnel["embedding_topk_ranked"]
        if int(r.get("rank") or 0) > len(described_ids)
    ]

    if not described_ids:
        result = make_step_result(
            status="unavailable",
            failure_reason="no_described_candidates",
            missing_outputs=list(step.get("required_outputs") or []),
            execution_ok=False,
            scope_match=False,
        )
        store.add_retrieve_result(step_id=str(step["id"]), result=result)
        from hk_equity_mcp.pe_fin_lite.checkpoint_observability import (
            enrich_retrieve_funnel_passive,
        )

        funnel = enrich_retrieve_funnel_passive(funnel, step=step, step_result=result)
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
        "runtime_recovered": binder_out.get("runtime_recovered"),
        "initial_parse_error": binder_out.get("initial_parse_error"),
        "call_observability": binder_out.get("call_observability") or [],
    }
    if not binder_out.get("ok"):
        # Structured binder failure: do not invoke MCP; do not crash runner.
        result = make_step_result(
            status="failed",
            failure_reason=f"binder_failed:{binder_out.get('error')}",
            missing_outputs=list(step.get("required_outputs") or []),
            execution_ok=False,
        )
        store.add_retrieve_result(step_id=str(step["id"]), result=result)
        from hk_equity_mcp.pe_fin_lite.checkpoint_observability import (
            enrich_retrieve_funnel_passive,
        )

        funnel = enrich_retrieve_funnel_passive(funnel, step=step, step_result=result)
        return {"step_result": result, "funnel": funnel, "usage": binder_out.get("usage")}

    binding = binder_out["binding"]
    primary = binding["primary"]
    fallback = binding.get("fallback")
    if provider_profile and provider_trace is not None:
        from hk_equity_mcp.pe_fin_replan.provider_availability import (
            ProviderAvailabilityError,
            validate_bound_operation,
        )

        try:
            for label, call in (("primary", primary), ("fallback", fallback)):
                if not call:
                    continue
                obs = validate_bound_operation(
                    operation_id=str(call.get("operation_id")),
                    arguments=call.get("arguments")
                    if isinstance(call.get("arguments"), dict)
                    else {},
                    step=step,
                    profile=provider_profile,
                    filter_trace=provider_trace,
                    question=str(question_for_provider or question or ""),
                    metadata_by_id=metadata_by_id,
                    raw_ops=provider_raw_ops,
                )
                if label == "primary":
                    funnel["provider_availability"] = {
                        **dict(provider_trace),
                        **obs,
                    }
        except ProviderAvailabilityError as exc:
            result = make_step_result(
                status="failed",
                failure_reason=str(exc),
                missing_outputs=list(step.get("required_outputs") or []),
                execution_ok=False,
                extra={"execution_failure_class": "unsupported_provider_capability"},
            )
            result["execution_failure_class"] = "unsupported_provider_capability"
            store.add_retrieve_result(step_id=str(step["id"]), result=result)
            from hk_equity_mcp.pe_fin_lite.checkpoint_observability import (
                enrich_retrieve_funnel_passive,
            )

            funnel = enrich_retrieve_funnel_passive(
                funnel, step=step, step_result=result
            )
            return {
                "step_result": result,
                "funnel": funnel,
                "usage": binder_out.get("usage"),
            }
    funnel["primary"] = {
        "operation_id": primary["operation_id"],
        "arguments": _redact_args(primary["arguments"]),
    }
    if fallback:
        funnel["fallback"] = {
            "operation_id": fallback["operation_id"],
            "arguments": _redact_args(fallback["arguments"]),
        }
    # Passive selection validation (does not alter binding).
    described_set = set(described_ids)
    selected_outside = False
    for oid in (
        primary.get("operation_id"),
        (fallback or {}).get("operation_id") if fallback else None,
    ):
        if oid and oid not in described_set:
            selected_outside = True
    funnel["selected_outside_described"] = selected_outside
    funnel["binder_selection_validation"] = {
        "primary_in_described": primary.get("operation_id") in described_set,
        "fallback_in_described": (
            None
            if not fallback
            else (fallback.get("operation_id") in described_set)
        ),
        "selected_outside_described": selected_outside,
    }

    required_outputs = list(step.get("required_outputs") or [])
    budget.primary_selected += 1
    primary_args = dict(primary["arguments"] or {})
    primary_invoke = _guarded_invoke(primary["operation_id"], primary_args,)
    funnel["invoked_operations"].append(primary["operation_id"])
    result = _evaluate_invoke(
        invoke_result=primary_invoke,
        required_outputs=required_outputs,
        step=step,
        operation_id=primary["operation_id"],
        arguments=primary_args,
        question_grounding=grounding,
        eea_enabled=eea_enabled,
        question=question,
    )
    funnel["invoke_gateway_success"] = {
        primary["operation_id"]: bool(result.get("gateway_success")),
    }

    eea_obs: Dict[str, Any] = {
        "execution_failure_class": None,
        "local_repair_attempted": False,
        "local_repair_kind": None,
        "local_repair_success": None,
        "fallback_attempted": False,
        "fallback_success": None,
        "transport_retry_attempted": False,
    }

    # EEA: classify + at most one local repair / transport retry before fallback.
    if eea_enabled and not retrieve_step_succeeded(result):
        from hk_equity_mcp.pe_fin_lite.execution_error_handling import (
            classify_execution_failure,
            empty_or_no_result,
            local_repair_arguments,
            should_argument_repair,
            should_skip_primary_retry_use_fallback,
            should_transport_retry,
        )

        failure_class = classify_execution_failure(
            invoke_result=primary_invoke,
            evaluated=result,
            arguments=primary_args,
            required_schema_props=list(
                (
                    (described_contracts.get(primary["operation_id"]) or {}).get(
                        "json_schema"
                    )
                    or {}
                ).get("required")
                or []
            ),
        )
        eea_obs["execution_failure_class"] = failure_class
        contract = described_contracts.get(primary["operation_id"]) or {}

        if should_transport_retry(failure_class):
            eea_obs["transport_retry_attempted"] = True
            retry_invoke = _guarded_invoke(primary["operation_id"], primary_args,)
            funnel["invoked_operations"].append(primary["operation_id"])
            result = _evaluate_invoke(
                invoke_result=retry_invoke,
                required_outputs=required_outputs,
                step=step,
                operation_id=primary["operation_id"],
                arguments=primary_args,
                question_grounding=grounding,
                eea_enabled=eea_enabled,
                question=question,
            )
            funnel["invoke_gateway_success"][primary["operation_id"]] = bool(
                result.get("gateway_success")
            )
            primary_invoke = retry_invoke
            if not retrieve_step_succeeded(result):
                failure_class = classify_execution_failure(
                    invoke_result=retry_invoke,
                    evaluated=result,
                    arguments=primary_args,
                )
                eea_obs["execution_failure_class"] = failure_class

        if (
            not retrieve_step_succeeded(result)
            and should_argument_repair(failure_class)
        ):
            repaired = local_repair_arguments(
                arguments=primary_args,
                step=step,
                grounding=grounding,
                contract=contract,
                evidence_values=store.values,
            )
            eea_obs["local_repair_attempted"] = True
            eea_obs["local_repair_kind"] = repaired.get("local_repair_kind")
            if repaired.get("changed"):
                primary_args = repaired["arguments"]
                repair_invoke = _guarded_invoke(primary["operation_id"], primary_args,)
                funnel["invoked_operations"].append(primary["operation_id"])
                result = _evaluate_invoke(
                    invoke_result=repair_invoke,
                    required_outputs=required_outputs,
                    step=step,
                    operation_id=primary["operation_id"],
                    arguments=primary_args,
                    question_grounding=grounding,
                    eea_enabled=eea_enabled,
                    question=question,
                )
                funnel["invoke_gateway_success"][primary["operation_id"]] = bool(
                    result.get("gateway_success")
                )
                eea_obs["local_repair_success"] = retrieve_step_succeeded(result)
                if not retrieve_step_succeeded(result):
                    failure_class = classify_execution_failure(
                        invoke_result=repair_invoke,
                        evaluated=result,
                        arguments=primary_args,
                    )
                    eea_obs["execution_failure_class"] = failure_class
            else:
                eea_obs["local_repair_success"] = False

        # Empty/no-result: allow one safe query relaxation OR fallback (not both extra tools beyond fallback).
        if (
            not retrieve_step_succeeded(result)
            and empty_or_no_result(failure_class)
            and fallback is None
        ):
            relaxed = local_repair_arguments(
                arguments=primary_args,
                step=step,
                grounding=grounding,
                contract=contract,
                evidence_values=store.values,
            )
            # Safe relaxation: drop overly-specific optional title filters only.
            args2 = dict(relaxed.get("arguments") or primary_args)
            for drop_key in ("title", "title_keyword", "keyword"):
                if drop_key in args2 and drop_key not in (
                    list((contract.get("json_schema") or {}).get("required") or [])
                ):
                    args2.pop(drop_key, None)
            if args2 != primary_args:
                eea_obs["local_repair_attempted"] = True
                eea_obs["local_repair_kind"] = (
                    str(eea_obs.get("local_repair_kind") or "") + "+query_relax"
                ).strip("+")
                relax_invoke = _guarded_invoke(primary["operation_id"], args2,)
                funnel["invoked_operations"].append(primary["operation_id"])
                result = _evaluate_invoke(
                    invoke_result=relax_invoke,
                    required_outputs=required_outputs,
                    step=step,
                    operation_id=primary["operation_id"],
                    arguments=args2,
                    question_grounding=grounding,
                    eea_enabled=eea_enabled,
                    question=question,
                )
                funnel["invoke_gateway_success"][primary["operation_id"]] = bool(
                    result.get("gateway_success")
                )
                eea_obs["local_repair_success"] = retrieve_step_succeeded(result)

        if should_skip_primary_retry_use_fallback(failure_class):
            # Do not further retry primary; fall through to fallback if any.
            pass

    # Local fallback at most once — merge partial verified values when scopes match.
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
            eea_obs["fallback_attempted"] = True
            fb_invoke = _guarded_invoke(fallback["operation_id"], fallback["arguments"],)
            funnel["invoked_operations"].append(fallback["operation_id"])
            fb_result = _evaluate_invoke(
                invoke_result=fb_invoke,
                required_outputs=required_outputs,
                step=step,
                operation_id=fallback["operation_id"],
                arguments=fallback["arguments"],
                question_grounding=grounding,
                eea_enabled=eea_enabled,
                question=question,
            )
            funnel["invoke_gateway_success"][fallback["operation_id"]] = bool(
                fb_result.get("gateway_success")
            )
            merged = merge_partial_retrieve_results(
                result, fb_result, eea_enabled=eea_enabled
            )
            eea_obs["fallback_success"] = retrieve_step_succeeded(merged)
            funnel["local_fallback"] = {
                "attempted": True,
                "reason": fallback_reason,
                "operation_id": fallback["operation_id"],
                "success": retrieve_step_succeeded(merged),
                "merged": True,
                "merge_conflicts": merged.get("merge_conflicts") or [],
                "primary_fallback_merge_status": merged.get("primary_fallback_merge_status"),
            }
            result = merged

    if eea_enabled:
        funnel["eea_execution"] = eea_obs
        result["execution_failure_class"] = eea_obs.get("execution_failure_class")
        result["local_repair_attempted"] = eea_obs.get("local_repair_attempted")
        result["local_repair_success"] = eea_obs.get("local_repair_success")
        result["primary_fallback_merge_status"] = (
            (funnel.get("local_fallback") or {}).get("primary_fallback_merge_status")
            or result.get("primary_fallback_merge_status")
        )

    store.add_retrieve_result(step_id=str(step["id"]), result=result)
    gateway_ok_ids = [
        oid
        for oid in funnel["invoked_operations"]
        if (funnel.get("invoke_gateway_success") or {}).get(oid)
    ]
    from hk_equity_mcp.pe_fin_lite.checkpoint_observability import (
        enrich_retrieve_funnel_passive,
    )

    funnel = enrich_retrieve_funnel_passive(funnel, step=step, step_result=result)
    return {
        "step_result": result,
        "funnel": funnel,
        "usage": binder_out.get("usage") or {},
        "successful_invoked": gateway_ok_ids,
        "gateway_successful_operations": gateway_ok_ids,
    }
