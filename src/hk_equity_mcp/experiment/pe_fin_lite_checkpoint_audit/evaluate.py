"""Deterministic offline checkpoint evaluation (C01–C14)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from hk_equity_mcp.experiment.pe_fin_lite_checkpoint_audit import CHECKPOINT_IDS
from hk_equity_mcp.pe_fin_lite.checkpoint_observability import (
    enrich_retrieve_funnel_passive,
)

Status = str  # pass | fail | N/A | unknown


def _status(ok: Optional[bool], *, applicable: bool = True, known: bool = True) -> Status:
    if not applicable:
        return "N/A"
    if not known:
        return "unknown"
    if ok is True:
        return "pass"
    if ok is False:
        return "fail"
    return "unknown"


def _is_na_status(status: Status) -> bool:
    return status == "N/A" or str(status).startswith("N/A_")


def _dual(
    *,
    checkpoint: str,
    contract_status: Status,
    stage_outcome_status: Status,
    downstream_task_effect: Optional[str] = None,
    detail: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Dual-layer checkpoint record: observability ≠ semantic stage success.

    Headline ``status`` follows stage_outcome_status so contract pass cannot
    masquerade as stage pass in funnel/top-level counts.
    """
    agg = stage_outcome_status
    return {
        "checkpoint": checkpoint,
        "status": agg,
        "contract_status": contract_status,
        "observability_status": contract_status,
        "stage_outcome_status": stage_outcome_status,
        "downstream_task_effect": downstream_task_effect,
        "detail": detail or {},
    }


def _binder_failed_before_invoke(row: Dict[str, Any]) -> bool:
    for s in row.get("step_statuses") or []:
        fr = str(s.get("failure_reason") or "")
        if fr.startswith("binder_failed"):
            return True
    for f in _funnels(row):
        binding = f.get("binding") if isinstance(f.get("binding"), dict) else {}
        if binding.get("ok") is False and not (
            f.get("invoked_operations") or row.get("invoked_operations")
        ):
            return True
    return False


def _temporal_no_tool_path(row: Dict[str, Any]) -> bool:
    err = str(row.get("error_type") or "")
    if "temporal_latest_current" in err:
        return True
    for t in row.get("trace") or []:
        if isinstance(t, dict) and t.get("stage") == "temporal_policy":
            return True
    return False


def _plan_steps(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    plan = row.get("initial_plan") or {}
    steps = plan.get("steps") if isinstance(plan, dict) else None
    return list(steps or [])


def _funnels(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw = list(row.get("retrieval_funnels") or [])
    return [enrich_retrieve_funnel_passive(dict(f)) for f in raw]


def _step_types(row: Dict[str, Any]) -> List[str]:
    return [str(s.get("type") or "") for s in _plan_steps(row)]


def _has_retrieve(row: Dict[str, Any]) -> bool:
    return "retrieve" in _step_types(row) or bool(row.get("retrieval_funnels"))


def eval_c01_planner(row: Dict[str, Any]) -> Dict[str, Any]:
    plan = row.get("initial_plan")
    obs = (row.get("checkpoint_observability") or {}).get("C01_planner") or {}
    ok = plan is not None and isinstance(plan, dict) and bool(plan.get("steps"))
    if row.get("status") == "failed" and not plan:
        ok = False
    repair = int(obs.get("repair_count") or row.get("parse_repair_count") or 0)
    detail = {
        "parse_or_schema_ok": ok,
        "repair_count": repair,
        "n_steps": len(_plan_steps(row)),
        "step_types": _step_types(row),
        "call_observability": obs.get("call_observability")
        or _extract_planner_obs_from_trace(row),
    }
    st = _status(ok)
    return _dual(
        checkpoint="C01_planner",
        contract_status=st,
        stage_outcome_status=st,
        detail=detail,
    )


def _extract_planner_obs_from_trace(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = []
    for t in row.get("trace") or []:
        if isinstance(t, dict) and t.get("stage") == "planner":
            if t.get("call_observability"):
                out.extend(list(t.get("call_observability") or []))
    return out


def eval_c02_filter(row: Dict[str, Any]) -> Dict[str, Any]:
    if not _has_retrieve(row):
        return _dual(
            checkpoint="C02_filter",
            contract_status="N/A",
            stage_outcome_status="N/A",
            detail={"reason": "no_retrieve"},
        )
    funnels = _funnels(row)
    if not funnels:
        return _dual(
            checkpoint="C02_filter",
            contract_status="unknown",
            stage_outcome_status="unknown",
            detail={"reason": "missing_funnel"},
        )
    structural_ok = True
    details = []
    for f in funnels:
        if f.get("disclosure_composite"):
            details.append({"disclosure_composite": True, "filter": "N/A_composite"})
            continue
        filt = (f.get("checkpoint_passive") or {}).get("filter") or f.get("hard_filtered")
        if not filt:
            structural_ok = False
            details.append({"missing_filter": True})
            continue
        leak = bool(
            (f.get("checkpoint_passive") or {})
            .get("filter", {})
            .get("blocked_or_excluded_leakage")
        )
        if leak:
            structural_ok = False
        details.append({"filter": filt, "blocked_leakage": leak})
    # Quality vs gold/accepted ops is not reliably available in D12 offline audit.
    return _dual(
        checkpoint="C02_filter",
        contract_status=_status(structural_ok),
        stage_outcome_status="not_evaluated",
        downstream_task_effect="filter_quality_not_evaluated_without_accepted_op_set",
        detail={
            "funnels": details,
            "structural_pass_means": "no_blocked_excluded_leakage_and_filter_trace_present",
            "quality_status": "not_evaluated",
        },
    )


def eval_c03_embedding(row: Dict[str, Any]) -> Dict[str, Any]:
    if not _has_retrieve(row):
        return _dual(
            checkpoint="C03_embedding",
            contract_status="N/A",
            stage_outcome_status="N/A",
        )
    funnels = _funnels(row)
    if not funnels:
        return _dual(
            checkpoint="C03_embedding",
            contract_status="unknown",
            stage_outcome_status="unknown",
        )
    structural_ok = True
    details = []
    for f in funnels:
        if f.get("disclosure_composite"):
            details.append({"disclosure_composite": True})
            continue
        emb = (f.get("checkpoint_passive") or {}).get("embedding") or {}
        topk = emb.get("top_k") or f.get("embedding_topk")
        if topk is None:
            structural_ok = False
            details.append({"missing_topk": True})
            continue
        size = len(topk)
        if size > 5:
            structural_ok = False
        details.append(
            {
                "query": emb.get("step_query") or f.get("search_query"),
                "top_k_size": size,
                "scores": emb.get("scores") or topk,
            }
        )
    return _dual(
        checkpoint="C03_embedding",
        contract_status=_status(structural_ok),
        stage_outcome_status="not_evaluated",
        downstream_task_effect="retrieval_quality_separated_from_recall_at_k_all155",
        detail={
            "funnels": details,
            "structural_pass_means": "ranked_candidates_present_topk_le_5",
            "note": "D12 cannot substitute all155 retrieval audit / Recall@K",
        },
    )


def eval_c04_describe(row: Dict[str, Any]) -> Dict[str, Any]:
    if not _has_retrieve(row):
        return _dual(
            checkpoint="C04_describe",
            contract_status="N/A",
            stage_outcome_status="N/A",
        )
    funnels = _funnels(row)
    if not funnels:
        return _dual(
            checkpoint="C04_describe",
            contract_status="unknown",
            stage_outcome_status="unknown",
        )
    ok = True
    details = []
    for f in funnels:
        if f.get("disclosure_composite"):
            details.append({"disclosure_composite": True})
            continue
        desc = (f.get("checkpoint_passive") or {}).get("describe") or {}
        ids = desc.get("described_ids") or f.get("described_candidates")
        if ids is None:
            ok = False
            details.append({"missing_described": True})
            continue
        n = len(ids)
        if n > 3:
            ok = False
        details.append(
            {
                "described_ids": ids,
                "described_count": n,
                "rank4_5_not_described": desc.get("rank4_5_not_described"),
            }
        )
    st = _status(ok)
    return _dual(
        checkpoint="C04_describe",
        contract_status=st,
        stage_outcome_status=st,
        detail={"funnels": details},
    )


def eval_c05_binder(row: Dict[str, Any]) -> Dict[str, Any]:
    if not _has_retrieve(row):
        return _dual(
            checkpoint="C05_binder",
            contract_status="N/A",
            stage_outcome_status="N/A",
        )
    funnels = _funnels(row)
    if not funnels:
        return _dual(
            checkpoint="C05_binder",
            contract_status="unknown",
            stage_outcome_status="unknown",
        )
    ok = True
    details = []
    for f in funnels:
        if f.get("disclosure_composite"):
            details.append({"disclosure_composite": True})
            continue
        binder = (f.get("checkpoint_passive") or {}).get("binder") or {}
        primary = binder.get("primary") or (f.get("primary") or {}).get("operation_id")
        described = binder.get("described_candidates") or f.get("described_candidates") or []
        outside = binder.get("selected_outside_described")
        if outside is None and primary and described:
            outside = primary not in described
        binding_ok = binder.get("binding_ok")
        if binding_ok is False or (f.get("binding") or {}).get("ok") is False:
            ok = False
        if outside:
            ok = False
        if primary is None and not f.get("disclosure_composite"):
            if f.get("described_candidates") or described:
                ok = False
        details.append(
            {
                "primary": primary,
                "fallback": binder.get("fallback")
                or (f.get("fallback") or {}).get("operation_id"),
                "described": described,
                "selected_outside_described": outside,
                "binding_ok": binding_ok,
            }
        )
    st = _status(ok)
    return _dual(
        checkpoint="C05_binder",
        contract_status=st,
        stage_outcome_status=st,
        detail={"funnels": details},
    )


def eval_c06_arguments(row: Dict[str, Any]) -> Dict[str, Any]:
    if not _has_retrieve(row):
        return _dual(
            checkpoint="C06_arguments",
            contract_status="N/A",
            stage_outcome_status="N/A",
        )
    funnels = _funnels(row)
    if not funnels:
        return _dual(
            checkpoint="C06_arguments",
            contract_status="unknown",
            stage_outcome_status="unknown",
        )
    ok = True
    details = []
    for f in funnels:
        if f.get("disclosure_composite"):
            details.append({"disclosure_composite": True})
            continue
        primary = f.get("primary") if isinstance(f.get("primary"), dict) else {}
        args = primary.get("arguments")
        if primary.get("operation_id") and args is None:
            ok = False
            details.append({"missing_arguments": True, "operation_id": primary.get("operation_id")})
            continue
        args_passive = (f.get("checkpoint_passive") or {}).get("arguments") or {}
        if args_passive.get("missing_or_invalid_argument_flag") is True:
            ok = False
        details.append(
            {
                "operation_id": primary.get("operation_id"),
                "argument_keys": sorted(str(k) for k in (args or {}).keys()),
                "grounding": args_passive.get("grounding_sources") or f.get("grounding"),
            }
        )
    st = _status(ok)
    return _dual(
        checkpoint="C06_arguments",
        contract_status=st,
        stage_outcome_status=st,
        detail={"funnels": details},
    )


def eval_c07_mcp(row: Dict[str, Any]) -> Dict[str, Any]:
    invoked = list(row.get("invoked_operations") or [])
    records = list(row.get("invoke_records") or row.get("invoke_calls") or [])
    mcp_count = int(row.get("real_mcp_execution_count") or 0)
    tool_success = row.get("tool_execution_success")
    all_ok = row.get("all_tool_invocations_successful")
    metrics_vacuous = mcp_count == 0 and tool_success is True

    # Valid temporal no-tool path (even if plan never scheduled retrieve).
    if not invoked and not records and mcp_count == 0 and _temporal_no_tool_path(row):
        return _dual(
            checkpoint="C07_mcp",
            contract_status="pass" if not metrics_vacuous else "fail",
            stage_outcome_status="N/A_not_required",
            downstream_task_effect="N/A_not_required",
            detail={
                "no_invocation": True,
                "classification": "N/A_not_required",
                "tool_success_claimed": tool_success,
                "metrics_vacuous_true": metrics_vacuous,
                "note": "temporal_or_valid_no_tool_path",
            },
        )

    if not _has_retrieve(row):
        return _dual(
            checkpoint="C07_mcp",
            contract_status="N/A",
            stage_outcome_status="N/A_not_planned",
            downstream_task_effect="N/A_not_planned",
        )

    if not invoked and not records and mcp_count == 0:
        if _binder_failed_before_invoke(row):
            return _dual(
                checkpoint="C07_mcp",
                contract_status="pass" if not metrics_vacuous else "fail",
                stage_outcome_status="N/A_not_reached",
                downstream_task_effect="N/A_not_reached",
                detail={
                    "no_invocation": True,
                    "classification": "N/A_not_reached",
                    "tool_success_claimed": tool_success,
                    "metrics_vacuous_true": metrics_vacuous,
                    "note": "upstream_binder_failed_before_mcp",
                },
            )
        funnels = _funnels(row)
        stopped = any(
            (f.get("disclosure_composite") or {}).get("stopped_after_list") for f in funnels
        )
        if stopped:
            return _dual(
                checkpoint="C07_mcp",
                contract_status="pass",
                stage_outcome_status="pass",
                detail={"stopped_after_list": True, "invoked": []},
            )
        return _dual(
            checkpoint="C07_mcp",
            contract_status="fail",
            stage_outcome_status="fail",
            detail={
                "no_invocation": True,
                "tool_success_claimed": tool_success,
                "metrics_vacuous_true": metrics_vacuous,
                "note": "zero_invocation_not_tool_success",
            },
        )
    if all_ok is True and int(row.get("failed_invocation_count") or 0) > 0:
        return _dual(
            checkpoint="C07_mcp",
            contract_status="fail",
            stage_outcome_status="fail",
            detail={"inconsistent_success_flags": True},
        )
    gw_ok = True
    for rec in records:
        if isinstance(rec, dict) and rec.get("gateway_success") is False:
            gw_ok = False
    # Also inspect funnel gateway maps when records sparse.
    if gw_ok:
        for f in _funnels(row):
            for okv in (f.get("invoke_gateway_success") or {}).values():
                if okv is False:
                    gw_ok = False
    st = _status(gw_ok)
    return _dual(
        checkpoint="C07_mcp",
        contract_status=st,
        stage_outcome_status=st,
        detail={
            "invoked": invoked,
            "records": records,
            "successful_invoked": row.get("successful_invoked_operations"),
            "real_mcp_execution_count": row.get("real_mcp_execution_count"),
            "tool_execution_success": tool_success,
            "all_tool_invocations_successful": all_ok,
        },
    )


def eval_c08_normalize(row: Dict[str, Any]) -> Dict[str, Any]:
    if not _has_retrieve(row):
        return _dual(
            checkpoint="C08_normalize",
            contract_status="N/A",
            stage_outcome_status="N/A",
        )
    statuses = list(row.get("step_statuses") or [])
    retrieve_steps = [s for s in statuses if s.get("type") == "retrieve"]
    if not retrieve_steps and not _funnels(row):
        return _dual(
            checkpoint="C08_normalize",
            contract_status="unknown",
            stage_outcome_status="unknown",
            detail={"reason": "observability_missing"},
        )
    details = []
    saw_explicit = False
    outcome_ok: Optional[bool] = None
    for s in retrieve_steps:
        ns = s.get("normalization_success")
        if ns is None:
            details.append(
                {
                    "step_id": s.get("id"),
                    "normalization_success": None,
                    "gateway_success": s.get("gateway_success"),
                    "note": "null_normalization_not_auto_pass",
                }
            )
            continue
        saw_explicit = True
        if s.get("gateway_success") and ns is False:
            outcome_ok = False
        elif ns is True and outcome_ok is not False:
            outcome_ok = True
        details.append(
            {
                "step_id": s.get("id"),
                "normalization_success": ns,
                "missing_outputs": s.get("missing_outputs"),
                "gateway_success": s.get("gateway_success"),
            }
        )
    for f in _funnels(row):
        norm = (f.get("checkpoint_passive") or {}).get("normalize") or {}
        if "normalization_success" in norm and norm.get("normalization_success") is not None:
            saw_explicit = True
            if norm.get("normalization_success") is False and any(
                (f.get("invoke_gateway_success") or {}).values()
            ):
                outcome_ok = False
        details.append({"funnel_normalize": norm})
    if not saw_explicit:
        return _dual(
            checkpoint="C08_normalize",
            contract_status="unknown",
            stage_outcome_status="unknown",
            detail={"items": details, "reason": "observability_missing"},
        )
    return _dual(
        checkpoint="C08_normalize",
        contract_status=_status(True),
        stage_outcome_status=_status(outcome_ok if outcome_ok is not None else True),
        detail={"items": details, "gateway_not_equal_normalization": True},
    )


def eval_c09_evidence(row: Dict[str, Any]) -> Dict[str, Any]:
    if not _has_retrieve(row):
        return _dual(
            checkpoint="C09_evidence_acceptance",
            contract_status="N/A",
            stage_outcome_status="N/A",
        )
    statuses = [s for s in (row.get("step_statuses") or []) if s.get("type") == "retrieve"]
    if not statuses:
        return _dual(
            checkpoint="C09_evidence_acceptance",
            contract_status="unknown",
            stage_outcome_status="unknown",
        )
    logic_ok = True
    retrieve_complete = True
    details = []
    for s in statuses:
        completed = s.get("step_completed")
        missing = list(s.get("missing_outputs") or [])
        scope = s.get("scope_match")
        classification = None
        if completed is True and (missing or scope is False):
            logic_ok = False
            classification = "false_accept"
            retrieve_complete = False
        elif missing and completed is False:
            classification = "correctly_rejected_incomplete"
            retrieve_complete = False
        elif (
            completed is False
            and not missing
            and scope is not False
            and s.get("gateway_success")
            and s.get("required_outputs_satisfied")
        ):
            logic_ok = False
            classification = "false_reject"
            retrieve_complete = False
        elif completed is not True:
            retrieve_complete = False
        details.append(
            {
                "step_id": s.get("id"),
                "step_completed": completed,
                "missing_outputs": missing,
                "scope_match": scope,
                "required_outputs_satisfied": s.get("required_outputs_satisfied"),
                "classification": classification,
            }
        )
    return _dual(
        checkpoint="C09_evidence_acceptance",
        contract_status=_status(logic_ok),
        stage_outcome_status=_status(retrieve_complete and logic_ok),
        downstream_task_effect=(
            "retrieve_incomplete" if not retrieve_complete else "retrieve_complete"
        ),
        detail={
            "steps": details,
            "acceptance_logic_pass": logic_ok,
            "retrieve_outcome_complete": retrieve_complete,
        },
    )


def eval_c10_fallback(row: Dict[str, Any]) -> Dict[str, Any]:
    # Always applicable at task level for unavailable/refusal semantics; retrieve fallbacks optional.
    funnels = _funnels(row)
    pred_status = str(row.get("predicted_status") or row.get("status") or "")
    details: Dict[str, Any] = {
        "predicted_status": pred_status,
        "fallback_used_count": row.get("fallback_used_count"),
        "local_fallback_count": row.get("local_fallback_count"),
        "funnels": [],
    }
    ok = True
    saw_fallback = False
    for f in funnels:
        fb = (f.get("checkpoint_passive") or {}).get("fallback") or f.get("local_fallback")
        if fb:
            saw_fallback = True
            details["funnels"].append(fb)
            # Silent lexical/mock fallback would be a fail if flagged.
            if str((fb or {}).get("reason") or "") in {"mock", "lexical", "silent"}:
                ok = False
    # No silent mock binder in live predictions.
    if row.get("mock") is True and row.get("live") is True:
        ok = False
        details["silent_or_mock_live"] = True
    # If answerability expects refuse and status is answered without evidence — fail.
    ans = str(row.get("answerability_class") or "")
    if ans in {"should_refuse_or_clarify", "intentionally_infeasible"}:
        if pred_status in {"answered", "success"} and not (
            row.get("evidence_store") or row.get("evidence_ledger")
        ):
            ok = False
            details["incorrect_non_refusal"] = True
    if not saw_fallback and not funnels and not _has_retrieve(row):
        return _dual(
            checkpoint="C10_fallback_unavailable",
            contract_status="N/A",
            stage_outcome_status="N/A",
            detail=details,
        )
    st = _status(ok)
    return _dual(
        checkpoint="C10_fallback_unavailable",
        contract_status=st,
        stage_outcome_status=st,
        detail=details,
    )


def eval_c11_compute(row: Dict[str, Any]) -> Dict[str, Any]:
    if "compute" not in _step_types(row):
        return _dual(
            checkpoint="C11_compute",
            contract_status="N/A",
            stage_outcome_status="N/A_not_planned",
            downstream_task_effect="N/A_not_planned",
            detail={"applicable": False},
        )
    records = list(row.get("compute_traces") or [])
    statuses = [s for s in (row.get("step_statuses") or []) if s.get("type") == "compute"]
    if not records and not statuses:
        return _dual(
            checkpoint="C11_compute",
            contract_status="unknown",
            stage_outcome_status="unknown",
        )
    details = []
    upstream_blocked = False
    direct_fail = False
    executed_ok = False
    for s in statuses:
        details.append(s)
        fr = str(s.get("failure_reason") or "")
        st = str(s.get("status") or "")
        if st == "skipped_due_to_dependency" or fr.startswith("depends_on_failed") or fr.startswith(
            "skipped_due_to_dependency"
        ):
            upstream_blocked = True
        elif st not in {"completed", "ok", "success"} or s.get("execution_ok") is False:
            direct_fail = True
        elif s.get("execution_ok") is True or st in {"completed", "ok", "success"}:
            executed_ok = True
    for r in records:
        details.append({"record": r})
        err = str((r or {}).get("error") or "")
        if err.startswith("skipped_due_to_dependency"):
            upstream_blocked = True
        elif err:
            direct_fail = True
        elif (r or {}).get("ok") is True:
            executed_ok = True
    if upstream_blocked and not executed_ok and not direct_fail:
        return _dual(
            checkpoint="C11_compute",
            contract_status="pass",
            stage_outcome_status="N/A_upstream_blocked",
            downstream_task_effect="N/A_upstream_blocked",
            detail={"items": details, "classification": "N/A_upstream_blocked"},
        )
    if direct_fail:
        return _dual(
            checkpoint="C11_compute",
            contract_status="pass",
            stage_outcome_status="fail",
            downstream_task_effect="direct_compute_failure",
            detail={"items": details, "classification": "direct_compute_failure"},
        )
    return _dual(
        checkpoint="C11_compute",
        contract_status="pass",
        stage_outcome_status=_status(executed_ok),
        detail={"items": details},
    )


def eval_c12_reason(row: Dict[str, Any]) -> Dict[str, Any]:
    if "reason" not in _step_types(row):
        return _dual(
            checkpoint="C12_reason",
            contract_status="N/A",
            stage_outcome_status="N/A_not_planned",
            downstream_task_effect="N/A_not_planned",
            detail={"applicable": False},
        )
    arts = list(row.get("reason_artifacts") or [])
    statuses = [s for s in (row.get("step_statuses") or []) if s.get("type") == "reason"]
    # Upstream skip takes precedence even when empty artifacts.
    for s in statuses:
        st = str(s.get("status") or "")
        fr = str(s.get("failure_reason") or "")
        if st == "skipped_due_to_dependency" or fr.startswith("depends_on_failed"):
            if not arts or all(
                isinstance(a, dict)
                and not a.get("summary")
                and not a.get("reason_parse_failed")
                for a in arts
            ):
                return _dual(
                    checkpoint="C12_reason",
                    contract_status="pass",
                    stage_outcome_status="N/A_upstream_blocked",
                    downstream_task_effect="N/A_upstream_blocked",
                    detail={"classification": "N/A_upstream_blocked", "steps": statuses},
                )
    if not arts and not statuses:
        return _dual(
            checkpoint="C12_reason",
            contract_status="unknown",
            stage_outcome_status="unknown",
        )
    # Only skipped statuses present.
    if statuses and all(
        str(s.get("status") or "") == "skipped_due_to_dependency" for s in statuses
    ):
        return _dual(
            checkpoint="C12_reason",
            contract_status="pass",
            stage_outcome_status="N/A_upstream_blocked",
            downstream_task_effect="N/A_upstream_blocked",
            detail={"classification": "N/A_upstream_blocked", "steps": statuses},
        )
    parse_failed = False
    runtime_recovered = False
    details = []
    for a in arts:
        details.append(a)
        if isinstance(a, dict):
            if a.get("reason_parse_failed") or "reason_parse_failed" in list(
                a.get("gaps") or []
            ):
                parse_failed = True
            if a.get("runtime_recovered"):
                runtime_recovered = True
            if a.get("unsupported_claim") or a.get("numeric_mutation"):
                parse_failed = True
    for s in statuses:
        if s.get("reason_parse_failed") or s.get("failure_reason") == "reason_parse_failed":
            parse_failed = True
        if s.get("runtime_recovered"):
            runtime_recovered = True
    stage = "fail" if parse_failed else "pass"
    return _dual(
        checkpoint="C12_reason",
        contract_status="pass",
        stage_outcome_status=stage,
        downstream_task_effect=(
            "runtime_recovered_verified_evidence"
            if runtime_recovered
            else ("reason_stage_fail" if parse_failed else "reason_stage_pass")
        ),
        detail={
            "artifacts": details,
            "reason_parse_failed": parse_failed,
            "runtime_recovered": runtime_recovered,
            "note": "runtime_recovered_not_equal_reason_stage_pass",
        },
    )


def eval_c13_final(row: Dict[str, Any]) -> Dict[str, Any]:
    payload = row.get("final_payload")
    status = row.get("predicted_status") or (
        payload.get("status") if isinstance(payload, dict) else None
    )
    if status is None and row.get("final_answer") is None:
        return _dual(
            checkpoint="C13_final",
            contract_status="fail",
            stage_outcome_status="fail",
            detail={"missing_final": True},
        )
    schema_ok = True
    if row.get("status") == "failed" and str(status).lower() in {"answered", "success"}:
        schema_ok = False
    semantic_correct = row.get("answer_correct") in (True, "True", "true")
    return _dual(
        checkpoint="C13_final",
        contract_status=_status(schema_ok),
        stage_outcome_status=_status(schema_ok),
        downstream_task_effect=(
            "semantically_correct" if semantic_correct else "schema_valid_not_necessarily_correct"
        ),
        detail={
            "answer_status": status,
            "task_status": row.get("status"),
            "schema_valid": schema_ok,
            "semantically_correct": semantic_correct,
            "evidence_ids": (payload or {}).get("evidence_ids")
            if isinstance(payload, dict)
            else None,
            "final_answer_present": bool(row.get("final_answer")),
        },
    )


def eval_c14_evaluator(row: Dict[str, Any]) -> Dict[str, Any]:
    has_fields = (
        "answer_correct" in row
        or "auto_grade_status" in row
        or "auto_grade_reason" in row
        or "manual_answer_correctness" in row
    )
    detail = {
        "answer_correct": row.get("answer_correct"),
        "auto_grade_reason": row.get("auto_grade_reason"),
        "auto_grade_status": row.get("auto_grade_status"),
        "manual_required": True,
        "human_fields_separate": True,
        "evaluator_executed": has_fields,
        "agent_error_type": row.get("error_type"),
        "note": "evaluator_executed_separated_from_answer_correct",
    }
    if not has_fields:
        return _dual(
            checkpoint="C14_evaluator",
            contract_status="fail",
            stage_outcome_status="fail",
            detail=detail,
        )
    return _dual(
        checkpoint="C14_evaluator",
        contract_status="pass",
        stage_outcome_status="pass",
        downstream_task_effect=(
            "answer_correct"
            if row.get("answer_correct") in (True, "True", "true")
            else "answer_incorrect_or_ungraded"
        ),
        detail=detail,
    )


EVALUATORS = {
    "C01_planner": eval_c01_planner,
    "C02_filter": eval_c02_filter,
    "C03_embedding": eval_c03_embedding,
    "C04_describe": eval_c04_describe,
    "C05_binder": eval_c05_binder,
    "C06_arguments": eval_c06_arguments,
    "C07_mcp": eval_c07_mcp,
    "C08_normalize": eval_c08_normalize,
    "C09_evidence_acceptance": eval_c09_evidence,
    "C10_fallback_unavailable": eval_c10_fallback,
    "C11_compute": eval_c11_compute,
    "C12_reason": eval_c12_reason,
    "C13_final": eval_c13_final,
    "C14_evaluator": eval_c14_evaluator,
}


def evaluate_task_checkpoints(row: Dict[str, Any]) -> Dict[str, Any]:
    results = []
    for cid in CHECKPOINT_IDS:
        results.append(EVALUATORS[cid](row))
    primary = None
    secondary: List[str] = []
    for r in results:
        # Prefer stage_outcome fail as blocking; ignore N/A_* and not_evaluated.
        stage = str(r.get("stage_outcome_status") or r.get("status") or "")
        if stage == "fail":
            if primary is None:
                primary = r["checkpoint"]
            else:
                secondary.append(r["checkpoint"])
    return {
        "task_id": row.get("task_id") or row.get("id"),
        "domain": row.get("domain"),
        "checkpoints": {r["checkpoint"]: r for r in results},
        "checkpoint_list": results,
        "primary_blocking_checkpoint": primary,
        "secondary_failures": secondary,
        "answer_correct": row.get("answer_correct"),
        "agent_error_type": row.get("error_type"),
        "status": row.get("status"),
        "engineering_vs_method_vs_eval": {
            "engineering_primary": primary,
            "method_outcome_status": row.get("status"),
            "evaluation_answer_correct": row.get("answer_correct"),
        },
    }
