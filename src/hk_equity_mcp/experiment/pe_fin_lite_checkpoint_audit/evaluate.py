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
    # Planner protocol engineering: valid plan schema/semantics present.
    # Accuracy is irrelevant here.
    detail = {
        "parse_or_schema_ok": ok,
        "repair_count": repair,
        "n_steps": len(_plan_steps(row)),
        "step_types": _step_types(row),
        "call_observability": obs.get("call_observability")
        or _extract_planner_obs_from_trace(row),
    }
    return {"checkpoint": "C01_planner", "status": _status(ok), "detail": detail}


def _extract_planner_obs_from_trace(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = []
    for t in row.get("trace") or []:
        if isinstance(t, dict) and t.get("stage") == "planner":
            if t.get("call_observability"):
                out.extend(list(t.get("call_observability") or []))
    return out


def eval_c02_filter(row: Dict[str, Any]) -> Dict[str, Any]:
    if not _has_retrieve(row):
        return {"checkpoint": "C02_filter", "status": "N/A", "detail": {"reason": "no_retrieve"}}
    funnels = _funnels(row)
    if not funnels:
        return {
            "checkpoint": "C02_filter",
            "status": "unknown",
            "detail": {"reason": "missing_funnel"},
        }
    # Pass if every retrieve funnel has filter observability and no blocked leakage into kept.
    ok = True
    details = []
    for f in funnels:
        if f.get("disclosure_composite"):
            details.append({"disclosure_composite": True, "filter": "N/A_composite"})
            continue
        filt = (f.get("checkpoint_passive") or {}).get("filter") or f.get("hard_filtered")
        if not filt:
            ok = False
            details.append({"missing_filter": True})
            continue
        leak = bool((f.get("checkpoint_passive") or {}).get("filter", {}).get("blocked_or_excluded_leakage"))
        if leak:
            ok = False
        details.append({"filter": filt, "blocked_leakage": leak})
    return {"checkpoint": "C02_filter", "status": _status(ok), "detail": {"funnels": details}}


def eval_c03_embedding(row: Dict[str, Any]) -> Dict[str, Any]:
    if not _has_retrieve(row):
        return {"checkpoint": "C03_embedding", "status": "N/A", "detail": {}}
    funnels = _funnels(row)
    if not funnels:
        return {"checkpoint": "C03_embedding", "status": "unknown", "detail": {}}
    ok = True
    details = []
    for f in funnels:
        if f.get("disclosure_composite"):
            details.append({"disclosure_composite": True})
            continue
        emb = (f.get("checkpoint_passive") or {}).get("embedding") or {}
        topk = emb.get("top_k") or f.get("embedding_topk")
        if topk is None:
            ok = False
            details.append({"missing_topk": True})
            continue
        # Engineering: Top-K recorded and size <= 5.
        size = len(topk)
        if size > 5:
            ok = False
        details.append(
            {
                "query": emb.get("step_query") or f.get("search_query"),
                "top_k_size": size,
                "scores": emb.get("scores") or topk,
            }
        )
    return {"checkpoint": "C03_embedding", "status": _status(ok), "detail": {"funnels": details}}


def eval_c04_describe(row: Dict[str, Any]) -> Dict[str, Any]:
    if not _has_retrieve(row):
        return {"checkpoint": "C04_describe", "status": "N/A", "detail": {}}
    funnels = _funnels(row)
    if not funnels:
        return {"checkpoint": "C04_describe", "status": "unknown", "detail": {}}
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
    return {"checkpoint": "C04_describe", "status": _status(ok), "detail": {"funnels": details}}


def eval_c05_binder(row: Dict[str, Any]) -> Dict[str, Any]:
    if not _has_retrieve(row):
        return {"checkpoint": "C05_binder", "status": "N/A", "detail": {}}
    funnels = _funnels(row)
    if not funnels:
        return {"checkpoint": "C05_binder", "status": "unknown", "detail": {}}
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
            # No primary after binder stage → fail if retrieve attempted without composite.
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
    return {"checkpoint": "C05_binder", "status": _status(ok), "detail": {"funnels": details}}


def eval_c06_arguments(row: Dict[str, Any]) -> Dict[str, Any]:
    if not _has_retrieve(row):
        return {"checkpoint": "C06_arguments", "status": "N/A", "detail": {}}
    funnels = _funnels(row)
    if not funnels:
        return {"checkpoint": "C06_arguments", "status": "unknown", "detail": {}}
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
        # Empty {} when operation selected is a soft signal; mark fail for engineering
        # only when binder recorded missing/invalid flag.
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
    return {"checkpoint": "C06_arguments", "status": _status(ok), "detail": {"funnels": details}}


def eval_c07_mcp(row: Dict[str, Any]) -> Dict[str, Any]:
    if not _has_retrieve(row):
        return {"checkpoint": "C07_mcp", "status": "N/A", "detail": {}}
    invoked = list(row.get("invoked_operations") or [])
    records = list(row.get("invoke_records") or row.get("invoke_calls") or [])
    # No invocation: engineering pass for MCP stage is fail if retrieve expected invoke,
    # but zero invocation must NOT be counted as tool success.
    tool_success = row.get("tool_execution_success")
    all_ok = row.get("all_tool_invocations_successful")
    if not invoked and not records:
        # Retrieve planned but nothing invoked → fail (unless disclosure stopped early).
        funnels = _funnels(row)
        stopped = any(
            (f.get("disclosure_composite") or {}).get("stopped_after_list") for f in funnels
        )
        if stopped:
            return {
                "checkpoint": "C07_mcp",
                "status": "pass",
                "detail": {"stopped_after_list": True, "invoked": []},
            }
        return {
            "checkpoint": "C07_mcp",
            "status": "fail",
            "detail": {
                "no_invocation": True,
                "tool_success_claimed": tool_success,
                "note": "zero_invocation_not_tool_success",
            },
        }
    # Consistency: do not treat failed/partial as strict success.
    if all_ok is True and int(row.get("failed_invocation_count") or 0) > 0:
        return {
            "checkpoint": "C07_mcp",
            "status": "fail",
            "detail": {"inconsistent_success_flags": True},
        }
    gw_ok = True
    for rec in records:
        if isinstance(rec, dict) and rec.get("gateway_success") is False:
            gw_ok = False
    return {
        "checkpoint": "C07_mcp",
        "status": _status(gw_ok),
        "detail": {
            "invoked": invoked,
            "records": records,
            "successful_invoked": row.get("successful_invoked_operations"),
            "real_mcp_execution_count": row.get("real_mcp_execution_count"),
        },
    }


def eval_c08_normalize(row: Dict[str, Any]) -> Dict[str, Any]:
    if not _has_retrieve(row):
        return {"checkpoint": "C08_normalize", "status": "N/A", "detail": {}}
    statuses = list(row.get("step_statuses") or [])
    retrieve_steps = [s for s in statuses if s.get("type") == "retrieve"]
    if not retrieve_steps and not _funnels(row):
        return {"checkpoint": "C08_normalize", "status": "unknown", "detail": {}}
    ok = True
    details = []
    for s in retrieve_steps:
        # If gateway claimed success but normalization missing and required outputs empty → fail.
        if s.get("gateway_success") and s.get("normalization_success") is False:
            ok = False
        details.append(
            {
                "step_id": s.get("id"),
                "normalization_success": s.get("normalization_success"),
                "missing_outputs": s.get("missing_outputs"),
                "gateway_success": s.get("gateway_success"),
            }
        )
    # Also inspect funnel passive normalize.
    for f in _funnels(row):
        norm = (f.get("checkpoint_passive") or {}).get("normalize") or {}
        if norm.get("normalization_success") is False and (
            f.get("invoke_gateway_success") or {}
        ):
            # Gateway success with normalize fail is a real engineering signal.
            if any((f.get("invoke_gateway_success") or {}).values()):
                ok = False
        details.append({"funnel_normalize": norm})
    return {"checkpoint": "C08_normalize", "status": _status(ok), "detail": {"items": details}}


def eval_c09_evidence(row: Dict[str, Any]) -> Dict[str, Any]:
    if not _has_retrieve(row):
        return {"checkpoint": "C09_evidence_acceptance", "status": "N/A", "detail": {}}
    statuses = [s for s in (row.get("step_statuses") or []) if s.get("type") == "retrieve"]
    if not statuses:
        return {"checkpoint": "C09_evidence_acceptance", "status": "unknown", "detail": {}}
    ok = True
    details = []
    for s in statuses:
        # Strict: step_completed requires missing empty + scope_match.
        completed = s.get("step_completed")
        missing = list(s.get("missing_outputs") or [])
        scope = s.get("scope_match")
        if completed is True and (missing or scope is False):
            ok = False  # false-accept
        if (
            completed is False
            and not missing
            and scope is not False
            and s.get("gateway_success")
            and s.get("required_outputs_satisfied")
        ):
            # Potential false-reject — still engineering-detectable; mark fail for review.
            ok = False
        details.append(
            {
                "step_id": s.get("id"),
                "step_completed": completed,
                "missing_outputs": missing,
                "scope_match": scope,
                "required_outputs_satisfied": s.get("required_outputs_satisfied"),
                "false_accept_or_reject_review": None,
            }
        )
    return {
        "checkpoint": "C09_evidence_acceptance",
        "status": _status(ok),
        "detail": {"steps": details},
    }


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
        return {
            "checkpoint": "C10_fallback_unavailable",
            "status": "N/A",
            "detail": details,
        }
    return {
        "checkpoint": "C10_fallback_unavailable",
        "status": _status(ok),
        "detail": details,
    }


def eval_c11_compute(row: Dict[str, Any]) -> Dict[str, Any]:
    if "compute" not in _step_types(row):
        return {"checkpoint": "C11_compute", "status": "N/A", "detail": {"applicable": False}}
    records = list(row.get("compute_traces") or [])
    statuses = [s for s in (row.get("step_statuses") or []) if s.get("type") == "compute"]
    if not records and not statuses:
        return {"checkpoint": "C11_compute", "status": "unknown", "detail": {}}
    ok = True
    details = []
    for s in statuses:
        if s.get("status") not in {"completed", "ok", "success"} and s.get("execution_ok") is not True:
            ok = False
        details.append(s)
    for r in records:
        if isinstance(r, dict) and r.get("error"):
            ok = False
        details.append({"record": r})
    return {"checkpoint": "C11_compute", "status": _status(ok), "detail": {"items": details}}


def eval_c12_reason(row: Dict[str, Any]) -> Dict[str, Any]:
    if "reason" not in _step_types(row):
        return {"checkpoint": "C12_reason", "status": "N/A", "detail": {"applicable": False}}
    arts = list(row.get("reason_artifacts") or [])
    statuses = [s for s in (row.get("step_statuses") or []) if s.get("type") == "reason"]
    if not arts and not statuses:
        return {"checkpoint": "C12_reason", "status": "unknown", "detail": {}}
    # Engineering: reason ran without inventing tool calls; unsupported claim flags if present.
    ok = True
    details = []
    for a in arts:
        if isinstance(a, dict) and a.get("unsupported_claim"):
            ok = False
        if isinstance(a, dict) and a.get("numeric_mutation"):
            ok = False
        details.append(a)
    return {"checkpoint": "C12_reason", "status": _status(ok), "detail": {"artifacts": details}}


def eval_c13_final(row: Dict[str, Any]) -> Dict[str, Any]:
    payload = row.get("final_payload")
    status = row.get("predicted_status") or (
        payload.get("status") if isinstance(payload, dict) else None
    )
    # Contract: must have a status string; failed/partial must not be marked strict success.
    if status is None and row.get("final_answer") is None:
        return {
            "checkpoint": "C13_final",
            "status": "fail",
            "detail": {"missing_final": True},
        }
    ok = True
    if row.get("status") == "failed" and str(status).lower() in {"answered", "success"}:
        # Inconsistent synthesis labeling.
        ok = False
    detail = {
        "answer_status": status,
        "task_status": row.get("status"),
        "evidence_ids": (payload or {}).get("evidence_ids")
        if isinstance(payload, dict)
        else None,
        "final_answer_present": bool(row.get("final_answer")),
        "full_partial_failed": row.get("status"),
    }
    return {"checkpoint": "C13_final", "status": _status(ok), "detail": detail}


def eval_c14_evaluator(row: Dict[str, Any]) -> Dict[str, Any]:
    # Evaluator errors are separate from agent errors. Presence of grading fields = pass
    # for engineering observability; do not conflate answer_correct with agent success.
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
        "potential_evaluation_failure": None,
        "agent_error_type": row.get("error_type"),
        "note": "evaluator_failure_separated_from_agent_error",
    }
    # If final looks correct-ish but evaluator missing → unknown/fail observability.
    if not has_fields:
        return {"checkpoint": "C14_evaluator", "status": "fail", "detail": detail}
    return {"checkpoint": "C14_evaluator", "status": "pass", "detail": detail}


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
        if r["status"] == "fail":
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
