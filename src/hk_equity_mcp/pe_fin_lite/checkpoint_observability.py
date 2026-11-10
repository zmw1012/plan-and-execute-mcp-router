"""Passive checkpoint observability helpers (no decision changes).

Records structured fields for C01–C14 audits. Never stores hidden CoT text
or secrets — only presence/length flags where reasoning may exist.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Optional, Sequence

CHECKPOINT_TRACE_CONTRACT_VERSION = "pe_fin_lite_checkpoint_trace_v1"
CHECKPOINT_OBSERVABILITY_VERSION = "pe_fin_lite_checkpoint_observability_v1"


def checkpoint_observability_hash() -> str:
    blob = json.dumps(
        {
            "trace_contract": CHECKPOINT_TRACE_CONTRACT_VERSION,
            "observability": CHECKPOINT_OBSERVABILITY_VERSION,
            "fields": [
                "filter",
                "embedding",
                "describe",
                "binder",
                "arguments",
                "mcp",
                "normalize",
                "evidence",
                "fallback",
                "compute",
                "reason",
                "final",
                "evaluator",
            ],
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _redact_secret_keys(obj: Any) -> Any:
    if isinstance(obj, dict):
        banned = {
            "api_key",
            "authorization",
            "reasoning_content",
            "reasoning_text",
            "thinking",
            "secret",
        }
        return {
            k: _redact_secret_keys(v)
            for k, v in obj.items()
            if str(k).lower() not in banned
        }
    if isinstance(obj, list):
        return [_redact_secret_keys(x) for x in obj]
    return obj


def enrich_retrieve_funnel_passive(
    funnel: Dict[str, Any],
    *,
    step: Optional[Dict[str, Any]] = None,
    step_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Attach passive C02–C10 fields derived from an existing retrieve funnel.

    Purely observational: does not change binder selection, invoke, or merge.
    """
    out = dict(funnel or {})
    filt = out.get("hard_filtered") if isinstance(out.get("hard_filtered"), dict) else {}
    kept = list(filt.get("kept") or [])
    rejected = list(filt.get("rejected") or [])
    topk = list(out.get("embedding_topk") or [])
    described = list(out.get("described_candidates") or [])
    ranked = []
    for i, row in enumerate(topk):
        if not isinstance(row, dict):
            continue
        ranked.append(
            {
                "rank": i + 1,
                "operation_id": row.get("operation_id"),
                "score": row.get("score"),
            }
        )
    not_described = [
        r["operation_id"]
        for r in ranked
        if r.get("rank", 0) > len(described) and r.get("operation_id")
    ]
    primary = out.get("primary") if isinstance(out.get("primary"), dict) else {}
    fallback = out.get("fallback") if isinstance(out.get("fallback"), dict) else None
    primary_oid = primary.get("operation_id")
    fallback_oid = (fallback or {}).get("operation_id") if fallback else None
    selected_outside = False
    for oid in (primary_oid, fallback_oid):
        if oid and oid not in described and described:
            selected_outside = True

    step_result = step_result or {}
    prov = step_result.get("provenance") if isinstance(step_result.get("provenance"), dict) else {}
    raw_obs = (
        step_result.get("raw_result_observability")
        or prov.get("raw_result_observability")
        or {}
    )
    local_fb = out.get("local_fallback") if isinstance(out.get("local_fallback"), dict) else {}

    passive = {
        "contract_version": CHECKPOINT_TRACE_CONTRACT_VERSION,
        "filter": {
            "operations_before_filter": int(filt.get("kept_count") or 0)
            + int(filt.get("rejected_count") or 0),
            "retained_operations": kept,
            "retained_count": int(filt.get("kept_count") or len(kept)),
            "rejected_operations": rejected,
            "rejected_count": int(filt.get("rejected_count") or len(rejected)),
            "blocked_or_excluded_leakage": any(
                str((r or {}).get("reason") or "") == "blocked_or_rejected"
                and str((r or {}).get("operation_id") or "") in kept
                for r in rejected
            ),
        },
        "embedding": {
            "step_query": out.get("search_query"),
            "candidate_operation_ids": [r.get("operation_id") for r in ranked],
            "scores": ranked,
            "ranks": [r.get("rank") for r in ranked],
            "top_k": ranked[:5],
            "top_k_size": len(ranked),
        },
        "describe": {
            "described_ids": described,
            "described_count": len(described),
            "rank4_5_not_described": not_described,
            "describe_n": len(described),
        },
        "binder": {
            "described_candidates": described,
            "primary": primary_oid,
            "fallback": fallback_oid,
            "binding_ok": (out.get("binding") or {}).get("ok"),
            "selected_outside_described": selected_outside,
            "repair_count": (out.get("binding") or {}).get("repair_count", 0),
        },
        "arguments": {
            "primary_arguments_present": bool(primary.get("arguments")),
            "fallback_arguments_present": bool((fallback or {}).get("arguments")),
            "primary_arguments_keys": sorted(
                str(k) for k in (primary.get("arguments") or {}).keys()
            ),
            # Grounding sources recorded earlier on funnel when available.
            "grounding_sources": out.get("grounding") or {},
            "hallucinated_argument_flag": None,  # offline audit may refine
            "missing_or_invalid_argument_flag": None,
        },
        "mcp": {
            "invoked_operations": list(out.get("invoked_operations") or []),
            "invoke_gateway_success": dict(out.get("invoke_gateway_success") or {}),
            "gateway_success": bool(step_result.get("gateway_success")),
            "mcp_transport_ok": step_result.get("mcp_transport_ok"),
            "protocol_or_transport_error": (
                False
                if step_result.get("mcp_transport_ok") is not False
                else True
            ),
            "underlying_composite": bool(out.get("disclosure_composite")),
        },
        "normalize": {
            "raw_result_ref": prov.get("raw_result_ref"),
            "normalization_success": step_result.get("normalization_success"),
            "normalization_error": (raw_obs or {}).get("normalization_error"),
            "pool_keys": (raw_obs or {}).get("pool_keys")
            or step_result.get("projection_pool_keys"),
            "missing_outputs": list(step_result.get("missing_outputs") or []),
        },
        "evidence": {
            "required_outputs": list((step or {}).get("required_outputs") or []),
            "observed_or_derived_values": sorted(
                str(k) for k in (step_result.get("values") or {}).keys()
            ),
            "missing_outputs": list(step_result.get("missing_outputs") or []),
            "scope_constraints": (step or {}).get("scope_constraints"),
            "scope_match": step_result.get("scope_match"),
            "required_outputs_satisfied": step_result.get(
                "required_outputs_satisfied"
            ),
            "step_completed": step_result.get("step_completed"),
            "false_accept_or_reject_review": None,  # human/offline review slot
        },
        "fallback": {
            "trigger_reason": local_fb.get("reason"),
            "fallback_invoked": bool(local_fb.get("attempted")),
            "incremental_evidence": bool(local_fb.get("merged")),
            "merge_conflicts": list(local_fb.get("merge_conflicts") or []),
            "recovered": bool(local_fb.get("success")),
            "unavailable_or_refusal_type": step_result.get("status")
            if str(step_result.get("status") or "")
            in {"unavailable", "clarification_needed", "failed"}
            else None,
        },
    }
    # Preserve non-null fields already recorded on the funnel (offline audit must
    # not wipe live-written normalize/evidence observability).
    existing = out.get("checkpoint_passive")
    if isinstance(existing, dict):
        for section, payload in passive.items():
            if section == "contract_version":
                continue
            prev = existing.get(section)
            if not isinstance(prev, dict) or not isinstance(payload, dict):
                continue
            for key, value in payload.items():
                if value in (None, [], {}) and prev.get(key) not in (None, [], {}):
                    payload[key] = prev.get(key)
    out["checkpoint_passive"] = _redact_secret_keys(passive)
    out["checkpoint_trace_contract_version"] = CHECKPOINT_TRACE_CONTRACT_VERSION
    return out


def build_task_checkpoint_observability(
    *,
    planner_out: Optional[Dict[str, Any]] = None,
    plan: Optional[Dict[str, Any]] = None,
    funnels: Optional[Sequence[Dict[str, Any]]] = None,
    step_statuses: Optional[Sequence[Dict[str, Any]]] = None,
    compute_traces: Optional[Sequence[Dict[str, Any]]] = None,
    reason_artifacts: Optional[Sequence[Dict[str, Any]]] = None,
    final_payload: Optional[Dict[str, Any]] = None,
    final_contract_valid: Optional[bool] = None,
    answer_correct: Optional[Any] = None,
    auto_grade_reason: Optional[str] = None,
    auto_grade_status: Optional[str] = None,
    invoke_records: Optional[Sequence[Dict[str, Any]]] = None,
    status: Optional[str] = None,
) -> Dict[str, Any]:
    """Aggregate passive per-task observability for prediction rows / audits."""
    planner_out = planner_out or {}
    steps = list((plan or {}).get("steps") or [])
    funnels = list(funnels or [])
    step_types = [str(s.get("type") or "") for s in steps]
    retrieve_funnels = [
        enrich_retrieve_funnel_passive(dict(f), step_result=None) for f in funnels
    ]
    return _redact_secret_keys(
        {
            "observability_version": CHECKPOINT_OBSERVABILITY_VERSION,
            "checkpoint_trace_contract_version": CHECKPOINT_TRACE_CONTRACT_VERSION,
            "C01_planner": {
                "ok": planner_out.get("ok"),
                "repair_count": planner_out.get("repair_count"),
                "error": planner_out.get("error"),
                "error_type": planner_out.get("error_type"),
                "n_steps": len(steps),
                "step_types": step_types,
                "depends_on": [list(s.get("depends_on") or []) for s in steps],
                "call_observability": planner_out.get("call_observability") or [],
                "planner_max_tokens": planner_out.get("planner_max_tokens"),
            },
            "retrieve_funnels_passive": [
                f.get("checkpoint_passive") for f in retrieve_funnels
            ],
            "step_statuses": list(step_statuses or []),
            "C11_compute": {
                "applicable": "compute" in step_types,
                "records": list(compute_traces or []),
            },
            "C12_reason": {
                "applicable": "reason" in step_types,
                "artifacts": list(reason_artifacts or []),
            },
            "C13_final": {
                "contract_valid": final_contract_valid,
                "answer_status": (final_payload or {}).get("status")
                if isinstance(final_payload, dict)
                else None,
                "evidence_ids": (final_payload or {}).get("evidence_ids")
                if isinstance(final_payload, dict)
                else None,
                "synthesis_error": None,
                "task_status": status,
            },
            "C14_evaluator": {
                "answer_correct": answer_correct,
                "auto_grade_reason": auto_grade_reason,
                "auto_grade_status": auto_grade_status,
                "manual_required": True,
                "human_fields_separate": True,
            },
            "invoke_records": list(invoke_records or []),
        }
    )
