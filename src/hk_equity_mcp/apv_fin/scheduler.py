"""Step scheduler: only retrieve steps enter the MCP operation resolver."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence

from hk_equity_mcp.agent.llm_client import LLMClient
from hk_equity_mcp.apv_fin.calculator import (
    CalculatorError,
    compute_trace_record,
    evaluate_compute_payload,
)
from hk_equity_mcp.apv_fin.evidence_ledger import EvidenceLedger
from hk_equity_mcp.apv_fin.plan_state import PlanStep, PlanState
from hk_equity_mcp.apv_fin.planner import _merge_usage
from hk_equity_mcp.apv_fin.prompts import (
    APV_FIN_COMPUTE_USER_TEMPLATE,
    APV_FIN_REASON_USER_TEMPLATE,
    APV_FIN_SYSTEM_PROMPT,
)
from hk_equity_mcp.apv_fin.reranker import RetrievalContext, rerank_candidates
from hk_equity_mcp.apv_fin.schemas import format_repair_user_message, parse_and_validate
from hk_equity_mcp.operation_resolver.server import OperationResolverServer


def build_retrieval_query(step: PlanStep, task_spec: Optional[Dict[str, Any]] = None) -> str:
    task_spec = task_spec or {}
    parts = [step.objective]
    if step.retrieval_query_hint:
        parts.append(str(step.retrieval_query_hint))
    for key in ("entities", "metrics", "periods"):
        vals = task_spec.get(key) or []
        if vals:
            parts.append(f"{key}: " + ", ".join(str(v) for v in vals))
    return " | ".join(p for p in parts if p).strip()


def context_from_task_spec(task_spec: Optional[Dict[str, Any]] = None) -> RetrievalContext:
    task_spec = task_spec or {}
    domain_hints: List[str] = []
    # Lightweight heuristic from metrics/constraints text — company-generic.
    blob = " ".join(
        [str(task_spec.get("goal") or "")]
        + list(task_spec.get("metrics") or [])
        + list(task_spec.get("constraints") or [])
    ).lower()
    if any(k in blob for k in ("price", "close", "ohlc", "volume", "return", "index")):
        domain_hints.append("market")
    if any(k in blob for k in ("revenue", "eps", "ifrs", "profit", "filing", "announcement")):
        domain_hints.append("financial")
    if any(k in blob for k in ("news", "media", "headline", "annual report")):
        domain_hints.append("news")
    temporal: List[str] = []
    if task_spec.get("periods"):
        temporal.append("fiscal_period")
        temporal.append("date_range")
        temporal.append("point_in_time")
    return RetrievalContext(
        domain_hints=domain_hints,
        market_scope="hk",
        entity_scope_hints=["hk_listed_issuer", "hk_equity", "index"],
        identifier_types=["hk_code_5digit", "hk_code_4digit", "hk_ticker_dot", "index_symbol"],
        temporal_needs=temporal,
        source_type_prefs=["official_disclosure", "exchange_feed", "structured_vendor"],
        metric_basis_hints=list(task_spec.get("metric_basis_hints") or []),
        cutoff_date=task_spec.get("cutoff_date") or "2026-08-12",
        disallow_post_cutoff_spot=True,
    )


class BudgetTracker:
    def __init__(self, limits: Dict[str, Any]):
        self.limits = {
            "max_plan_revisions": int(limits.get("max_plan_revisions", 2)),
            "max_targeted_retrievals_per_step": int(
                limits.get("max_targeted_retrievals_per_step", 2)
            ),
            "max_total_mcp_invocations": int(limits.get("max_total_mcp_invocations", 8)),
            "max_llm_monitor_rounds": int(limits.get("max_llm_monitor_rounds", 6)),
            "max_same_operation_repeats": int(limits.get("max_same_operation_repeats", 1)),
            "max_final_repair_rounds": int(limits.get("max_final_repair_rounds", 1)),
            "max_idle_monitor_rounds": int(limits.get("max_idle_monitor_rounds", 1)),
        }
        self.plan_revisions = 0
        self.monitor_rounds = 0
        self.idle_monitor_rounds = 0
        self.mcp_invocations = 0
        self.retrievals_per_step: Dict[str, int] = {}
        self.operation_invoke_counts: Dict[str, int] = {}
        self.final_repair_rounds = 0
        self.llm_calls = 0
        self.stopping_reason: Optional[str] = None

    def snapshot(self) -> Dict[str, Any]:
        return {
            "limits": dict(self.limits),
            "plan_revisions": self.plan_revisions,
            "monitor_rounds": self.monitor_rounds,
            "idle_monitor_rounds": self.idle_monitor_rounds,
            "mcp_invocations": self.mcp_invocations,
            "retrievals_per_step": dict(self.retrievals_per_step),
            "operation_invoke_counts": dict(self.operation_invoke_counts),
            "final_repair_rounds": self.final_repair_rounds,
            "llm_calls": self.llm_calls,
            "stopping_reason": self.stopping_reason,
        }

    def exhausted(self) -> Optional[str]:
        if self.plan_revisions > self.limits["max_plan_revisions"]:
            return "max_plan_revisions"
        if self.monitor_rounds > self.limits["max_llm_monitor_rounds"]:
            return "max_llm_monitor_rounds"
        if self.mcp_invocations > self.limits["max_total_mcp_invocations"]:
            return "max_total_mcp_invocations"
        if self.idle_monitor_rounds > self.limits["max_idle_monitor_rounds"]:
            return "max_idle_monitor_rounds"
        return None


def execute_retrieve_step(
    *,
    step: PlanStep,
    task_spec: Dict[str, Any],
    resolver: OperationResolverServer,
    metadata_by_id: Dict[str, Dict[str, Any]],
    runtime_operation_ids: Sequence[str],
    ledger: EvidenceLedger,
    budget: BudgetTracker,
    top_k: int = 20,
    invoke_arguments: Optional[Dict[str, Any]] = None,
    select_operation_id: Optional[str] = None,
    allow_forced_select: bool = False,
) -> Dict[str, Any]:
    """Retrieve path: find → hard-filter/rerank → describe → invoke via gateway."""
    if select_operation_id and not allow_forced_select:
        return {
            "ok": False,
            "error": "forced_select_forbidden_in_live_mode",
            "step_id": step.step_id,
            "mcp_called": False,
        }
    count = budget.retrievals_per_step.get(step.step_id, 0)
    if count >= budget.limits["max_targeted_retrievals_per_step"]:
        return {
            "ok": False,
            "error": "max_targeted_retrievals_per_step",
            "step_id": step.step_id,
            "mcp_called": False,
        }
    budget.retrievals_per_step[step.step_id] = count + 1

    query = build_retrieval_query(step, task_spec)
    find_out = resolver.find_operations(query, top_k=top_k)
    candidates = list(find_out.get("candidates") or [])
    ctx = context_from_task_spec(task_spec)
    ranked = rerank_candidates(
        candidates,
        metadata_by_id,
        ctx,
        runtime_operation_ids=runtime_operation_ids,
    )
    reranked = ranked["reranked"]
    chosen = None
    forced_select_used = False
    if select_operation_id:
        chosen = next(
            (r for r in reranked if r["operation_id"] == select_operation_id), None
        )
        if chosen is None and select_operation_id in metadata_by_id:
            if not allow_forced_select:
                return {
                    "ok": False,
                    "error": "forced_select_forbidden_in_live_mode",
                    "step_id": step.step_id,
                    "query": query,
                    "find": find_out,
                    "rerank": ranked,
                    "mcp_called": False,
                }
            # Offline mock / unit-test override when hash retrieval misses the target.
            forced_select_used = True
            chosen = {
                "operation_id": select_operation_id,
                "original_rank": None,
                "base_score": 0.0,
                "compatibility_tier": "A",
                "compatibility_reasons": {
                    "supports": ["forced_select_override"],
                    "partial": [],
                    "conflicts": [],
                },
                "hard_filtered": False,
                "final_rank": 0,
                "candidate": {"operation_id": select_operation_id, "score": 0.0},
            }
    if chosen is None and reranked:
        # Prefer Tier A/B; allow C only as fallback.
        for tier in ("A", "B", "C"):
            chosen = next(
                (r for r in reranked if r["compatibility_tier"] == tier), None
            )
            if chosen is not None:
                break

    describe_out = None
    invoke_out = None
    mcp_called = False
    if chosen is not None:
        oid = chosen["operation_id"]
        describe_out = resolver.describe_operation(oid)
        repeats = budget.operation_invoke_counts.get(oid, 0)
        if repeats > budget.limits["max_same_operation_repeats"]:
            return {
                "ok": False,
                "error": "max_same_operation_repeats",
                "step_id": step.step_id,
                "query": query,
                "find": find_out,
                "rerank": ranked,
                "describe": describe_out,
                "mcp_called": False,
            }
        if describe_out.get("success"):
            if budget.mcp_invocations >= budget.limits["max_total_mcp_invocations"]:
                return {
                    "ok": False,
                    "error": "max_total_mcp_invocations",
                    "step_id": step.step_id,
                    "query": query,
                    "find": find_out,
                    "rerank": ranked,
                    "describe": describe_out,
                    "mcp_called": False,
                }
            args = dict(invoke_arguments or {})
            invoke_out = resolver.invoke_api(oid, args)
            mcp_called = True
            budget.mcp_invocations += 1
            budget.operation_invoke_counts[oid] = repeats + 1
            ev = ledger.add(
                step_id=step.step_id,
                claim=f"retrieve/invoke {oid}",
                source_type="mcp_invoke",
                operation_id=oid,
                raw_value=invoke_out,
                normalized_value=(invoke_out or {}).get("result"),
                provenance="UnifiedRealMcpGateway via OperationResolverServer",
                validation_status="valid" if invoke_out.get("success") else "invalid",
            )
            return {
                "ok": bool(invoke_out.get("success")),
                "step_id": step.step_id,
                "query": query,
                "find": find_out,
                "rerank": ranked,
                "chosen_operation_id": oid,
                "describe": describe_out,
                "invoke": invoke_out,
                "evidence_id": ev.evidence_id,
                "mcp_called": mcp_called,
                "forced_select_used": forced_select_used,
                "error": None if invoke_out.get("success") else invoke_out.get("error"),
            }

    return {
        "ok": False,
        "step_id": step.step_id,
        "query": query,
        "find": find_out,
        "rerank": ranked,
        "chosen_operation_id": None,
        "describe": describe_out,
        "invoke": invoke_out,
        "mcp_called": mcp_called,
        "forced_select_used": forced_select_used,
        "error": "no_compatible_operation",
    }


def execute_compute_step(
    *,
    llm: LLMClient,
    question: str,
    step: PlanStep,
    ledger: EvidenceLedger,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    scripted_payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Compute steps never call MCP."""
    if scripted_payload is not None:
        payload = scripted_payload
        llm_meta = None
        repair_count = 0
    else:
        messages = [
            {"role": "system", "content": APV_FIN_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": APV_FIN_COMPUTE_USER_TEMPLATE.format(
                    question=question,
                    step_id=step.step_id,
                    objective=step.objective,
                    evidence_json=json.dumps(ledger.to_list(), ensure_ascii=False)[:8000],
                ),
            },
        ]
        resp = llm.complete(messages, temperature=temperature, max_tokens=max_tokens)
        payload, repair_count, error = parse_and_validate(resp["content"], "apv_fin_compute_v1")
        llm_meta = resp
        if error is not None:
            messages.append({"role": "assistant", "content": resp["content"]})
            messages.append(
                {
                    "role": "user",
                    "content": format_repair_user_message("apv_fin_compute_v1", error),
                }
            )
            repair = llm.complete(
                messages, temperature=temperature, max_tokens=max_tokens
            )
            payload, repair_count, error2 = parse_and_validate(
                resp["content"],
                "apv_fin_compute_v1",
                repair_content=repair["content"],
            )
            llm_meta = {
                "content": repair["content"],
                "usage": _merge_usage(resp.get("usage"), repair.get("usage")),
                "latency_ms": float(resp.get("latency_ms") or 0)
                + float(repair.get("latency_ms") or 0),
                "provider": repair.get("provider"),
                "model": repair.get("model"),
            }
            if error2 is not None:
                return {
                    "ok": False,
                    "mcp_called": False,
                    "error": error2,
                    "repair_count": repair_count,
                    "llm": llm_meta,
                    "compute_trace": compute_trace_record(
                        step_id=step.step_id, payload={}, error=error2
                    ),
                }

    try:
        evaluated = evaluate_compute_payload(payload)
        trace = compute_trace_record(
            step_id=step.step_id, payload=payload, evaluated=evaluated
        )
        ev = ledger.add(
            step_id=step.step_id,
            claim=f"compute {payload.get('expression')}",
            source_type="compute",
            raw_value=payload,
            normalized_value=evaluated.get("result"),
            currency_or_unit=payload.get("unit"),
            provenance="deterministic_safe_arithmetic",
            validation_status="valid",
            formula=payload.get("expression"),
            operands=payload.get("operands"),
        )
        return {
            "ok": True,
            "mcp_called": False,
            "error": None,
            "repair_count": repair_count,
            "llm": llm_meta,
            "payload": payload,
            "evaluated": evaluated,
            "compute_trace": trace,
            "evidence_id": ev.evidence_id,
        }
    except CalculatorError as exc:
        trace = compute_trace_record(
            step_id=step.step_id, payload=payload, error=str(exc)
        )
        return {
            "ok": False,
            "mcp_called": False,
            "error": str(exc),
            "repair_count": repair_count,
            "llm": llm_meta,
            "payload": payload,
            "compute_trace": trace,
        }


def execute_reason_step(
    *,
    llm: LLMClient,
    question: str,
    step: PlanStep,
    ledger: EvidenceLedger,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    scripted_payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Reason/verify steps never call MCP; store concise summary only."""
    if scripted_payload is not None:
        parsed = scripted_payload
        llm_meta = None
    else:
        messages = [
            {"role": "system", "content": APV_FIN_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": APV_FIN_REASON_USER_TEMPLATE.format(
                    question=question,
                    step_id=step.step_id,
                    step_type=step.step_type,
                    objective=step.objective,
                    evidence_json=json.dumps(ledger.to_list(), ensure_ascii=False)[:8000],
                ),
            },
        ]
        resp = llm.complete(messages, temperature=temperature, max_tokens=max_tokens)
        llm_meta = resp
        try:
            parsed = json.loads(resp["content"])
        except Exception:  # noqa: BLE001
            parsed = {
                "summary": str(resp["content"])[:500],
                "decision": "continue",
                "resolved": True,
                "claims": [],
            }

    summary = str(parsed.get("summary") or parsed.get("decision_reason") or "")
    ev = ledger.add(
        step_id=step.step_id,
        claim=summary or f"{step.step_type} summary",
        source_type="reason" if step.step_type == "reason" else "verify",
        raw_value=parsed,
        normalized_value=summary,
        provenance="concise_reasoning_summary",
        validation_status="unchecked",
    )
    return {
        "ok": bool(parsed.get("resolved", True)),
        "mcp_called": False,
        "error": None,
        "llm": llm_meta,
        "payload": parsed,
        "evidence_id": ev.evidence_id,
    }


def select_next_step(
    plan_state: PlanState, preferred_ids: Optional[Sequence[str]] = None
) -> Optional[PlanStep]:
    ready = plan_state.ready_steps()
    if not ready:
        return None
    if preferred_ids:
        for sid in preferred_ids:
            for step in ready:
                if step.step_id == sid:
                    return step
    # Prefer retrieve → compute → reason/verify → answer/clarify/refuse
    priority = {
        "retrieve": 0,
        "compute": 1,
        "reason": 2,
        "verify": 3,
        "answer": 4,
        "clarify": 5,
        "refuse": 6,
    }
    ready = sorted(
        ready, key=lambda s: (priority.get(s.step_type, 9), s.step_id)
    )
    return ready[0]
