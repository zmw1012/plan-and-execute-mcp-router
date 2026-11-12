"""PE-Fin RePlan agent: Planner → Execute-one → Replanner → … → Response.

Graph edges (no LangGraph dependency):
  START → Planner → Executor(one) → Replanner
                                   ├─ plan → Executor(one)
                                   └─ response → END
No Executor→Executor edge. No Monitor.
"""

from __future__ import annotations

import time
from copy import deepcopy
from typing import Any, Dict, List, Optional

from hk_equity_mcp.agent.llm_client import LLMClient
from hk_equity_mcp.experiment.normalize import (
    agent_completion_to_fallback,
    normalize_final_payload,
)
from hk_equity_mcp.operation_resolver.server import OperationResolverServer
from hk_equity_mcp.pe_fin_lite.answerability_eea import classify_answerability
from hk_equity_mcp.pe_fin_lite.budget import (
    BudgetedLLMClient,
    BudgetTracker,
    LlmCallBudgetExhausted,
    McpBudgetExhausted,
)
from hk_equity_mcp.pe_fin_lite.capability_profile import profile_for_planner
from hk_equity_mcp.pe_fin_lite.executor import (
    execute_compute_step,
    execute_reason_step,
    execute_retrieve_step,
)
from hk_equity_mcp.pe_fin_lite.planner import run_planner
from hk_equity_mcp.pe_fin_lite.prompts import assert_prompts_clean
from hk_equity_mcp.pe_fin_lite.schemas import PlanValidationError
from hk_equity_mcp.pe_fin_lite.step_result import retrieve_step_succeeded
from hk_equity_mcp.pe_fin_lite.temporal_policy import DEFAULT_DATASET_CUTOFF
from hk_equity_mcp.pe_fin_replan import (
    PE_FIN_REPLAN_ARCHITECTURE,
    PE_FIN_REPLAN_CONDITION,
    PE_FIN_REPLAN_METHOD_VERSION,
    PE_FIN_REPLAN_PROTOCOL_VERSION,
    REPLAN_FLAGS,
)
from hk_equity_mcp.pe_fin_replan.grounding import validate_replanner_response
from hk_equity_mcp.pe_fin_replan.identity import DEFAULT_BUDGETS, replan_protocol_identity
from hk_equity_mcp.pe_fin_replan.plan_ops import (
    remove_step,
    select_one_ready_step,
    step_fingerprint,
    validate_updated_remaining_plan,
)
from hk_equity_mcp.pe_fin_replan.replanner import run_replanner
from hk_equity_mcp.pe_fin_replan.state import PlanExecuteState, failed_signature


def _merge_usage(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = {
        "prompt_tokens": int((a or {}).get("prompt_tokens") or 0)
        + int((b or {}).get("prompt_tokens") or 0),
        "completion_tokens": int((a or {}).get("completion_tokens") or 0)
        + int((b or {}).get("completion_tokens") or 0),
        "total_tokens": int((a or {}).get("total_tokens") or 0)
        + int((b or {}).get("total_tokens") or 0),
    }
    return out


class PeFinReplanAgent:
    def __init__(
        self,
        llm: LLMClient,
        resolver: OperationResolverServer,
        *,
        capability_profile: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None,
        metadata_by_id: Optional[Dict[str, Dict[str, Any]]] = None,
        runtime_operation_ids: Optional[List[str]] = None,
        live_mode: bool = True,
        allow_hkex_download: bool = False,
    ):
        assert_prompts_clean()
        self._raw_llm = llm
        self.llm = llm
        self.resolver = resolver
        self.capability_profile = capability_profile
        self.config = config or {}
        self.limits = dict(self.config.get("limits") or {})
        for k, v in DEFAULT_BUDGETS.items():
            self.limits.setdefault(k, v)
        self.temperature = float(self.config.get("llm", {}).get("temperature", 0.0))
        self.max_tokens = int(self.config.get("llm", {}).get("max_tokens", 2048))
        from hk_equity_mcp.pe_fin_lite.schemas import PLANNER_MAX_TOKENS

        llm_cfg = self.config.get("llm") or {}
        self.planner_max_tokens = int(
            llm_cfg.get("planner_max_tokens", PLANNER_MAX_TOKENS)
        )
        self.metadata_by_id = metadata_by_id or {}
        self.runtime_operation_ids = list(
            runtime_operation_ids
            or sorted(self.resolver.list_underlying_operation_ids())
        )
        self.live_mode = bool(live_mode)
        self.allow_hkex_download = bool(allow_hkex_download)
        self.budget = BudgetTracker(self.limits)
        self._use_mock_binder = bool(self.config.get("use_mock_binder", False))
        if self.live_mode and self._use_mock_binder:
            raise RuntimeError("RePlan fail-closed: use_mock_binder forbidden in live_mode")

    def _sync_state_counters(self, state: PlanExecuteState) -> None:
        state.llm_calls = int(self.budget.llm_calls)
        state.mcp_calls = int(self.budget.mcp_invocations)
        state.embedding_queries = int(self.budget.embedding_query_calls)

    def _execute_one_step(
        self,
        *,
        state: PlanExecuteState,
        step: Dict[str, Any],
        question: str,
    ) -> Dict[str, Any]:
        """Execute exactly one step; never continues to the next step."""
        stype = step.get("type")
        sid = str(step.get("id"))
        funnel = None
        usage: Dict[str, Any] = {}
        if stype == "retrieve":
            out = execute_retrieve_step(
                step=step,
                question=question,
                resolver=self.resolver,
                metadata_by_id=self.metadata_by_id,
                runtime_operation_ids=self.runtime_operation_ids,
                store=state.evidence_store,
                budget=self.budget,
                llm=self.llm,
                allow_hkex_download=self.allow_hkex_download,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                use_mock_binder=self._use_mock_binder,
                eea_enabled=True,
            )
            usage = dict(out.get("usage") or {})
            funnel = out.get("funnel")
            result = out["step_result"]
            # Record failed signatures from invoke map
            gw = (funnel or {}).get("invoke_gateway_success") or {}
            primary = (funnel or {}).get("primary") or {}
            for oid in (funnel or {}).get("invoked_operations") or []:
                if gw.get(oid) is False:
                    args = primary.get("arguments") if oid == primary.get("operation_id") else (
                        ((funnel or {}).get("fallback") or {}).get("arguments")
                    )
                    state.failed_signatures.add(
                        failed_signature(str(oid), args if isinstance(args, dict) else {})
                    )
            ok = retrieve_step_succeeded(result)
        elif stype == "compute":
            out = execute_compute_step(
                step=step, store=state.evidence_store, eea_enabled=True
            )
            result = out["step_result"]
            ok = result.get("status") == "completed"
        elif stype == "reason":
            out = execute_reason_step(
                step=step,
                store=state.evidence_store,
                llm=self.llm,
                question=question,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            usage = dict(out.get("usage") or {})
            result = out["step_result"]
            ok = result.get("status") == "completed"
        else:
            result = {
                "status": "failed",
                "execution_ok": False,
                "failure_reason": f"unknown_step_type:{stype}",
                "missing_outputs": [],
                "values": {},
            }
            ok = False

        past = {
            "step_id": sid,
            "step_type": stype,
            "status": result.get("status"),
            "execution_ok": bool(result.get("execution_ok")),
            "failure_reason": result.get("failure_reason"),
            "missing_outputs": list(result.get("missing_outputs") or []),
            "values": dict(result.get("values") or {}),
            "execution_failure_class": result.get("execution_failure_class")
            or ((funnel or {}).get("eea_execution") or {}).get("execution_failure_class"),
            "ok": ok,
            "funnel": funnel,
        }
        return {"past": past, "result": result, "usage": usage, "funnel": funnel, "ok": ok}

    def run(
        self,
        question: str,
        *,
        answerability_class: Optional[str] = None,
    ) -> Dict[str, Any]:
        started = time.perf_counter()
        self.budget = BudgetTracker(self.limits)
        self.llm = BudgetedLLMClient(self._raw_llm, self.budget)
        state = PlanExecuteState(
            question=question,
            max_llm_calls=int(self.limits["max_total_llm_calls_per_task"]),
            max_mcp_calls=int(self.limits["max_total_mcp_invocations"]),
            max_executed_steps=int(self.limits.get("max_executed_steps", 6)),
            max_replan_rounds=int(self.limits.get("max_replan_rounds", 6)),
            max_new_steps_per_replan=int(self.limits.get("max_new_steps_per_replan", 2)),
        )
        total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        funnels: List[Dict[str, Any]] = []
        planner_calls = 0
        replan_calls = 0
        replan_repairs = 0
        trace: List[Dict[str, Any]] = []
        error: Optional[str] = None
        error_type: Optional[str] = None
        status = "failed"
        predicted_status = "unavailable"
        final_text = ""
        early_answer_count = 0
        recovery_attempts = 0
        recovery_successes = 0
        steps_added = 0
        steps_removed = 0
        steps_modified = 0

        try:
            # ---- Planner exactly once ----
            planner_out = run_planner(
                llm=self.llm,
                question=question,
                capability_profile=profile_for_planner(self.capability_profile),
                limits=self.limits,
                max_plan_steps=int(self.limits.get("max_plan_steps", 6)),
                temperature=self.temperature,
                max_tokens=self.planner_max_tokens,
            )
            planner_calls = 1 + int(planner_out.get("repair_count") or 0)
            # BudgetedLLM already counted; track usage merge
            total_usage = _merge_usage(total_usage, planner_out.get("usage") or {})
            self._sync_state_counters(state)
            trace.append(
                {
                    "stage": "planner",
                    "ok": planner_out.get("ok"),
                    "repair_count": planner_out.get("repair_count"),
                    "error": planner_out.get("error"),
                }
            )
            if not planner_out.get("ok"):
                state.stopping_reason = str(
                    planner_out.get("stopping_reason") or "planner_invalid"
                )
                state.final_response = {
                    "status": "unavailable",
                    "final_answer": "",
                    "reasoning_brief": "Planner failed; controlled stop.",
                    "refusal_reason": state.stopping_reason,
                }
                state.final_response_source = "planner_failure"
                error = planner_out.get("error")
                error_type = str(planner_out.get("error_type") or "planner_schema_or_semantics")
            else:
                plan = planner_out["plan"]
                state.original_plan = deepcopy(plan)
                state.remaining_plan = deepcopy(list(plan.get("steps") or []))
                state.remaining_plan_history.append(
                    {"round": 0, "remaining_steps": deepcopy(state.remaining_plan)}
                )

            # ---- Loop: Executor(one) → Replanner ----
            while state.final_response is None:
                self._sync_state_counters(state)
                if state.llm_calls >= state.max_llm_calls:
                    state.stopping_reason = "max_total_llm_calls"
                    state.resource_limit_failure = True
                    break
                if state.mcp_calls >= state.max_mcp_calls:
                    state.stopping_reason = "max_total_mcp_calls"
                    state.resource_limit_failure = True
                    break
                if state.executed_step_count >= state.max_executed_steps:
                    state.stopping_reason = "max_executed_steps"
                    state.resource_limit_failure = True
                    break
                if state.replan_round_count >= state.max_replan_rounds:
                    state.stopping_reason = "max_replan_rounds"
                    state.resource_limit_failure = True
                    break
                if state.no_progress_round_count >= state.max_no_progress_rounds:
                    state.stopping_reason = "two_consecutive_no_progress_rounds"
                    state.loop_prevention_trigger = "no_progress"
                    break

                step = select_one_ready_step(
                    state.remaining_plan, state.completed_step_ids
                )
                if step is None:
                    # No ready step — force Replanner with empty remaining signal
                    latest = {
                        "step_id": None,
                        "status": "no_executable_ready_step",
                        "note": "remaining_plan_blocked_or_empty",
                    }
                else:
                    # Block repeating failed signatures for binder-selected ops is
                    # enforced post-invoke; execute one step now.
                    before_progress = (
                        len(state.evidence_store.values),
                        state.executed_step_count,
                    )
                    exec_out = self._execute_one_step(
                        state=state, step=step, question=question
                    )
                    total_usage = _merge_usage(total_usage, exec_out.get("usage") or {})
                    if exec_out.get("funnel"):
                        funnels.append(exec_out["funnel"])
                    past = exec_out["past"]
                    state.past_steps.append(past)
                    state.executed_step_count += 1
                    sid = str(step.get("id"))
                    state.remaining_plan = remove_step(state.remaining_plan, sid)
                    if exec_out.get("ok"):
                        state.completed_step_ids.add(sid)
                        state.completed_objectives.add(
                            str(step.get("objective") or "").strip().lower()
                        )
                    latest = {
                        "step_id": sid,
                        "step_type": step.get("type"),
                        "status": past.get("status"),
                        "ok": past.get("ok"),
                        "failure_reason": past.get("failure_reason"),
                        "missing_outputs": past.get("missing_outputs"),
                        "execution_failure_class": past.get("execution_failure_class"),
                        "values_keys": sorted((past.get("values") or {}).keys()),
                    }
                    after_progress = (
                        len(state.evidence_store.values),
                        state.executed_step_count,
                    )
                    if after_progress == before_progress or (
                        not exec_out.get("ok") and not (past.get("values") or {})
                    ):
                        # progress measured after replan too; mark tentative
                        pass
                    trace.append(
                        {
                            "stage": "executor",
                            "step_id": sid,
                            "step_type": step.get("type"),
                            "status": past.get("status"),
                            "ok": past.get("ok"),
                        }
                    )
                    self._sync_state_counters(state)

                # Answerability signal for Replanner (does NOT auto-stop)
                ans = classify_answerability(
                    plan={"steps": list(state.remaining_plan) + [
                        {"id": p["step_id"], "type": p["step_type"], "required_outputs": p.get("missing_outputs") or []}
                        for p in state.past_steps
                    ]},
                    store=state.evidence_store,
                    step_statuses=[
                        {
                            "id": p["step_id"],
                            "type": p["step_type"],
                            "status": p["status"],
                        }
                        for p in state.past_steps
                    ],
                    answerability_class=answerability_class,
                )
                state.answerability_snapshots.append(ans)

                # ---- Replanner (mandatory after executor / no-ready) ----
                if state.llm_calls >= state.max_llm_calls:
                    state.stopping_reason = "max_total_llm_calls"
                    state.resource_limit_failure = True
                    break

                compact = state.compact_for_replanner(latest=latest)
                compact["answerability_latest"] = ans
                replan_out = run_replanner(
                    llm=self.llm,
                    compact_state=compact,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    benchmark_cutoff=DEFAULT_DATASET_CUTOFF,
                )
                replan_calls += 1
                replan_repairs += int(replan_out.get("repair_count") or 0)
                total_usage = _merge_usage(total_usage, replan_out.get("usage") or {})
                state.replan_round_count += 1
                self._sync_state_counters(state)

                if not replan_out.get("ok"):
                    state.stopping_reason = str(
                        replan_out.get("stopping_reason")
                        or "invalid_replanner_output_after_repair"
                    )
                    state.resource_limit_failure = True
                    error = replan_out.get("error")
                    error_type = "replanner_schema_or_semantics"
                    break

                payload = replan_out["payload"]
                action = payload.get("action")
                remaining_before_ids = [s.get("id") for s in state.remaining_plan]
                decision = {
                    "round": state.replan_round_count,
                    "latest_step_id": (latest or {}).get("step_id"),
                    "latest_step_status": (latest or {}).get("status"),
                    "remaining_steps_before": remaining_before_ids,
                    "action": action,
                    "decision_reason": payload.get("decision_reason"),
                    "repair_count": replan_out.get("repair_count"),
                }

                if action == "response":
                    resp = dict(payload.get("response") or {})
                    ground = validate_replanner_response(
                        response=resp,
                        question=question,
                        evidence_values=dict(state.evidence_store.values),
                        answerability=ans,
                    )
                    if not ground.get("ok"):
                        # Controlled failure — do not pretend normal unavailable
                        state.stopping_reason = "ungrounded_replanner_response"
                        state.loop_prevention_trigger = "grounding_validator"
                        error = ";".join(ground.get("reasons") or [])
                        error_type = "replanner_grounding_failure"
                        decision["grounding_ok"] = False
                        decision["grounding_reasons"] = ground.get("reasons")
                        state.replan_decisions.append(decision)
                        trace.append({"stage": "replanner", **decision})
                        break
                    state.final_response = resp
                    state.final_response_source = "replanner"
                    if resp.get("status") == "answered" and state.executed_step_count < len(
                        (state.original_plan or {}).get("steps") or []
                    ):
                        early_answer_count += 1
                    decision["grounding_ok"] = True
                    decision["remaining_steps_after"] = []
                    state.replan_decisions.append(decision)
                    trace.append({"stage": "replanner", **decision})
                    break

                # action == plan
                try:
                    new_remaining, diff = validate_updated_remaining_plan(
                        proposed=list(payload.get("remaining_steps") or []),
                        previous_remaining=state.remaining_plan,
                        completed_ids=state.completed_step_ids,
                        completed_objectives=state.completed_objectives,
                        max_new_steps=state.max_new_steps_per_replan,
                    )
                except PlanValidationError as exc:
                    state.stopping_reason = f"invalid_plan_update:{exc}"
                    state.resource_limit_failure = True
                    error = str(exc)
                    error_type = "replanner_plan_validation"
                    decision["plan_validation_error"] = str(exc)
                    state.replan_decisions.append(decision)
                    trace.append({"stage": "replanner", **decision})
                    break

                # Progress / recovery bookkeeping
                if diff.get("changed"):
                    state.no_progress_round_count = 0
                    if not (latest or {}).get("ok"):
                        recovery_attempts += 1
                        if diff.get("n_added") or diff.get("n_modified"):
                            recovery_successes += 1  # provisional: revised after failure
                else:
                    state.no_progress_round_count += 1

                steps_added += int(diff.get("n_added") or 0)
                steps_removed += int(diff.get("n_removed") or 0)
                steps_modified += int(diff.get("n_modified") or 0)
                state.plan_diff_history.append(diff)
                state.remaining_plan = new_remaining
                state.remaining_plan_history.append(
                    {
                        "round": state.replan_round_count,
                        "remaining_steps": deepcopy(new_remaining),
                    }
                )
                decision["remaining_steps_after"] = [s.get("id") for s in new_remaining]
                decision["added_step_ids"] = diff.get("added_step_ids")
                decision["removed_step_ids"] = diff.get("removed_step_ids")
                decision["modified_step_ids"] = diff.get("modified_step_ids")
                state.replan_decisions.append(decision)
                trace.append({"stage": "replanner", **decision})

                if not new_remaining:
                    # Empty plan without response — force another replan round via loop
                    # but count as no progress if repeats
                    state.no_progress_round_count += 1
                    continue

            # End while

        except (LlmCallBudgetExhausted, McpBudgetExhausted) as exc:
            state.stopping_reason = "resource_limit_failure"
            state.resource_limit_failure = True
            error = str(exc)
            error_type = "resource_limit_failure"
            trace.append({"stage": "budget_failure", "error": str(exc)})
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
            error_type = "task_exception"
            state.stopping_reason = state.stopping_reason or "task_exception"
            trace.append({"stage": "exception", "error": error})

        # Finalize prediction fields
        if state.final_response is None:
            # Safety stop — not a normal unavailable
            state.final_response = {
                "status": "unavailable",
                "final_answer": "",
                "reasoning_brief": (
                    f"Controlled stop: {state.stopping_reason or 'unknown'}; "
                    "not a normal Replanner unavailable decision."
                ),
                "refusal_reason": state.stopping_reason,
            }
            if state.final_response_source is None:
                state.final_response_source = "controlled_failure"

        predicted_status, final_text = normalize_final_payload(
            state.final_response,
            fallback_status=agent_completion_to_fallback("partial"),
        )
        if state.final_response_source == "replanner" and state.final_response.get(
            "status"
        ) == "answered":
            status = "full" if state.evidence_store.values and not state.resource_limit_failure else "partial"
        elif state.final_response_source == "replanner":
            status = "partial"
        else:
            status = "failed"

        self._sync_state_counters(state)
        identity = replan_protocol_identity()
        return {
            "status": status,
            "error": error,
            "error_type": error_type,
            "final_answer": final_text,
            "predicted_status": predicted_status,
            "final_payload": state.final_response,
            "tokens": total_usage,
            "llm_calls": self.budget.llm_calls,
            "latency_ms": (time.perf_counter() - started) * 1000.0,
            "trace": trace,
            "initial_plan": state.original_plan,
            "original_plan_hash": state.original_plan_hash(),
            "executed_steps": list(state.past_steps),
            "remaining_plan_history": list(state.remaining_plan_history),
            "past_steps": list(state.past_steps),
            "replan_calls": replan_calls,
            "replan_repairs": replan_repairs,
            "replan_decisions": list(state.replan_decisions),
            "plan_diff_history": list(state.plan_diff_history),
            "steps_added": steps_added,
            "steps_removed": steps_removed,
            "steps_modified": steps_modified,
            "early_answer_count": early_answer_count,
            "recovery_attempts": recovery_attempts,
            "recovery_successes": recovery_successes,
            "failed_signature_blocks": len(state.failed_signatures),
            "no_progress_rounds": state.no_progress_round_count,
            "final_response_source": state.final_response_source,
            "stopping_reason": state.stopping_reason,
            "resource_limit_failure": state.resource_limit_failure,
            "loop_prevention_trigger": state.loop_prevention_trigger,
            "retrieval_funnels": funnels,
            "evidence_store": state.evidence_store.to_dict(),
            "evidence_ledger": state.evidence_store.to_dict(),
            "compute_traces": state.evidence_store.compute_records,
            "reason_artifacts": state.evidence_store.reason_artifacts,
            "step_statuses": [
                {
                    "id": p.get("step_id"),
                    "type": p.get("step_type"),
                    "status": p.get("status"),
                    "failure_reason": p.get("failure_reason"),
                    "missing_outputs": p.get("missing_outputs"),
                }
                for p in state.past_steps
            ],
            "budget": self.budget.snapshot(),
            "underlying_operation_calls": self.budget.mcp_invocations,
            "invoked_operations": [
                oid
                for f in funnels
                for oid in (f.get("invoked_operations") or [])
            ],
            "predicted_operations": [
                oid
                for f in funnels
                for oid in (f.get("invoked_operations") or [])
            ],
            "selected_operations": [
                oid
                for f in funnels
                for oid in (f.get("invoked_operations") or [])
            ],
            "successful_invoked_operations": [
                oid
                for f in funnels
                for oid, ok in ((f.get("invoke_gateway_success") or {}).items())
                if ok
            ],
            "resolver_calls": {
                "find_operations": sum(1 for f in funnels if f.get("search_query")),
                "describe_operation": sum(
                    len(f.get("described_candidates") or []) for f in funnels
                ),
            },
            "embedding_query_calls": self.budget.embedding_query_calls,
            "binding_calls": self.budget.binder_calls,
            "architecture": PE_FIN_REPLAN_ARCHITECTURE,
            "condition": PE_FIN_REPLAN_CONDITION,
            "method_version": PE_FIN_REPLAN_METHOD_VERSION,
            "protocol_version": PE_FIN_REPLAN_PROTOCOL_VERSION,
            "posthoc_exploratory": True,
            "formal_primary_result": False,
            "replan_flags": dict(REPLAN_FLAGS),
            "replan_identity": identity,
            "planner_calls": planner_calls,
            "monitor_decisions": [],
            "plan_revisions": list(state.plan_diff_history),
            "answerability_snapshots": list(state.answerability_snapshots),
            "graph_edges": [
                "START→Planner",
                "Planner→Executor",
                "Executor→Replanner",
                "Replanner→Executor|END",
            ],
            **self.budget.llm_budget_fields(),
        }
