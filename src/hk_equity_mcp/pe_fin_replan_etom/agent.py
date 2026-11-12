"""PE-Fin RePlan-ETOM agent: Planner (once) -> [Executor (one step) -> Replanner]*7.

Graph (hard edges)::

    START -> Planner -> Executor -> Replanner -> Executor | END

* Planner runs exactly once and is not an execution round.
* Every round executes at most ONE ready step, then the Replanner decides
  ``plan`` (updated remaining steps) or ``response`` (final answer).
* Round ``max_execution_rounds`` (7) is response-only; there is no eighth round.
* The final response is written by the Replanner only. Any other termination is a
  *controlled failure* (never reported as a normal ``insufficient_evidence``).
* No Monitor, no global replan, no RePlan-4, no LangGraph.
* ETOM is routing-only: ``invoke`` records a predicted route node; no ETOM MCP
  tool is executed. Predicted tools / edges are read from the gateway nodes.

All LLM calls go through ``BudgetedLLMClient`` (one task-level counter <= 16);
MCP "invocations" (route nodes) are capped at 8 by the gateway budget.
"""

from __future__ import annotations

import time
from copy import deepcopy
from typing import Any, Dict, List, Optional, Sequence

from hk_equity_mcp.etom.adapter import EtomRoutingResolver
from hk_equity_mcp.operation_resolver.schemas import OperationSpec
from hk_equity_mcp.pe_fin_lite.budget import (
    BudgetedLLMClient,
    BudgetTracker,
    LlmCallBudgetExhausted,
    McpBudgetExhausted,
)
from hk_equity_mcp.pe_fin_lite.schemas import PlanValidationError
from hk_equity_mcp.pe_fin_replan.failed_invocation import FailedInvocationGuard
from hk_equity_mcp.pe_fin_replan.plan_ops import remove_step, select_one_ready_step
from hk_equity_mcp.pe_fin_replan.state import PlanExecuteState
from hk_equity_mcp.pe_fin_replan_etom import ETOM_FLAGS
from hk_equity_mcp.pe_fin_replan_etom.capability_profile import profile_for_planner
from hk_equity_mcp.pe_fin_replan_etom.evidence import EtomEvidenceStore
from hk_equity_mcp.pe_fin_replan_etom.executor import (
    ExecutorContext,
    execute_compute_step,
    execute_reason_step,
    execute_retrieve_step,
)
from hk_equity_mcp.pe_fin_replan_etom.gateway import EtomGateway
from hk_equity_mcp.pe_fin_replan_etom.grounding import (
    is_premature_insufficient,
    validate_replanner_response,
)
from hk_equity_mcp.pe_fin_replan_etom.identity import (
    DEFAULT_BUDGETS,
    METHOD_IDENTITY,
)
from hk_equity_mcp.pe_fin_replan_etom.planner import run_planner
from hk_equity_mcp.pe_fin_replan_etom.prompts import assert_prompts_clean
from hk_equity_mcp.pe_fin_replan_etom.replanner import (
    run_replanner,
    run_replanner_semantic_repair,
    validate_remaining_plan_update,
)
from hk_equity_mcp.pe_fin_replan_etom.schemas import (
    PLANNER_MAX_TOKENS,
    map_completion_to_predicted_status,
)

GRAPH_EDGES = (
    "START->Planner",
    "Planner->Executor",
    "Executor->Replanner",
    "Replanner->Executor|END",
)


def _merge_usage(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(a or {})
    for key, value in (b or {}).items():
        if isinstance(value, (int, float)) and isinstance(out.get(key), (int, float)):
            out[key] = out[key] + value
        else:
            out[key] = value
    return out


class PeFinReplanEtomAgent:
    """One instance may run many tasks; all per-task state is created in ``run``."""

    def __init__(
        self,
        llm: Any,
        resolver: EtomRoutingResolver,
        *,
        capability_profile: Dict[str, Any],
        registry: Dict[str, OperationSpec],
        official: Dict[str, Dict[str, Any]],
        vectors: Dict[str, Sequence[float]],
        embedder: Any,
        limits: Optional[Dict[str, Any]] = None,
        metadata_by_id: Optional[Dict[str, Dict[str, Any]]] = None,
        runtime_operation_ids: Optional[Sequence[str]] = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        planner_max_tokens: int = PLANNER_MAX_TOKENS,
    ):
        assert_prompts_clean()
        self._raw_llm = llm
        self.resolver = resolver
        self.capability_profile = capability_profile
        self.registry = registry
        self.official = official
        self.vectors = vectors
        self.embedder = embedder
        self.limits: Dict[str, Any] = dict(DEFAULT_BUDGETS)
        self.limits.update(dict(limits or {}))
        self.limits.setdefault("max_plan_steps", 6)
        self.metadata_by_id = metadata_by_id or {}
        self.runtime_operation_ids = (
            list(runtime_operation_ids) if runtime_operation_ids is not None else None
        )
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.planner_max_tokens = int(planner_max_tokens)
        # Populated per run (exposed for tests / audits).
        self.budget: Optional[BudgetTracker] = None
        self.gateway: Optional[EtomGateway] = None
        self.llm: Optional[BudgetedLLMClient] = None

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _sync(state: PlanExecuteState, budget: BudgetTracker) -> None:
        state.llm_calls = budget.llm_calls
        state.mcp_calls = budget.mcp_invocations
        state.embedding_queries = budget.embedding_query_calls

    def _execute_one_step(
        self, *, step: Dict[str, Any], question: str, ctx: ExecutorContext
    ) -> Dict[str, Any]:
        """Executor: exactly one step; never calls another executor function."""
        stype = step.get("type")
        if stype == "retrieve":
            out = execute_retrieve_step(step=step, question=question, ctx=ctx)
        elif stype == "compute":
            out = execute_compute_step(step=step, ctx=ctx)
        elif stype == "reason":
            out = execute_reason_step(step=step, question=question, ctx=ctx)
        else:
            out = {
                "step_result": {
                    "status": "failed",
                    "execution_ok": False,
                    "failure_reason": f"unknown_step_type:{stype}",
                    "values": {},
                    "missing_outputs": list(step.get("required_outputs") or []),
                },
                "usage": {},
            }
        result = out.get("step_result") or {}
        ok = (
            bool(result.get("execution_ok"))
            and not list(result.get("missing_outputs") or [])
            and result.get("status") == "completed"
        )
        past = {
            "step_id": str(step.get("id")),
            "step_type": stype,
            "status": result.get("status"),
            "ok": ok,
            "failure_reason": result.get("failure_reason"),
            "missing_outputs": list(result.get("missing_outputs") or []),
            "execution_failure_class": result.get("execution_failure_class"),
            "values": dict(result.get("values") or {}),
            "routing_only": result.get("routing_only"),
            "operation_id": (result.get("provenance") or {}).get("operation_id")
            or result.get("operation_id"),
            "route_order": result.get("route_order"),
        }
        return {
            "past": past,
            "ok": ok,
            "usage": out.get("usage") or {},
            "funnel": out.get("funnel"),
            "result": result,
        }

    def _compact_state(
        self,
        *,
        state: PlanExecuteState,
        gateway: EtomGateway,
        latest: Optional[Dict[str, Any]],
        response_only: bool,
    ) -> Dict[str, Any]:
        compact = state.compact_for_replanner(latest=latest)
        compact["forced_final_round"] = bool(response_only)
        compact["routing_only"] = True
        compact["predicted_route"] = [
            {
                "step_id": row.get("step_id"),
                "operation_id": row.get("operation_id"),
                "order": row.get("order"),
            }
            for row in gateway.invocation_log
            if row.get("route_recorded")
        ]
        return compact

    # --------------------------------------------------------------------- run
    def run(self, question: str) -> Dict[str, Any]:
        started = time.perf_counter()
        budget = BudgetTracker(self.limits)
        llm = BudgetedLLMClient(self._raw_llm, budget)
        guard = FailedInvocationGuard()
        gateway = EtomGateway(self.resolver, budget=budget, guard=guard)
        gateway.reset_route()
        store = EtomEvidenceStore()
        self.budget, self.llm, self.gateway = budget, llm, gateway

        max_rounds = int(
            self.limits.get(
                "max_execution_rounds", DEFAULT_BUDGETS["max_execution_rounds"]
            )
        )
        state = PlanExecuteState(
            question=question,
            evidence_store=store,
            invocation_guard=guard,
            max_llm_calls=int(self.limits["max_total_llm_calls_per_task"]),
            max_mcp_calls=int(self.limits["max_total_mcp_invocations"]),
            max_execution_rounds=max_rounds,
            max_executed_steps=int(self.limits.get("max_executed_steps", max_rounds)),
            max_replan_rounds=int(self.limits.get("max_replan_rounds", max_rounds)),
            max_new_steps_per_replan=int(self.limits.get("max_new_steps_per_replan", 2)),
        )
        ctx = ExecutorContext(
            llm=llm,
            budget=budget,
            gateway=gateway,
            registry=self.registry,
            official=self.official,
            vectors=self.vectors,
            embedder=self.embedder,
            store=store,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_k=int(self.limits.get("retrieval_top_k", 5)),
            describe_n=int(self.limits.get("describe_top_n", 3)),
            metadata_by_id=self.metadata_by_id,
            runtime_operation_ids=self.runtime_operation_ids,
        )

        total_usage: Dict[str, Any] = {}
        funnels: List[Dict[str, Any]] = []
        trace: List[Dict[str, Any]] = []
        error: Optional[str] = None
        error_type: Optional[str] = None
        planner_calls = 0
        replan_calls = 0
        replan_repairs = 0
        completion_out: Optional[str] = None
        executed_ids = set()

        try:
            planner_out = run_planner(
                llm=llm,
                question=question,
                capability_profile=profile_for_planner(self.capability_profile),
                limits=self.limits,
                max_plan_steps=int(self.limits.get("max_plan_steps", 6)),
                temperature=self.temperature,
                max_tokens=self.planner_max_tokens,
            )
            planner_calls = 1 + int(planner_out.get("repair_count") or 0)
            budget.planner_repair_count = int(planner_out.get("repair_count") or 0)
            total_usage = _merge_usage(total_usage, planner_out.get("usage") or {})
            trace.append(
                {
                    "stage": "planner",
                    "ok": planner_out.get("ok"),
                    "repair_count": planner_out.get("repair_count"),
                    "error": planner_out.get("error"),
                    "counts_as_execution_round": False,
                }
            )
            if not planner_out.get("ok"):
                state.stopping_reason = str(
                    planner_out.get("stopping_reason") or "planner_invalid"
                )
                error = planner_out.get("error")
                error_type = str(
                    planner_out.get("error_type") or "planner_schema_or_semantics"
                )
            else:
                plan = planner_out["plan"]
                state.original_plan = deepcopy(plan)
                state.remaining_plan = deepcopy(list(plan.get("steps") or []))
                state.remaining_plan_history.append(
                    {"round": 0, "remaining_steps": deepcopy(state.remaining_plan)}
                )

            # ---- Executor(one step) -> Replanner loop (only if the Planner succeeded)
            while state.original_plan is not None and state.final_response is None:
                self._sync(state, budget)
                if state.llm_calls >= state.max_llm_calls:
                    state.stopping_reason = "max_total_llm_calls"
                    state.resource_limit_failure = True
                    break
                if state.mcp_calls >= state.max_mcp_calls:
                    state.stopping_reason = "max_total_mcp_calls"
                    state.resource_limit_failure = True
                    break
                if state.execution_round_count >= state.max_execution_rounds:
                    state.stopping_reason = "max_execution_rounds"
                    state.resource_limit_failure = True
                    break
                if state.no_progress_round_count >= state.max_no_progress_rounds:
                    state.stopping_reason = "two_consecutive_no_progress_rounds"
                    state.loop_prevention_trigger = "no_progress"
                    break

                next_round = state.execution_round_count + 1
                response_only = next_round >= state.max_execution_rounds
                state.forced_final_round = bool(response_only)

                # ---------------- Executor: at most ONE ready step -----------------
                step = select_one_ready_step(state.remaining_plan, state.completed_step_ids)
                if step is None:
                    latest: Optional[Dict[str, Any]] = {
                        "step_id": None,
                        "status": "no_executable_ready_step",
                        "note": "remaining_plan_blocked_or_empty",
                    }
                    trace.append(
                        {
                            "stage": "executor",
                            "round": next_round,
                            "step_id": None,
                            "status": "no_executable_ready_step",
                            "ok": False,
                        }
                    )
                else:
                    if state.executed_step_count >= state.max_executed_steps:
                        state.stopping_reason = "max_executed_steps"
                        state.resource_limit_failure = True
                        break
                    exec_out = self._execute_one_step(step=step, question=question, ctx=ctx)
                    total_usage = _merge_usage(total_usage, exec_out.get("usage") or {})
                    if exec_out.get("funnel"):
                        funnels.append(exec_out["funnel"])
                    past = exec_out["past"]
                    sid = past["step_id"]
                    state.past_steps.append(past)
                    state.executed_step_count += 1
                    executed_ids.add(sid)
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
                        "operation_id": past.get("operation_id"),
                        "routing_only": past.get("routing_only"),
                    }
                    trace.append(
                        {
                            "stage": "executor",
                            "round": next_round,
                            "step_id": sid,
                            "step_type": step.get("type"),
                            "status": past.get("status"),
                            "ok": past.get("ok"),
                        }
                    )
                    self._sync(state, budget)

                if state.llm_calls >= state.max_llm_calls:
                    state.stopping_reason = "max_total_llm_calls"
                    state.resource_limit_failure = True
                    break

                # ---------------- Replanner ----------------------------------------
                compact = self._compact_state(
                    state=state, gateway=gateway, latest=latest, response_only=response_only
                )
                replan_out = run_replanner(
                    llm=llm,
                    compact_state=compact,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    response_only=response_only,
                    execution_round=next_round,
                    max_execution_rounds=state.max_execution_rounds,
                )
                replan_calls += 1
                replan_repairs += int(replan_out.get("repair_count") or 0)
                total_usage = _merge_usage(total_usage, replan_out.get("usage") or {})
                state.execution_round_count = next_round
                state.replan_round_count = next_round
                self._sync(state, budget)

                if not replan_out.get("ok"):
                    state.stopping_reason = str(
                        replan_out.get("stopping_reason")
                        or "invalid_replanner_output_after_repair"
                    )
                    state.resource_limit_failure = True
                    error = replan_out.get("error")
                    error_type = str(
                        replan_out.get("error_type") or "replanner_schema_or_semantics"
                    )
                    trace.append(
                        {
                            "stage": "replanner",
                            "round": next_round,
                            "ok": False,
                            "response_only": response_only,
                            "error": error,
                            "error_type": error_type,
                        }
                    )
                    break

                payload = replan_out["payload"]
                action = payload.get("action")
                decision: Dict[str, Any] = {
                    "round": next_round,
                    "latest_step_id": (latest or {}).get("step_id"),
                    "latest_step_status": (latest or {}).get("status"),
                    "remaining_steps_before": [s.get("id") for s in state.remaining_plan],
                    "action": action,
                    "decision_reason": payload.get("decision_reason"),
                    "repair_count": replan_out.get("repair_count"),
                    "response_only": response_only,
                }

                if action == "plan" and response_only:
                    state.stopping_reason = "round7_plan_not_allowed"
                    state.resource_limit_failure = True
                    error = "round7_plan_not_allowed"
                    error_type = "replanner_schema_or_semantics"
                    state.replan_decisions.append(decision)
                    trace.append({"stage": "replanner", **decision})
                    break

                if action == "response":
                    resp = dict(payload.get("response") or {})
                    completion = str(resp.get("completion") or "")
                    if completion == "insufficient_evidence" and is_premature_insufficient(
                        execution_round=next_round,
                        max_execution_rounds=state.max_execution_rounds,
                        remaining_plan=state.remaining_plan,
                        completed_ids=state.completed_step_ids,
                        decision_reason=str(payload.get("decision_reason") or ""),
                    ):
                        state.premature_insufficient_blocked_count += 1
                        decision["premature_insufficient_blocked"] = True
                        state.replan_decisions.append(decision)
                        trace.append({"stage": "replanner", **decision})
                        if response_only:
                            state.stopping_reason = "premature_insufficient_on_final_round"
                            state.resource_limit_failure = True
                            error = "premature_insufficient_blocked"
                            error_type = "premature_insufficient"
                            break
                        state.no_progress_round_count += 1
                        continue

                    ground = validate_replanner_response(
                        response=resp,
                        question=question,
                        evidence_values=dict(store.values),
                        provenance_class_by_key=dict(store.provenance_class_by_key),
                    )
                    decision["grounding_ok"] = bool(ground.get("ok"))
                    decision["grounding_classifications"] = ground.get("classifications")
                    if not ground.get("ok"):
                        state.stopping_reason = "ungrounded_replanner_response"
                        state.loop_prevention_trigger = "grounding_validator"
                        error = ";".join(ground.get("reasons") or [])
                        error_type = "replanner_grounding_failure"
                        decision["grounding_reasons"] = ground.get("reasons")
                        state.replan_decisions.append(decision)
                        trace.append({"stage": "replanner", **decision})
                        break
                    state.final_response = resp
                    state.final_response_source = "replanner"
                    completion_out = completion
                    decision["completion"] = completion
                    decision["remaining_steps_after"] = []
                    state.replan_decisions.append(decision)
                    trace.append({"stage": "replanner", **decision})
                    break

                # action == "plan" (rounds 1..6 only)
                validation_kwargs = dict(
                    previous_remaining=state.remaining_plan,
                    completed_ids=state.completed_step_ids,
                    completed_objectives=state.completed_objectives,
                    max_new_steps=state.max_new_steps_per_replan,
                    executed_ids=executed_ids,
                )
                try:
                    new_remaining, diff = validate_remaining_plan_update(
                        proposed=list(payload.get("remaining_steps") or []),
                        **validation_kwargs,
                    )
                except PlanValidationError as exc:
                    # Shared single repair budget: schema repair already used -> stop.
                    if int(replan_out.get("repair_count") or 0) != 0:
                        state.stopping_reason = f"invalid_plan_update:{exc}"
                        state.resource_limit_failure = True
                        error, error_type = str(exc), "replanner_plan_validation"
                        decision["plan_validation_error"] = str(exc)
                        state.replan_decisions.append(decision)
                        trace.append({"stage": "replanner", **decision})
                        break
                    if state.llm_calls >= state.max_llm_calls:
                        state.stopping_reason = "max_total_llm_calls"
                        state.resource_limit_failure = True
                        error, error_type = str(exc), "replanner_plan_validation"
                        decision["plan_validation_error"] = str(exc)
                        state.replan_decisions.append(decision)
                        trace.append({"stage": "replanner", **decision})
                        break
                    semantic_out = run_replanner_semantic_repair(
                        llm=llm,
                        messages=list(replan_out.get("messages") or []),
                        previous_content=str(replan_out.get("assistant_content") or ""),
                        semantic_error=str(exc),
                        max_new_steps=state.max_new_steps_per_replan,
                        execution_round=next_round,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        response_only=response_only,
                    )
                    replan_repairs += 1
                    total_usage = _merge_usage(total_usage, semantic_out.get("usage") or {})
                    self._sync(state, budget)
                    decision["semantic_repair"] = True
                    decision["repair_count"] = 1
                    decision["plan_validation_error_before_repair"] = str(exc)
                    if not semantic_out.get("ok"):
                        state.stopping_reason = str(
                            semantic_out.get("stopping_reason")
                            or f"invalid_plan_update_after_semantic_repair:{exc}"
                        )
                        state.resource_limit_failure = True
                        error = str(semantic_out.get("error") or exc)
                        error_type = "replanner_plan_validation"
                        decision["plan_validation_error"] = str(exc)
                        state.replan_decisions.append(decision)
                        trace.append({"stage": "replanner", **decision})
                        break
                    payload = semantic_out["payload"]
                    decision["action"] = payload.get("action")
                    if payload.get("action") != "plan":
                        state.stopping_reason = (
                            f"semantic_repair_expected_plan_got_{payload.get('action')}"
                        )
                        state.resource_limit_failure = True
                        error, error_type = state.stopping_reason, "replanner_plan_validation"
                        state.replan_decisions.append(decision)
                        trace.append({"stage": "replanner", **decision})
                        break
                    try:
                        new_remaining, diff = validate_remaining_plan_update(
                            proposed=list(payload.get("remaining_steps") or []),
                            **validation_kwargs,
                        )
                    except PlanValidationError as exc2:
                        state.stopping_reason = (
                            f"invalid_plan_update_after_semantic_repair:{exc2}"
                        )
                        state.resource_limit_failure = True
                        error, error_type = str(exc2), "replanner_plan_validation"
                        decision["plan_validation_error"] = str(exc2)
                        state.replan_decisions.append(decision)
                        trace.append({"stage": "replanner", **decision})
                        break

                # ETOM progress rule: a round that executed an ok step, or whose plan
                # update changed the remaining plan, is progress. (HK counted only plan
                # diffs, which would stop a multi-step ETOM route after two rounds in
                # which the Replanner simply kept the remaining steps unchanged.)
                if diff.get("changed") or bool((latest or {}).get("ok")):
                    state.no_progress_round_count = 0
                else:
                    state.no_progress_round_count += 1
                state.plan_diff_history.append(diff)
                state.remaining_plan = new_remaining
                state.remaining_plan_history.append(
                    {"round": next_round, "remaining_steps": deepcopy(new_remaining)}
                )
                decision["remaining_steps_after"] = [s.get("id") for s in new_remaining]
                decision["added_step_ids"] = diff.get("added_step_ids")
                decision["removed_step_ids"] = diff.get("removed_step_ids")
                state.replan_decisions.append(decision)
                trace.append({"stage": "replanner", **decision})
                if not new_remaining:
                    state.no_progress_round_count += 1

        except (LlmCallBudgetExhausted, McpBudgetExhausted) as exc:
            state.stopping_reason = "resource_limit_failure"
            state.resource_limit_failure = True
            error, error_type = str(exc), "resource_limit_failure"
            trace.append({"stage": "budget_failure", "error": str(exc)})
        except Exception as exc:  # noqa: BLE001 - task-level isolation
            error = f"{type(exc).__name__}: {exc}"
            error_type = "task_exception"
            state.stopping_reason = state.stopping_reason or "task_exception"
            trace.append({"stage": "exception", "error": error})

        # ---------------------------------------------------------------- finalize
        if state.final_response is None:
            state.final_response = {
                "completion": "insufficient_evidence",
                "final_answer": "",
                "reasoning_brief": (
                    f"Controlled stop: {state.stopping_reason or 'unknown'}; "
                    "not a normal Replanner insufficient_evidence decision."
                ),
            }
            state.final_response_source = state.final_response_source or "controlled_failure"
        completion_out = completion_out or str(
            state.final_response.get("completion") or "insufficient_evidence"
        )
        if state.final_response_source == "replanner":
            predicted_status = map_completion_to_predicted_status(completion_out)
            status = "full" if completion_out == "answered" else "partial"
        else:
            predicted_status = "unavailable"
            status = "failed"
        self._sync(state, budget)

        stage_seq = [t.get("stage") for t in trace if t.get("stage") in ("executor", "replanner")]
        executor_to_executor = any(
            a == "executor" and b == "executor" for a, b in zip(stage_seq, stage_seq[1:])
        )
        identity = dict(METHOD_IDENTITY)
        return {
            "status": status,
            "error": error,
            "error_type": error_type,
            "final_answer": str(state.final_response.get("final_answer") or ""),
            "predicted_status": predicted_status,
            "completion": completion_out,
            "final_payload": state.final_response,
            "final_response_source": state.final_response_source,
            "tokens": total_usage,
            "latency_ms": (time.perf_counter() - started) * 1000.0,
            "trace": trace,
            "initial_plan": state.original_plan,
            "original_plan_hash": state.original_plan_hash(),
            "past_steps": list(state.past_steps),
            "executed_steps": list(state.past_steps),
            "remaining_plan_history": list(state.remaining_plan_history),
            "replan_decisions": list(state.replan_decisions),
            "plan_diff_history": list(state.plan_diff_history),
            "planner_calls": planner_calls,
            "replan_calls": replan_calls,
            "replan_repairs": replan_repairs,
            "execution_round_count": state.execution_round_count,
            "executed_step_count": state.executed_step_count,
            "max_execution_rounds": state.max_execution_rounds,
            "replan_round_count": state.replan_round_count,
            "forced_final_round": state.forced_final_round,
            "round7_response_only": state.max_execution_rounds == 7,
            "no_eighth_round": state.execution_round_count <= state.max_execution_rounds,
            "executor_to_executor_edge": executor_to_executor,
            "premature_insufficient_blocked_count": state.premature_insufficient_blocked_count,
            "no_progress_rounds": state.no_progress_round_count,
            "stopping_reason": state.stopping_reason,
            "resource_limit_failure": state.resource_limit_failure,
            "loop_prevention_trigger": state.loop_prevention_trigger,
            # ---- budgets
            "llm_calls": budget.llm_calls,
            "llm_calls_by_stage": dict(budget.calls_by_stage),
            "mcp_invocations": budget.mcp_invocations,
            "embedding_query_calls": budget.embedding_query_calls,
            "binding_calls": budget.binder_calls,
            "budget": budget.snapshot(),
            # ---- routing outputs (predictions come from gateway nodes, never gold)
            "predicted_tools": gateway.predicted_tools,
            "predicted_nodes": gateway.predicted_nodes,
            "dependency_edges": gateway.dependency_edges,
            "real_tool_executions": gateway.real_tool_executions,
            "invocation_log": list(gateway.invocation_log),
            "failed_signature_blocks": len(guard.signatures),
            "signature_blocked_count": guard.signature_blocked_count,
            "blocked_signature_reasons": dict(guard.blocked_reason_by_key),
            "retrieval_funnels": funnels,
            "evidence_store": store.to_dict(),
            "graph_edges": list(GRAPH_EDGES),
            # ---- identity labels
            "condition": identity["condition"],
            "method_version": identity["method_version"],
            "protocol_version": identity["protocol_version"],
            "architecture": identity["architecture"],
            "posthoc_exploratory": True,
            "formal_primary_result": False,
            "flags": dict(ETOM_FLAGS),
        }
