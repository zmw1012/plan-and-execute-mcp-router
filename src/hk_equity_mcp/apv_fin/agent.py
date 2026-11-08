"""APV-Fin agent: dynamic Plan-Visible + ReAct hybrid (proposed method).

Canonical live order:
  initial planner
  → scheduler selects ready steps
  → execute retrieve/compute/reason/verify
  → write evidence ledger
  → monitor (continue/replan/answer/clarify/refuse/fail)
  → final validator
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional, Set

from hk_equity_mcp.agent.llm_client import LLMClient
from hk_equity_mcp.apv_fin.evidence_ledger import EvidenceLedger
from hk_equity_mcp.apv_fin.metadata import load_and_validate_metadata
from hk_equity_mcp.apv_fin.monitor import run_monitor
from hk_equity_mcp.apv_fin.planner import _merge_usage, run_initial_planner
from hk_equity_mcp.apv_fin.prompts import (
    APV_FIN_FINAL_USER_TEMPLATE,
    APV_FIN_SYSTEM_PROMPT,
    assert_prompts_clean,
)
from hk_equity_mcp.apv_fin.scheduler import (
    BudgetTracker,
    execute_compute_step,
    execute_reason_step,
    execute_retrieve_step,
    select_next_step,
)
from hk_equity_mcp.apv_fin.structured_parse import parse_structured, repair_user_message
from hk_equity_mcp.apv_fin.validator import controlled_failure_payload, validate_final_state
from hk_equity_mcp.experiment.normalize import (
    agent_completion_to_fallback,
    normalize_final_payload,
)
from hk_equity_mcp.operation_resolver.server import OperationResolverServer


class ApvFinAgent:
    def __init__(
        self,
        llm: LLMClient,
        resolver: OperationResolverServer,
        *,
        config: Optional[Dict[str, Any]] = None,
        metadata_by_id: Optional[Dict[str, Dict[str, Any]]] = None,
        runtime_operation_ids: Optional[List[str]] = None,
        live_mode: bool = True,
    ):
        assert_prompts_clean()
        self.llm = llm
        self.resolver = resolver
        self.config = config or {}
        self.live_mode = bool(live_mode)
        self.limits = dict(self.config.get("limits") or {})
        self.budget = BudgetTracker(self.limits)
        self.temperature = float(self.config.get("llm", {}).get("temperature", 0.0))
        self.max_tokens = int(self.config.get("llm", {}).get("max_tokens", 2048))
        self.top_k = int((self.config.get("retrieval") or {}).get("top_k", 20))
        self.metadata_by_id = metadata_by_id or load_and_validate_metadata()
        self.runtime_operation_ids = list(
            runtime_operation_ids
            or sorted(self.resolver.list_underlying_operation_ids())
        )
        # Mock-only hooks — must remain unreachable in live mode.
        self._allow_mock_hooks = bool(self.config.get("allow_mock_hooks", False))
        mock_hook_keys = (
            "mock_retrieve_arguments",
            "mock_compute_payload",
            "mock_reason_payload",
            "mock_select_operation_id",
        )
        if self.live_mode and self._allow_mock_hooks:
            raise RuntimeError("APV-Fin fail-closed: allow_mock_hooks forbidden in live_mode")
        if self.live_mode and any(self.config.get(k) for k in mock_hook_keys):
            raise RuntimeError(
                "APV-Fin fail-closed: mock hooks present while live_mode=True"
            )
        self._scripted_retrieve_args = (
            dict(self.config.get("mock_retrieve_arguments") or {})
            if self._allow_mock_hooks
            else {}
        )
        self._scripted_compute = (
            self.config.get("mock_compute_payload") if self._allow_mock_hooks else None
        )
        self._scripted_reason = (
            self.config.get("mock_reason_payload") if self._allow_mock_hooks else None
        )
        self._scripted_select_op = (
            self.config.get("mock_select_operation_id") if self._allow_mock_hooks else None
        )

    def run(self, question: str) -> Dict[str, Any]:
        started = time.perf_counter()
        self.budget = BudgetTracker(self.limits)
        trace: List[Dict[str, Any]] = []
        ledger = EvidenceLedger()
        compute_traces: List[Dict[str, Any]] = []
        observations: List[Dict[str, Any]] = []
        scheduler_decisions: List[Dict[str, Any]] = []
        monitor_decisions: List[Dict[str, Any]] = []
        plan_revisions: List[Dict[str, Any]] = []
        total_in = 0
        total_out = 0
        parse_repair_count = 0
        resolver_calls = {"find_operations": 0, "describe_operation": 0}
        predicted_operations: List[str] = []
        selected_operations: List[str] = []
        invoked_operations: List[str] = []
        successful_invoked: List[str] = []
        retrieved_candidates: List[Any] = []
        executed_step_ids: Set[str] = set()
        status = "failed"
        error: Optional[str] = None
        final_obj: Optional[Dict[str, Any]] = None
        task_spec: Dict[str, Any] = {}
        plan_state = None
        validator_result: Optional[Dict[str, Any]] = None
        observations_since_monitor = 0

        def _acc_llm(llm_meta: Optional[Dict[str, Any]]) -> None:
            nonlocal total_in, total_out
            if not llm_meta:
                return
            self.budget.llm_calls += 1
            usage = llm_meta.get("usage") or {}
            total_in += int(usage.get("input_tokens") or 0)
            total_out += int(usage.get("output_tokens") or 0)

        # ---- 1) Initial planner -------------------------------------------------
        plan_out = run_initial_planner(
            self.llm,
            question,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        _acc_llm(plan_out.get("llm"))
        parse_repair_count += int(plan_out.get("repair_count") or 0)
        trace.append(
            {
                "type": "planner",
                "raw_contents": plan_out.get("raw_contents"),
                "result": {
                    k: v for k, v in plan_out.items() if k not in {"plan_state", "llm"}
                },
            }
        )
        if not plan_out.get("ok"):
            final_obj = controlled_failure_payload(
                reason=plan_out.get("error") or "planner_fail_closed"
            )
            status = "failed"
            error = plan_out.get("error")
            self.budget.stopping_reason = "planner_fail_closed"
        else:
            plan_state = plan_out["plan_state"]
            task_spec = dict(plan_out.get("task_spec") or {})
            ledger.add(
                step_id="planner",
                claim="initial task_spec",
                source_type="planner",
                raw_value=task_spec,
                normalized_value=task_spec.get("goal"),
                provenance="apv_fin_plan_v1",
                validation_status="unchecked",
            )
            plan_revisions.append(
                {"revision": plan_state.revision, "note": "initial_plan"}
            )

            # ---- 2) Execute-ready → monitor loop --------------------------------
            while True:
                exhausted = self.budget.exhausted()
                if exhausted:
                    self.budget.stopping_reason = exhausted
                    final_obj = controlled_failure_payload(reason=exhausted)
                    status = "failed"
                    error = exhausted
                    break

                # Scheduler: execute currently ready steps BEFORE calling monitor.
                # Never call monitor in a loop with zero new observations.
                executed_this_round = 0
                while True:
                    step = select_next_step(plan_state)
                    if step is None:
                        break
                    if step.step_id in executed_step_ids and step.status == "running":
                        # Defensive: do not re-execute the same running step.
                        plan_state.mark_failed(step.step_id, "duplicate_execution_guard")
                        break
                    sched_event = {
                        "type": "scheduler",
                        "selected_step_id": step.step_id,
                        "step_type": step.step_type,
                        "ready_step_ids": [s.step_id for s in plan_state.ready_steps()],
                    }
                    scheduler_decisions.append(sched_event)
                    trace.append(sched_event)

                    plan_state.mark_running(step.step_id)
                    executed_step_ids.add(step.step_id)
                    step_event: Dict[str, Any] = {
                        "type": "step_execution",
                        "step_id": step.step_id,
                        "step_type": step.step_type,
                    }

                    if step.step_type == "retrieve":
                        out = execute_retrieve_step(
                            step=step,
                            task_spec=task_spec,
                            resolver=self.resolver,
                            metadata_by_id=self.metadata_by_id,
                            runtime_operation_ids=self.runtime_operation_ids,
                            ledger=ledger,
                            budget=self.budget,
                            top_k=self.top_k,
                            invoke_arguments=(
                                self._scripted_retrieve_args.get(step.step_id)
                                or self._scripted_retrieve_args.get("*")
                            ),
                            select_operation_id=self._scripted_select_op,
                            allow_forced_select=self._allow_mock_hooks,
                        )
                        if out.get("forced_select_used") and self.live_mode:
                            final_obj = controlled_failure_payload(
                                reason="forced_select_in_live_mode"
                            )
                            status = "failed"
                            error = "forced_select_in_live_mode"
                            self.budget.stopping_reason = error
                            step_event["result"] = out
                            trace.append(step_event)
                            break
                        resolver_calls["find_operations"] += 1
                        if out.get("describe") is not None:
                            resolver_calls["describe_operation"] += 1
                        if out.get("rerank"):
                            retrieved_candidates.append(
                                {
                                    "step_id": step.step_id,
                                    "query": out.get("query"),
                                    "original_candidates": [
                                        {
                                            "operation_id": c.get("operation_id"),
                                            "score": c.get("score"),
                                            "original_rank": i + 1,
                                        }
                                        for i, c in enumerate(
                                            (out.get("find") or {}).get("candidates")
                                            or []
                                        )
                                    ],
                                    "reranked": [
                                        {
                                            "operation_id": r["operation_id"],
                                            "original_rank": r["original_rank"],
                                            "final_rank": r.get("final_rank"),
                                            "compatibility_tier": r.get(
                                                "compatibility_tier"
                                            ),
                                            "compatibility_reasons": r.get(
                                                "compatibility_reasons"
                                            ),
                                            "base_score": r.get("base_score"),
                                        }
                                        for r in (out["rerank"].get("reranked") or [])[
                                            :20
                                        ]
                                    ],
                                    "hard_filtered": out["rerank"].get("hard_filtered"),
                                }
                            )
                        if out.get("chosen_operation_id"):
                            selected_operations.append(out["chosen_operation_id"])
                            predicted_operations.append(out["chosen_operation_id"])
                        if out.get("invoke") is not None:
                            oid = out.get("chosen_operation_id")
                            if oid:
                                invoked_operations.append(oid)
                            if out.get("ok") and oid:
                                successful_invoked.append(oid)
                        step_event["result"] = {
                            k: v for k, v in out.items() if k not in {"find"}
                        }
                        observations.append(step_event["result"])
                        observations_since_monitor += 1
                        trace.extend(_ov_compatible_retrieve_events(out))
                        if out.get("ok"):
                            plan_state.mark_resolved(
                                step.step_id,
                                observation_ids=[out.get("evidence_id")]
                                if out.get("evidence_id")
                                else None,
                            )
                        else:
                            plan_state.mark_failed(
                                step.step_id, out.get("error") or "retrieve_failed"
                            )

                    elif step.step_type == "compute":
                        out = execute_compute_step(
                            llm=self.llm,
                            question=question,
                            step=step,
                            ledger=ledger,
                            temperature=self.temperature,
                            max_tokens=self.max_tokens,
                            scripted_payload=self._scripted_compute,
                        )
                        _acc_llm(out.get("llm"))
                        parse_repair_count += int(out.get("repair_count") or 0)
                        if out.get("compute_trace"):
                            compute_traces.append(out["compute_trace"])
                        assert out.get("mcp_called") is False
                        step_event["result"] = {
                            k: v for k, v in out.items() if k != "llm"
                        }
                        observations.append(step_event["result"])
                        observations_since_monitor += 1
                        if out.get("ok"):
                            plan_state.mark_resolved(
                                step.step_id,
                                observation_ids=[out.get("evidence_id")]
                                if out.get("evidence_id")
                                else None,
                            )
                        else:
                            plan_state.mark_failed(
                                step.step_id, out.get("error") or "compute_failed"
                            )

                    elif step.step_type in {"reason", "verify"}:
                        out = execute_reason_step(
                            llm=self.llm,
                            question=question,
                            step=step,
                            ledger=ledger,
                            temperature=self.temperature,
                            max_tokens=self.max_tokens,
                            scripted_payload=self._scripted_reason,
                        )
                        _acc_llm(out.get("llm"))
                        assert out.get("mcp_called") is False
                        step_event["result"] = {
                            k: v for k, v in out.items() if k != "llm"
                        }
                        observations.append(step_event["result"])
                        observations_since_monitor += 1
                        if out.get("ok"):
                            plan_state.mark_resolved(
                                step.step_id,
                                observation_ids=[out.get("evidence_id")]
                                if out.get("evidence_id")
                                else None,
                            )
                        else:
                            plan_state.mark_failed(
                                step.step_id, out.get("error") or "reason_failed"
                            )

                    elif step.step_type in {"answer", "clarify", "refuse"}:
                        plan_state.mark_resolved(step.step_id)
                        step_event["result"] = {
                            "ok": True,
                            "mcp_called": False,
                            "terminal": step.step_type,
                        }
                        observations.append(step_event["result"])
                        observations_since_monitor += 1
                    else:
                        plan_state.mark_failed(
                            step.step_id, f"unknown_step_type:{step.step_type}"
                        )
                        step_event["result"] = {"ok": False, "mcp_called": False}
                        observations_since_monitor += 1

                    trace.append(step_event)
                    executed_this_round += 1
                    # Execute one ready step per scheduler pass, then monitor.
                    break

                if error == "forced_select_in_live_mode":
                    break

                # If plan complete after execution, synthesize without extra idle monitors.
                if plan_state.is_complete() and executed_this_round > 0:
                    final_pack = self._synthesize_final(
                        question=question,
                        plan_state=plan_state,
                        ledger=ledger,
                        task_spec=task_spec,
                        compute_traces=compute_traces,
                    )
                    _acc_llm(final_pack.pop("_llm", None))
                    parse_repair_count += int(final_pack.pop("_repair_count", 0) or 0)
                    validator_result = final_pack.get("validator")
                    if not final_pack.get("ok"):
                        status = "failed"
                        error = final_pack.get("error")
                        final_obj = final_pack.get("payload") or controlled_failure_payload(
                            reason=error or "final_fail_closed"
                        )
                    else:
                        final_obj = final_pack["payload"]
                        status = "full"
                        # Validator must pass for answered finals.
                        if (
                            final_obj.get("status") == "answered"
                            and validator_result
                            and not validator_result.get("ok")
                        ):
                            if (
                                self.budget.final_repair_rounds
                                < self.budget.limits["max_final_repair_rounds"]
                            ):
                                self.budget.final_repair_rounds += 1
                                final_obj = controlled_failure_payload(
                                    reason="validator_failed:"
                                    + ",".join(validator_result.get("failed_checks") or [])
                                )
                                self.budget.stopping_reason = "validator_controlled_failure"
                            else:
                                final_obj = controlled_failure_payload(
                                    reason="validator_failed_no_budget"
                                )
                                status = "failed"
                                error = "validator_failed_no_budget"
                                self.budget.stopping_reason = error
                        else:
                            self.budget.stopping_reason = "plan_complete"
                    break

                # Idle-monitor guard: no new observations and nothing ready → stop.
                if executed_this_round == 0 and observations_since_monitor == 0:
                    self.budget.idle_monitor_rounds += 1
                    if self.budget.idle_monitor_rounds > self.budget.limits[
                        "max_idle_monitor_rounds"
                    ]:
                        self.budget.stopping_reason = "max_idle_monitor_rounds"
                        final_obj = controlled_failure_payload(
                            reason="max_idle_monitor_rounds"
                        )
                        status = "failed"
                        error = "max_idle_monitor_rounds"
                        break
                    # One allowed idle monitor only when plan may need replan/clarify.
                else:
                    self.budget.idle_monitor_rounds = 0

                # Monitor uses real plan state + observations (after execution).
                self.budget.monitor_rounds += 1
                mon = run_monitor(
                    self.llm,
                    question=question,
                    plan_state=plan_state,
                    observations=observations,
                    evidence_summary=ledger.summary(),
                    budget=self.budget.snapshot(),
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                _acc_llm(mon.get("llm"))
                parse_repair_count += int(mon.get("repair_count") or 0)
                observations_since_monitor = 0
                if mon.get("patch_summary"):
                    self.budget.plan_revisions = plan_state.revision
                    plan_revisions.append(mon["patch_summary"])
                mon_rec = {
                    "decision": mon.get("decision"),
                    "decision_reason": mon.get("decision_reason"),
                    "ready_step_ids": mon.get("ready_step_ids"),
                    "ok": mon.get("ok"),
                    "error": mon.get("error"),
                    "patch_summary": mon.get("patch_summary"),
                    "raw_contents": mon.get("raw_contents"),
                }
                monitor_decisions.append(mon_rec)
                trace.append({"type": "monitor", "result": mon_rec})
                if not mon.get("ok"):
                    final_obj = controlled_failure_payload(
                        reason=mon.get("error") or "monitor_fail_closed"
                    )
                    status = "failed"
                    error = mon.get("error")
                    self.budget.stopping_reason = "monitor_fail_closed"
                    break

                decision = mon["decision"]
                if decision == "replan":
                    self.budget.plan_revisions = plan_state.revision
                    if self.budget.plan_revisions > self.budget.limits["max_plan_revisions"]:
                        self.budget.stopping_reason = "max_plan_revisions"
                        final_obj = controlled_failure_payload(reason="max_plan_revisions")
                        status = "failed"
                        error = "max_plan_revisions"
                        break
                    # After replan, execute newly ready steps on next loop iteration.
                    continue

                if decision in {"clarify", "refuse", "fail", "answer"}:
                    self.budget.stopping_reason = f"monitor_{decision}"
                    if decision == "clarify":
                        final_obj = {
                            "schema_version": "apv_fin_final_v1",
                            "status": "clarification_needed",
                            "final_answer": mon["payload"].get("clarify_question")
                            or mon.get("decision_reason")
                            or "Clarification needed.",
                            "reasoning_brief": mon.get("decision_reason") or "",
                            "evidence_ids": [i.evidence_id for i in ledger.items],
                            "clarification_question": mon["payload"].get(
                                "clarify_question"
                            ),
                            "refusal_reason": None,
                            "currency": None,
                            "unit": None,
                            "metric_basis": None,
                            "period": None,
                            "entity": None,
                        }
                        status = "full"
                        break
                    if decision in {"refuse", "fail"}:
                        reason = (
                            mon["payload"].get("refuse_reason")
                            or mon["payload"].get("fail_reason")
                            or mon.get("decision_reason")
                            or decision
                        )
                        final_obj = controlled_failure_payload(reason=str(reason))
                        status = "full" if decision == "refuse" else "failed"
                        break
                    final_pack = self._synthesize_final(
                        question=question,
                        plan_state=plan_state,
                        ledger=ledger,
                        task_spec=task_spec,
                        compute_traces=compute_traces,
                    )
                    _acc_llm(final_pack.pop("_llm", None))
                    parse_repair_count += int(final_pack.pop("_repair_count", 0) or 0)
                    validator_result = final_pack.get("validator")
                    if not final_pack.get("ok"):
                        status = "failed"
                        error = final_pack.get("error")
                        final_obj = final_pack.get("payload") or controlled_failure_payload(
                            reason=error or "final_fail_closed"
                        )
                    else:
                        final_obj = final_pack["payload"]
                        status = "full"
                        if (
                            final_obj.get("status") == "answered"
                            and validator_result
                            and not validator_result.get("ok")
                        ):
                            # answer_ready / answered but validator fails → controlled failure
                            if (
                                self.budget.final_repair_rounds
                                < self.budget.limits["max_final_repair_rounds"]
                            ):
                                self.budget.final_repair_rounds += 1
                                final_obj = controlled_failure_payload(
                                    reason="validator_failed:"
                                    + ",".join(
                                        validator_result.get("failed_checks") or []
                                    )
                                )
                                self.budget.stopping_reason = (
                                    "validator_controlled_failure"
                                )
                            else:
                                final_obj = controlled_failure_payload(
                                    reason="validator_failed_no_budget"
                                )
                                status = "failed"
                                error = "validator_failed_no_budget"
                                self.budget.stopping_reason = error
                    break

                if decision in {"continue", "verify"}:
                    # Continue executing newly ready steps; do not re-monitor without work.
                    continue

                final_obj = controlled_failure_payload(
                    reason=f"unknown_monitor_decision:{decision}"
                )
                status = "failed"
                error = f"unknown_monitor_decision:{decision}"
                self.budget.stopping_reason = error
                break

        if final_obj is None:
            final_obj = controlled_failure_payload(reason="no_final_payload")
            status = "failed"
            error = "no_final_payload"
            self.budget.stopping_reason = error

        if validator_result is None and plan_state is not None:
            validator_result = validate_final_state(
                plan_state=plan_state,
                ledger=ledger,
                task_spec=task_spec,
                final_payload=final_obj,
                compute_traces=compute_traces,
            )

        predicted_status, final_text = normalize_final_payload(
            final_obj, fallback_status=agent_completion_to_fallback(status)
        )
        latency_ms = (time.perf_counter() - started) * 1000.0

        def _dedupe(seq: List[str]) -> List[str]:
            seen: Set[str] = set()
            out: List[str] = []
            for x in seq:
                if x not in seen:
                    seen.add(x)
                    out.append(x)
            return out

        return {
            "status": status,
            "error": error,
            "final_answer": final_text,
            "predicted_status": predicted_status,
            "final_payload": final_obj,
            "task_spec": task_spec,
            "initial_plan": (plan_out.get("initial_plan") if plan_out else None),
            "plan_state": plan_state.to_dict() if plan_state else None,
            "plan_revisions": plan_state.revision if plan_state else 0,
            "plan_revision_log": plan_revisions,
            "scheduler_decisions": scheduler_decisions,
            "monitor_decisions": monitor_decisions,
            "evidence_ledger": ledger.to_list(),
            "evidence_summary": ledger.summary(),
            "compute_traces": compute_traces,
            "validator": validator_result,
            "stopping_reason": self.budget.stopping_reason,
            "budget": self.budget.snapshot(),
            "predicted_operations": _dedupe(predicted_operations),
            "selected_operations": _dedupe(selected_operations),
            "invoked_operations": _dedupe(invoked_operations),
            "successful_invoked_operations": _dedupe(successful_invoked),
            "retrieved_candidates": retrieved_candidates,
            "describe_calls": [
                t for t in trace if t.get("type") == "step_execution"
            ],
            "invoke_calls": [
                t
                for t in trace
                if t.get("type") == "step_execution"
                and (t.get("result") or {}).get("invoke") is not None
            ],
            "dependency_edges_predicted": _dependency_edges(plan_state),
            "tool_execution_success": _tool_execution_success(trace),
            "all_tool_invocations_successful": _tool_execution_success(trace),
            "successful_invocation_count": _invoke_counts(trace)[0],
            "failed_invocation_count": _invoke_counts(trace)[1],
            "recovered_after_tool_error": False,
            "protocol_success_rate": None,
            "final_operation_success_by_operation_id": {},
            "evidence": [
                {
                    "operation_id": i.operation_id,
                    "fields": [i.claim],
                    "note": i.provenance,
                    "evidence_id": i.evidence_id,
                }
                for i in ledger.items
                if i.source_type == "mcp_invoke"
            ],
            "citations": [],
            "llm_calls": self.budget.llm_calls,
            "tokens": {
                "input_tokens": total_in,
                "output_tokens": total_out,
                "total_tokens": total_in + total_out,
            },
            "embedding_calls": getattr(
                getattr(getattr(self.resolver, "retriever", None), "backend", None),
                "call_count",
                0,
            ),
            "embedding_tokens": getattr(
                getattr(getattr(self.resolver, "retriever", None), "backend", None),
                "token_count",
                0,
            ),
            "latency_ms": round(latency_ms, 3),
            "retry_count": parse_repair_count,
            "parse_repair_count": parse_repair_count,
            "trace": trace,
            "resolver_calls": resolver_calls,
            "underlying_operation_calls": self.budget.mcp_invocations,
            "real_mcp_execution_count": self.budget.mcp_invocations,
            "provider_metadata": {
                "llm_provider": getattr(self.llm, "provider", None),
                "llm_model": getattr(self.llm, "model", None),
                "live_mode": self.live_mode,
                "allow_mock_hooks": self._allow_mock_hooks,
            },
            "condition": "apv_fin",
            "execution_order": [
                "planner",
                "scheduler_ready",
                "execute_steps",
                "evidence_ledger",
                "monitor",
                "final_validator",
            ],
        }

    def _synthesize_final(
        self,
        *,
        question: str,
        plan_state,
        ledger: EvidenceLedger,
        task_spec: Dict[str, Any],
        compute_traces: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        coverage = plan_state.mandatory_coverage()
        pre_validator = validate_final_state(
            plan_state=plan_state,
            ledger=ledger,
            task_spec=task_spec,
            final_payload={"status": "answered", "evidence_ids": []},
            compute_traces=compute_traces,
        )
        messages = [
            {"role": "system", "content": APV_FIN_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": APV_FIN_FINAL_USER_TEMPLATE.format(
                    question=question,
                    coverage_json=json.dumps(coverage, ensure_ascii=False),
                    evidence_json=json.dumps(ledger.to_list(), ensure_ascii=False)[
                        :12000
                    ],
                    validator_json=json.dumps(pre_validator, ensure_ascii=False),
                ),
            },
        ]
        resp = self.llm.complete(
            messages, temperature=self.temperature, max_tokens=self.max_tokens
        )
        parsed = parse_structured(resp.get("content"), "apv_fin_final_v1")
        llm_meta = resp
        repair_count = int(parsed.get("repair_count") or 0)
        if not parsed["ok"]:
            messages.append({"role": "assistant", "content": str(resp.get("content") or "")})
            messages.append(
                {
                    "role": "user",
                    "content": repair_user_message(
                        "apv_fin_final_v1", parsed.get("error") or "parse_failed"
                    ),
                }
            )
            repair = self.llm.complete(
                messages, temperature=self.temperature, max_tokens=self.max_tokens
            )
            parsed = parse_structured(
                resp.get("content"),
                "apv_fin_final_v1",
                repair_content=repair.get("content"),
            )
            repair_count = int(parsed.get("repair_count") or 1)
            llm_meta = {
                "content": repair.get("content"),
                "raw_content": repair.get("raw_content", repair.get("content")),
                "usage": _merge_usage(resp.get("usage"), repair.get("usage")),
                "latency_ms": float(resp.get("latency_ms") or 0)
                + float(repair.get("latency_ms") or 0),
                "provider": repair.get("provider"),
                "model": repair.get("model"),
            }
            if not parsed["ok"]:
                return {
                    "ok": False,
                    "error": parsed.get("error") or "final_parse_fail_closed",
                    "payload": controlled_failure_payload(
                        reason=parsed.get("error") or "final_parse_fail_closed"
                    ),
                    "validator": pre_validator,
                    "_llm": llm_meta,
                    "_repair_count": repair_count,
                    "raw_contents": [
                        parsed.get("raw_content"),
                        parsed.get("repair_raw_content"),
                    ],
                }

        payload = parsed["payload"]
        validator = validate_final_state(
            plan_state=plan_state,
            ledger=ledger,
            task_spec=task_spec,
            final_payload=payload,
            compute_traces=compute_traces,
        )
        return {
            "ok": True,
            "error": None,
            "payload": payload,
            "validator": validator,
            "_llm": llm_meta,
            "_repair_count": repair_count,
            "raw_contents": [
                parsed.get("raw_content"),
                parsed.get("repair_raw_content"),
            ],
        }


def _ov_compatible_retrieve_events(out: Dict[str, Any]) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    find_out = out.get("find") or {}
    events.append(
        {
            "type": "tool_result",
            "name": "find_operations",
            "arguments": {"query": out.get("query"), "top_k": find_out.get("top_k")},
            "result": find_out,
            "success": bool(find_out.get("success", True)),
        }
    )
    if out.get("describe") is not None:
        oid = out.get("chosen_operation_id")
        events.append(
            {
                "type": "tool_result",
                "name": "describe_operation",
                "arguments": {"operation_id": oid},
                "result": out.get("describe"),
                "success": bool((out.get("describe") or {}).get("success")),
            }
        )
    if out.get("invoke") is not None:
        oid = out.get("chosen_operation_id")
        events.append(
            {
                "type": "tool_result",
                "name": "invoke_api",
                "arguments": {
                    "operation_id": oid,
                    "arguments": (out.get("invoke") or {}).get("arguments") or {},
                },
                "result": out.get("invoke"),
                "success": bool((out.get("invoke") or {}).get("success")),
            }
        )
    return events


def _dependency_edges(plan_state) -> List[Dict[str, str]]:
    if plan_state is None:
        return []
    edges = []
    for step in plan_state.steps.values():
        for dep in step.depends_on:
            edges.append({"from": dep, "to": step.step_id})
    return edges


def _invoke_counts(trace: List[Dict[str, Any]]) -> tuple:
    ok = 0
    fail = 0
    for t in trace:
        if t.get("type") != "step_execution":
            continue
        inv = (t.get("result") or {}).get("invoke")
        if inv is None:
            continue
        if inv.get("success"):
            ok += 1
        else:
            fail += 1
    return ok, fail


def _tool_execution_success(trace: List[Dict[str, Any]]) -> bool:
    ok, fail = _invoke_counts(trace)
    if ok + fail == 0:
        return True
    return fail == 0
