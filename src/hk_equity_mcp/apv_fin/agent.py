"""APV-Fin agent: dynamic Plan-Visible + ReAct hybrid (proposed method)."""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

from hk_equity_mcp.agent.llm_client import LLMClient
from hk_equity_mcp.apv_fin.evidence_ledger import EvidenceLedger
from hk_equity_mcp.apv_fin.metadata import load_and_validate_metadata
from hk_equity_mcp.apv_fin.monitor import run_monitor
from hk_equity_mcp.apv_fin.planner import run_initial_planner
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
from hk_equity_mcp.apv_fin.schemas import format_repair_user_message, parse_and_validate
from hk_equity_mcp.apv_fin.validator import controlled_failure_payload, validate_final_state
from hk_equity_mcp.experiment.normalize import (
    agent_completion_to_fallback,
    normalize_final_payload,
)
from hk_equity_mcp.operation_resolver.server import OperationResolverServer
from hk_equity_mcp.apv_fin.planner import _merge_usage


class ApvFinAgent:
    def __init__(
        self,
        llm: LLMClient,
        resolver: OperationResolverServer,
        *,
        config: Optional[Dict[str, Any]] = None,
        metadata_by_id: Optional[Dict[str, Dict[str, Any]]] = None,
        runtime_operation_ids: Optional[List[str]] = None,
    ):
        assert_prompts_clean()
        self.llm = llm
        self.resolver = resolver
        self.config = config or {}
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
        # Optional offline hooks for unit tests / mock scripts.
        self._scripted_retrieve_args = dict(
            self.config.get("mock_retrieve_arguments") or {}
        )
        self._scripted_compute = self.config.get("mock_compute_payload")
        self._scripted_reason = self.config.get("mock_reason_payload")
        self._scripted_select_op = self.config.get("mock_select_operation_id")

    def run(self, question: str) -> Dict[str, Any]:
        started = time.perf_counter()
        # Fresh budgets/trace state per task (agent instance may be reused by runner).
        self.budget = BudgetTracker(self.limits)
        trace: List[Dict[str, Any]] = []
        ledger = EvidenceLedger()
        compute_traces: List[Dict[str, Any]] = []
        observations: List[Dict[str, Any]] = []
        total_in = 0
        total_out = 0
        parse_repair_count = 0
        resolver_calls = {"find_operations": 0, "describe_operation": 0}
        predicted_operations: List[str] = []
        retrieved_candidates: List[Any] = []
        status = "failed"
        error: Optional[str] = None
        final_obj: Optional[Dict[str, Any]] = None
        task_spec: Dict[str, Any] = {}
        plan_state = None
        validator_result: Optional[Dict[str, Any]] = None

        def _acc_llm(llm_meta: Optional[Dict[str, Any]]) -> None:
            nonlocal total_in, total_out
            if not llm_meta:
                return
            self.budget.llm_calls += 1
            usage = llm_meta.get("usage") or {}
            total_in += int(usage.get("input_tokens") or 0)
            total_out += int(usage.get("output_tokens") or 0)

        # 1) Initial planner
        plan_out = run_initial_planner(
            self.llm,
            question,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        _acc_llm(plan_out.get("llm"))
        parse_repair_count += int(plan_out.get("repair_count") or 0)
        trace.append({"type": "planner", "result": {k: v for k, v in plan_out.items() if k != "plan_state"}})
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

            # 2) Monitor/execute loop
            while True:
                exhausted = self.budget.exhausted()
                if exhausted:
                    self.budget.stopping_reason = exhausted
                    final_obj = controlled_failure_payload(reason=exhausted)
                    status = "failed"
                    error = exhausted
                    break

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
                if mon.get("patch_summary"):
                    self.budget.plan_revisions = plan_state.revision
                trace.append(
                    {
                        "type": "monitor",
                        "result": {
                            k: v
                            for k, v in mon.items()
                            if k != "llm" or True
                        },
                    }
                )
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
                        final_obj = controlled_failure_payload(
                            reason="max_plan_revisions"
                        )
                        status = "failed"
                        error = "max_plan_revisions"
                        break
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
                    # answer → synthesize below
                    final_obj = self._synthesize_final(
                        question=question,
                        plan_state=plan_state,
                        ledger=ledger,
                        task_spec=task_spec,
                        compute_traces=compute_traces,
                    )
                    _acc_llm(final_obj.pop("_llm", None))
                    parse_repair_count += int(final_obj.pop("_repair_count", 0) or 0)
                    validator_result = final_obj.get("validator")
                    if not final_obj.get("ok"):
                        status = "failed"
                        error = final_obj.get("error")
                        final_obj = final_obj.get("payload") or controlled_failure_payload(
                            reason=error or "final_fail_closed"
                        )
                    else:
                        final_obj = final_obj["payload"]
                        status = "full"
                    break

                if decision in {"continue", "verify"}:
                    preferred = list(mon.get("ready_step_ids") or [])
                    step = select_next_step(plan_state, preferred)
                    if step is None:
                        # Nothing ready — ask monitor next round or stop.
                        if plan_state.is_complete():
                            final_obj = self._synthesize_final(
                                question=question,
                                plan_state=plan_state,
                                ledger=ledger,
                                task_spec=task_spec,
                                compute_traces=compute_traces,
                            )
                            _acc_llm(final_obj.pop("_llm", None))
                            parse_repair_count += int(
                                final_obj.pop("_repair_count", 0) or 0
                            )
                            if not final_obj.get("ok"):
                                status = "failed"
                                error = final_obj.get("error")
                                final_obj = final_obj.get(
                                    "payload"
                                ) or controlled_failure_payload(
                                    reason=error or "final_fail_closed"
                                )
                            else:
                                final_obj = final_obj["payload"]
                                status = "full"
                            self.budget.stopping_reason = "plan_complete"
                            break
                        continue

                    plan_state.mark_running(step.step_id)
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
                            invoke_arguments=self._scripted_retrieve_args.get(
                                step.step_id
                            )
                            or self._scripted_retrieve_args.get("*"),
                            select_operation_id=self._scripted_select_op,
                        )
                        resolver_calls["find_operations"] += 1
                        if out.get("describe") is not None:
                            resolver_calls["describe_operation"] += 1
                        if out.get("rerank"):
                            retrieved_candidates.append(
                                {
                                    "step_id": step.step_id,
                                    "query": out.get("query"),
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
                            predicted_operations.append(out["chosen_operation_id"])
                        step_event["result"] = {
                            k: v for k, v in out.items() if k not in {"find"}
                        }
                        observations.append(step_event["result"])
                        # Emit Operation-Visible-compatible tool_result events so
                        # evaluator v1.2 extractors reuse without baseline changes.
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
                        # Terminal plan nodes: resolve and let monitor/final handle.
                        plan_state.mark_resolved(step.step_id)
                        step_event["result"] = {
                            "ok": True,
                            "mcp_called": False,
                            "terminal": step.step_type,
                        }
                        observations.append(step_event["result"])
                    else:
                        plan_state.mark_failed(
                            step.step_id, f"unknown_step_type:{step.step_type}"
                        )
                        step_event["result"] = {"ok": False, "mcp_called": False}

                    trace.append(step_event)
                    continue

                # Unknown decision — fail closed.
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
            # One optional final repair if validator fails and budget allows.
            if (
                not validator_result.get("ok")
                and self.budget.final_repair_rounds
                < self.budget.limits["max_final_repair_rounds"]
                and status == "full"
                and final_obj.get("status") == "answered"
            ):
                self.budget.final_repair_rounds += 1
                # Convert to controlled refusal rather than fabricating.
                final_obj = controlled_failure_payload(
                    reason="validator_failed:"
                    + ",".join(validator_result.get("failed_checks") or [])
                )
                status = "full"
                self.budget.stopping_reason = "validator_controlled_failure"
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

        # Deduplicate predicted operations preserving order.
        seen = set()
        pred_ops: List[str] = []
        for oid in predicted_operations:
            if oid not in seen:
                seen.add(oid)
                pred_ops.append(oid)

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
            "evidence_ledger": ledger.to_list(),
            "evidence_summary": ledger.summary(),
            "compute_traces": compute_traces,
            "validator": validator_result,
            "stopping_reason": self.budget.stopping_reason,
            "budget": self.budget.snapshot(),
            "predicted_operations": pred_ops,
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
            "embedding_calls": getattr(self.resolver.retriever.backend, "call_count", 0)
            if hasattr(getattr(self.resolver, "retriever", None), "backend")
            else 0,
            "embedding_tokens": getattr(
                self.resolver.retriever.backend, "token_count", 0
            )
            if hasattr(getattr(self.resolver, "retriever", None), "backend")
            else 0,
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
            },
            "condition": "apv_fin",
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
        payload, repair_count, error = parse_and_validate(
            resp["content"], "apv_fin_final_v1"
        )
        llm_meta = resp
        if error is not None:
            messages.append({"role": "assistant", "content": resp["content"]})
            messages.append(
                {
                    "role": "user",
                    "content": format_repair_user_message("apv_fin_final_v1", error),
                }
            )
            repair = self.llm.complete(
                messages, temperature=self.temperature, max_tokens=self.max_tokens
            )
            payload, repair_count, error2 = parse_and_validate(
                resp["content"],
                "apv_fin_final_v1",
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
                    "error": error2,
                    "payload": controlled_failure_payload(reason=error2),
                    "validator": pre_validator,
                    "_llm": llm_meta,
                    "_repair_count": repair_count,
                }

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
        }


def _ov_compatible_retrieve_events(out: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Map APV-Fin retrieve output into OV tool_result trace shape."""
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
