"""APV-Fin agent: LLM planning/binding/exceptional replan/final + deterministic progress.

Canonical live order (v1.1):
  initial planner
  → scheduler selects ready steps
  → execute retrieve/compute/reason/verify
  → project observations + deterministic progressor (auto-resolve)
  → exceptional Monitor ONLY on failure/missing/unsupported/no-ready
  → final validator + Final Answer LLM (compact ledger)
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional, Set

from hk_equity_mcp.agent.llm_client import LLMClient
from hk_equity_mcp.apv_fin.evidence_ledger import EvidenceLedger
from hk_equity_mcp.apv_fin.llm_budget import BudgetedLLMClient, LlmCallBudgetExhausted
from hk_equity_mcp.apv_fin.capability_contract import (
    load_and_validate_capability_contracts,
)
from hk_equity_mcp.apv_fin.metadata import load_and_validate_metadata
from hk_equity_mcp.apv_fin.monitor import run_monitor
from hk_equity_mcp.apv_fin.planner import _merge_usage, run_initial_planner
from hk_equity_mcp.apv_fin.progressor import (
    apply_deterministic_progress,
    classify_after_step,
    needs_exceptional_replan,
)
from hk_equity_mcp.apv_fin.projection import compact_ledger_for_prompt
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
        capability_contracts: Optional[Dict[str, Dict[str, Any]]] = None,
        live_mode: bool = True,
    ):
        assert_prompts_clean()
        self._raw_llm = llm
        self.llm = llm  # replaced per-run with BudgetedLLMClient
        self.resolver = resolver
        self.config = config or {}
        self.live_mode = bool(live_mode)
        self.limits = dict(self.config.get("limits") or {})
        self.budget = BudgetTracker(self.limits)
        self.temperature = float(self.config.get("llm", {}).get("temperature", 0.0))
        self.max_tokens = int(self.config.get("llm", {}).get("max_tokens", 2048))
        self.top_k = int((self.config.get("retrieval") or {}).get("top_k", 20))
        self.metadata_by_id = metadata_by_id or load_and_validate_metadata()
        self.capability_contracts = (
            capability_contracts or load_and_validate_capability_contracts()
        )
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
        # All generation helpers must use this wrapper — never self._raw_llm.
        self.llm = BudgetedLLMClient(self._raw_llm, self.budget)
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
        binding_calls = 0
        binding_repair_count = 0
        binding_decisions: List[Any] = []
        described_candidate_ids: List[str] = []
        binding_selected_operations: List[str] = []
        grounded_arguments_log: List[Any] = []
        argument_sources_log: List[Any] = []
        argument_validation_log: List[Any] = []
        retrieval_reformulation_count = 0
        executed_step_ids: Set[str] = set()
        status = "failed"
        error: Optional[str] = None
        error_type: Optional[str] = None
        final_obj: Optional[Dict[str, Any]] = None
        task_spec: Dict[str, Any] = {}
        plan_state = None
        plan_out: Dict[str, Any] = {}
        validator_result: Optional[Dict[str, Any]] = None
        observations_since_monitor = 0
        last_classification: Optional[Dict[str, Any]] = None

        def _acc_tokens(llm_meta: Optional[Dict[str, Any]]) -> None:
            """Accumulate tokens only — call counts live solely in BudgetedLLMClient."""
            nonlocal total_in, total_out
            if not llm_meta:
                return
            usage = llm_meta.get("usage") or {}
            total_in += int(usage.get("input_tokens") or 0)
            total_out += int(usage.get("output_tokens") or 0)

        def _fail_llm_budget(exc: LlmCallBudgetExhausted) -> None:
            nonlocal final_obj, status, error, error_type
            self.budget.stopping_reason = "resource_limit_failure"
            error_type = LlmCallBudgetExhausted.error_type
            error = str(exc)
            status = "failed"
            final_obj = controlled_failure_payload(reason="resource_limit_failure")
            trace.append(
                {
                    "type": "llm_budget",
                    "error_type": error_type,
                    "used": exc.used,
                    "limit": exc.limit,
                    "stage": exc.stage,
                }
            )

        def _compact_obs_for_monitor() -> List[Dict[str, Any]]:
            compact: List[Dict[str, Any]] = []
            for obs in observations[-8:]:
                if isinstance(obs, dict) and obs.get("compact_observation"):
                    compact.append(obs["compact_observation"])
                elif isinstance(obs, dict):
                    compact.append(
                        {
                            k: obs.get(k)
                            for k in (
                                "step_id",
                                "ok",
                                "error",
                                "chosen_operation_id",
                                "projection",
                                "value",
                                "missing_operands",
                            )
                            if k in obs
                        }
                    )
            return compact

        # ---- 1) Initial planner -------------------------------------------------
        try:
            plan_out = run_initial_planner(
                self.llm,
                question,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        except LlmCallBudgetExhausted as exc:
            _fail_llm_budget(exc)
            plan_out = {"ok": False, "error": str(exc)}
        else:
            _acc_tokens(plan_out.get("llm"))
            parse_repair_count += int(plan_out.get("repair_count") or 0)
            trace.append(
                {
                    "type": "planner",
                    "raw_contents": plan_out.get("raw_contents"),
                    "result": {
                        k: v
                        for k, v in plan_out.items()
                        if k not in {"plan_state", "llm"}
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

        if plan_out.get("ok"):
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
                            llm=self.llm,
                            question=question,
                            invoke_arguments=(
                                self._scripted_retrieve_args.get(step.step_id)
                                or self._scripted_retrieve_args.get("*")
                            ),
                            select_operation_id=self._scripted_select_op,
                            allow_forced_select=self._allow_mock_hooks,
                            # Mock/offline: synthetic binding through the same validator;
                            # never empty-arg invoke and never skip Tool Binding.
                            offline_tool_binding=bool(self._allow_mock_hooks),
                            contracts=self.capability_contracts,
                        )
                        _acc_tokens((out.get("binding") or {}).get("llm"))
                        binding_meta = out.get("binding") or {}
                        if out.get("binding_mode") == "deterministic":
                            binding_decisions.append(
                                {
                                    "step_id": step.step_id,
                                    "decision": "select",
                                    "selected_operation_id": out.get(
                                        "binding_selected_operation"
                                    ),
                                    "ok": True,
                                    "binding_mode": "deterministic",
                                }
                            )
                        elif binding_meta:
                            binding_calls += 1
                            binding_repair_count += int(
                                binding_meta.get("repair_count") or 0
                            )
                            binding_decisions.append(
                                {
                                    "step_id": step.step_id,
                                    "decision": binding_meta.get("decision"),
                                    "selected_operation_id": binding_meta.get(
                                        "selected_operation_id"
                                    ),
                                    "ok": binding_meta.get("ok"),
                                    "error": binding_meta.get("error"),
                                    "binding_mode": out.get("binding_mode"),
                                    "raw_contents": binding_meta.get("raw_contents"),
                                }
                            )
                        if out.get("described_candidate_ids"):
                            described_candidate_ids.extend(
                                out.get("described_candidate_ids") or []
                            )
                        if out.get("binding_selected_operation"):
                            binding_selected_operations.append(
                                out["binding_selected_operation"]
                            )
                        if out.get("grounded_arguments") is not None:
                            grounded_arguments_log.append(
                                {
                                    "step_id": step.step_id,
                                    "arguments": out.get("grounded_arguments"),
                                }
                            )
                        if out.get("argument_sources") is not None:
                            argument_sources_log.append(
                                {
                                    "step_id": step.step_id,
                                    "sources": out.get("argument_sources"),
                                }
                            )
                        if out.get("argument_validation") is not None:
                            argument_validation_log.append(
                                {
                                    "step_id": step.step_id,
                                    "validation": out.get("argument_validation"),
                                }
                            )
                        retrieval_reformulation_count += int(
                            out.get("retrieval_reformulation_count") or 0
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
                        last_classification = classify_after_step(
                            step=step,
                            execution_ok=bool(out.get("ok") or out.get("mcp_execution_success")),
                            projection=out.get("projection"),
                            task_spec=task_spec,
                            error=out.get("error"),
                            adapted=out.get("adapted"),
                        )
                        # MCP failed → exceptional; MCP ok but missing outputs → exception
                        # unless narrative list expansion.
                        if out.get("mcp_execution_success") and not out.get("ok"):
                            if last_classification.get("action") != "expand_dependency":
                                last_classification = {
                                    "action": "exception",
                                    "trigger": "required_output_missing",
                                    "reason": out.get("error") or "required_output_missing",
                                    "missing_outputs": (
                                        (out.get("required_outputs_check") or {}).get(
                                            "missing"
                                        )
                                    ),
                                }
                        progress = apply_deterministic_progress(
                            plan_state=plan_state,
                            step=step,
                            classification=last_classification,
                            observation_ids=[out.get("evidence_id")]
                            if out.get("evidence_id")
                            else None,
                            projection=out.get("projection") or {"adapted": out.get("adapted")},
                            contracts=self.capability_contracts,
                            available_deps={},
                            already_invoked=invoked_operations,
                        )
                        if progress.get("resolved") or progress.get(
                            "dependency_auto_expansion_count"
                        ):
                            self.budget.deterministic_step_transitions += 1
                        self.budget.dependency_auto_expansion_count += int(
                            progress.get("dependency_auto_expansion_count") or 0
                        )
                        step_event["progress"] = progress
                        step.selected_operation = out.get("chosen_operation_id")
                        step.invoked_operation = out.get("chosen_operation_id")
                        step.execution_result = {
                            "ok": out.get("ok"),
                            "error": out.get("error"),
                        }

                    elif step.step_type == "compute":
                        out = execute_compute_step(
                            step=step,
                            ledger=ledger,
                            scripted_payload=self._scripted_compute,
                        )
                        if out.get("compute_trace"):
                            compute_traces.append(out["compute_trace"])
                        assert out.get("mcp_called") is False
                        assert out.get("llm_called") is False
                        step_event["result"] = {
                            k: v for k, v in out.items() if k != "llm"
                        }
                        observations.append(step_event["result"])
                        observations_since_monitor += 1
                        last_classification = classify_after_step(
                            step=step,
                            execution_ok=bool(out.get("ok")),
                            projection={
                                "ok": out.get("ok"),
                                "value": out.get("value"),
                            },
                            task_spec=task_spec,
                            error=out.get("error"),
                        )
                        if out.get("error") == "missing_compute_operands":
                            last_classification = {
                                "action": "exception",
                                "trigger": "missing_compute_operands",
                                "reason": "missing_compute_operands",
                            }
                        progress = apply_deterministic_progress(
                            plan_state=plan_state,
                            step=step,
                            classification=last_classification,
                            observation_ids=[out.get("evidence_id")]
                            if out.get("evidence_id")
                            else None,
                            projection={"ok": out.get("ok"), "value": out.get("value")},
                        )
                        if progress.get("resolved"):
                            self.budget.deterministic_step_transitions += 1
                        if out.get("ok"):
                            self.budget.deterministic_compute_count += 1
                        step_event["progress"] = progress

                    elif step.step_type in {"reason", "verify"}:
                        try:
                            out = execute_reason_step(
                                llm=self.llm,
                                question=question,
                                step=step,
                                ledger=ledger,
                                temperature=self.temperature,
                                max_tokens=self.max_tokens,
                                scripted_payload=self._scripted_reason,
                            )
                        except LlmCallBudgetExhausted as exc:
                            _fail_llm_budget(exc)
                            break
                        _acc_tokens(out.get("llm"))
                        assert out.get("mcp_called") is False
                        step_event["result"] = {
                            k: v for k, v in out.items() if k != "llm"
                        }
                        observations.append(step_event["result"])
                        observations_since_monitor += 1
                        last_classification = classify_after_step(
                            step=step,
                            execution_ok=bool(out.get("ok")),
                            projection={"ok": out.get("ok")},
                            task_spec=task_spec,
                            error=out.get("error"),
                        )
                        progress = apply_deterministic_progress(
                            plan_state=plan_state,
                            step=step,
                            classification=last_classification,
                            observation_ids=[out.get("evidence_id")]
                            if out.get("evidence_id")
                            else None,
                            projection={"ok": out.get("ok")},
                        )
                        if progress.get("resolved"):
                            self.budget.deterministic_step_transitions += 1
                        step_event["progress"] = progress

                    elif step.step_type == "answer":
                        # Answer is synthesized by Final Answer LLM after non-answer
                        # mandatory steps resolve — do not auto-resolve here.
                        step_event["result"] = {
                            "ok": True,
                            "mcp_called": False,
                            "terminal": "answer_ready",
                        }
                        observations.append(step_event["result"])
                        observations_since_monitor += 1
                        last_classification = {
                            "action": "resolve",
                            "trigger": None,
                            "reason": "answer_ready_deferred_to_final",
                        }

                    elif step.step_type in {"clarify", "refuse"}:
                        plan_state.mark_resolved(step.step_id)
                        self.budget.deterministic_step_transitions += 1
                        step_event["result"] = {
                            "ok": True,
                            "mcp_called": False,
                            "terminal": step.step_type,
                        }
                        observations.append(step_event["result"])
                        observations_since_monitor += 1
                        last_classification = {
                            "action": "resolve",
                            "trigger": None,
                            "reason": step.step_type,
                        }
                    else:
                        plan_state.mark_failed(
                            step.step_id, f"unknown_step_type:{step.step_type}"
                        )
                        step_event["result"] = {"ok": False, "mcp_called": False}
                        observations_since_monitor += 1
                        last_classification = {
                            "action": "exception",
                            "trigger": "capability_unsupported",
                            "reason": f"unknown_step_type:{step.step_type}",
                        }

                    trace.append(step_event)
                    executed_this_round += 1
                    break

                if error == "forced_select_in_live_mode" or error_type == "resource_limit_failure":
                    break

                # Final path: all mandatory non-answer steps resolved → Final Answer LLM.
                if plan_state.non_answer_mandatory_resolved():
                    try:
                        final_pack = self._synthesize_final(
                            question=question,
                            plan_state=plan_state,
                            ledger=ledger,
                            task_spec=task_spec,
                            compute_traces=compute_traces,
                        )
                    except LlmCallBudgetExhausted as exc:
                        _fail_llm_budget(exc)
                        break
                    _acc_tokens(final_pack.pop("_llm", None))
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
                        for sid, st in plan_state.steps.items():
                            if st.step_type == "answer" and st.status != "resolved":
                                plan_state.mark_resolved(sid)
                                self.budget.deterministic_step_transitions += 1
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

                # Exceptional Monitor only — never on normal deterministic success.
                trigger = needs_exceptional_replan(
                    plan_state=plan_state, last_classification=last_classification
                )
                if trigger is None:
                    # Normal progress: continue scheduling without Monitor LLM.
                    if executed_this_round == 0:
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
                    else:
                        self.budget.idle_monitor_rounds = 0
                    continue

                self.budget.exception_trigger = trigger
                self.budget.exception_replan_calls += 1
                self.budget.monitor_rounds += 1
                if self.budget.exception_replan_calls > self.budget.limits[
                    "max_exception_replan_calls_per_task"
                ]:
                    self.budget.stopping_reason = "max_exception_replan_calls_per_task"
                    final_obj = controlled_failure_payload(
                        reason="max_exception_replan_calls_per_task"
                    )
                    status = "failed"
                    error = "max_exception_replan_calls_per_task"
                    break

                try:
                    mon = run_monitor(
                        self.llm,
                        question=question,
                        plan_state=plan_state,
                        observations=_compact_obs_for_monitor(),
                        evidence_summary=compact_ledger_for_prompt(ledger.to_list()),
                        budget=self.budget.snapshot(),
                        exception_trigger=trigger,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                    )
                except LlmCallBudgetExhausted as exc:
                    _fail_llm_budget(exc)
                    break
                _acc_tokens(mon.get("llm"))
                parse_repair_count += int(mon.get("repair_count") or 0)
                observations_since_monitor = 0
                last_classification = None
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
                    "exception_trigger": trigger,
                    "raw_contents": mon.get("raw_contents"),
                }
                monitor_decisions.append(mon_rec)
                trace.append({"type": "exceptional_monitor", "result": mon_rec})
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
                    continue

                if decision in {"clarify", "refuse", "fail"}:
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
                    reason = (
                        mon["payload"].get("refuse_reason")
                        or mon["payload"].get("fail_reason")
                        or mon.get("decision_reason")
                        or decision
                    )
                    final_obj = controlled_failure_payload(reason=str(reason))
                    status = "full" if decision == "refuse" else "failed"
                    break

                if decision in {"continue", "verify", "answer"}:
                    # continue: resume deterministic scheduling; answer is Final path.
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
            "error_type": error_type
            or (
                "resource_limit_failure"
                if self.budget.stopping_reason
                in {
                    "resource_limit_failure",
                    "max_exception_replan_calls_per_task",
                    "max_plan_revisions",
                    "max_total_mcp_invocations",
                    "max_llm_monitor_rounds",
                }
                else None
            ),
            "final_answer": final_text,
            "predicted_status": predicted_status,
            "final_payload": final_obj,
            **self.budget.llm_budget_fields(),
            "task_spec": task_spec,
            "initial_plan": (plan_out.get("initial_plan") if plan_out else None),
            "plan_state": plan_state.to_dict() if plan_state else None,
            "plan_revisions": plan_state.revision if plan_state else 0,
            "plan_revision_log": plan_revisions,
            "scheduler_decisions": scheduler_decisions,
            "monitor_decisions": monitor_decisions,
            "exception_replan_calls": self.budget.exception_replan_calls,
            "deterministic_step_transitions": self.budget.deterministic_step_transitions,
            "dependency_auto_expansion_count": self.budget.dependency_auto_expansion_count,
            "schema_description_expansion_count": self.budget.schema_description_expansion_count,
            "repeated_signature_block_count": self.budget.repeated_signature_block_count,
            "raw_observation_chars": self.budget.raw_observation_chars,
            "projected_observation_chars": self.budget.projected_observation_chars,
            "compression_ratio": (
                (self.budget.projected_observation_chars / self.budget.raw_observation_chars)
                if self.budget.raw_observation_chars
                else None
            ),
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
            "binding_calls": binding_calls,
            "binding_repair_count": binding_repair_count,
            "binding_decisions": binding_decisions,
            "described_candidate_ids": _dedupe(described_candidate_ids),
            "binding_selected_operation": binding_selected_operations[-1]
            if binding_selected_operations
            else None,
            "binding_selected_operations": _dedupe(binding_selected_operations),
            "grounded_arguments": grounded_arguments_log,
            "argument_sources": argument_sources_log,
            "argument_validation": argument_validation_log,
            "retrieval_reformulation_count": retrieval_reformulation_count,
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
            "method_version": "apv_fin_v1_3_capability_compiler",
            "deterministic_binding_count": self.budget.deterministic_binding_count,
            "llm_binding_count": self.budget.llm_binding_count,
            "capability_injected_candidate_count": self.budget.capability_injected_candidate_count,
            "capability_filter_candidate_count": self.budget.capability_filter_candidate_count,
            "output_adapter_success_count": self.budget.output_adapter_success_count,
            "required_output_missing_count": self.budget.required_output_missing_count,
            "deterministic_compute_count": self.budget.deterministic_compute_count,
            "execution_order": [
                "planner",
                "capability_filter",
                "scheduler_ready",
                "execute_steps",
                "deterministic_progressor",
                "exceptional_monitor",
                "final_validator",
                "final_answer",
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
        compact = compact_ledger_for_prompt(ledger.to_list())
        citation_ids = [
            i.evidence_id
            for i in ledger.items
            if str(i.validation_status or "") in {"valid", "verified"}
        ]
        messages = [
            {"role": "system", "content": APV_FIN_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": APV_FIN_FINAL_USER_TEMPLATE.format(
                    question=question,
                    task_spec_json=json.dumps(task_spec, ensure_ascii=False),
                    coverage_json=json.dumps(coverage, ensure_ascii=False),
                    evidence_json=json.dumps(compact, ensure_ascii=False),
                    compute_json=json.dumps(compute_traces, ensure_ascii=False)[:4000],
                    citation_json=json.dumps(citation_ids, ensure_ascii=False),
                    validator_json=json.dumps(pre_validator, ensure_ascii=False),
                ),
            },
        ]
        resp = self.llm.complete(
            messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stage="final",
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
                messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stage="final_repair",
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


def _tool_execution_success(trace: List[Dict[str, Any]]) -> Optional[bool]:
    """True when every MCP invoke succeeded; null/N/A when zero invocations."""
    ok, fail = _invoke_counts(trace)
    if ok == 0 and fail == 0:
        return None
    if fail > 0:
        return False
    return True
