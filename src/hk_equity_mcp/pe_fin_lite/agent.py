"""PE-Fin Lite agent: Planner → step executor → Final Synthesizer.

No global Monitor. No continuous Replanner.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional, Set

from hk_equity_mcp.agent.llm_client import LLMClient
from hk_equity_mcp.experiment.normalize import (
    agent_completion_to_fallback,
    normalize_final_payload,
)
from hk_equity_mcp.operation_resolver.server import OperationResolverServer
from hk_equity_mcp.pe_fin_lite.budget import (
    BudgetedLLMClient,
    BudgetTracker,
    LlmCallBudgetExhausted,
    McpBudgetExhausted,
    structured_budget_failure,
)
from hk_equity_mcp.pe_fin_lite.capability_profile import profile_for_planner
from hk_equity_mcp.pe_fin_lite.evidence_store import EvidenceStore
from hk_equity_mcp.pe_fin_lite.executor import (
    execute_compute_step,
    execute_reason_step,
    execute_retrieve_step,
)
from hk_equity_mcp.pe_fin_lite.planner import run_planner, topological_ready_order
from hk_equity_mcp.pe_fin_lite.prompts import (
    PE_FIN_LITE_SYSTEM_PROMPT,
    assert_prompts_clean,
    render_final_user,
)
from hk_equity_mcp.pe_fin_lite.schemas import (
    format_repair_user_message,
    parse_and_validate,
)
from hk_equity_mcp.pe_fin_lite.step_result import retrieve_step_succeeded


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


class PeFinLiteAgent:
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
        self.temperature = float(self.config.get("llm", {}).get("temperature", 0.0))
        self.max_tokens = int(self.config.get("llm", {}).get("max_tokens", 2048))
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
            raise RuntimeError("PE-Fin Lite fail-closed: use_mock_binder forbidden in live_mode")

    def run(self, question: str) -> Dict[str, Any]:
        started = time.perf_counter()
        self.budget = BudgetTracker(self.limits)
        self.llm = BudgetedLLMClient(self._raw_llm, self.budget)
        store = EvidenceStore()
        trace: List[Dict[str, Any]] = []
        step_statuses: List[Dict[str, Any]] = []
        funnels: List[Dict[str, Any]] = []
        total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        parse_repair_count = 0
        predicted_operations: List[str] = []
        selected_operations: List[str] = []
        invoked_operations: List[str] = []
        successful_invoked: List[str] = []
        resolver_calls = {"find_operations": 0, "describe_operation": 0}
        status = "failed"
        error: Optional[str] = None
        error_type: Optional[str] = None
        final_obj: Optional[Dict[str, Any]] = None
        plan: Optional[Dict[str, Any]] = None
        step_state: Dict[str, str] = {}

        try:
            planner_out = run_planner(
                llm=self.llm,
                question=question,
                capability_profile=profile_for_planner(self.capability_profile),
                limits=self.limits,
                max_plan_steps=int(self.limits.get("max_plan_steps", 6)),
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            parse_repair_count += int(planner_out.get("repair_count") or 0)
            self.budget.planner_repair_count = int(planner_out.get("repair_count") or 0)
            total_usage = _merge_usage(total_usage, planner_out.get("usage") or {})
            trace.append({"stage": "planner", "ok": planner_out.get("ok")})
            if not planner_out.get("ok"):
                raise RuntimeError(f"planner_failed:{planner_out.get('error')}")
            plan = planner_out["plan"]
            steps = list(plan.get("steps") or [])
            ordered = topological_ready_order(steps)
            completed_ok: Set[str] = set()
            failed_deps: Set[str] = set()

            for step in ordered:
                sid = str(step["id"])
                deps = [str(d) for d in (step.get("depends_on") or [])]
                bad_deps = [d for d in deps if d not in completed_ok]
                if bad_deps:
                    # Independent siblings continue; only dependents of failed steps skip.
                    step_state[sid] = "skipped_due_to_dependency"
                    failed_deps.add(sid)
                    sr = {
                        "status": "skipped_due_to_dependency",
                        "execution_ok": False,
                        "missing_outputs": [],
                        "failure_reason": f"depends_on_failed:{bad_deps}",
                    }
                    step_statuses.append({"id": sid, "type": step.get("type"), **sr})
                    trace.append({"stage": "step", "step_id": sid, "status": sr["status"]})
                    continue

                stype = step.get("type")
                if stype == "retrieve":
                    out = execute_retrieve_step(
                        step=step,
                        question=question,
                        resolver=self.resolver,
                        metadata_by_id=self.metadata_by_id,
                        runtime_operation_ids=self.runtime_operation_ids,
                        store=store,
                        budget=self.budget,
                        llm=self.llm,
                        allow_hkex_download=self.allow_hkex_download,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        use_mock_binder=self._use_mock_binder,
                    )
                    total_usage = _merge_usage(total_usage, out.get("usage") or {})
                    funnel = out.get("funnel") or {}
                    funnels.append(funnel)
                    resolver_calls["find_operations"] += 1 if funnel.get("search_query") else 0
                    resolver_calls["describe_operation"] += len(
                        funnel.get("described_candidates") or []
                    )
                    if funnel.get("disclosure_composite"):
                        resolver_calls["describe_operation"] += len(
                            (funnel["disclosure_composite"].get("trace") or [])
                        )
                    for oid in funnel.get("invoked_operations") or []:
                        invoked_operations.append(oid)
                        selected_operations.append(oid)
                        predicted_operations.append(oid)
                    result = out["step_result"]
                    if retrieve_step_succeeded(result):
                        completed_ok.add(sid)
                        step_state[sid] = "completed"
                        prov = result.get("provenance") or {}
                        if prov.get("operation_id"):
                            successful_invoked.append(str(prov["operation_id"]))
                    else:
                        step_state[sid] = str(result.get("status") or "failed")
                        failed_deps.add(sid)
                    step_statuses.append(
                        {
                            "id": sid,
                            "type": "retrieve",
                            "status": result.get("status"),
                            "execution_ok": result.get("execution_ok"),
                            "missing_outputs": result.get("missing_outputs"),
                            "scope_match": result.get("scope_match"),
                            "failure_reason": result.get("failure_reason"),
                        }
                    )
                    trace.append(
                        {
                            "stage": "retrieve",
                            "step_id": sid,
                            "status": result.get("status"),
                            "funnel": {
                                "search_query": funnel.get("search_query"),
                                "topk": funnel.get("embedding_topk"),
                                "described": funnel.get("described_candidates"),
                                "primary": funnel.get("primary"),
                                "fallback": funnel.get("fallback"),
                                "local_fallback": funnel.get("local_fallback"),
                            },
                        }
                    )
                elif stype == "compute":
                    out = execute_compute_step(step=step, store=store)
                    result = out["step_result"]
                    if result.get("status") == "completed":
                        completed_ok.add(sid)
                        step_state[sid] = "completed"
                    else:
                        step_state[sid] = str(result.get("status") or "failed")
                        failed_deps.add(sid)
                    step_statuses.append(
                        {
                            "id": sid,
                            "type": "compute",
                            "status": result.get("status"),
                            "execution_ok": result.get("execution_ok"),
                            "failure_reason": result.get("failure_reason"),
                        }
                    )
                    trace.append({"stage": "compute", "step_id": sid, "status": result.get("status")})
                elif stype == "reason":
                    out = execute_reason_step(
                        step=step,
                        store=store,
                        llm=self.llm,
                        question=question,
                        temperature=self.temperature,
                        max_tokens=min(1024, self.max_tokens),
                    )
                    total_usage = _merge_usage(total_usage, out.get("usage") or {})
                    result = out["step_result"]
                    completed_ok.add(sid)
                    step_state[sid] = "completed"
                    step_statuses.append(
                        {
                            "id": sid,
                            "type": "reason",
                            "status": result.get("status"),
                            "execution_ok": True,
                        }
                    )
                    trace.append({"stage": "reason", "step_id": sid, "status": "completed"})
                else:
                    step_state[sid] = "failed"
                    failed_deps.add(sid)
                    step_statuses.append(
                        {
                            "id": sid,
                            "type": stype,
                            "status": "failed",
                            "failure_reason": f"unknown_step_type:{stype}",
                        }
                    )

            # Final synthesizer (1 + optional 1 repair)
            missing_notes = [
                f"{m.get('step_id')}:{m.get('failure_reason') or m.get('missing_outputs')}"
                for m in store.missing
            ]
            final_user = render_final_user(
                question=question,
                step_statuses=step_statuses,
                evidence=store.compact_for_prompt(),
                compute=store.compute_records,
                reason=store.reason_artifacts,
                missing=missing_notes,
            )
            messages = [
                {"role": "system", "content": PE_FIN_LITE_SYSTEM_PROMPT},
                {"role": "user", "content": final_user},
            ]
            final_completion = self.llm.complete(
                messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stage="final",
            )
            total_usage = _merge_usage(total_usage, final_completion.get("usage") or {})
            parsed = parse_and_validate(str(final_completion.get("content") or ""), "final")
            if parsed[0] is None:
                repair_msg = format_repair_user_message("final", parsed[2] or "invalid final")
                repair = self.llm.complete(
                    messages
                    + [
                        {
                            "role": "assistant",
                            "content": str(final_completion.get("content") or ""),
                        },
                        {"role": "user", "content": repair_msg},
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    stage="final_repair",
                )
                self.budget.final_repair_rounds += 1
                parse_repair_count += 1
                total_usage = _merge_usage(total_usage, repair.get("usage") or {})
                parsed = parse_and_validate(
                    str(final_completion.get("content") or ""),
                    "final",
                    repair_content=str(repair.get("content") or ""),
                )
            if parsed[0] is None:
                # Fail closed: do not invent; emit unavailable.
                final_obj = {
                    "status": "unavailable",
                    "final_answer": "",
                    "reasoning_brief": "Final synthesizer could not produce a valid contract JSON.",
                }
                status = "failed"
                error = parsed[2] or "final_parse_failed"
                error_type = "final_schema_or_synthesis"
            else:
                final_obj = parsed[0]
                # If evidence missing and model answered with numbers, prefer unavailable
                # only when model ignored gaps — trust model status otherwise.
                status = "full" if final_obj.get("status") == "answered" else "partial"
                if final_obj.get("status") == "unavailable":
                    status = "failed" if store.missing and not store.values else "partial"
                if final_obj.get("status") == "answered" and not store.missing:
                    status = "full"
                elif final_obj.get("status") == "answered" and store.missing:
                    # Still accept answered if some evidence exists; grader decides correctness.
                    status = "partial"
            trace.append({"stage": "final", "status": final_obj.get("status")})

        except (LlmCallBudgetExhausted, McpBudgetExhausted) as exc:
            fail = structured_budget_failure(
                exc,
                extra={
                    "plan": plan,
                    "step_statuses": step_statuses,
                    "evidence_store": store.to_dict(),
                    "funnels": funnels,
                    "budget": self.budget.snapshot(),
                },
            )
            predicted_status, final_text = normalize_final_payload(fail.get("final_answer"))
            return {
                **fail,
                "final_answer": final_text
                if isinstance(fail.get("final_answer"), str)
                else (fail.get("final_answer") or {}).get("final_answer", ""),
                "predicted_status": predicted_status,
                "tokens": total_usage,
                "llm_calls": self.budget.llm_calls,
                "latency_ms": (time.perf_counter() - started) * 1000.0,
                "trace": trace + [{"stage": "budget_failure", "error": str(exc)}],
                "resolver_calls": resolver_calls,
                "underlying_operation_calls": self.budget.mcp_invocations,
                "predicted_operations": predicted_operations,
                "selected_operations": selected_operations,
                "invoked_operations": invoked_operations,
                "successful_invoked_operations": successful_invoked,
                "initial_plan": plan,
                "step_statuses": step_statuses,
                "retrieval_funnels": funnels,
                "evidence_store": store.to_dict(),
                "compute_traces": store.compute_records,
                "reason_artifacts": store.reason_artifacts,
                "parse_repair_count": parse_repair_count,
                "budget": self.budget.snapshot(),
                **self.budget.llm_budget_fields(),
                "monitor_decisions": [],
                "plan_revisions": [],
                "architecture": "plan_first_step_level_retrieval_v1",
            }
        except Exception as exc:  # noqa: BLE001 — task isolation
            status = "failed"
            error = f"{type(exc).__name__}: {exc}"
            error_type = "task_exception"
            final_obj = {
                "status": "unavailable",
                "final_answer": "",
                "reasoning_brief": "Task failed due to an internal exception; no answer invented.",
            }
            trace.append({"stage": "exception", "error": error})

        predicted_status, final_text = normalize_final_payload(
            final_obj, fallback_status=agent_completion_to_fallback(status)
        )
        succ = len(successful_invoked)
        fail_inv = max(0, len(invoked_operations) - succ)
        all_ok = None if not invoked_operations else (fail_inv == 0)

        return {
            "status": status,
            "error": error,
            "error_type": error_type,
            "final_answer": final_text,
            "predicted_status": predicted_status,
            "final_payload": final_obj,
            "tokens": total_usage,
            "llm_calls": self.budget.llm_calls,
            "latency_ms": (time.perf_counter() - started) * 1000.0,
            "trace": trace,
            "resolver_calls": resolver_calls,
            "underlying_operation_calls": self.budget.mcp_invocations,
            "predicted_operations": predicted_operations,
            "selected_operations": selected_operations,
            "invoked_operations": invoked_operations,
            "successful_invoked_operations": successful_invoked,
            "initial_plan": plan,
            "step_statuses": step_statuses,
            "retrieval_funnels": funnels,
            "evidence_store": store.to_dict(),
            "evidence_ledger": store.to_dict(),
            "compute_traces": store.compute_records,
            "reason_artifacts": store.reason_artifacts,
            "parse_repair_count": parse_repair_count,
            "binding_calls": self.budget.binder_calls,
            "binding_repair_count": 0,
            "describe_calls": resolver_calls["describe_operation"],
            "invoke_calls": self.budget.mcp_invocations,
            "tool_execution_success": all_ok if all_ok is not None else True,
            "all_tool_invocations_successful": all_ok,
            "successful_invocation_count": succ,
            "failed_invocation_count": fail_inv,
            "real_mcp_execution_count": self.budget.mcp_invocations,
            "budget": self.budget.snapshot(),
            "stopping_reason": self.budget.stopping_reason,
            "monitor_decisions": [],
            "plan_revisions": [],
            "architecture": "plan_first_step_level_retrieval_v1",
            "primary_selected_count": self.budget.primary_selected,
            "fallback_used_count": self.budget.fallback_used,
            "local_fallback_count": self.budget.local_fallback_count,
            "embedding_query_calls": self.budget.embedding_query_calls,
            **self.budget.llm_budget_fields(),
            "provider_metadata": {},
        }
