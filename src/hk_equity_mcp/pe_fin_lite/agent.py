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
        # Binder / Reason / Final default budget (unchanged at 2048 unless configured).
        self.max_tokens = int(self.config.get("llm", {}).get("max_tokens", 2048))
        # Planner + planner_repair only (protocol v3 default 4096).
        from hk_equity_mcp.pe_fin_lite.schemas import PLANNER_MAX_TOKENS

        llm_cfg = self.config.get("llm") or {}
        if "planner_max_tokens" in llm_cfg:
            self.planner_max_tokens = int(llm_cfg["planner_max_tokens"])
        elif "planner_max_tokens" in self.config:
            self.planner_max_tokens = int(self.config["planner_max_tokens"])
        else:
            self.planner_max_tokens = int(PLANNER_MAX_TOKENS)
        self.metadata_by_id = metadata_by_id or {}
        self.runtime_operation_ids = list(
            runtime_operation_ids
            or sorted(self.resolver.list_underlying_operation_ids())
        )
        self.live_mode = bool(live_mode)
        self.allow_hkex_download = bool(allow_hkex_download)
        self.budget = BudgetTracker(self.limits)
        self._use_mock_binder = bool(self.config.get("use_mock_binder", False))
        self.eea_enabled = bool(self.config.get("eea_enabled", False))
        if self.live_mode and self._use_mock_binder:
            raise RuntimeError("PE-Fin Lite fail-closed: use_mock_binder forbidden in live_mode")

    def run(
        self,
        question: str,
        *,
        answerability_class: Optional[str] = None,
    ) -> Dict[str, Any]:
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
        invoke_records: List[Dict[str, Any]] = []
        resolver_calls = {"find_operations": 0, "describe_operation": 0}
        status = "failed"
        error: Optional[str] = None
        error_type: Optional[str] = None
        stopping_reason: Optional[str] = None
        final_obj: Optional[Dict[str, Any]] = None
        plan: Optional[Dict[str, Any]] = None
        step_state: Dict[str, str] = {}
        skip_execution = False
        planner_out: Dict[str, Any] = {}
        answerability_info: Optional[Dict[str, Any]] = None

        try:
            planner_out = run_planner(
                llm=self.llm,
                question=question,
                capability_profile=profile_for_planner(self.capability_profile),
                limits=self.limits,
                max_plan_steps=int(self.limits.get("max_plan_steps", 6)),
                temperature=self.temperature,
                max_tokens=self.planner_max_tokens,
            )
            parse_repair_count += int(planner_out.get("repair_count") or 0)
            self.budget.planner_repair_count = int(planner_out.get("repair_count") or 0)
            total_usage = _merge_usage(total_usage, planner_out.get("usage") or {})
            trace.append(
                {
                    "stage": "planner",
                    "ok": planner_out.get("ok"),
                    "error": planner_out.get("error"),
                    "error_type": planner_out.get("error_type"),
                    "stopping_reason": planner_out.get("stopping_reason"),
                    "repair_count": planner_out.get("repair_count"),
                    "planner_max_tokens": planner_out.get("planner_max_tokens"),
                    "call_observability": planner_out.get("call_observability"),
                }
            )
            if not planner_out.get("ok"):
                # Structured planner failure — do not raise a runner exception.
                error = f"planner_failed:{planner_out.get('error')}"
                error_type = str(
                    planner_out.get("error_type") or "planner_schema_or_semantics"
                )
                stopping_reason = str(
                    planner_out.get("stopping_reason") or "planner_invalid"
                )
                final_obj = {
                    "status": "unavailable",
                    "final_answer": "",
                    "reasoning_brief": (
                        "Planner could not produce a valid plan under the output contract."
                    ),
                    "refusal_reason": stopping_reason,
                }
                status = "failed"
                skip_execution = True
            else:
                plan = planner_out["plan"]
            if not skip_execution:
                steps = list((plan or {}).get("steps") or [])
                ordered = topological_ready_order(steps)
                completed_ok: Set[str] = set()
                failed_deps: Set[str] = set()

            for step in ordered if not skip_execution else []:
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
                        eea_enabled=self.eea_enabled,
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
                    gw_map = dict(funnel.get("invoke_gateway_success") or {})
                    for oid in funnel.get("invoked_operations") or []:
                        invoked_operations.append(oid)
                        selected_operations.append(oid)
                        predicted_operations.append(oid)
                        gw_ok = bool(gw_map.get(oid, out.get("gateway_successful_operations") and oid in (out.get("gateway_successful_operations") or [])))
                        if oid in (out.get("gateway_successful_operations") or []):
                            gw_ok = True
                        if oid in gw_map:
                            gw_ok = bool(gw_map[oid])
                        invoke_records.append(
                            {
                                "operation_id": oid,
                                "step_id": sid,
                                "source": "gateway",
                                "gateway_success": gw_ok,
                                "success": gw_ok,  # gateway/protocol success (not step completion)
                                "step_completed": False,
                            }
                        )
                        if gw_ok:
                            successful_invoked.append(oid)
                    result = out["step_result"]
                    step_done = retrieve_step_succeeded(result)
                    for rec in invoke_records:
                        if rec.get("step_id") == sid:
                            rec["step_completed"] = step_done
                            rec["required_outputs_satisfied"] = bool(
                                result.get("required_outputs_satisfied")
                            )
                            rec["scope_match"] = bool(result.get("scope_match", True))
                            rec["normalization_success"] = result.get(
                                "normalization_success"
                            )
                    if step_done:
                        completed_ok.add(sid)
                        step_state[sid] = "completed"
                    else:
                        step_state[sid] = str(result.get("status") or "failed")
                        failed_deps.add(sid)
                        if not error_type and result.get("failure_reason"):
                            error_type = "semantic_or_structured_retrieve_failure"
                            stopping_reason = stopping_reason or str(
                                result.get("failure_reason")
                            )
                    step_statuses.append(
                        {
                            "id": sid,
                            "type": "retrieve",
                            "status": result.get("status"),
                            "execution_ok": result.get("execution_ok"),
                            "missing_outputs": result.get("missing_outputs"),
                            "scope_match": result.get("scope_match"),
                            "failure_reason": result.get("failure_reason"),
                            "gateway_success": result.get("gateway_success"),
                            "required_outputs_satisfied": result.get(
                                "required_outputs_satisfied"
                            ),
                            "step_completed": step_done,
                        }
                    )
                    trace.append(
                        {
                            "stage": "retrieve",
                            "step_id": sid,
                            "status": result.get("status"),
                            "funnel": {
                                "search_query": funnel.get("search_query"),
                                "hard_filtered": funnel.get("hard_filtered"),
                                "topk": funnel.get("embedding_topk"),
                                "topk_ranked": funnel.get("embedding_topk_ranked"),
                                "described": funnel.get("described_candidates"),
                                "rank4_5_not_described": funnel.get(
                                    "rank4_5_not_described"
                                ),
                                "primary": funnel.get("primary"),
                                "fallback": funnel.get("fallback"),
                                "local_fallback": funnel.get("local_fallback"),
                                "selected_outside_described": funnel.get(
                                    "selected_outside_described"
                                ),
                                "binder_selection_validation": funnel.get(
                                    "binder_selection_validation"
                                ),
                                "invoke_gateway_success": funnel.get(
                                    "invoke_gateway_success"
                                ),
                                "checkpoint_passive": funnel.get("checkpoint_passive"),
                            },
                        }
                    )
                elif stype == "compute":
                    out = execute_compute_step(
                        step=step, store=store, eea_enabled=self.eea_enabled
                    )
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
                    compute_rec = (out.get("step_result") or {}) if isinstance(out, dict) else {}
                    trace.append(
                        {
                            "stage": "compute",
                            "step_id": sid,
                            "status": result.get("status"),
                            "failure_reason": result.get("failure_reason"),
                            "expression": (step.get("compute_spec") or {}).get(
                                "expression"
                            ),
                            "operand_refs": (step.get("compute_spec") or {}).get(
                                "operand_refs"
                            ),
                            "passive": {
                                "execution_ok": result.get("execution_ok"),
                                "values_keys": sorted(
                                    str(k) for k in (result.get("values") or {}).keys()
                                ),
                            },
                        }
                    )
                    _ = compute_rec  # keep local for future passive hooks
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
                    # Reason parse failure does not block dependents of verified evidence,
                    # but stage_outcome is fail when reason_parse_failed.
                    completed_ok.add(sid)
                    step_state[sid] = str(result.get("status") or "completed")
                    step_statuses.append(
                        {
                            "id": sid,
                            "type": "reason",
                            "status": result.get("status"),
                            "execution_ok": bool(result.get("execution_ok")),
                            "failure_reason": result.get("failure_reason"),
                            "reason_parse_failed": bool(
                                out.get("reason_parse_failed")
                                or result.get("reason_parse_failed")
                            ),
                            "runtime_recovered": bool(
                                out.get("runtime_recovered")
                                or result.get("runtime_recovered")
                            ),
                        }
                    )
                    artifact = out.get("reason_artifact") or {}
                    trace.append(
                        {
                            "stage": "reason",
                            "step_id": sid,
                            "status": result.get("status"),
                            "passive": {
                                "summary_present": bool(artifact.get("summary")),
                                "unsupported_claim": False,
                                "gaps": artifact.get("gaps"),
                                "reason_parse_failed": bool(
                                    out.get("reason_parse_failed")
                                ),
                                "runtime_recovered": bool(
                                    out.get("runtime_recovered")
                                ),
                                "repair_count": int(out.get("repair_count") or 0),
                            },
                        }
                    )
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

            # Final synthesizer (1 + optional 1 repair). Skipped on structured planner failure.
            if not skip_execution:
                from hk_equity_mcp.pe_fin_lite.temporal_policy import (
                    assess_latest_current_refusal,
                )

                temporal = assess_latest_current_refusal(
                    question=question,
                    evidence_store=store.to_dict(),
                )
                if temporal and temporal.get("should_refuse"):
                    final_obj = {
                        "status": "unavailable",
                        "final_answer": "",
                        "reasoning_brief": (
                            "Question requests latest/current/today, but the "
                            "benchmark temporal cutoff forbids silently "
                            "substituting an earlier historical quote."
                        ),
                        "refusal_reason": temporal.get("reason"),
                    }
                    status = "partial"
                    stopping_reason = stopping_reason or str(temporal.get("reason"))
                    error_type = error_type or "temporal_latest_current_unsatisfied"
                    trace.append(
                        {
                            "stage": "temporal_policy",
                            "status": "unavailable",
                            "detail": temporal,
                        }
                    )

            if not skip_execution and self.eea_enabled and final_obj is None:
                from hk_equity_mcp.pe_fin_lite.answerability_eea import (
                    classify_answerability,
                    final_status_from_answerability,
                )
                from hk_equity_mcp.pe_fin_lite.temporal_policy import (
                    assess_latest_current_refusal,
                )

                temporal2 = assess_latest_current_refusal(
                    question=question,
                    evidence_store=store.to_dict(),
                )
                merge_conflicts = []
                for f in funnels:
                    lf = f.get("local_fallback") or {}
                    merge_conflicts.extend(list(lf.get("merge_conflicts") or []))
                answerability_info = classify_answerability(
                    plan=plan,
                    store=store,
                    step_statuses=step_statuses,
                    merge_conflicts=merge_conflicts,
                    answerability_class=answerability_class,
                    temporal_blocked=bool(temporal2 and temporal2.get("should_refuse")),
                )
                trace.append({"stage": "answerability_eea", **answerability_info})
                guided = final_status_from_answerability(
                    str(answerability_info.get("answerability_state"))
                )
                if guided == "unavailable" and answerability_info.get(
                    "answerability_state"
                ) == "infeasible":
                    final_obj = {
                        "status": "unavailable",
                        "final_answer": "",
                        "reasoning_brief": (
                            "EEA answerability=infeasible; refuse/clarify without inventing."
                        ),
                        "refusal_reason": answerability_info.get(
                            "answerability_reason_code"
                        ),
                    }
                    status = "partial"
                    stopping_reason = stopping_reason or str(
                        answerability_info.get("answerability_reason_code")
                    )
                elif answerability_info.get("answerability_state") == "contradictory":
                    final_obj = {
                        "status": "answered",
                        "final_answer": (
                            "Conflicting verified evidence for the same entity/period/basis; "
                            "cannot deterministically resolve. "
                            f"Conflicts: {answerability_info.get('infeasibility_evidence')}"
                        ),
                        "reasoning_brief": "EEA answerability=contradictory; report conflict.",
                        "limitations": ["conflicting_evidence"],
                    }
                    status = "partial"

            if not skip_execution and final_obj is None:
                missing_notes = [
                    f"{m.get('step_id')}:{m.get('failure_reason') or m.get('missing_outputs')}"
                    for m in store.missing
                ]
                if answerability_info and answerability_info.get(
                    "answerability_state"
                ) == "partial_but_answerable":
                    missing_notes.append(
                        "eea_partial_but_answerable:"
                        + str(answerability_info.get("optional_outputs_missing"))
                    )
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
                parsed = parse_and_validate(
                    str(final_completion.get("content") or ""), "final"
                )
                if parsed[0] is None:
                    repair_msg = format_repair_user_message(
                        "final", parsed[2] or "invalid final"
                    )
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
                    final_obj = {
                        "status": "unavailable",
                        "final_answer": "",
                        "reasoning_brief": (
                            "Final synthesizer could not produce a valid contract JSON."
                        ),
                    }
                    status = "failed"
                    error = parsed[2] or "final_parse_failed"
                    error_type = "final_schema_or_synthesis"
                    stopping_reason = stopping_reason or "final_invalid"
                else:
                    final_obj = parsed[0]
                    status = "full" if final_obj.get("status") == "answered" else "partial"
                    if final_obj.get("status") == "unavailable":
                        status = (
                            "failed" if store.missing and not store.values else "partial"
                        )
                    if final_obj.get("status") == "answered" and not store.missing:
                        status = "full"
                    elif final_obj.get("status") == "answered" and store.missing:
                        status = "partial"
                trace.append({"stage": "final", "status": (final_obj or {}).get("status")})
            else:
                trace.append(
                    {
                        "stage": "final",
                        "status": "unavailable",
                        "skipped": True,
                        "reason": stopping_reason or "planner_failed",
                    }
                )

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

        if final_obj is None:
            final_obj = {
                "status": "unavailable",
                "final_answer": "",
                "reasoning_brief": "No final payload produced.",
            }
        predicted_status, final_text = normalize_final_payload(
            final_obj, fallback_status=agent_completion_to_fallback(status)
        )
        mcp_count = int(self.budget.mcp_invocations)
        # Prefer recorded gateway invocations; ensure count matches budget.
        if len(invoke_records) < mcp_count:
            for oid in invoked_operations[len(invoke_records) :]:
                invoke_records.append(
                    {
                        "operation_id": oid,
                        "source": "gateway",
                        "gateway_success": oid in successful_invoked,
                        "success": oid in successful_invoked,
                    }
                )
        # Deduplicate successful gateway ops while preserving order.
        seen_succ = set()
        successful_unique = []
        for oid in successful_invoked:
            if oid not in seen_succ:
                seen_succ.add(oid)
                successful_unique.append(oid)
        succ = sum(
            1
            for rec in invoke_records
            if rec.get("gateway_success") or rec.get("success")
        )
        fail_inv = max(0, len(invoke_records) - succ)
        # Zero MCP invocations must not vacuous-true tool success.
        if mcp_count == 0 and not invoke_records:
            all_ok = None
            tool_execution_success: Optional[bool] = None
        else:
            all_ok = fail_inv == 0
            tool_execution_success = all_ok

        from hk_equity_mcp.pe_fin_lite.checkpoint_observability import (
            CHECKPOINT_TRACE_CONTRACT_VERSION,
            build_task_checkpoint_observability,
            checkpoint_observability_hash,
        )

        checkpoint_obs = build_task_checkpoint_observability(
            planner_out=planner_out,
            plan=plan,
            funnels=funnels,
            step_statuses=step_statuses,
            compute_traces=store.compute_records,
            reason_artifacts=store.reason_artifacts,
            final_payload=final_obj,
            final_contract_valid=isinstance(final_obj, dict)
            and bool(final_obj.get("status")),
            invoke_records=invoke_records,
            status=status,
        )
        out_row = {
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
            "underlying_operation_calls": mcp_count,
            "predicted_operations": predicted_operations,
            "selected_operations": selected_operations,
            "invoked_operations": invoked_operations,
            "successful_invoked_operations": successful_unique,
            "initial_plan": plan,
            "step_statuses": step_statuses,
            "retrieval_funnels": funnels,
            "evidence_store": store.to_dict(),
            "evidence_ledger": store.to_dict(),
            "compute_traces": store.compute_records,
            "reason_artifacts": store.reason_artifacts,
            "parse_repair_count": parse_repair_count,
            "binding_calls": self.budget.binder_calls,
            "binding_repair_count": sum(
                int((f.get("binding") or {}).get("repair_count") or 0)
                for f in funnels
                if isinstance(f, dict)
            ),
            "describe_calls": resolver_calls["describe_operation"],
            "invoke_calls": list(invoke_records),
            "invoke_records": list(invoke_records),
            "tool_execution_success": tool_execution_success,
            "all_tool_invocations_successful": all_ok,
            "successful_invocation_count": succ,
            "failed_invocation_count": fail_inv,
            "real_mcp_execution_count": mcp_count,
            "budget": self.budget.snapshot(),
            "stopping_reason": stopping_reason or self.budget.stopping_reason,
            "monitor_decisions": [],
            "plan_revisions": [],
            "architecture": "plan_first_step_level_retrieval_v1",
            "primary_selected_count": self.budget.primary_selected,
            "fallback_used_count": self.budget.fallback_used,
            "local_fallback_count": self.budget.local_fallback_count,
            "embedding_query_calls": self.budget.embedding_query_calls,
            **self.budget.llm_budget_fields(),
            "provider_metadata": {},
            # Passive observability only — does not affect decisions.
            "checkpoint_observability": checkpoint_obs,
            "checkpoint_trace_contract_version": CHECKPOINT_TRACE_CONTRACT_VERSION,
            "checkpoint_observability_hash": checkpoint_observability_hash(),
        }
        if self.eea_enabled:
            from hk_equity_mcp.pe_fin_lite.eea_identity import (
                PE_FIN_LITE_EEA_METHOD_VERSION,
                PE_FIN_LITE_EEA_PROTOCOL_VERSION,
            )

            out_row["eea_enabled"] = True
            out_row["method_version_eea"] = PE_FIN_LITE_EEA_METHOD_VERSION
            out_row["protocol_version"] = PE_FIN_LITE_EEA_PROTOCOL_VERSION
            out_row["posthoc_exploratory"] = True
            out_row["formal_primary_result"] = False
            if answerability_info:
                out_row["answerability_state"] = answerability_info.get(
                    "answerability_state"
                )
                out_row["answerability_reason_code"] = answerability_info.get(
                    "answerability_reason_code"
                )
                out_row["core_outputs_satisfied"] = answerability_info.get(
                    "core_outputs_satisfied"
                )
                out_row["optional_outputs_missing"] = answerability_info.get(
                    "optional_outputs_missing"
                )
                out_row["infeasibility_evidence"] = answerability_info.get(
                    "infeasibility_evidence"
                )
        return out_row
