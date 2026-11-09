"""APV-Fin aggregate metrics on top of evaluator v1.2 fields."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence


def _mean(values: Sequence[Optional[float]]) -> Optional[float]:
    nums = [float(v) for v in values if v is not None]
    if not nums:
        return None
    return sum(nums) / len(nums)


def _rate(flags: Sequence[bool]) -> Optional[float]:
    if not flags:
        return None
    return sum(1 for f in flags if f) / len(flags)


def _plan_steps(row: Dict[str, Any]) -> Dict[str, Any]:
    plan = row.get("plan_state") or {}
    if not isinstance(plan, dict):
        return {}
    steps = plan.get("steps") or {}
    return steps if isinstance(steps, dict) else {}


def evidence_chain_complete(row: Dict[str, Any]) -> bool:
    """True only when mandatory retrieves resolved with valid/verified MCP evidence.

    Failed MCP invocations, planner task_spec alone, or empty ledgers do not count.
    """
    steps = _plan_steps(row)
    mandatory_retrieve = [
        s
        for s in steps.values()
        if s.get("mandatory") and s.get("step_type") == "retrieve"
    ]
    if mandatory_retrieve:
        if not all(s.get("status") == "resolved" for s in mandatory_retrieve):
            return False
        # Dependency edges among mandatory retrieves must be satisfied via resolved deps.
        for s in mandatory_retrieve:
            for dep in s.get("depends_on") or []:
                dep_step = steps.get(dep) or {}
                if dep_step.get("mandatory") and dep_step.get("status") != "resolved":
                    return False

    ledger = row.get("evidence_ledger") or []
    if isinstance(ledger, dict):
        items = ledger.get("items") or []
    else:
        items = ledger
    valid_mcp = [
        i
        for i in items
        if isinstance(i, dict)
        and i.get("source_type") == "mcp_invoke"
        and str(i.get("validation_status") or "") in {"valid", "verified"}
    ]
    if not valid_mcp:
        return False
    # Any failed invocation without a later success for that step blocks completion.
    if int(row.get("failed_invocation_count") or 0) > 0 and int(
        row.get("successful_invocation_count") or 0
    ) == 0:
        return False
    if str(row.get("status") or "").lower() in {"failed", "error", "fail"}:
        # Failed task cannot claim completed evidence chain.
        return False
    return True


def calculation_validation_flag(row: Dict[str, Any]) -> Optional[bool]:
    """null when no mandatory compute; false if present but not ok; true if ok."""
    steps = _plan_steps(row)
    mandatory_compute = [
        s
        for s in steps.values()
        if s.get("mandatory") and s.get("step_type") == "compute"
    ]
    if not mandatory_compute:
        # Optional compute absence → N/A
        any_compute = [s for s in steps.values() if s.get("step_type") == "compute"]
        if not any_compute:
            return None
        # Non-mandatory compute present: require traces if executed; else N/A.
        traces = row.get("compute_traces") or []
        if not traces:
            return None
        return all(bool(t.get("ok")) for t in traces)

    # Mandatory compute must be resolved with successful deterministic traces.
    if not all(s.get("status") == "resolved" for s in mandatory_compute):
        return False
    traces = row.get("compute_traces") or []
    if not traces:
        return False
    if not all(bool(t.get("ok")) for t in traces):
        return False
    validator = row.get("validator") or {}
    calc_check = next(
        (
            c
            for c in (validator.get("checks") or [])
            if c.get("name") == "calculation_operands"
        ),
        None,
    )
    if calc_check is not None and not bool(calc_check.get("ok")):
        return False
    return True


def strict_answer_success(row: Dict[str, Any]) -> Optional[bool]:
    """Common/strict answer success — failed/partial tasks are false, never gold-matched refusal credit."""
    status = str(row.get("status") or row.get("result_status") or "").lower()
    if status in {"failed", "error", "fail", "partial"}:
        return False
    predicted_status = str(row.get("predicted_status") or "").lower()
    if status == "failed" or predicted_status in {"failed"}:
        return False
    ac = row.get("answer_correct")
    if ac is True:
        # Guard: unavailable/failed trajectory must not count as strict success.
        if predicted_status in {"unavailable", "clarification_needed"} and status in {
            "failed",
            "partial",
        }:
            return False
        return True
    if ac is False:
        return False
    return None


def tool_execution_success_flag(row: Dict[str, Any]) -> Optional[bool]:
    """null/N/A when zero MCP invocations; else True iff all invokes succeeded."""
    failed = int(row.get("failed_invocation_count") or 0)
    successful = int(row.get("successful_invocation_count") or 0)
    mcp_n = int(row.get("real_mcp_execution_count") or 0)
    flag = row.get("all_tool_invocations_successful")
    if flag is None and successful == 0 and failed == 0 and mcp_n == 0:
        return None
    if failed > 0:
        return False
    if flag is False:
        return False
    if successful == 0 and mcp_n > 0:
        return False
    if successful == 0 and failed == 0:
        return None
    return True if flag is True or (successful > 0 and failed == 0) else False


def plan_completion_flag(row: Dict[str, Any]) -> bool:
    steps = _plan_steps(row)
    if not steps:
        return False
    mandatory = [s for s in steps.values() if s.get("mandatory")]
    if not mandatory:
        return False
    return all(s.get("status") == "resolved" for s in mandatory)


def aggregate_apv_fin_metrics(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    rows = list(rows)
    n = len(rows)

    answer_correct_flags: List[bool] = []
    descriptive_auto_grade_flags: List[bool] = []
    for r in rows:
        strict = strict_answer_success(r)
        if strict is not None:
            answer_correct_flags.append(strict)
        # Descriptive auto-grade retained separately; not used as common strict answer.
        if r.get("auto_grade_status") == "graded" and r.get("answer_correct") is True:
            if str(r.get("status") or "").lower() not in {"failed", "error", "fail", "partial"}:
                descriptive_auto_grade_flags.append(True)
            else:
                descriptive_auto_grade_flags.append(False)
        elif r.get("auto_grade_status") == "graded" and r.get("answer_correct") is False:
            descriptive_auto_grade_flags.append(False)

    plan_complete = [plan_completion_flag(r) for r in rows]
    mandatory_cov: List[Optional[float]] = []
    evidence_complete = [evidence_chain_complete(r) for r in rows]
    calc_valid: List[Optional[bool]] = []
    plan_revisions = []
    resource_limit_failures = 0
    refusal_flags = []
    tool_exec = [
        f for f in (tool_execution_success_flag(r) for r in rows) if f is not None
    ]

    for r in rows:
        steps = _plan_steps(r)
        if steps:
            mandatory = [s for s in steps.values() if s.get("mandatory")]
            resolved = [s for s in mandatory if s.get("status") == "resolved"]
            mandatory_cov.append(
                (len(resolved) / len(mandatory)) if mandatory else None
            )
        else:
            mandatory_cov.append(None)

        calc_valid.append(calculation_validation_flag(r))

        plan_revisions.append(int(r.get("plan_revisions") or 0))
        stop = str(r.get("stopping_reason") or r.get("error") or "")
        err_type = str(r.get("error_type") or "")
        if (
            err_type
            in {
                "resource_limit_failure",
                "llm_call_budget_exhausted",  # legacy v4 label
            }
            or stop
            in {
                "resource_limit_failure",
                "llm_call_budget_exhausted",
                "max_plan_revisions",
                "max_llm_monitor_rounds",
                "max_exception_replan_calls_per_task",
                "max_total_mcp_invocations",
            }
            or stop.startswith("max_")
        ):
            resource_limit_failures += 1

        if r.get("refusal_correctness") is not None and r.get("refusal_correctness") != "":
            try:
                refusal_flags.append(float(r.get("refusal_correctness")) >= 1.0)
            except (TypeError, ValueError):
                refusal_flags.append(bool(r.get("refusal_correctness")))

    calc_bools = [c for c in calc_valid if c is not None]

    return {
        "n_tasks": n,
        "final_answer_accuracy": _rate(answer_correct_flags),
        "common_strict_answer_success": _rate(answer_correct_flags),
        "descriptive_auto_grade_accuracy": _rate(descriptive_auto_grade_flags),
        "plan_completion_rate": _rate(plan_complete),
        "mandatory_node_coverage_mean": _mean(mandatory_cov),
        "evidence_chain_completion_rate": _rate(evidence_complete),
        "tool_execution_success_rate": _rate(tool_exec),
        "tool_set_exact_rate": _rate(
            [bool(r.get("exact_tool_set")) for r in rows if r.get("exact_tool_set") is not None]
        ),
        "tool_set_f1_mean": _mean(
            [r.get("operation_f1") for r in rows if r.get("operation_f1") is not None]
        ),
        "dependency_correctness_mean": _mean(
            [
                r.get("graph_dependency_accuracy")
                for r in rows
                if r.get("graph_dependency_accuracy") not in (None, "")
            ]
        ),
        "refusal_clarification_accuracy": _rate(refusal_flags),
        "calculation_validation_rate": _rate(calc_bools) if calc_bools else None,
        "plan_revisions_per_task_mean": _mean(plan_revisions),
        "llm_calls_mean": _mean([r.get("llm_calls") for r in rows]),
        "mcp_invocations_mean": _mean(
            [r.get("real_mcp_execution_count") for r in rows]
        ),
        "tokens_mean": _mean(
            [
                (r.get("tokens") or {}).get("total_tokens")
                if isinstance(r.get("tokens"), dict)
                else r.get("total_tokens")
                for r in rows
            ]
        ),
        "latency_ms_mean": _mean([r.get("latency_ms") for r in rows]),
        "resource_limit_failures": resource_limit_failures,
        "resource_limit_failure_rate": (resource_limit_failures / n) if n else None,
        "deterministic_step_transitions_mean": _mean(
            [int(r.get("deterministic_step_transitions") or 0) for r in rows]
        ),
        "exception_replan_calls_mean": _mean(
            [int(r.get("exception_replan_calls") or 0) for r in rows]
        ),
        "raw_observation_chars_mean": _mean(
            [int(r.get("raw_observation_chars") or 0) for r in rows]
        ),
        "projected_observation_chars_mean": _mean(
            [int(r.get("projected_observation_chars") or 0) for r in rows]
        ),
        "compression_ratio_mean": _mean(
            [
                float(r["compression_ratio"])
                if r.get("compression_ratio") is not None
                else (
                    (int(r.get("projected_observation_chars") or 0) / int(r.get("raw_observation_chars")))
                    if int(r.get("raw_observation_chars") or 0) > 0
                    else 0.0
                )
                for r in rows
            ]
        ),
        "repeated_signature_block_count_sum": sum(
            int(r.get("repeated_signature_block_count") or 0) for r in rows
        ),
        "dependency_auto_expansion_count_sum": sum(
            int(r.get("dependency_auto_expansion_count") or 0) for r in rows
        ),
        "schema_description_expansion_count_sum": sum(
            int(r.get("schema_description_expansion_count") or 0) for r in rows
        ),
        "deterministic_binding_count_sum": sum(
            int(r.get("deterministic_binding_count") or 0) for r in rows
        ),
        "llm_binding_count_sum": sum(int(r.get("llm_binding_count") or 0) for r in rows),
        "capability_injected_candidate_count_sum": sum(
            int(r.get("capability_injected_candidate_count") or 0) for r in rows
        ),
        "capability_filter_candidate_count_sum": sum(
            int(r.get("capability_filter_candidate_count") or 0) for r in rows
        ),
        "output_adapter_success_count_sum": sum(
            int(r.get("output_adapter_success_count") or 0) for r in rows
        ),
        "required_output_missing_count_sum": sum(
            int(r.get("required_output_missing_count") or 0) for r in rows
        ),
        "deterministic_compute_count_sum": sum(
            int(r.get("deterministic_compute_count") or 0) for r in rows
        ),
        "monitor_revision_count_mean": _mean(
            [len(r.get("monitor_decisions") or []) for r in rows]
        ),
        "exception_replan_calls_sum": sum(
            int(r.get("exception_replan_calls") or 0) for r in rows
        ),
        "cost_summary": _cost_summary(rows),
    }


def _sum_int(rows: Sequence[Dict[str, Any]], *keys: str) -> int:
    total = 0
    for r in rows:
        for key in keys:
            if key == "tokens.input_tokens":
                total += int((r.get("tokens") or {}).get("input_tokens") or 0)
            elif key == "tokens.output_tokens":
                total += int((r.get("tokens") or {}).get("output_tokens") or 0)
            elif key == "tokens.total_tokens":
                total += int((r.get("tokens") or {}).get("total_tokens") or 0)
            else:
                val = r.get(key)
                if val is not None:
                    total += int(val or 0)
                    break
    return total


def _cost_summary(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Always emit cost observability — including all-failed / zero-success runs."""
    used_vals: List[int] = []
    hitting_cap = 0
    stage_totals: Dict[str, int] = {}
    latencies: List[float] = []
    for r in rows:
        used = r.get("llm_calls_used")
        if used is None:
            used = r.get("llm_calls")
        if used is not None:
            used_i = int(used)
            used_vals.append(used_i)
            budget = r.get("llm_call_budget")
            exhausted = r.get("llm_budget_exhausted")
            if exhausted or (
                budget is not None and used_i >= int(budget)
            ) or str(r.get("error_type") or "") == "llm_call_budget_exhausted":
                hitting_cap += 1
        for stage, count in (r.get("calls_by_stage") or {}).items():
            stage_totals[str(stage)] = stage_totals.get(str(stage), 0) + int(count or 0)
        if r.get("latency_ms") is not None:
            try:
                latencies.append(float(r.get("latency_ms")))
            except (TypeError, ValueError):
                pass

    return {
        "n_tasks": len(rows),
        "successful_tasks": sum(
            1
            for r in rows
            if str(r.get("status") or r.get("result_status") or "").lower()
            in {"ok", "success", "answered", "completed", "full"}
        ),
        "failed_tasks": sum(
            1
            for r in rows
            if str(r.get("status") or r.get("result_status") or "").lower()
            in {"failed", "error", "fail", "partial"}
        ),
        "planner_calls": int(stage_totals.get("planner") or 0),
        "planner_repair_calls": int(stage_totals.get("planner_repair") or 0),
        "tool_binding_calls": int(stage_totals.get("tool_binding") or 0),
        "tool_binding_repair_calls": int(stage_totals.get("tool_binding_repair") or 0),
        "monitor_calls": int(stage_totals.get("monitor") or 0),
        "monitor_repair_calls": int(stage_totals.get("monitor_repair") or 0),
        "calls_by_stage": stage_totals,
        "total_input_tokens": _sum_int(rows, "tokens.input_tokens"),
        "total_output_tokens": _sum_int(rows, "tokens.output_tokens"),
        "total_tokens": _sum_int(rows, "tokens.total_tokens"),
        "total_llm_calls": _sum_int(rows, "llm_calls_used", "llm_calls"),
        "total_real_mcp_executions": _sum_int(rows, "real_mcp_execution_count"),
        "total_embedding_query_calls": _sum_int(rows, "embedding_calls"),
        "catalog_index_embedding_calls": _sum_int(
            rows, "catalog_index_embedding_calls", "embedding_index_calls"
        ),
        "mean_latency_ms": _mean(latencies),
        "total_latency_ms": sum(latencies) if latencies else 0.0,
        "tasks_hitting_llm_call_cap": hitting_cap,
        "max_llm_calls_observed": max(used_vals) if used_vals else None,
        "mean_llm_calls_per_task": _mean(used_vals),
        "estimated_cost_usd": None,
        "note": (
            "Token/call totals only; estimated_cost_usd is null without a provider "
            "price table. Catalog index embedding reuse is recorded separately from "
            "per-task embedding query calls. Artifacts are written even when all "
            "tasks fail."
        ),
    }
