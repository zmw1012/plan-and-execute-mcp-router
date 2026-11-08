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


def aggregate_apv_fin_metrics(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    rows = list(rows)
    n = len(rows)

    answer_correct_flags = []
    for r in rows:
        ac = r.get("answer_correct")
        if ac is True:
            answer_correct_flags.append(True)
        elif ac is False:
            answer_correct_flags.append(False)

    plan_complete = []
    mandatory_cov = []
    evidence_complete = []
    calc_valid = []
    plan_revisions = []
    resource_limit_failures = 0
    refusal_flags = []

    for r in rows:
        plan = r.get("plan_state") or {}
        steps = (plan.get("steps") or {}) if isinstance(plan, dict) else {}
        if steps:
            mandatory = [s for s in steps.values() if s.get("mandatory")]
            resolved = [s for s in mandatory if s.get("status") == "resolved"]
            plan_complete.append(len(resolved) == len(mandatory))
            mandatory_cov.append(
                (len(resolved) / len(mandatory)) if mandatory else 1.0
            )
        else:
            plan_complete.append(False)
            mandatory_cov.append(None)

        ledger = r.get("evidence_ledger") or r.get("evidence") or []
        evidence_complete.append(bool(ledger))

        traces = r.get("compute_traces") or []
        if traces:
            calc_valid.append(all(bool(t.get("ok")) for t in traces))
        validator = r.get("validator") or {}
        calc_check = next(
            (
                c
                for c in (validator.get("checks") or [])
                if c.get("name") == "calculation_operands"
            ),
            None,
        )
        if calc_check is not None:
            calc_valid.append(bool(calc_check.get("ok")))

        plan_revisions.append(int(r.get("plan_revisions") or 0))
        stop = str(r.get("stopping_reason") or r.get("error") or "")
        if "max_" in stop or stop in {
            "max_plan_revisions",
            "max_llm_monitor_rounds",
            "max_total_mcp_invocations",
        }:
            resource_limit_failures += 1

        if r.get("refusal_correctness") is not None and r.get("refusal_correctness") != "":
            try:
                refusal_flags.append(float(r.get("refusal_correctness")) >= 1.0)
            except (TypeError, ValueError):
                refusal_flags.append(bool(r.get("refusal_correctness")))

    return {
        "n_tasks": n,
        "final_answer_accuracy": _rate(answer_correct_flags),
        "common_strict_answer_success": _rate(answer_correct_flags),
        "plan_completion_rate": _rate(plan_complete),
        "mandatory_node_coverage_mean": _mean(mandatory_cov),
        "evidence_chain_completion_rate": _rate(evidence_complete),
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
        "calculation_validation_rate": _rate(calc_valid) if calc_valid else None,
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
        "cost_summary": _cost_summary(rows),
    }


def _cost_summary(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    used_vals: List[int] = []
    hitting_cap = 0
    stage_totals: Dict[str, int] = {}
    for r in rows:
        used = r.get("llm_calls_used")
        if used is None:
            used = r.get("llm_calls")
        if used is None:
            continue
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
    return {
        "tasks_hitting_llm_call_cap": hitting_cap,
        "max_llm_calls_observed": max(used_vals) if used_vals else None,
        "mean_llm_calls_per_task": _mean(used_vals),
        "calls_by_stage": stage_totals,
    }
