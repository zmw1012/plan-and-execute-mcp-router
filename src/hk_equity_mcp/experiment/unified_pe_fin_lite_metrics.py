"""PE-Fin Lite aggregate metrics on top of evaluator v1.2 fields."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Set


def _mean(values: Sequence[Optional[float]]) -> Optional[float]:
    nums = [float(v) for v in values if v is not None]
    if not nums:
        return None
    return sum(nums) / len(nums)


def _rate(flags: Sequence[bool]) -> Optional[float]:
    if not flags:
        return None
    return sum(1 for f in flags if f) / len(flags)


def _gold_ops(row: Dict[str, Any]) -> Set[str]:
    gold = row.get("gold_operations") or []
    if isinstance(gold, str):
        return {gold} if gold else set()
    return {str(x) for x in gold if x}


def retrieval_recall_at_k(row: Dict[str, Any], *, k: int = 5) -> Optional[float]:
    """Gold-grounded Recall@K for evaluation only (not used as answer success)."""
    gold = _gold_ops(row)
    if not gold:
        return None
    funnels = row.get("retrieval_funnels") or []
    retrieved: Set[str] = set()
    for funnel in funnels:
        for item in (funnel.get("embedding_topk") or [])[:k]:
            oid = item.get("operation_id") if isinstance(item, dict) else None
            if oid:
                retrieved.add(str(oid))
        for oid in funnel.get("described_candidates") or []:
            retrieved.add(str(oid))
        for oid in funnel.get("invoked_operations") or []:
            retrieved.add(str(oid))
    if not retrieved:
        # Also consider predicted ops
        retrieved = {str(x) for x in (row.get("predicted_operations") or []) if x}
    if not retrieved:
        return 0.0
    hit = len(gold & retrieved)
    return hit / len(gold)


def planner_valid(row: Dict[str, Any]) -> bool:
    plan = row.get("initial_plan") or {}
    steps = plan.get("steps") if isinstance(plan, dict) else None
    return isinstance(steps, list) and len(steps) > 0 and row.get("error_type") != "planner_parse_or_schema"


def retrieve_step_completion(row: Dict[str, Any]) -> Optional[float]:
    statuses = row.get("step_statuses") or []
    retrieves = [s for s in statuses if s.get("type") == "retrieve"]
    if not retrieves:
        return None
    ok = sum(1 for s in retrieves if s.get("status") == "completed" and s.get("execution_ok"))
    return ok / len(retrieves)


def compute_completion(row: Dict[str, Any]) -> Optional[float]:
    statuses = row.get("step_statuses") or []
    computes = [s for s in statuses if s.get("type") == "compute"]
    if not computes:
        return None
    ok = sum(1 for s in computes if s.get("status") == "completed")
    return ok / len(computes)


def required_output_satisfaction(row: Dict[str, Any]) -> Optional[float]:
    statuses = row.get("step_statuses") or []
    retrieves = [s for s in statuses if s.get("type") == "retrieve"]
    if not retrieves:
        return None
    ok = sum(1 for s in retrieves if not (s.get("missing_outputs") or []) and s.get("status") == "completed")
    return ok / len(retrieves)


def scope_match_rate(row: Dict[str, Any]) -> Optional[float]:
    statuses = row.get("step_statuses") or []
    retrieves = [s for s in statuses if s.get("type") == "retrieve" and "scope_match" in s]
    if not retrieves:
        return None
    return sum(1 for s in retrieves if s.get("scope_match")) / len(retrieves)


def final_contract_valid(row: Dict[str, Any]) -> bool:
    status = str(row.get("predicted_status") or "")
    return status in {"answered", "clarification_needed", "unavailable"} and not (
        row.get("error_type") == "final_schema_or_synthesis"
    )


def aggregate_pe_fin_lite_metrics(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    planner_flags = [planner_valid(r) for r in rows]
    recall = [retrieval_recall_at_k(r, k=5) for r in rows]
    exact = [
        bool(r.get("exact_tool_set")) if r.get("exact_tool_set") is not None else None
        for r in rows
    ]
    f1 = [r.get("operation_f1") for r in rows]
    precision = [r.get("operation_precision") for r in rows]
    recall_ops = [r.get("operation_recall") for r in rows]
    primary_rate = []
    fallback_rate = []
    for r in rows:
        budget = r.get("budget") or {}
        primary_rate.append(int(budget.get("primary_selected") or r.get("primary_selected_count") or 0) > 0)
        fallback_rate.append(int(budget.get("fallback_used") or r.get("fallback_used_count") or 0) > 0)
    inv_success = [
        bool(r.get("all_tool_invocations_successful"))
        for r in rows
        if r.get("all_tool_invocations_successful") is not None
    ]
    req_out = [required_output_satisfaction(r) for r in rows]
    scope = [scope_match_rate(r) for r in rows]
    retrieve_c = [retrieve_step_completion(r) for r in rows]
    compute_c = [compute_completion(r) for r in rows]
    final_ok = [final_contract_valid(r) for r in rows]
    answer_correct = [
        bool(r.get("answer_correct"))
        for r in rows
        if r.get("answer_correct") is not None and str(r.get("answer_correct")) != ""
    ]
    strict = [
        bool(r.get("answer_correct"))
        and str(r.get("predicted_status") or "") == "answered"
        and str(r.get("status") or "") in {"full", "partial"}
        for r in rows
    ]

    llm_calls = [r.get("llm_calls") for r in rows]
    mcp = [r.get("underlying_operation_calls") or r.get("real_mcp_execution_count") for r in rows]
    latency = [r.get("latency_ms") for r in rows]
    emb_q = [
        (r.get("budget") or {}).get("embedding_query_calls")
        or r.get("embedding_query_calls")
        for r in rows
    ]

    tokens_in = []
    tokens_out = []
    tokens_total = []
    for r in rows:
        tok = r.get("tokens") or {}
        tokens_in.append(tok.get("prompt_tokens") or tok.get("input_tokens"))
        tokens_out.append(tok.get("completion_tokens") or tok.get("output_tokens"))
        tokens_total.append(tok.get("total_tokens"))

    cost_summary = {
        "n_tasks": n,
        "successful_tasks": sum(1 for r in rows if str(r.get("status")) == "full"),
        "failed_tasks": sum(1 for r in rows if str(r.get("status")) in {"failed", "error"}),
        "llm_calls_total": sum(int(x or 0) for x in llm_calls),
        "llm_calls_mean": _mean([float(x) for x in llm_calls if x is not None]),
        "mcp_invocations_total": sum(int(x or 0) for x in mcp),
        "mcp_invocations_mean": _mean([float(x) for x in mcp if x is not None]),
        "embedding_query_calls_total": sum(int(x or 0) for x in emb_q if x is not None),
        "tokens_in_total": sum(int(x or 0) for x in tokens_in if x is not None),
        "tokens_out_total": sum(int(x or 0) for x in tokens_out if x is not None),
        "tokens_total": sum(int(x or 0) for x in tokens_total if x is not None),
        "latency_ms_mean": _mean([float(x) for x in latency if x is not None]),
        "estimated_cost_usd": None,
    }

    funnel = {
        "planner_valid": sum(1 for f in planner_flags if f),
        "retrieve_step_completed_mean": _mean(retrieve_c),
        "required_output_satisfied_mean": _mean(req_out),
        "scope_match_mean": _mean(scope),
        "compute_completion_mean": _mean(compute_c),
        "final_contract_valid": sum(1 for f in final_ok if f),
        "strict_answer_correct": sum(1 for f in strict if f),
        "n": n,
    }

    return {
        "metrics_version": "pe_fin_lite_metrics_v1",
        "n_tasks": n,
        "planner_validity_rate": _rate(planner_flags),
        "retrieval_recall_at_5_mean": _mean(recall),
        "exact_tool_set_rate": _rate([bool(x) for x in exact if x is not None]),
        "operation_precision_mean": _mean(precision),
        "operation_recall_mean": _mean(recall_ops),
        "operation_f1_mean": _mean(f1),
        "primary_selection_rate": _rate(primary_rate),
        "fallback_use_rate": _rate(fallback_rate),
        "invocation_success_rate": _rate(inv_success),
        "required_output_satisfaction_mean": _mean(req_out),
        "scope_match_rate_mean": _mean(scope),
        "retrieve_step_completion_mean": _mean(retrieve_c),
        "compute_completion_mean": _mean(compute_c),
        "final_contract_validity_rate": _rate(final_ok),
        "answer_correctness_rate": _rate(answer_correct) if answer_correct else None,
        "strict_answer_success_rate": _rate(strict),
        "mechanism_funnel": funnel,
        "cost_summary": cost_summary,
        "notes": (
            "Retrieval success is NOT equated with answer success. "
            "Recall@K is gold-grounded evaluation-only."
        ),
    }
