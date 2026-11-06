"""Layered Operation-Visible failure taxonomy (offline; no LLM/MCP).

Primary categories (stage order):
1. retrieval
2. agent_selection
3. description_argument_grounding
4. dependency_order
5. tool_execution
6. answer_synthesis_calculation
7. feasibility_refusal
8. resource_limit
9. evaluation_failure
10. success
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from hk_equity_mcp.experiment.unified_evaluator import REFUSAL_CLASSES, REFUSAL_STATUSES

TAXONOMY_VERSION = "error_taxonomy_v1.1"

PRIMARY_CATEGORIES = (
    "retrieval",
    "agent_selection",
    "description_argument_grounding",
    "dependency_order",
    "tool_execution",
    "answer_synthesis_calculation",
    "feasibility_refusal",
    "resource_limit",
    "evaluation_failure",
    "answer_success",
    "strict_task_success",
)

RESOURCE_ERRORS = frozenset(
    {"max_llm_turns", "max_search_reformulations", "max_invocations"}
)


def _as_set(values: Optional[Sequence[Any]]) -> Set[str]:
    return {str(v) for v in (values or []) if v is not None and str(v).strip()}


def _gold_in_retrieval(row: Dict[str, Any], retrieval_row: Optional[Dict[str, Any]]) -> bool:
    gold = _as_set(row.get("gold_operations"))
    if not gold:
        return True
    cands: List[Any] = []
    if retrieval_row:
        cands = retrieval_row.get("retrieved_candidates") or []
    if not cands:
        cands = row.get("retrieved_candidates") or []
    retrieved = {
        str(c.get("operation_id"))
        for c in cands
        if isinstance(c, dict) and c.get("operation_id")
    }
    return bool(gold & retrieved) if retrieved else False


def _predicted_ops(row: Dict[str, Any]) -> Set[str]:
    ops = row.get("predicted_operations")
    if ops:
        return _as_set(ops)
    return _as_set(
        [
            (r.get("operation_id") if isinstance(r, dict) else None)
            for r in (row.get("invoke_calls") or [])
        ]
    )


def _invoke_records(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    recs = row.get("invoke_calls") or row.get("invoke_records") or []
    return [r for r in recs if isinstance(r, dict)]


def _has_arg_grounding_issue(row: Dict[str, Any]) -> bool:
    """Heuristic: gold ops selected, but invoke args look empty/mismatched vs success path."""
    gold = _as_set(row.get("gold_operations"))
    pred = _predicted_ops(row)
    if not gold or not gold.issubset(pred):
        return False
    recs = _invoke_records(row)
    if not recs:
        return False
    # Failed validation-like errors on gold ops.
    for r in recs:
        oid = str(r.get("operation_id") or "")
        if oid not in gold:
            continue
        if r.get("success"):
            continue
        err = str(r.get("error") or r.get("error_type") or "").lower()
        if any(
            k in err
            for k in (
                "argument",
                "validation",
                "invalid",
                "missing",
                "schema",
                "date",
                "symbol",
                "adjust",
            )
        ):
            return True
        args = r.get("arguments") or {}
        if isinstance(args, dict) and not args:
            return True
    return False


def _dependency_issue(row: Dict[str, Any]) -> bool:
    graph = row.get("dependency_graph")
    if not graph:
        acc = row.get("graph_dependency_accuracy")
        return isinstance(acc, (int, float)) and acc < 1.0 and _as_set(row.get("gold_operations"))
    pred = _predicted_ops(row)
    gold = _as_set(row.get("gold_operations"))
    if gold and not gold.issubset(pred):
        return False
    acc = row.get("graph_dependency_accuracy")
    if isinstance(acc, (int, float)) and acc < 1.0:
        return True
    # Order: if gold has ≥2 ops and predicted order disagrees with first gold edge order.
    gold_list = [str(x) for x in (row.get("gold_operations") or [])]
    pred_list = [str(x) for x in (row.get("predicted_operations") or [])]
    if len(gold_list) >= 2 and len(pred_list) >= 2:
        try:
            if pred_list.index(gold_list[0]) > pred_list.index(gold_list[1]):
                return True
        except ValueError:
            return False
    return False


def _should_refuse(row: Dict[str, Any]) -> bool:
    ac = str(row.get("answerability_class") or "")
    gs = str(row.get("gold_status") or "").lower()
    gold_ops = _as_set(row.get("gold_operations"))
    return (
        ac in REFUSAL_CLASSES
        or gs in {"unavailable", "unanswerable", "clarification_needed", "refuse"}
        or (not gold_ops and ac not in {"structured_answerable", "evidence_answerable"})
    )


def classify_task_error(
    row: Dict[str, Any],
    *,
    retrieval_row: Optional[Dict[str, Any]] = None,
    v11_answer_correct: Optional[bool] = None,
    v11_auto_status: Optional[str] = None,
) -> Dict[str, Any]:
    """Assign primary + secondary error categories for one enriched prediction.

    Distinguishes:
      - ``strict_task_success``: full rubric pass
      - ``answer_success``: final answer correct but strict rubric failed
      - ``evaluation_failure``: only while answer is still unresolved (no human decision)
    Extra non-gold tool calls → secondary ``agent_selection``.
    """
    secondary: List[str] = []
    gold = _as_set(row.get("gold_operations"))
    pred = _predicted_ops(row)
    # Prefer final (post-human) answer flag when present.
    answer_correct = row.get("final_answer_correct")
    if answer_correct is None:
        answer_correct = row.get("answer_correct")
    auto_status = row.get("auto_grade_status")
    status = str(row.get("status") or "")
    err = str(row.get("error") or row.get("error_type") or "").strip()
    predicted_status = str(row.get("predicted_status") or "").lower()
    tool_ok = bool(row.get("all_tool_invocations_successful", row.get("tool_execution_success", True)))
    exact = bool(row.get("exact_tool_set"))
    human_resolved = bool(row.get("human_decision_applied"))
    strict = row.get("strict_task_success")

    # Evaluation failure only while still unresolved — never after human confirm.
    if (
        not human_resolved
        and auto_status == "manual_required"
        and answer_correct is None
        and (
            status == "full"
            or predicted_status in {"answered", "unavailable", "clarification_needed"}
        )
    ):
        primary = "evaluation_failure"
        reason = "manual_required_structured_gold_insufficient"
        return _pack(primary, secondary, reason, row)

    should_refuse = _should_refuse(row)

    def _answer_secondaries() -> None:
        if err in RESOURCE_ERRORS:
            secondary.append("resource_limit")
        if gold and not _gold_in_retrieval(row, retrieval_row):
            secondary.append("retrieval")
        # Extra or missing ops (incl. over-selection) → agent_selection.
        if gold and (not exact):
            secondary.append("agent_selection")
        elif pred - gold:
            secondary.append("agent_selection")
        if not tool_ok:
            secondary.append("tool_execution")
        if should_refuse and predicted_status not in REFUSAL_STATUSES:
            secondary.append("feasibility_refusal")
        if _dependency_issue(row):
            secondary.append("dependency_order")

    # Strict / answer success paths (do not label answer-correct as strict success).
    if answer_correct is True:
        _answer_secondaries()
        if strict is True:
            return _pack(
                "strict_task_success",
                secondary,
                "strict_task_success",
                row,
            )
        return _pack(
            "answer_success",
            secondary,
            "answer_correct_but_not_strict_success",
            row,
        )

    # Resource limits when the run hard-stopped without a correct answer.
    if err in RESOURCE_ERRORS or (
        status == "failed" and err in RESOURCE_ERRORS
    ):
        primary = "resource_limit"
        reason = f"resource_limit:{err or 'unknown'}"
        if gold and not _gold_in_retrieval(row, retrieval_row):
            secondary.append("retrieval")
        elif gold and gold.isdisjoint(pred):
            secondary.append("agent_selection")
        return _pack(primary, secondary, reason, row)

    # Refusal / feasibility failures.
    if should_refuse:
        if predicted_status not in REFUSAL_STATUSES or answer_correct is False:
            primary = "feasibility_refusal"
            reason = "failed_refusal_or_substitute_answer"
            if gold and not _gold_in_retrieval(row, retrieval_row):
                secondary.append("retrieval")
            return _pack(primary, secondary, reason, row)

    # Retrieval miss.
    if gold and not _gold_in_retrieval(row, retrieval_row):
        primary = "retrieval"
        reason = "gold_operations_not_in_find_operations_topk"
        if pred and gold.isdisjoint(pred):
            secondary.append("agent_selection")
        if answer_correct is False:
            secondary.append("answer_synthesis_calculation")
        return _pack(primary, secondary, reason, row)

    # Agent selection miss (gold was retrievable / present).
    if gold and not gold.issubset(pred):
        primary = "agent_selection"
        reason = "gold_ops_retrieved_or_available_but_not_fully_selected"
        if not tool_ok:
            secondary.append("tool_execution")
        if answer_correct is False:
            secondary.append("answer_synthesis_calculation")
        return _pack(primary, secondary, reason, row)

    # Argument / describe grounding.
    if _has_arg_grounding_issue(row):
        primary = "description_argument_grounding"
        reason = "gold_ops_selected_but_argument_or_validation_failed"
        if answer_correct is False:
            secondary.append("answer_synthesis_calculation")
        return _pack(primary, secondary, reason, row)

    # Dependency / order.
    if _dependency_issue(row):
        primary = "dependency_order"
        reason = "multi_step_dependency_or_order_failure"
        if answer_correct is False:
            secondary.append("answer_synthesis_calculation")
        return _pack(primary, secondary, reason, row)

    # Tool execution.
    if not tool_ok:
        primary = "tool_execution"
        reason = "invoke_api_failure_without_full_recovery"
        if answer_correct is False:
            secondary.append("answer_synthesis_calculation")
        return _pack(primary, secondary, reason, row)

    # Answer synthesis / calculation (tools OK, answer wrong).
    if answer_correct is False:
        primary = "answer_synthesis_calculation"
        reason = "tools_ok_but_final_answer_incorrect"
        if not exact:
            secondary.append("agent_selection")
        return _pack(primary, secondary, reason, row)

    # Soft failures: status failed without taxonomy match.
    if status == "failed" or err:
        primary = "resource_limit" if err in RESOURCE_ERRORS else "tool_execution"
        return _pack(primary, secondary, f"unclassified_failure:{err or status}", row)

    # Evaluation change signal (optional annotation only).
    if (
        v11_answer_correct is False
        and answer_correct is True
        and v11_auto_status == "graded"
    ):
        secondary.append("evaluation_failure")

    if answer_correct is None:
        return _pack(
            "evaluation_failure",
            secondary,
            "answer_correct_unavailable",
            row,
        )

    # Should be unreachable when answer_correct is bool; keep fail-safe.
    if answer_correct is True:
        return _pack("answer_success", secondary, "default_answer_success", row)
    return _pack(
        "answer_synthesis_calculation",
        secondary,
        "default_answer_incorrect",
        row,
    )


def _pack(
    primary: str,
    secondary: Sequence[str],
    reason: str,
    row: Dict[str, Any],
) -> Dict[str, Any]:
    # Dedupe secondary, exclude primary.
    sec = []
    seen = set()
    for s in secondary:
        if not s or s == primary or s in seen:
            continue
        seen.add(s)
        sec.append(s)
    return {
        "task_id": row.get("task_id") or row.get("id"),
        "primary_error_category": primary,
        "secondary_error_categories": sec,
        "error_reason": reason,
        "taxonomy_version": TAXONOMY_VERSION,
    }


def summarize_taxonomy(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    counts: Dict[str, int] = {k: 0 for k in PRIMARY_CATEGORIES}
    secondary_counts: Dict[str, int] = {}
    by_domain: Dict[str, Dict[str, int]] = {}
    by_diff: Dict[str, Dict[str, int]] = {}
    for r in rows:
        primary = str(r.get("primary_error_category") or "success")
        counts[primary] = counts.get(primary, 0) + 1
        for s in r.get("secondary_error_categories") or []:
            secondary_counts[str(s)] = secondary_counts.get(str(s), 0) + 1
        dom = str(r.get("domain") or "null")
        diff = str(r.get("project_difficulty") or "null")
        by_domain.setdefault(dom, {})
        by_domain[dom][primary] = by_domain[dom].get(primary, 0) + 1
        by_diff.setdefault(diff, {})
        by_diff[diff][primary] = by_diff[diff].get(primary, 0) + 1
    n = len(rows) or 1
    return {
        "taxonomy_version": TAXONOMY_VERSION,
        "n": len(rows),
        "primary_counts": counts,
        "primary_rates": {k: counts.get(k, 0) / n for k in PRIMARY_CATEGORIES},
        "secondary_counts": secondary_counts,
        "by_domain": by_domain,
        "by_project_difficulty": by_diff,
    }
