"""Automatic metrics for unified HK Operation-Visible runs.

Answer/evidence correctness remain manual. Gold-path execution success is NOT
treated as Agent success.

Tool execution definitions
--------------------------
``all_tool_invocations_successful`` / ``tool_execution_success``:
    True iff every ``invoke_api`` in the trajectory succeeded.
    Vacuous True when there were zero invocations (e.g. correct refusal with
    no tool calls is NOT treated as an execution failure).

``recovered_after_tool_error``:
    True when an earlier invoke failed, a later invoke succeeded, and the
    agent still produced a complete final answer (``status == "full"``).
    This is execution recovery only — not answer correctness.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

EVALUATOR_VERSION = "hk_unified_evaluator_v1.2"
EVALUATOR_MODULE_PATH = Path(__file__).resolve()


REFUSAL_CLASSES = frozenset(
    {
        "intentionally_infeasible",
        "should_refuse_or_clarify",
        "capability_gap",
        "robustness",
    }
)
REFUSAL_STATUSES = frozenset({"unavailable", "clarification_needed"})


def evaluator_source_hash() -> str:
    return hashlib.sha256(EVALUATOR_MODULE_PATH.read_bytes()).hexdigest()


def evaluator_identity() -> Dict[str, str]:
    return {
        "evaluator_version": EVALUATOR_VERSION,
        "evaluator_source_sha256": evaluator_source_hash(),
        "evaluator_module": str(EVALUATOR_MODULE_PATH.name),
    }


def _as_set(values: Optional[Iterable[Any]]) -> Set[str]:
    if not values:
        return set()
    out: Set[str] = set()
    for v in values:
        if v is None:
            continue
        out.add(str(v))
    return out


def tool_prf(predicted: Set[str], gold: Set[str]) -> Tuple[float, float, float]:
    if not predicted and not gold:
        return 1.0, 1.0, 1.0
    tp = len(predicted & gold)
    precision = tp / len(predicted) if predicted else 0.0
    recall = tp / len(gold) if gold else 0.0
    if precision + recall == 0:
        return precision, recall, 0.0
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def extract_predicted_operations(trace: Sequence[Dict[str, Any]]) -> List[str]:
    """Collect invoke_api operation_ids in call order (unique preserve)."""
    seen: Set[str] = set()
    ordered: List[str] = []
    for event in trace or []:
        if event.get("type") != "tool_result":
            continue
        if event.get("name") != "invoke_api":
            continue
        args = event.get("arguments") or {}
        oid = args.get("operation_id")
        if not oid:
            continue
        if oid not in seen:
            seen.add(str(oid))
            ordered.append(str(oid))
    return ordered


def extract_retrieved_candidates(trace: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for event in trace or []:
        if event.get("type") != "tool_result":
            continue
        # Operation-Visible: find_operations; MCP-Zero: mcp_zero_retrieve.
        if event.get("name") not in {"find_operations", "mcp_zero_retrieve"}:
            continue
        result = event.get("result") or {}
        for cand in result.get("candidates") or []:
            rows.append(
                {
                    "operation_id": cand.get("operation_id"),
                    "score": cand.get("score"),
                    "cosine": cand.get("cosine"),
                    "lexical": cand.get("lexical"),
                    "server_score": cand.get("server_score"),
                    "tool_score": cand.get("tool_score"),
                    "rank": cand.get("rank"),
                    "server_id": cand.get("server_id"),
                    "query": result.get("query"),
                }
            )
    return rows


def extract_describe_calls(trace: Sequence[Dict[str, Any]]) -> List[str]:
    out: List[str] = []
    for event in trace or []:
        if event.get("type") != "tool_result":
            continue
        if event.get("name") != "describe_operation":
            continue
        args = event.get("arguments") or {}
        oid = args.get("operation_id")
        if oid:
            out.append(str(oid))
    return out


def extract_invoke_records(trace: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for event in trace or []:
        if event.get("type") != "tool_result":
            continue
        if event.get("name") != "invoke_api":
            continue
        args = event.get("arguments") or {}
        result = event.get("result") or {}
        nested = result.get("result") if isinstance(result.get("result"), dict) else {}
        nested = nested or {}
        success = bool(result.get("success"))
        protocol_success = nested.get("protocol_success")
        if protocol_success is None:
            protocol_success = result.get("protocol_success")
        execution_success = nested.get("execution_success")
        if execution_success is None:
            execution_success = result.get("execution_success")
        out.append(
            {
                "operation_id": args.get("operation_id"),
                "arguments": args.get("arguments") or {},
                "success": success,
                "protocol_success": protocol_success,
                "execution_success": execution_success,
                "error": result.get("error") or nested.get("error"),
                "error_type": result.get("error_type") or nested.get("error_type"),
                "parallel": bool(event.get("parallel")),
            }
        )
    return out


def count_parse_repairs(trace: Sequence[Dict[str, Any]]) -> int:
    return sum(
        1 for e in trace or [] if e.get("type") == "llm_turn" and e.get("parse_error")
    )


def extract_evidence_from_answer(final_answer: Any) -> List[Dict[str, Any]]:
    if isinstance(final_answer, dict):
        ev = final_answer.get("evidence")
        if isinstance(ev, list):
            return [e for e in ev if isinstance(e, dict)]
    return []


def dependency_edge_accuracy(
    predicted_ops: Sequence[str],
    gold_graph: Optional[Sequence[Dict[str, Any]]],
) -> Optional[float]:
    """Approximate graph accuracy: producer/consumer ops both predicted when gold edges exist."""
    if not gold_graph:
        return None
    pred = set(predicted_ops)
    edges = list(gold_graph)
    if not edges:
        return 1.0
    ok = 0
    for edge in edges:
        producer = edge.get("producer_operation") or edge.get("from_operation")
        consumer = edge.get("consumer_operation") or edge.get("to_operation")
        producer = producer or edge.get("producer_operation_id")
        consumer = consumer or edge.get("consumer_operation_id")
        if producer is None and consumer is None:
            ops = edge.get("operations") or []
            if ops and set(map(str, ops)).issubset(pred):
                ok += 1
            continue
        if producer and consumer and str(producer) in pred and str(consumer) in pred:
            ok += 1
    return ok / len(edges)


def refusal_correctness(
    *,
    answerability_class: Optional[str],
    gold_status: Optional[str],
    predicted_status: Optional[str],
    gold_operations: Sequence[str],
) -> Optional[float]:
    """Deterministic subset only; returns None when human judgment is required."""
    ac = (answerability_class or "").strip()
    gs = (gold_status or "").strip().lower()
    ps = (predicted_status or "").strip().lower()
    gold_ops = _as_set(gold_operations)

    should_refuse = (
        ac in REFUSAL_CLASSES
        or gs in {"unavailable", "clarification_needed", "refuse"}
        or (not gold_ops and ac not in {"structured_answerable", "evidence_answerable"})
    )
    if should_refuse:
        return 1.0 if ps in REFUSAL_STATUSES else 0.0
    if ps in REFUSAL_STATUSES and gold_ops:
        return 0.0
    return None


def _complete_final_answer(result: Dict[str, Any]) -> bool:
    """Agent produced a completed final-answer turn (not a hard crash)."""
    if result.get("status") == "full":
        return True
    # Offline rows: treat non-empty final_answer without failed/timeout as complete.
    if result.get("status") in {"failed"}:
        return False
    return result.get("final_answer") not in (None, "")


def compute_tool_execution_metrics(
    invoke_records: Sequence[Dict[str, Any]],
    *,
    result: Dict[str, Any],
) -> Dict[str, Any]:
    """Split tool execution metrics (execution recovery ≠ answer correctness)."""
    successful = [r for r in invoke_records if r.get("success")]
    failed = [r for r in invoke_records if not r.get("success")]
    # Vacuous True when zero invocations: refusal without tools is not exec failure.
    all_ok = all(r.get("success") for r in invoke_records) if invoke_records else True

    recovered = False
    if failed and successful and _complete_final_answer(result):
        # Early failure then later success (order-preserving).
        first_fail_idx = next(
            i for i, r in enumerate(invoke_records) if not r.get("success")
        )
        recovered = any(
            r.get("success") for r in invoke_records[first_fail_idx + 1 :]
        )

    protocol_vals = [
        bool(r.get("protocol_success"))
        for r in invoke_records
        if r.get("protocol_success") is not None
    ]
    if protocol_vals:
        protocol_success_rate = sum(protocol_vals) / len(protocol_vals)
    else:
        # Fall back to top-level invoke success when protocol flags absent.
        protocol_success_rate = (
            (sum(1 for r in invoke_records if r.get("success")) / len(invoke_records))
            if invoke_records
            else None
        )

    by_oid: Dict[str, bool] = {}
    for rec in invoke_records:
        oid = rec.get("operation_id")
        if not oid:
            continue
        key = str(oid)
        by_oid[key] = bool(by_oid.get(key)) or bool(rec.get("success"))

    return {
        # Canonical name; tool_execution_success kept as alias with same meaning.
        "all_tool_invocations_successful": all_ok,
        "tool_execution_success": all_ok,
        "successful_invocation_count": len(successful),
        "failed_invocation_count": len(failed),
        "recovered_after_tool_error": recovered,
        "protocol_success_rate": (
            round(protocol_success_rate, 6)
            if protocol_success_rate is not None
            else None
        ),
        "final_operation_success_by_operation_id": by_oid,
    }


def evaluate_task(
    *,
    item: Dict[str, Any],
    result: Dict[str, Any],
) -> Dict[str, Any]:
    trace = result.get("trace") or []
    predicted_ops = extract_predicted_operations(trace)
    gold_ops = [str(x) for x in (item.get("gold_operations") or [])]
    # Prefer gold already stored on a prediction row during re-eval.
    if not gold_ops and result.get("gold_operations"):
        gold_ops = [str(x) for x in (result.get("gold_operations") or [])]
    pred_set, gold_set = set(predicted_ops), set(gold_ops)
    precision, recall, f1 = tool_prf(pred_set, gold_set)
    invoke_records = extract_invoke_records(trace)
    tool_metrics = compute_tool_execution_metrics(invoke_records, result=result)

    wrong = 0.0
    if gold_set:
        wrong = len(pred_set - gold_set) / max(len(pred_set), 1)
    elif pred_set:
        wrong = 1.0

    graph_acc = dependency_edge_accuracy(
        predicted_ops, item.get("dependency_graph")
    )
    refusal = refusal_correctness(
        answerability_class=item.get("answerability_class")
        or result.get("answerability_class"),
        gold_status=item.get("gold_status") or result.get("gold_status"),
        predicted_status=result.get("predicted_status"),
        gold_operations=gold_ops,
    )

    tokens = result.get("tokens") or {}
    evidence = extract_evidence_from_answer(
        result.get("final_answer_obj")
    ) or extract_evidence_from_answer(result.get("final_answer"))
    repair_count = int(result.get("repair_count") or 0)
    if not repair_count:
        repair_count = int(result.get("protocol_repair_count") or 0) + int(
            result.get("parse_repair_count") or count_parse_repairs(trace)
        )
    total_turn_count = int(result.get("total_turn_count") or result.get("llm_calls") or 0)
    valid_turn_count = result.get("valid_turn_count")
    if valid_turn_count is None:
        valid_turn_count = sum(
            1
            for e in trace
            if e.get("type") == "llm_turn" and e.get("turn_valid") is True
        )
    terminal_contract_valid = result.get("terminal_contract_valid")
    if terminal_contract_valid is None:
        terminal_contract_valid = (
            result.get("status") == "full"
            and isinstance(result.get("final_answer_obj"), dict)
            and result.get("final_answer_obj", {}).get("status") is not None
        )
    task_protocol_success = result.get("task_protocol_success")
    if task_protocol_success is None:
        task_protocol_success = bool(
            terminal_contract_valid
            and result.get("status") == "full"
            and not result.get("error")
        )
    # Task-level protocol success must be false on max_llm_turns / failed runs
    # even when gateway invoke-level protocol_success flags are all True.
    if str(result.get("status") or "") == "failed":
        task_protocol_success = False
        terminal_contract_valid = False

    out = {
        "predicted_operations": predicted_ops,
        "retrieved_candidates": extract_retrieved_candidates(trace),
        "describe_calls": extract_describe_calls(trace),
        "invoke_calls": invoke_records,
        "dependency_edges_predicted": [
            {
                "producer": predicted_ops[i],
                "consumer": predicted_ops[i + 1],
            }
            for i in range(len(predicted_ops) - 1)
        ],
        "evidence": evidence,
        "citations": evidence,
        "gold_operations": gold_ops,
        "operation_precision": round(precision, 6),
        "operation_recall": round(recall, 6),
        "operation_f1": round(f1, 6),
        "exact_tool_set": pred_set == gold_set,
        "graph_dependency_accuracy": graph_acc,
        "wrong_tool_rate": round(wrong, 6),
        "refusal_correctness": refusal,
        "parse_repair_count": int(
            result.get("parse_repair_count") or count_parse_repairs(trace)
        ),
        "protocol_repair_count": int(result.get("protocol_repair_count") or 0),
        "repair_count": repair_count,
        "valid_turn_count": int(valid_turn_count or 0),
        "total_turn_count": total_turn_count,
        "terminal_contract_valid": bool(terminal_contract_valid),
        "task_protocol_success": bool(task_protocol_success),
        "real_mcp_execution_count": len(invoke_records),
        "input_tokens": tokens.get("input_tokens"),
        "output_tokens": tokens.get("output_tokens"),
        "total_tokens": tokens.get("total_tokens"),
    }
    out.update(tool_metrics)
    return out


def _program_error_key(row: Dict[str, Any]) -> Optional[str]:
    """Only non-empty program errors; never treat status=full as an error."""
    for key in ("error", "error_type"):
        val = row.get(key)
        if val is None:
            continue
        text = str(val).strip()
        if text and text.lower() not in {"none", "null"}:
            return text
    return None


def summarize_predictions(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {
            "n": 0,
            "error_counts": {},
            "result_status_counts": {},
            "note": (
                "error_counts only include non-empty error/error_type; "
                "result_status_counts are independent (full is not an error)."
            ),
        }

    def _avg(key: str) -> Optional[float]:
        vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
        if not vals:
            return None
        return sum(vals) / len(vals)

    error_counts: Dict[str, int] = {}
    result_status_counts: Dict[str, int] = {}
    for r in rows:
        status = r.get("status")
        status_key = str(status) if status is not None and str(status).strip() else "missing"
        result_status_counts[status_key] = result_status_counts.get(status_key, 0) + 1
        err_key = _program_error_key(r)
        if err_key is not None:
            error_counts[err_key] = error_counts.get(err_key, 0) + 1

    attempt_rows = [
        r for r in rows if int(r.get("real_mcp_execution_count") or 0) > 0
    ]
    successful_invocations = sum(
        int(r.get("successful_invocation_count") or 0) for r in rows
    )
    total_invocations = sum(int(r.get("real_mcp_execution_count") or 0) for r in rows)
    zero_invocation_failed = sum(
        1
        for r in rows
        if str(r.get("status") or "") == "failed"
        and int(r.get("real_mcp_execution_count") or 0) == 0
    )
    inv_conditioned_success = (
        sum(1 for r in attempt_rows if r.get("all_tool_invocations_successful"))
        / len(attempt_rows)
        if attempt_rows
        else None
    )
    task_protocol_rate = (
        sum(1 for r in rows if r.get("task_protocol_success")) / len(rows)
    )

    return {
        "n": len(rows),
        "processed_count": len(rows),
        "full_count": result_status_counts.get("full", 0),
        "completed_success_count": result_status_counts.get("full", 0),
        "failed_count": result_status_counts.get("failed", 0),
        "mean_operation_f1": _avg("operation_f1"),
        "mean_operation_precision": _avg("operation_precision"),
        "mean_operation_recall": _avg("operation_recall"),
        "exact_tool_set_rate": sum(1 for r in rows if r.get("exact_tool_set"))
        / len(rows),
        # Compatibility: vacuous True on zero-invocation tasks still included.
        "all_tool_invocations_successful_rate": sum(
            1 for r in rows if r.get("all_tool_invocations_successful")
        )
        / len(rows),
        "execution_success_rate": sum(
            1 for r in rows if r.get("tool_execution_success")
        )
        / len(rows),
        "invocation_attempt_task_count": len(attempt_rows),
        "invocation_conditioned_task_success_rate": (
            round(inv_conditioned_success, 6)
            if inv_conditioned_success is not None
            else None
        ),
        "successful_invocations": successful_invocations,
        "total_invocations": total_invocations,
        "zero_invocation_failed_task_count": zero_invocation_failed,
        "recovered_after_tool_error_rate": sum(
            1 for r in rows if r.get("recovered_after_tool_error")
        )
        / len(rows),
        # Invoke-transport protocol average (may be high even when task failed).
        "mean_protocol_success_rate": _avg("protocol_success_rate"),
        # Task-level protocol contract (False on max_llm_turns / failed).
        "task_protocol_success_rate": round(task_protocol_rate, 6),
        "mean_valid_turn_rate": (
            None
            if not any(isinstance(r.get("total_turn_count"), (int, float)) for r in rows)
            else (
                sum(float(r.get("valid_turn_count") or 0) for r in rows)
                / max(sum(float(r.get("total_turn_count") or 0) for r in rows), 1.0)
            )
        ),
        "mean_repair_count": _avg("repair_count"),
        "terminal_contract_valid_rate": round(
            sum(1 for r in rows if r.get("terminal_contract_valid")) / len(rows),
            6,
        ),
        "mean_wrong_tool_rate": _avg("wrong_tool_rate"),
        "mean_latency_ms": _avg("latency_ms"),
        "mean_llm_calls": _avg("llm_calls"),
        "mean_total_tokens": _avg("total_tokens"),
        "mean_real_mcp_execution_count": _avg("real_mcp_execution_count"),
        "error_counts": error_counts,
        "result_status_counts": result_status_counts,
        "note": (
            "error_counts only include non-empty error/error_type; "
            "result_status_counts are independent (full is not an error). "
            "tool_execution_success == all_tool_invocations_successful "
            "(vacuous True when zero invocations). Prefer "
            "invocation_conditioned_task_success_rate and "
            "task_protocol_success_rate for formal reporting."
        ),
    }


def reevaluate_prediction_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Recompute metrics from an existing prediction row (offline; no Agent)."""
    item = {
        "gold_operations": row.get("gold_operations") or [],
        "dependency_graph": row.get("dependency_graph"),
        "answerability_class": row.get("answerability_class"),
        "gold_status": row.get("gold_status"),
    }
    metrics = evaluate_task(item=item, result=row)
    # Return metrics only — never mutate identity fields of the prediction.
    return metrics
