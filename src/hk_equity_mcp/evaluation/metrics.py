"""Metric interfaces for HK equity agent experiments.

Implementations are intentionally lightweight placeholders that can be
extended once real LLM trajectories are available.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Set


class MetricCalculator:
    """Compute aggregate metrics from per-example evaluation rows."""

    def strict_task_success_rate(self, rows: List[Dict[str, Any]]) -> float:
        """Fraction of examples where tool, args, execution, and answer all match."""
        if not rows:
            return 0.0
        ok = sum(1 for r in rows if r.get("strict_success"))
        return ok / len(rows)

    def tool_selection_accuracy(self, rows: List[Dict[str, Any]]) -> float:
        if not rows:
            return 0.0
        ok = sum(1 for r in rows if r.get("tool_correct"))
        return ok / len(rows)

    def tool_f1(self, rows: List[Dict[str, Any]]) -> float:
        """Micro F1 over predicted vs gold tool sets (single-tool cases for now)."""
        tp = fp = fn = 0
        for row in rows:
            pred = set(row.get("predicted_tools") or [])
            gold = set(row.get("gold_tools") or [])
            tp += len(pred & gold)
            fp += len(pred - gold)
            fn += len(gold - pred)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    def argument_accuracy(self, rows: List[Dict[str, Any]]) -> float:
        if not rows:
            return 0.0
        ok = sum(1 for r in rows if r.get("args_correct"))
        return ok / len(rows)

    def execution_success_rate(self, rows: List[Dict[str, Any]]) -> float:
        if not rows:
            return 0.0
        ok = sum(1 for r in rows if r.get("execution_success"))
        return ok / len(rows)

    def answer_correctness(self, rows: List[Dict[str, Any]]) -> float:
        if not rows:
            return 0.0
        ok = sum(1 for r in rows if r.get("answer_correct"))
        return ok / len(rows)

    def evidence_correctness(self, rows: List[Dict[str, Any]]) -> float:
        if not rows:
            return 0.0
        ok = sum(1 for r in rows if r.get("evidence_correct"))
        return ok / len(rows)

    def wrong_or_unnecessary_tool_rate(self, rows: List[Dict[str, Any]]) -> float:
        if not rows:
            return 0.0
        bad = sum(1 for r in rows if r.get("wrong_or_unnecessary_tool"))
        return bad / len(rows)

    def average_tool_calls(self, rows: List[Dict[str, Any]]) -> float:
        if not rows:
            return 0.0
        return sum(float(r.get("tool_calls", 0)) for r in rows) / len(rows)

    def average_tokens(self, rows: List[Dict[str, Any]]) -> float:
        if not rows:
            return 0.0
        return sum(float(r.get("tokens", 0)) for r in rows) / len(rows)

    def average_latency_ms(self, rows: List[Dict[str, Any]]) -> float:
        if not rows:
            return 0.0
        return sum(float(r.get("latency_ms", 0)) for r in rows) / len(rows)

    def summarize(self, rows: List[Dict[str, Any]]) -> Dict[str, float]:
        return {
            "strict_task_success_rate": self.strict_task_success_rate(rows),
            "tool_selection_accuracy": self.tool_selection_accuracy(rows),
            "tool_f1": self.tool_f1(rows),
            "argument_accuracy": self.argument_accuracy(rows),
            "execution_success_rate": self.execution_success_rate(rows),
            "answer_correctness": self.answer_correctness(rows),
            "evidence_correctness": self.evidence_correctness(rows),
            "wrong_or_unnecessary_tool_rate": self.wrong_or_unnecessary_tool_rate(rows),
            "tool_calls": self.average_tool_calls(rows),
            "tokens": self.average_tokens(rows),
            "latency": self.average_latency_ms(rows),
        }


def args_match(predicted: Dict[str, Any], gold: Dict[str, Any]) -> bool:
    """Exact match on gold keys (predicted may contain extras)."""
    for key, value in gold.items():
        if predicted.get(key) != value:
            return False
    return True


def evidence_overlap(
    predicted: Iterable[str], supporting: Iterable[str]
) -> bool:
    """Placeholder: any string overlap / containment counts as correct."""
    pred_list = list(predicted or [])
    gold_list = list(supporting or [])
    if not gold_list:
        return True
    for g in gold_list:
        for p in pred_list:
            if g in p or p in g:
                return True
    return False


def answer_contains_reference(answer: Optional[str], reference: str) -> bool:
    if answer is None:
        return False
    return reference.strip().lower() in str(answer).strip().lower()
