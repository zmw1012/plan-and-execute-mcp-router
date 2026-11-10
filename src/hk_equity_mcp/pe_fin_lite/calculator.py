"""Safe deterministic calculator for PE-Fin Lite (reuses APV-Fin AST evaluator)."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Union

from hk_equity_mcp.apv_fin.calculator import (  # noqa: F401 — re-export
    CalculatorError,
    evaluate_compute_payload,
    safe_eval_expression,
)

Number = Union[int, float]


def percent_change(old: Number, new: Number) -> float:
    if old == 0:
        raise CalculatorError("percent_change division by zero")
    return (float(new) - float(old)) / float(old) * 100.0


def ratio(numerator: Number, denominator: Number) -> float:
    if denominator == 0:
        raise CalculatorError("ratio division by zero")
    return float(numerator) / float(denominator)


def margin(part: Number, whole: Number) -> float:
    return ratio(part, whole)


def average(values: Mapping[str, Number] | list) -> float:
    if isinstance(values, Mapping):
        nums = list(values.values())
    else:
        nums = list(values)
    if not nums:
        raise CalculatorError("average of empty set")
    return float(sum(float(x) for x in nums)) / len(nums)


def compare(a: Number, b: Number) -> Dict[str, Any]:
    return {
        "a": a,
        "b": b,
        "diff": float(a) - float(b),
        "a_gt_b": float(a) > float(b),
        "a_eq_b": float(a) == float(b),
    }


def evaluate_named_op(op: str, operands: Mapping[str, Number]) -> Number:
    op = str(op).strip().lower()
    vals = list(operands.values())
    if op in {"add", "sum"}:
        return float(sum(float(v) for v in vals))
    if op in {"subtract", "sub"}:
        if len(vals) != 2:
            raise CalculatorError("subtract requires exactly 2 operands")
        return float(vals[0]) - float(vals[1])
    if op in {"multiply", "mul", "product"}:
        out = 1.0
        for v in vals:
            out *= float(v)
        return out
    if op in {"divide", "div"}:
        if len(vals) != 2:
            raise CalculatorError("divide requires exactly 2 operands")
        return ratio(vals[0], vals[1])
    if op in {"percent_change", "pct_change", "growth"}:
        if len(vals) != 2:
            raise CalculatorError("percent_change requires exactly 2 operands")
        return percent_change(vals[0], vals[1])
    if op in {"ratio"}:
        if len(vals) != 2:
            raise CalculatorError("ratio requires exactly 2 operands")
        return ratio(vals[0], vals[1])
    if op in {"margin"}:
        if len(vals) != 2:
            raise CalculatorError("margin requires exactly 2 operands")
        return margin(vals[0], vals[1])
    if op in {"average", "mean"}:
        return average(vals)
    raise CalculatorError(f"unsupported named op: {op}")


def run_compute_spec(
    spec: Dict[str, Any],
    *,
    resolved_operands: Mapping[str, Number],
) -> Dict[str, Any]:
    expression = str(spec.get("expression") or "").strip()
    # Allow named ops via expression like "percent_change(a,b)" mapped separately
    # by binder/planner using plain AST expression with operand names.
    evaluated = evaluate_compute_payload(
        {
            "expression": expression,
            "operands": dict(resolved_operands),
            "unit": spec.get("output_unit"),
        }
    )
    return {
        "expression": expression,
        "operands": dict(resolved_operands),
        "operand_refs": dict(spec.get("operand_refs") or {}),
        "result": evaluated["result"],
        "unit": spec.get("output_unit"),
        "output_name": spec.get("output_name"),
        "ok": True,
    }


def compute_record(
    *,
    step_id: str,
    spec: Dict[str, Any],
    evaluated: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
) -> Dict[str, Any]:
    evaluated = evaluated or {}
    return {
        "step_id": step_id,
        "formula": evaluated.get("expression") or spec.get("expression"),
        "operand_refs": dict(spec.get("operand_refs") or {}),
        "operands": evaluated.get("operands"),
        "result": evaluated.get("result"),
        "unit": spec.get("output_unit") if spec.get("output_unit") is not None else evaluated.get("unit"),
        "output_name": spec.get("output_name"),
        "ok": bool(evaluated.get("ok")) and error is None,
        "error": error,
    }
