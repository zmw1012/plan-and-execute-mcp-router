"""Safe deterministic calculator for PE-Fin Lite.

Extends the APV-Fin AST evaluator with a small whitelist of Call functions:
abs, min, max, round_half_up. Attribute access, imports, and arbitrary calls
remain forbidden. Unknown functions fail closed.
"""

from __future__ import annotations

import ast
import decimal
import math
import operator
from typing import Any, Dict, Mapping, Optional, Union

from hk_equity_mcp.apv_fin.calculator import CalculatorError  # noqa: F401 — re-export

Number = Union[int, float]

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}
_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

SAFE_FUNCTION_NAMES = frozenset({"abs", "min", "max", "round_half_up"})
COMPUTE_FUNCTION_CONTRACT_VERSION = "pe_fin_lite_compute_functions_v1"


def round_half_up(value: Number, ndigits: Number = 0) -> float:
    """Bankers-unsafe half-up rounding (explicitly requested by planners)."""
    digits = int(ndigits)
    q = decimal.Decimal("1").scaleb(-digits) if digits >= 0 else decimal.Decimal(10) ** (-digits)
    d = decimal.Decimal(str(float(value)))
    return float(d.quantize(q, rounding=decimal.ROUND_HALF_UP))


_SAFE_FUNCS = {
    "abs": lambda *a: abs(float(a[0])),
    "min": lambda *a: float(min(float(x) for x in a)),
    "max": lambda *a: float(max(float(x) for x in a)),
    "round_half_up": lambda *a: round_half_up(
        a[0], a[1] if len(a) > 1 else 0
    ),
}


def _eval_node(node: ast.AST, operands: Mapping[str, Number]) -> Number:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body, operands)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise CalculatorError(f"unsupported constant: {node.value!r}")
        return node.value
    if isinstance(node, getattr(ast, "Num", ())):  # type: ignore[arg-type]
        return float(node.n)  # type: ignore[attr-defined]
    if isinstance(node, ast.Name):
        if node.id not in operands:
            raise CalculatorError(f"unknown operand: {node.id}")
        return operands[node.id]
    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _BIN_OPS:
            raise CalculatorError(f"unsupported binary operator: {op_type.__name__}")
        left = _eval_node(node.left, operands)
        right = _eval_node(node.right, operands)
        try:
            return _BIN_OPS[op_type](left, right)
        except ZeroDivisionError as exc:
            raise CalculatorError("division by zero") from exc
    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _UNARY_OPS:
            raise CalculatorError(f"unsupported unary operator: {op_type.__name__}")
        return _UNARY_OPS[op_type](_eval_node(node.operand, operands))
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise CalculatorError("only simple function names are allowed")
        fname = node.func.id
        if fname not in _SAFE_FUNCS:
            raise CalculatorError(f"disallowed function: {fname}")
        if node.keywords:
            raise CalculatorError("keyword arguments are not allowed")
        args = [_eval_node(a, operands) for a in node.args]
        try:
            return _SAFE_FUNCS[fname](*args)
        except (TypeError, ValueError, decimal.InvalidOperation) as exc:
            raise CalculatorError(f"function {fname} failed: {exc}") from exc
    raise CalculatorError(f"unsupported expression node: {type(node).__name__}")


def safe_eval_expression(expression: str, operands: Mapping[str, Number]) -> Number:
    if not isinstance(expression, str) or not expression.strip():
        raise CalculatorError("expression must be a non-empty string")
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError as exc:
        raise CalculatorError(f"invalid expression syntax: {exc}") from exc
    for node in ast.walk(tree):
        if isinstance(
            node,
            (
                ast.Attribute,
                ast.Subscript,
                ast.List,
                ast.Dict,
                ast.Tuple,
                ast.Set,
                ast.ListComp,
                ast.DictComp,
                ast.SetComp,
                ast.GeneratorExp,
                ast.Lambda,
                ast.Await,
                ast.Yield,
                ast.Import,
                ast.ImportFrom,
            ),
        ):
            raise CalculatorError(
                f"disallowed expression construct: {type(node).__name__}"
            )
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in SAFE_FUNCTION_NAMES:
                raise CalculatorError(
                    f"disallowed expression construct: Call"
                    + (
                        f"({getattr(node.func, 'id', '?')})"
                        if isinstance(node.func, ast.Name)
                        else ""
                    )
                )
    result = _eval_node(tree, operands)
    if isinstance(result, bool) or not isinstance(result, (int, float)):
        raise CalculatorError("expression did not evaluate to a number")
    if isinstance(result, float) and (math.isnan(result) or math.isinf(result)):
        raise CalculatorError("non-finite result")
    return result


def evaluate_compute_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    expression = str(payload.get("expression") or "")
    raw_operands = dict(payload.get("operands") or {})
    operands: Dict[str, Number] = {}
    for key, value in raw_operands.items():
        if not isinstance(key, str) or not key:
            raise CalculatorError("operand names must be non-empty strings")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise CalculatorError(f"operand {key!r} must be numeric")
        operands[key] = value
    result = safe_eval_expression(expression, operands)
    return {
        "expression": expression,
        "operands": operands,
        "result": result,
        "unit": payload.get("unit"),
        "notes": payload.get("notes"),
        "ok": True,
    }


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
        "unit": spec.get("output_unit")
        if spec.get("output_unit") is not None
        else evaluated.get("unit"),
        "output_name": spec.get("output_name"),
        "ok": bool(evaluated.get("ok")) and error is None,
        "error": error,
    }
