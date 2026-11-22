"""Deterministic safe arithmetic evaluator for APV-Fin compute steps.

No eval()/exec(). Supports +, -, *, /, **, unary +/-, parentheses, and
named operands substituted before evaluation.
"""

from __future__ import annotations

import ast
import math
import operator
from typing import Any, Dict, Mapping, Optional, Tuple, Union

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


class CalculatorError(ValueError):
    pass


def _eval_node(node: ast.AST, operands: Mapping[str, Number]) -> Number:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body, operands)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise CalculatorError(f"unsupported constant: {node.value!r}")
        return node.value
    # Python 3.9 may still produce Num in some paths; keep defensive.
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
    raise CalculatorError(f"unsupported expression node: {type(node).__name__}")


def safe_eval_expression(expression: str, operands: Mapping[str, Number]) -> Number:
    if not isinstance(expression, str) or not expression.strip():
        raise CalculatorError("expression must be a non-empty string")
    # Reject attribute access / calls / comprehensions by AST whitelist.
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError as exc:
        raise CalculatorError(f"invalid expression syntax: {exc}") from exc
    for node in ast.walk(tree):
        if isinstance(
            node,
            (
                ast.Call,
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
            ),
        ):
            raise CalculatorError(
                f"disallowed expression construct: {type(node).__name__}"
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


def compute_trace_record(
    *,
    step_id: str,
    payload: Dict[str, Any],
    evaluated: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
) -> Dict[str, Any]:
    evaluated = evaluated or {}
    return {
        "step_id": step_id,
        "formula": payload.get("expression"),
        "operands": payload.get("operands") or evaluated.get("operands"),
        "result": evaluated.get("result"),
        "unit": payload.get("unit") if payload.get("unit") is not None else evaluated.get("unit"),
        "ok": bool(evaluated.get("ok")) and error is None,
        "error": error,
    }
