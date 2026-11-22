"""Deterministic calculator tests (no eval)."""

from __future__ import annotations

import pytest

from hk_equity_mcp.apv_fin.calculator import (
    CalculatorError,
    evaluate_compute_payload,
    safe_eval_expression,
)


def test_deterministic_arithmetic():
    assert safe_eval_expression("(a + b) / c", {"a": 10, "b": 5, "c": 3}) == 5.0
    out = evaluate_compute_payload(
        {
            "expression": "end / start - 1",
            "operands": {"end": 110.0, "start": 100.0},
            "unit": "decimal_return",
            "notes": None,
        }
    )
    assert out["ok"] is True
    assert abs(out["result"] - 0.1) < 1e-12


def test_rejects_call_and_attribute():
    with pytest.raises(CalculatorError, match="disallowed|unsupported"):
        safe_eval_expression("__import__('os').system('x')", {})
    with pytest.raises(CalculatorError):
        safe_eval_expression("a.b", {"a": 1})


def test_unknown_operand_and_div_zero():
    with pytest.raises(CalculatorError, match="unknown operand"):
        safe_eval_expression("a + b", {"a": 1})
    with pytest.raises(CalculatorError, match="division by zero"):
        safe_eval_expression("a / b", {"a": 1, "b": 0})
