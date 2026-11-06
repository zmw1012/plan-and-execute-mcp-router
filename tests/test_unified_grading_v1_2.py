"""Unit tests for deterministic grading v1.2 (no LLM / MCP)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hk_equity_mcp.experiment.unified_grading import (
    GRADER_VERSION,
    extract_numbers,
    grade_task,
    half_up,
)


ROOT = Path(__file__).resolve().parents[1]
PRED_PATH = (
    ROOT
    / "outputs"
    / "hk_unified_exploratory155_operation_visible_v1"
    / "predictions.jsonl"
)


def _load_pred(task_id: str) -> dict:
    if not PRED_PATH.exists():
        pytest.skip("baseline predictions not present")
    for line in PRED_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("task_id") == task_id:
            return row
    raise AssertionError(f"missing fixture task {task_id}")


def test_grader_version():
    assert GRADER_VERSION == "deterministic_v1.2"


def test_scalar_numeric_and_half_up():
    assert half_up(1.2345, 3) == 1.235
    item = {
        "task_id": "t_scalar",
        "answerability_class": "structured_answerable",
        "gold_status": "answered",
        "gold_values": [{"value": 751766, "unit": "CNY million", "tolerance_abs": 0.5}],
    }
    ok = grade_task(
        item,
        {
            "task_id": "t_scalar",
            "predicted_status": "answered",
            "final_answer": "Revenue was RMB 751,766 million.",
        },
    )
    assert ok["answer_correct"] is True


def test_percent_vs_decimal_and_unicode_minus():
    item = {
        "task_id": "t_pct",
        "answerability_class": "structured_answerable",
        "gold_status": "answered",
        "gold_values": [
            {
                "value": -0.107368,
                "unit": "decimal_return",
                "representation": "decimal_not_percent",
                "tolerance_abs": 1e-5,
            }
        ],
    }
    text = "simple return = −0.107368 (decimal)"
    assert extract_numbers(text)[0] == pytest.approx(-0.107368)
    ok = grade_task(
        item,
        {"task_id": "t_pct", "predicted_status": "answered", "final_answer": text},
    )
    assert ok["answer_correct"] is True

    pct_item = {
        "task_id": "t_pct2",
        "answerability_class": "structured_answerable",
        "gold_status": "answered",
        "gold_values": [
            {
                "value": 10.4603,
                "unit": "percent",
                "representation": "percentage_not_decimal",
                "tolerance_abs": 0.01,
            }
        ],
    }
    ok2 = grade_task(
        pct_item,
        {
            "task_id": "t_pct2",
            "predicted_status": "answered",
            "final_answer": "Return is +10.4603% over the window.",
        },
    )
    assert ok2["answer_correct"] is True


def test_percent_tolerance_not_leaking_to_decimal_form():
    """35.90% must not match gold 33.88% via unscaled decimal tolerance."""
    item = {
        "task_id": "t_bad_pct",
        "answerability_class": "structured_answerable",
        "gold_status": "answered",
        "gold_values": [{"value": 33.88, "unit": "percent", "tolerance_abs": 0.05}],
    }
    bad = grade_task(
        item,
        {
            "task_id": "t_bad_pct",
            "predicted_status": "answered",
            "final_answer": "Margin = 0.3590 = 35.90%.",
        },
    )
    assert bad["answer_correct"] is False


def test_multiple_labeled_values_and_extra_intermediates():
    item = {
        "task_id": "t_ohlc",
        "answerability_class": "structured_answerable",
        "gold_status": "answered",
        "gold_values": [
            {
                "value": {"open": 138.0, "high": 138.0, "low": 131.1, "close": 132.3},
                "unit": "HKD",
                "tolerance_abs": 0.05,
            }
        ],
    }
    text = (
        "Trade date 2026-05-15. Intermediate MA=135.0 ignored. "
        "OHLC in HKD: Open 138.00, High 138.00, Low 131.10, Close 132.30."
    )
    ok = grade_task(
        item,
        {"task_id": "t_ohlc", "predicted_status": "answered", "final_answer": text},
    )
    assert ok["answer_correct"] is True


def test_refusal_with_fabricated_substitute_wrong():
    item = {
        "task_id": "t_refuse_bad",
        "answerability_class": "intentionally_infeasible",
        "gold_status": "unavailable",
        "gold_values": None,
    }
    bad = grade_task(
        item,
        {
            "task_id": "t_refuse_bad",
            "predicted_status": "answered",
            "final_answer": (
                "FTShare blocked, but close was HKD 113.40 on 2026-07-15."
            ),
        },
    )
    assert bad["answer_correct"] is False


def test_correct_capability_gap_refusal():
    item = {
        "task_id": "t_refuse_ok",
        "answerability_class": "capability_gap",
        "gold_status": "unavailable",
        "gold_values": None,
    }
    ok = grade_task(
        item,
        {
            "task_id": "t_refuse_ok",
            "predicted_status": "unavailable",
            "final_answer": (
                "Unavailable: no reliable operation exposes live quotes under the frozen catalog."
            ),
        },
    )
    assert ok["answer_correct"] is True


def test_cannot_alone_does_not_auto_pass_when_answered_with_number():
    item = {
        "task_id": "t_cannot",
        "answerability_class": "should_refuse_or_clarify",
        "gold_status": "unavailable",
        "gold_values": None,
    }
    bad = grade_task(
        item,
        {
            "task_id": "t_cannot",
            "predicted_status": "answered",
            "final_answer": (
                "I cannot provide a live price; latest close was HKD 161.4 on 12 Aug 2026."
            ),
        },
    )
    assert bad["answer_correct"] is False


@pytest.mark.parametrize(
    "task_id,expect",
    [
        ("mkt_ht_011", True),
        ("mkt_dev_008", True),
        ("mkt_ht_015", True),
        ("mkt_ht_014", True),
        ("mkt_dev_013", True),
        ("mkt_dev_012", False),
        ("mkt_ht_013", False),
        ("mkt_dev_011", False),
        ("mkt_ht_020", False),
        ("mkt_ht_017", False),
        ("mkt_dev_014", False),
    ],
)
def test_baseline_fixture_ids_as_unit_tests_only(task_id: str, expect: bool):
    """Fixture IDs are allowed in tests only — never in production scoring branches."""
    row = _load_pred(task_id)
    out = grade_task(row, row)
    assert out["auto_grade_status"] == "graded"
    assert out["answer_correct"] is expect
    # Guard: production grader source must not special-case these IDs.
    src = (
        ROOT / "src" / "hk_equity_mcp" / "experiment" / "unified_grading.py"
    ).read_text(encoding="utf-8")
    assert f'task_id == "{task_id}"' not in src
    assert f"task_id == '{task_id}'" not in src


def test_components_are_separate_fields():
    item = {
        "task_id": "t_sep",
        "answerability_class": "structured_answerable",
        "gold_status": "answered",
        "gold_values": [{"value": 1.0, "unit": "HKD", "tolerance_abs": 0.01}],
    }
    out = grade_task(
        item,
        {
            "task_id": "t_sep",
            "predicted_status": "answered",
            "final_answer": "Price 1.00 HKD",
            "exact_tool_set": True,
        },
    )
    assert out["answer_correct"] is True
    assert "evidence_correct" in out
    assert out["overall_task_success"] is None  # never synthesized
    assert out["tool_selection_correct"] is True
