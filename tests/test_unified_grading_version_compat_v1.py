"""Regression: shared grader v1.2.1 vs historical MCP-Free v1.2 semantics.

Frozen MCP-Free HK-155 audits remain pinned to ``hk_unified_grading_v1.2``.
The integrated release ships ``hk_unified_grading_v1.2.1`` (grounded-numeric
refusal heuristic + status_failed hard-fail). Regrading sealed predictions
under the wrong version changes refusal metrics; historical
``grading_version=`` must reproduce the sealed outcome without rewriting
current identity strings or frozen audit metadata.
"""

from __future__ import annotations

from hk_equity_mcp.experiment.unified_grading import (
    GRADING_VERSION,
    GRADING_VERSION_V1_2,
    GRADER_VERSION,
    GRADER_VERSION_V1_2,
    HISTORICAL_GRADING_VERSIONS,
    grade_task,
    resolve_grading_identity,
)


def test_current_shared_grader_reports_v1_2_1():
    assert GRADING_VERSION == "hk_unified_grading_v1.2.1"
    assert GRADER_VERSION == "deterministic_v1.2.1"
    ident = resolve_grading_identity()
    assert ident["grading_version"] == GRADING_VERSION
    assert ident["grader_version"] == GRADER_VERSION


def test_historical_registry_pins_mcp_free_to_v1_2():
    assert HISTORICAL_GRADING_VERSIONS["mcp_free_hk155"] == GRADING_VERSION_V1_2
    hist = resolve_grading_identity(GRADING_VERSION_V1_2)
    assert hist == {
        "grader_version": GRADER_VERSION_V1_2,
        "grading_version": GRADING_VERSION_V1_2,
    }


def test_grounded_numeric_refusal_differs_between_v1_2_and_v1_2_1():
    """Numbers already present in the question are grounded in v1.2.1 only."""
    item = {
        "task_id": "synthetic_refusal_grounding",
        "question": (
            "Using the 2026-08-07 close for Alibaba (09988), what changed?"
        ),
        "answerability_class": "robustness",
        "gold_status": "answered",
        "gold_values": [
            {
                "expected_behavior": "refuse_post_cutoff",
                "cutoff": "2026-08-12",
            }
        ],
    }
    row = {
        "task_id": "synthetic_refusal_grounding",
        "status": "full",
        "predicted_status": "unavailable",
        "question": item["question"],
        "final_answer": (
            "Unavailable: I cannot verify the 2026-08-07 close for Alibaba (09988), "
            "so I cannot use it as formal gold."
        ),
    }
    hist = grade_task(item, row, grading_version=GRADING_VERSION_V1_2)
    cur = grade_task(item, row)
    assert hist["grading_version"] == GRADING_VERSION_V1_2
    assert cur["grading_version"] == GRADING_VERSION
    # Historical v1.2 treats any money/share/price cue + number as fabricated.
    assert hist["answer_correct"] is False
    assert hist["reason"] == "refusal_with_fabricated_numeric"
    # Current v1.2.1 excludes question-grounded numbers → refusal_status_ok.
    assert cur["answer_correct"] is True
    assert cur["reason"] == "refusal_status_ok"


def test_status_failed_hard_fail_is_v1_2_1_only():
    item = {
        "task_id": "synthetic_failed",
        "question": "Q",
        "gold_values": [{"kind": "number", "value": 1.0, "tolerance": 0.0}],
    }
    row = {
        "task_id": "synthetic_failed",
        "status": "failed",
        "final_answer": "1.0",
        "exact_tool_set": True,
    }
    hist = grade_task(item, row, grading_version=GRADING_VERSION_V1_2)
    cur = grade_task(item, row)
    assert cur["reason"] == "status_failed"
    assert cur["answer_correct"] is False
    # v1.2 did not apply the status_failed gate; may grade the numeric answer.
    assert hist["reason"] != "status_failed"


def test_default_grade_task_does_not_alias_to_historical_string():
    item = {
        "task_id": "t",
        "gold_values": [{"kind": "number", "value": 2.0, "tolerance": 0.0}],
    }
    row = {"task_id": "t", "status": "ok", "final_answer": "2"}
    out = grade_task(item, row)
    assert out["grading_version"] == "hk_unified_grading_v1.2.1"
    assert out["grader"] == "deterministic_v1.2.1"
