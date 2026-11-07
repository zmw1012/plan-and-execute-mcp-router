"""Offline tests for Flat All Tools final audit v1 (no network / paid APIs)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hk_equity_mcp.experiment.unified_flat_final_audit import (
    EXPECTED_MANUAL_TASKS,
    EXPECTED_PROMPT_HASH,
    EXPECTED_QUESTION_COUNT,
    KNOWN_RAW_SHA256,
    assert_known_raw_hashes,
    classify_flat_error,
    fingerprint_raw_artifacts,
    run_flat_final_audit,
    strict_answer_correct,
)
from hk_equity_mcp.experiment.unified_flat_metrics import RETRIEVAL_NA

ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "outputs" / "hk_unified_flat_all_tools_155_v1"


def _require_run() -> Path:
    if not (RUN_DIR / "predictions.jsonl").exists():
        pytest.skip(f"Flat 155 run missing: {RUN_DIR}")
    return RUN_DIR


def test_raw_hash_freeze_matches_known():
    run = _require_run()
    fps = fingerprint_raw_artifacts(run)
    assert_known_raw_hashes(fps)
    assert fps["predictions.jsonl"]["row_count"] == EXPECTED_QUESTION_COUNT
    assert fps["predictions.jsonl"]["unique_task_ids"] == EXPECTED_QUESTION_COUNT
    assert fps["predictions.jsonl"]["sha256"] == KNOWN_RAW_SHA256["predictions.jsonl"]
    assert fps["run_metadata.json"]["sha256"] == KNOWN_RAW_SHA256["run_metadata.json"]


def test_flat_retrieval_na_constant():
    assert RETRIEVAL_NA == "N/A"
    assert RETRIEVAL_NA != 1.0
    assert RETRIEVAL_NA != True  # noqa: E712


def test_classify_max_invocations_priority():
    row = {
        "status": "failed",
        "error": "max_invocations",
        "auto_grade_status": "graded",
        "answer_correct": False,
        "gold_operations": ["aktools.stock_hk_hist"],
        "selected_operations": ["ah_disclosure.resolve_company"],
        "invoked_operations": ["ah_disclosure.resolve_company"],
        "answerability_class": "structured_answerable",
        "predicted_status": "unavailable",
        "trace": [],
    }
    tax = classify_flat_error(row)
    assert tax["primary_error_category"] == "resource_limit_max_invocations"


def test_classify_max_llm_turns_priority():
    row = {
        "status": "failed",
        "error": "max_llm_turns",
        "auto_grade_status": "graded",
        "answer_correct": False,
        "gold_operations": [],
        "selected_operations": [],
        "invoked_operations": [],
        "answerability_class": "structured_answerable",
        "predicted_status": "unavailable",
        "trace": [],
    }
    tax = classify_flat_error(row)
    assert tax["primary_error_category"] == "resource_limit_max_llm_turns"


def test_partial_and_failed_strict_incorrect():
    failed = {
        "status": "failed",
        "error": "max_invocations",
        "auto_grade_status": "graded",
        "answer_correct": True,  # frozen refusal-on-limit edge case
    }
    assert strict_answer_correct(failed) is False
    partial = {
        "status": "partial",
        "error": None,
        "auto_grade_status": "graded",
        "answer_correct": False,
        "gold_operations": ["a"],
        "selected_operations": ["a"],
        "invoked_operations": ["a"],
        "answerability_class": "structured_answerable",
        "predicted_status": "unavailable",
        "trace": [],
    }
    assert strict_answer_correct(partial) is False
    tax = classify_flat_error(partial)
    assert tax["primary_error_category"] == "protocol_or_parse_failure"


def test_no_task_id_special_case_in_classifier_source():
    src = (
        ROOT
        / "src"
        / "hk_equity_mcp"
        / "experiment"
        / "unified_flat_final_audit.py"
    ).read_text(encoding="utf-8")
    # Classifier must not branch on concrete task ids.
    assert 'if tid == "fin_cf_' not in src
    assert "task_id ==" not in src.split("def classify_flat_error", 1)[1].split(
        "def strict_answer_correct", 1
    )[0]


def test_full_audit_deterministic_and_raw_unchanged(tmp_path: Path):
    run = _require_run()
    before = fingerprint_raw_artifacts(run)
    out1 = tmp_path / "audit_a"
    out2 = tmp_path / "audit_b"
    r1 = run_flat_final_audit(run, audit_outdir=out1, docs_mirror=None)
    r2 = run_flat_final_audit(run, audit_outdir=out2, docs_mirror=None)
    after = fingerprint_raw_artifacts(run)
    assert before == after
    assert r1["raw_unchanged"] is True
    assert r2["n_tasks"] == 155
    assert r1["manual_task_ids"] == list(EXPECTED_MANUAL_TASKS)
    assert r2["manual_task_ids"] == ["fin_cf_022"]

    m1 = json.loads((out1 / "final_metrics.json").read_text(encoding="utf-8"))
    m2 = json.loads((out2 / "final_metrics.json").read_text(encoding="utf-8"))
    # Drop timestamps-only fields via shared numeric identity.
    for key in (
        "auto_correct",
        "auto_gradable",
        "auto_accuracy",
        "conservative_lower_bound_accuracy",
        "upper_bound_if_manual_correct",
        "exact_tool_set_rate",
        "total_tokens",
        "total_llm_calls",
        "total_real_mcp_executions",
        "retrieval_metrics",
    ):
        assert m1[key] == m2[key]
    assert m1["retrieval_metrics"] == "N/A"
    assert m1["overall_accuracy"] is None
    assert m1["auto_correct"] == 87
    assert m1["auto_gradable"] == 154
    assert abs(m1["auto_accuracy"] - (87 / 154)) < 1e-12
    assert abs(m1["conservative_lower_bound_accuracy"] - (87 / 155)) < 1e-12
    assert abs(m1["upper_bound_if_manual_correct"] - (88 / 155)) < 1e-12
    assert m1["status_counts"]["full"] == 118
    assert m1["status_counts"]["failed"] == 36
    assert m1["status_counts"]["partial"] == 1
    assert m1["error_counts"]["max_invocations"] == 34
    assert m1["error_counts"]["max_llm_turns"] == 2
    assert m1["total_embedding_calls"] == 0
    assert m1["total_retrieval_calls"] == 0
    assert m1["system_prompt_hash"] == EXPECTED_PROMPT_HASH

    # Manual queue exactly one task; human fields empty.
    queue = (out1 / "manual_review_queue.csv").read_text(encoding="utf-8").splitlines()
    assert len(queue) == 2  # header + fin_cf_022
    assert "fin_cf_022" in queue[1]
    assert queue[1].endswith(",,,,") or ",,,," in queue[1]

    g = json.loads((out1 / "grouped_metrics.json").read_text(encoding="utf-8"))
    assert g["domain"]["financial"]["auto_graded_correct"] == 18
    assert g["domain"]["financial"]["auto_gradable_count"] == 59
    assert g["domain"]["market"]["auto_accuracy"] == pytest.approx(26 / 35)
    assert g["domain"]["news"]["auto_accuracy"] == pytest.approx(29 / 35)
    assert g["domain"]["cross_domain"]["auto_accuracy"] == pytest.approx(14 / 25)
    assert g["project_difficulty"]["D1"]["auto_accuracy"] == pytest.approx(23 / 35)
    assert g["answerability_class"]["should_refuse_or_clarify"]["auto_graded_correct"] == 0
    assert g["answerability_class"]["should_refuse_or_clarify"]["auto_gradable_count"] == 6
    assert g["answerability_class"]["evidence_answerable"]["auto_graded_correct"] == 3

    res = json.loads((out1 / "resource_analysis.json").read_text(encoding="utf-8"))
    assert res["total_input_tokens"] == 9231332
    assert res["total_output_tokens"] == 232792
    assert res["total_tokens"] == 9464124
    assert res["total_llm_calls"] == 460
    assert res["total_real_mcp_executions"] == 572
    assert res["estimated_usd"] is None
    assert res["schema_bundle_char_count"] == 33874
    assert res["schema_bundle_token_estimate"] == 8469

    tools = json.loads((out1 / "tool_selection_analysis.json").read_text(encoding="utf-8"))
    assert tools["retrieval_metrics"] == "N/A"
    assert abs(tools["exact_tool_set_rate"] - 0.23225806451612904) < 1e-12

    tax = json.loads((out1 / "error_taxonomy.json").read_text(encoding="utf-8"))
    assert tax["primary_counts"]["resource_limit_max_invocations"] == 34
    assert tax["primary_counts"]["resource_limit_max_llm_turns"] == 2


def test_official_audit_dir_if_present():
    run = _require_run()
    official = run / "final_audit_flat_v1"
    if not official.exists():
        pytest.skip("official audit dir not generated yet")
    m = json.loads((official / "final_metrics.json").read_text(encoding="utf-8"))
    assert m["n_tasks"] == 155
    assert m["manual_task_ids"] == ["fin_cf_022"]
    assert m["retrieval_metrics"] == "N/A"
    hashes = json.loads((official / "raw_artifact_hashes.json").read_text(encoding="utf-8"))
    assert hashes["unchanged"] is True
