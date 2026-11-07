"""Offline MCP-Zero final-audit determinism, human-grade seal, and raw-SHA protection."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from hk_equity_mcp.experiment.mcp_zero_final_audit import (
    COMMON_STRICT_ANSWER_SUCCESS_FORMULA,
    EXPECTED_METHOD_HASH,
    EXPECTED_PROMPT_HASH,
    STRICT_OVERALL_FORMULA,
    compute_common_strict_answer_success,
    compute_strict_overall_task_success,
    fingerprint_raw_artifacts,
    load_human_grades,
    run_mcp_zero_final_audit,
    sha256_file,
)
from hk_equity_mcp.mcp_zero import method_identity_sha256
from hk_equity_mcp.mcp_zero.prompt import mcp_zero_prompt_hash

ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "outputs" / "hk_unified_mcp_zero_exploratory155_v2"
KNOWN_RAW_SHA = {
    "predictions.jsonl": "f35f30aa712df505025f39d3b4d5043da2786439531ace9a7f5e6df92d353d1a",
    "run_metadata.json": "be398592cfa295feaa94acda89e08e499ce19f09fdc22782f6762b938b5d5705",
    "per_task_metrics.csv": "9a1ecce074e868dd85743fb24dfe298c5766e2885fe6a048735057f24aed443e",
    "manual_grading.csv": "762253da705007357bcab92fd0934a050454b594d02730132b3538f21a8958eb",
    "cost_summary.json": "59c9057f3fd83102c07183d60bdecfdc8dc50bb784933d412a8051473ead3389",
    "error_summary.json": "3f9ec4d805a7b62e33b2ee67d1ff057e4b0e001a0841837443a3938cae183ea1",
    "retrieval_analysis.json": "aa9082cbd6f6696566ffbf6f72e4e17e2879172feb946fe65574d072feacffc1",
    "grouped_aggregates.json": "6c65a45929e44f04d02461ba46eb98138b205fb37b540c40d0a57050f278ec50",
    "grading_summary.json": "c3dd494f9179280d9b4d6d0df6550e74194084fe9b5b191f9d65a06495bed77f",
}


@pytest.fixture(scope="module")
def require_run():
    if not (RUN_DIR / "predictions.jsonl").exists():
        pytest.skip("MCP-Zero exploratory155_v2 run directory missing")


@pytest.fixture
def sealed_human_grades(tmp_path: Path) -> Path:
    path = tmp_path / "human_grades.json"
    path.write_text(
        json.dumps(
            {
                "condition": "mcp_zero",
                "grader_version": "deterministic_v1.2",
                "grading_version": "hk_unified_grading_v1.2",
                "sealed": True,
                "tasks": {
                    "fin_cf_022": {
                        "answer_correct": True,
                        "evidence_correct": True,
                        "overall_task_success": True,
                        "notes": "pytest sealed grade",
                        "graded_by": "pytest",
                        "graded_at": "2026-11-07T00:00:00Z",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def test_prompt_and_method_hashes_match_formal_identity():
    assert mcp_zero_prompt_hash() == EXPECTED_PROMPT_HASH
    assert method_identity_sha256() == EXPECTED_METHOD_HASH


def test_raw_sha_stable_across_two_audits(tmp_path: Path, require_run):
    before = fingerprint_raw_artifacts(RUN_DIR)
    out1 = tmp_path / "a"
    out2 = tmp_path / "b"
    r1 = run_mcp_zero_final_audit(
        RUN_DIR, root=ROOT, audit_outdir=out1, docs_mirror_dir=None
    )
    r2 = run_mcp_zero_final_audit(
        RUN_DIR, root=ROOT, audit_outdir=out2, docs_mirror_dir=None
    )
    after = fingerprint_raw_artifacts(RUN_DIR)
    assert before == after
    assert r1["raw_sha256"] == {k: v["sha256"] for k, v in before.items()}
    assert r1["raw_unchanged"] is True
    assert r1["ok"] is True and r2["ok"] is True
    fp1 = json.loads((out1 / "audit_run_result.json").read_text(encoding="utf-8"))[
        "content_fingerprint"
    ]
    fp2 = json.loads((out2 / "audit_run_result.json").read_text(encoding="utf-8"))[
        "content_fingerprint"
    ]
    assert fp1 == fp2
    for name in (
        "strict_metrics.json",
        "grading_summary.json",
        "mcp_zero_error_taxonomy_summary.json",
        "tool_retrieval_protocol_metrics.json",
        "manual_review_queue.csv",
        "per_task_audit.csv",
    ):
        assert hashlib.sha256((out1 / name).read_bytes()).hexdigest() == hashlib.sha256(
            (out2 / name).read_bytes()
        ).hexdigest()


def test_failed_forced_incorrect_and_manual_queue(tmp_path: Path, require_run):
    # No docs mirror / human path → unsealed (docs sealed grades not auto-loaded).
    out = tmp_path / "audit"
    result = run_mcp_zero_final_audit(
        RUN_DIR, root=ROOT, audit_outdir=out, docs_mirror_dir=None
    )
    assert result["manual_required"] == ["fin_cf_022"]
    assert result.get("human_grades_applied") == []
    queue = (out / "manual_review_queue.csv").read_text(encoding="utf-8")
    assert "fin_cf_022" in queue
    assert "human_confirmed_answer_correct" in queue
    grades = json.loads((out / "human_grades.json").read_text(encoding="utf-8"))
    assert grades["tasks"]["fin_cf_022"]["answer_correct"] is None
    strict = json.loads((out / "strict_metrics.json").read_text(encoding="utf-8"))
    assert strict["n"] == 155
    assert strict["strict_answer_correct"] == 89
    assert strict["final_answer_accuracy"] == pytest.approx(89 / 155)
    assert result["failed_n"] == 25
    audit_csv = (out / "per_task_audit.csv").read_text(encoding="utf-8")
    assert "mcp_zero_primary_error" in audit_csv
    assert "common_strict_answer_success" in audit_csv


def test_predictions_sha_matches_known_pre_audit_value(require_run):
    assert sha256_file(RUN_DIR / "predictions.jsonl") == KNOWN_RAW_SHA["predictions.jsonl"]


def test_raw_artifact_hashes_match_e0fbf8a_audit(require_run):
    now = fingerprint_raw_artifacts(RUN_DIR)
    for name, expected in KNOWN_RAW_SHA.items():
        assert now[name]["sha256"] == expected, name


def test_human_grade_loading(sealed_human_grades: Path):
    filled, src = load_human_grades(sealed_human_grades)
    assert src == sealed_human_grades
    assert filled["fin_cf_022"]["answer_correct"] is True
    empty = sealed_human_grades.parent / "blank.json"
    empty.write_text(
        json.dumps({"tasks": {"fin_cf_022": {"answer_correct": None}}}),
        encoding="utf-8",
    )
    filled2, src2 = load_human_grades(empty)
    assert filled2 == {}
    assert src2 is None


def test_metric_formula_definitions_and_helpers():
    assert "tool_selection_correct" in STRICT_OVERALL_FORMULA
    assert "all_tool_invocations_successful" in STRICT_OVERALL_FORMULA
    assert "status == 'full'" in COMMON_STRICT_ANSWER_SUCCESS_FORMULA
    assert compute_strict_overall_task_success(
        status="full",
        answer_correct=True,
        tool_selection_correct=True,
        all_tool_invocations_successful=True,
    )
    assert not compute_strict_overall_task_success(
        status="full",
        answer_correct=True,
        tool_selection_correct=False,
        all_tool_invocations_successful=True,
    )
    assert compute_common_strict_answer_success(status="full", answer_correct=True)
    assert not compute_common_strict_answer_success(status="failed", answer_correct=True)


def test_sealed_human_grades_yield_90_of_155(
    tmp_path: Path, require_run, sealed_human_grades: Path
):
    out = tmp_path / "sealed_audit"
    result = run_mcp_zero_final_audit(
        RUN_DIR,
        root=ROOT,
        audit_outdir=out,
        docs_mirror_dir=None,
        human_grades_path=sealed_human_grades,
    )
    strict = json.loads((out / "strict_metrics.json").read_text(encoding="utf-8"))
    assert strict["final_answer_correct"] == 90
    assert strict["final_answer_accuracy"] == pytest.approx(90 / 155)
    assert strict["strict_answer_correct"] == 90
    assert strict["strict_answer_accuracy"] == pytest.approx(90 / 155)
    assert strict["common_strict_answer_success_n"] == 90
    assert strict["common_strict_answer_success"] == pytest.approx(90 / 155)
    # fin_cf_022 is not exact_tool_set → MCP-Zero strict_overall stays 57
    assert strict["strict_overall_task_success_n"] == 57
    assert strict["strict_overall_task_success"] == pytest.approx(57 / 155)
    assert result["human_grades_applied"] == ["fin_cf_022"]
    grades = json.loads((out / "human_grades.json").read_text(encoding="utf-8"))
    assert grades["tasks"]["fin_cf_022"]["answer_correct"] is True
    queue = (out / "manual_review_queue.csv").read_text(encoding="utf-8")
    assert ",1,1,1," in queue or queue.splitlines()[1].count(",1") >= 3
    defs = json.loads((out / "metric_definitions.json").read_text(encoding="utf-8"))
    assert defs["strict_overall_task_success"]["comparability_label"] == (
        "condition-specific"
    )
    assert defs["common_strict_answer_success"]["comparability_label"] == (
        "cross-baseline-comparable"
    )
    comp = json.loads(
        (out / "cross_condition_comparability.json").read_text(encoding="utf-8")
    )
    assert comp["strict_overall_comparability_label"] == "condition-specific"
    assert comp["common_metric_for_cross_baseline"] == "common_strict_answer_success"
    grouped = json.loads(
        (out / "grouped_strict_metrics.json").read_text(encoding="utf-8")
    )
    assert grouped["by_domain"]["financial"]["strict_answer_accuracy"] == pytest.approx(
        24 / 60
    )
    assert grouped["by_difficulty"]["D2"]["strict_answer_accuracy"] == pytest.approx(
        16 / 24
    )
    assert grouped["by_previously_reserved"]["development_origin"][
        "strict_answer_accuracy"
    ] == pytest.approx(38 / 70)
    # Raw unchanged after sealed audit
    after = fingerprint_raw_artifacts(RUN_DIR)
    for name, expected in KNOWN_RAW_SHA.items():
        assert after[name]["sha256"] == expected
