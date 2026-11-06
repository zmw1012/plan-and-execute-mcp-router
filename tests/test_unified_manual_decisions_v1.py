"""Tests for human manual-decision merge (fail-closed; offline)."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from hk_equity_mcp.experiment.unified_final_audit import run_final_audit, sha256_file
from hk_equity_mcp.experiment.unified_manual_decisions import (
    ManualDecisionError,
    compute_strict_task_success,
    load_manual_decisions,
    require_full_queue_coverage,
    summarize_final_metrics,
)

ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "outputs" / "hk_unified_exploratory155_operation_visible_v1"
DECISIONS = (
    ROOT
    / "docs"
    / "hk_unified_baseline_final_audit_v2"
    / "manual_review_decisions_v1.csv"
)
BASELINE_COMMIT = "39bd31b651deb1d064942ee5c6f5e9068946ac15"
QUEUE = [
    "mkt_dev_003",
    "xd_ht_003",
    "mkt_ht_004",
    "mkt_dev_007",
    "mkt_ht_006",
    "mkt_ht_008",
    "fin_cf_022",
    "mkt_ht_016",
]


@pytest.fixture(scope="module")
def require_baseline():
    if not (RUN_DIR / "predictions.jsonl").exists():
        pytest.skip("baseline run directory missing")
    if not DECISIONS.exists():
        pytest.skip("manual decisions file missing")


def test_load_manual_decisions_ok(require_baseline):
    decisions = load_manual_decisions(DECISIONS, allowed_task_ids=QUEUE)
    require_full_queue_coverage(decisions, QUEUE)
    assert len(decisions) == 8
    assert decisions["fin_cf_022"]["answer_correct"] is True
    assert decisions["fin_cf_022"]["evidence_na"] is True
    assert decisions["fin_cf_022"]["evidence_correct"] is None
    assert decisions["fin_cf_022"]["overall_task_success"] is False
    assert decisions["mkt_dev_003"]["overall_task_success"] is True


def test_duplicate_task_fail_closed(tmp_path: Path, require_baseline):
    src = DECISIONS.read_text(encoding="utf-8")
    bad = tmp_path / "dup.csv"
    # Duplicate first data row.
    lines = src.strip().splitlines()
    bad.write_text("\n".join(lines + [lines[1]]) + "\n", encoding="utf-8")
    with pytest.raises(ManualDecisionError, match="duplicate"):
        load_manual_decisions(bad, allowed_task_ids=QUEUE)


def test_unknown_task_fail_closed(tmp_path: Path, require_baseline):
    bad = tmp_path / "unknown.csv"
    bad.write_text(
        "task_id,answer_correct,evidence_correct,overall_task_success,notes\n"
        "not_in_queue,1,,1,x\n",
        encoding="utf-8",
    )
    with pytest.raises(ManualDecisionError, match="unknown task_id"):
        load_manual_decisions(bad, allowed_task_ids=QUEUE)


def test_blank_evidence_is_na(require_baseline):
    decisions = load_manual_decisions(DECISIONS, allowed_task_ids=QUEUE)
    assert decisions["mkt_ht_008"]["evidence_na"] is True
    assert decisions["mkt_ht_008"]["evidence_correct"] is None


def test_strict_success_separated_from_answer():
    row = {
        "exact_tool_set": False,
        "all_tool_invocations_successful": True,
        "graph_dependency_accuracy": None,
        "refusal_correctness": None,
    }
    out = compute_strict_task_success(
        row, answer_correct=True, evidence_correct=None, evidence_na=True
    )
    assert out["strict_task_success"] is False
    assert "exact_tool_set" in out["strict_failed_checks"]

    row2 = dict(row)
    row2["exact_tool_set"] = True
    ok = compute_strict_task_success(
        row2, answer_correct=True, evidence_correct=None, evidence_na=True
    )
    assert ok["strict_task_success"] is True


def test_summarize_keeps_abc_separate():
    rows = [
        {
            "final_answer_correct": True,
            "exact_tool_set": False,
            "strict_task_success": False,
        },
        {
            "final_answer_correct": True,
            "exact_tool_set": True,
            "strict_task_success": True,
        },
        {
            "final_answer_correct": False,
            "exact_tool_set": True,
            "strict_task_success": False,
        },
    ]
    s = summarize_final_metrics(rows)
    assert s["final_answer_correct_count"] == 2
    assert s["final_answer_accuracy"] == pytest.approx(2 / 3)
    assert s["exact_tool_set_success_count"] == 2
    assert s["strict_task_success_count"] == 1
    assert s["final_answer_accuracy"] != s["strict_task_success_rate"]


def test_merge_audit_with_decisions_deterministic(tmp_path: Path, require_baseline):
    before_raw = {
        name: sha256_file(RUN_DIR / name)
        for name in (
            "predictions.jsonl",
            "run_metadata.json",
            "per_task_metrics.csv",
            "manual_grading.csv",
            "retrieval_analysis.json",
        )
    }
    before_decisions = sha256_file(DECISIONS)

    out1 = tmp_path / "a"
    out2 = tmp_path / "b"
    r1 = run_final_audit(
        RUN_DIR,
        baseline_git_commit=BASELINE_COMMIT,
        audit_outdir=out1,
        manual_decisions_path=DECISIONS,
    )
    r2 = run_final_audit(
        RUN_DIR,
        baseline_git_commit=BASELINE_COMMIT,
        audit_outdir=out2,
        manual_decisions_path=DECISIONS,
    )

    after_raw = {name: sha256_file(RUN_DIR / name) for name in before_raw}
    after_decisions = sha256_file(DECISIONS)
    assert before_raw == after_raw
    assert before_decisions == after_decisions
    assert r1["content_fingerprint"] == r2["content_fingerprint"]

    final = r1["final_grading_summary_v2"]
    assert final is not None
    # Programmatic expectation: 94 auto-correct + fin_cf_022 human = 95/155
    assert final["final_answer_correct_count"] == 95
    assert final["n"] == 155
    assert final["final_answer_accuracy"] == pytest.approx(95 / 155)
    assert final["strict_task_success_rate"] != final["final_answer_accuracy"]

    for name in (
        "final_grading_summary_v2.json",
        "final_grouped_aggregates_v2.json",
        "final_per_task_audit_v2.csv",
        "final_baseline_error_analysis_v2.md",
        "final_report_ready_baseline_tables_v2.csv",
    ):
        assert (out1 / name).exists()
        assert hashlib.sha256((out1 / name).read_bytes()).hexdigest() == hashlib.sha256(
            (out2 / name).read_bytes()
        ).hexdigest()

    # Audit must not write the decisions file into out_dir under the same name
    # as an overwrite target of the docs path.
    assert not (out1 / "manual_review_decisions_v1.csv").exists()

    # Taxonomy must distinguish answer_success vs strict_task_success.
    tax = json.loads((out1 / "error_taxonomy_summary_v1.json").read_text())
    assert "answer_success" in tax["primary_counts"]
    assert "strict_task_success" in tax["primary_counts"]
    assert tax["primary_counts"].get("evaluation_failure", 0) == 0

    # Financial domain includes fin_cf_022 human answer.
    grouped = json.loads((out1 / "final_grouped_aggregates_v2.json").read_text())
    fin = grouped["domain"]["financial"]
    assert fin["final_answer_correct_count"] >= 1


def test_incomplete_coverage_fail_closed(tmp_path: Path, require_baseline):
    partial = tmp_path / "partial.csv"
    with DECISIONS.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    with partial.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "task_id",
                "answer_correct",
                "evidence_correct",
                "overall_task_success",
                "notes",
            ],
        )
        writer.writeheader()
        writer.writerow(rows[0])
    decisions = load_manual_decisions(partial, allowed_task_ids=QUEUE)
    with pytest.raises(ManualDecisionError, match="incomplete"):
        require_full_queue_coverage(decisions, QUEUE)
