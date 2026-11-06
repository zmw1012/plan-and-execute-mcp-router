"""Offline final-audit determinism and raw-SHA protection tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from hk_equity_mcp.experiment.unified_final_audit import (
    V11_MANUAL_REQUIRED_EXPECTED,
    run_final_audit,
    sha256_file,
)

ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "outputs" / "hk_unified_exploratory155_operation_visible_v1"
BASELINE_COMMIT = "39bd31b651deb1d064942ee5c6f5e9068946ac15"


@pytest.fixture(scope="module")
def require_baseline():
    if not (RUN_DIR / "predictions.jsonl").exists():
        pytest.skip("baseline run directory missing")


def test_raw_sha_stable_across_two_audits(tmp_path: Path, require_baseline):
    before = {
        name: sha256_file(RUN_DIR / name)
        for name in (
            "predictions.jsonl",
            "run_metadata.json",
            "per_task_metrics.csv",
            "manual_grading.csv",
            "retrieval_analysis.json",
        )
    }
    out1 = tmp_path / "audit_a"
    out2 = tmp_path / "audit_b"
    r1 = run_final_audit(
        RUN_DIR, baseline_git_commit=BASELINE_COMMIT, audit_outdir=out1
    )
    r2 = run_final_audit(
        RUN_DIR, baseline_git_commit=BASELINE_COMMIT, audit_outdir=out2
    )
    after = {
        name: sha256_file(RUN_DIR / name)
        for name in before
    }
    assert before == after
    assert r1["raw_sha256"] == before
    assert r2["content_fingerprint"] == r1["content_fingerprint"]

    # Core metric artifacts deterministic.
    for name in (
        "grading_summary_v2.json",
        "error_taxonomy_summary_v1.json",
        "retrieval_stage_summary_v1.json",
        "per_task_audit_v2.csv",
        "error_taxonomy_v1.csv",
        "report_ready_baseline_tables_v1.csv",
        "baseline_error_analysis_v1.md",
    ):
        assert hashlib.sha256((out1 / name).read_bytes()).hexdigest() == hashlib.sha256(
            (out2 / name).read_bytes()
        ).hexdigest()


def test_v11_manual_required_list_matches_frozen_csv(require_baseline):
    import csv

    path = RUN_DIR / "manual_grading.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    manual = sorted(
        r["task_id"] for r in rows if r.get("auto_grade_status") == "manual_required"
    )
    assert manual == sorted(V11_MANUAL_REQUIRED_EXPECTED)
    assert len(manual) == 8


def test_predictions_line_count_and_unique_ids(require_baseline):
    lines = [
        ln
        for ln in (RUN_DIR / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    assert len(lines) == 155
    ids = [json.loads(ln)["task_id"] for ln in lines]
    assert len(set(ids)) == 155
