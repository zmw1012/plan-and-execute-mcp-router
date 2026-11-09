"""Offline tests for APV-Fin Diagnostic-12 final audit v1 (no APIs)."""

from __future__ import annotations

import json
from pathlib import Path

from hk_equity_mcp.experiment.apv_fin_diagnostic12_final_audit_v1 import (
    EXPECTED_COMMIT,
    EXPECTED_METHOD,
    RAW_ARTIFACTS,
    descriptive_auto_correct,
    file_stats,
    run_final_audit,
    strict_answer_correct,
)

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "outputs" / "apv_fin_diagnostic12_live_v1_4_1"


def test_raw_artifacts_present_and_hashable():
    assert RUN.exists()
    for name in RAW_ARTIFACTS:
        st = file_stats(RUN / name)
        assert st["bytes"] > 0
        assert len(st["sha256"]) == 64


def test_strict_vs_descriptive_rules():
    failed_match = {
        "status": "failed",
        "answer_correct": True,
        "error_type": None,
        "stopping_reason": None,
    }
    assert descriptive_auto_correct(failed_match) is True
    assert strict_answer_correct(failed_match) is False
    full_ok = {
        "status": "full",
        "answer_correct": True,
        "error_type": None,
        "stopping_reason": "monitor_refuse",
    }
    # monitor_refuse full+correct can be strict true under rule (status full + answer_correct).
    assert strict_answer_correct(full_ok) is True
    resource = {
        "status": "failed",
        "answer_correct": True,
        "error_type": "resource_limit_failure",
        "stopping_reason": "max_exception_replan_calls_per_task",
    }
    assert strict_answer_correct(resource) is False


def test_run_final_audit_offline(tmp_path: Path):
    out = tmp_path / "final_audit_v1"
    docs = tmp_path / "docs_mirror"
    before = {n: file_stats(RUN / n)["sha256"] for n in RAW_ARTIFACTS}
    result = run_final_audit(root=ROOT, run_dir=RUN, out_dir=out, docs_dir=docs)
    after = {n: file_stats(RUN / n)["sha256"] for n in RAW_ARTIFACTS}
    assert before == after
    assert result["raw_hashes_unchanged"] is True
    assert result["identity_all_passed"] is True
    summary = json.loads((out / "audit_summary.json").read_text(encoding="utf-8"))
    assert summary["method_version"] == EXPECTED_METHOD
    assert summary["git_commit"] == EXPECTED_COMMIT
    sm = summary["strict_metrics"]
    assert sm["strict_correct"] == 0
    assert sm["descriptive_auto_correct"] == 1
    assert sm["full"] == 1
    assert sm["failed"] == 11
    assert sm["resource_limit"] == 8
    assert sm["exact_tool_set"] == 3
    assert abs(sm["operation_f1_mean"] - 0.4488095) < 1e-9
    for name in (
        "audit_summary.json",
        "per_task_audit.csv",
        "error_taxonomy.json",
        "mechanism_funnel.json",
        "baseline_comparison.json",
        "resource_summary.json",
        "raw_hashes.json",
        "final_audit_report.md",
    ):
        assert (out / name).exists()
        assert (docs / name).exists()
    base = json.loads((out / "baseline_comparison.json").read_text(encoding="utf-8"))
    ver = base["expected_correct_verification"]
    assert ver["operation_visible"]["match"] is True
    assert ver["mcp_zero"]["match"] is True
    assert ver["flat_all_tools"]["match"] is True
    assert ver["mcp_free"]["match"] is True
    assert ver["apv_fin"]["match"] is True
