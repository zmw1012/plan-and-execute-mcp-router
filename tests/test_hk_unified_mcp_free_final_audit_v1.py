"""Offline tests for MCP-Free final audit (no paid APIs / no network)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hk_equity_mcp.experiment.unified_final_audit import run_final_audit
from hk_equity_mcp.experiment.unified_mcp_free_final_audit import (
    EXPECTED_HOST,
    EXPECTED_MODEL,
    EXPECTED_PROVIDER,
    EXPECTED_QUESTIONS_HASH,
    TOOL_NA,
    assert_mcp_free_run_identity,
    run_mcp_free_final_audit,
    sha256_file,
    try_parse_mcp_free_content,
)
from hk_equity_mcp.experiment.unified_mcp_free_prompt import unified_mcp_free_prompt_hash

ROOT = Path(__file__).resolve().parents[1]
LIVE = ROOT / "outputs" / "hk_unified_exploratory155_mcp_free_live_v1"
OV_LIVE = ROOT / "outputs" / "hk_unified_exploratory155_operation_visible_v1"
RAW_FILES = (
    "predictions.jsonl",
    "run_metadata.json",
    "per_task_metrics.csv",
    "manual_grading.csv",
    "cost_summary.json",
    "error_summary.json",
    "grouped_aggregates.json",
)


@pytest.fixture(scope="module")
def require_mcp_free_live():
    if not (LIVE / "predictions.jsonl").exists():
        pytest.skip("MCP-Free live run directory missing")


def test_mcp_free_allows_null_catalog_fingerprint(require_mcp_free_live):
    meta = json.loads((LIVE / "run_metadata.json").read_text(encoding="utf-8"))
    assert meta["condition"] == "mcp_free"
    assert meta["catalog_fingerprint"] is None
    assert_mcp_free_run_identity(meta)


def test_operation_visible_still_requires_catalog_fingerprint(
    require_mcp_free_live, tmp_path: Path
):
    """OV audit must refuse MCP-Free run (null catalog) — frozen OV semantics."""
    with pytest.raises(ValueError, match="Missing catalog_fingerprint"):
        run_final_audit(
            LIVE,
            baseline_git_commit="test",
            audit_outdir=tmp_path / "ov_should_fail",
        )


def test_ov_baseline_still_auditable_when_present(tmp_path: Path):
    if not (OV_LIVE / "predictions.jsonl").exists():
        pytest.skip("OV baseline run missing")
    meta = json.loads((OV_LIVE / "run_metadata.json").read_text(encoding="utf-8"))
    assert meta.get("catalog_fingerprint")
    # Smoke: OV audit entrypoint still works (writes under tmp only).
    result = run_final_audit(
        OV_LIVE,
        baseline_git_commit="39bd31b651deb1d064942ee5c6f5e9068946ac15",
        audit_outdir=tmp_path / "ov_ok",
    )
    assert result["raw_sha256"]["predictions.jsonl"]


def test_tool_metrics_na_not_vacuous_one(require_mcp_free_live):
    rows = [
        json.loads(line)
        for line in (LIVE / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for row in rows:
        for key in (
            "operation_precision",
            "operation_recall",
            "operation_f1",
            "exact_tool_set",
            "tool_execution_success",
            "all_tool_invocations_successful",
        ):
            assert row.get(key) in (TOOL_NA, "N/A", None) or str(row.get(key)).upper() == "N/A"
            assert row.get(key) not in (True, 1, 1.0)
        assert int(row.get("embedding_calls") or 0) == 0
        assert int(row.get("real_mcp_execution_count") or 0) == 0


def test_provider_host_model_locked(require_mcp_free_live):
    meta = json.loads((LIVE / "run_metadata.json").read_text(encoding="utf-8"))
    assert meta["generation_provider"] == EXPECTED_PROVIDER
    assert (meta.get("provider_host") or meta.get("generation_base_url_host")) == EXPECTED_HOST
    assert meta["generation_model"] == EXPECTED_MODEL
    assert meta["questions_identity_hash"] == EXPECTED_QUESTIONS_HASH
    assert meta["system_prompt_hash"] == unified_mcp_free_prompt_hash()


def test_generic_parser_no_task_id_special_case():
    ok = try_parse_mcp_free_content(
        '{"status":"unavailable","final_answer":"x","reasoning_brief":"y"}'
    )
    assert ok["ok"] is True
    assert ok["method"] == "json.loads"
    truncated = (
        '{"status":"answered","final_answer":"Revenue is 1.",'
        '"reasoning_brief":"This is the sum of Alibaba\u2019'
    )
    rec = try_parse_mcp_free_content(truncated)
    assert rec["ok"] is True
    assert rec["method"] == "truncated_completed_fields"
    assert rec["predicted_status"] == "answered"
    assert "Revenue is 1." in rec["final_answer"]
    empty = try_parse_mcp_free_content("")
    assert empty["ok"] is False


def test_mcp_free_audit_preserves_raw_hashes_and_no_rerun(
    require_mcp_free_live, tmp_path: Path
):
    before = {name: sha256_file(LIVE / name) for name in RAW_FILES}
    out = tmp_path / "audit"
    result = run_mcp_free_final_audit(
        LIVE, audit_outdir=out, docs_mirror=tmp_path / "docs_mirror"
    )
    after = {name: sha256_file(LIVE / name) for name in RAW_FILES}
    assert before == after
    assert result["raw_unchanged"] is True
    assert result["raw_sha256"] == before
    hashes = json.loads((out / "raw_artifact_hashes.json").read_text(encoding="utf-8"))
    assert hashes["unchanged"] is True
    assert hashes["before"] == before
    summary = json.loads((out / "audit_summary.json").read_text(encoding="utf-8"))
    assert summary["no_llm_embedding_mcp"] is True
    assert summary["tasks_rerun"] is False
    assert summary["raw_predictions_modified"] is False
    assert summary["catalog_fingerprint"] is None
    assert summary["metrics"]["tool_metrics"] == TOOL_NA
    assert summary["metrics"]["embedding_calls"] == 0
    assert summary["metrics"]["n_tasks"] == 155
    assert summary["parse_error_task_id"] == "fin_ht_030"
    # Parse audit must not imply LLM repair.
    parse = summary["parse_error_audit"]
    assert parse["task_id"] == "fin_ht_030"
    assert "llm" not in str(parse.get("recovery_method") or "").lower()


def test_human_grade_confirmation_updates_final_accuracy(
    require_mcp_free_live, tmp_path: Path
):
    before = {name: sha256_file(LIVE / name) for name in RAW_FILES}
    grades = tmp_path / "human_grades.json"
    grades.write_text(
        json.dumps(
            {
                "by_task_id": {
                    "fin_cf_022": {
                        "human_confirmed_answer_correct": 1,
                        "human_confirmed_overall_task_success": 1,
                        "human_notes": "Correct; FY2026 covers 1 Apr 2025–31 Mar 2026.",
                    }
                }
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    out = tmp_path / "audit_human"
    result = run_mcp_free_final_audit(
        LIVE,
        audit_outdir=out,
        docs_mirror=tmp_path / "docs_human",
        human_grades_file=grades,
    )
    after = {name: sha256_file(LIVE / name) for name in RAW_FILES}
    assert before == after
    metrics = result["metrics"]
    assert metrics["correct"] == 20
    assert metrics["overall_accuracy"] == pytest.approx(20 / 155)
    assert metrics["auto_graded_correct"] == 19
    assert metrics["auto_gradable_accuracy"] == pytest.approx(19 / 154)
    assert metrics["final_accuracy_by_domain"]["financial"]["correct"] == 8
    assert metrics["final_accuracy_by_domain"]["financial"]["n"] == 60
    assert metrics["final_accuracy_by_difficulty"]["D2"]["correct"] == 2
    assert metrics["final_accuracy_by_difficulty"]["D2"]["n"] == 24
    assert metrics["final_accuracy_by_origin"]["development"]["correct"] == 10
    assert metrics["final_accuracy_by_origin"]["development"]["n"] == 70
    # Parse failure remains incorrect / evaluation failure.
    assert result["parse_error_task_id"] == "fin_ht_030"
    queue = (out / "manual_review_queue.csv").read_text(encoding="utf-8")
    assert "fin_cf_022" in queue and ",1,1," in queue.replace("\r", "")
    assert "parse_evaluation_failure" in queue
