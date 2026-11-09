"""Offline replay of incomplete Smoke-4 v8 artifacts (APV-Fin v1.4.1)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from hk_equity_mcp.apv_fin.llm_budget import LlmCallBudgetExhausted
from hk_equity_mcp.apv_fin.runtime_safety import (
    structured_llm_budget_failure,
    summarize_prediction_outcomes,
)

ROOT = Path(__file__).resolve().parents[1]
V8 = ROOT / "outputs" / "apv_fin_real_smoke4_v8"
V8_PRED = V8 / "predictions.jsonl"
V8_META = V8 / "run_metadata.json"
V8_INVALIDATED = V8 / "INVALIDATED.md"
EXPECTED_SHA = "52fd2dd864aad5b8805a8c79bcae891f9945433cd1a0754c7f9a9c3a66cff087"
EXPECTED_IDS = ["fin_cf_003", "fin_cf_023", "xd_dev_002", "mkt_dev_014"]


def test_v8_invalidated_marker_and_raw_hashes():
    assert V8_INVALIDATED.exists()
    text = V8_INVALIDATED.read_text(encoding="utf-8")
    assert "invalidated_incomplete_run_due_to_uncaught_llm_budget_exception" in text
    assert V8_PRED.exists()
    sha = hashlib.sha256(V8_PRED.read_bytes()).hexdigest()
    assert sha == EXPECTED_SHA
    rows = [
        json.loads(l)
        for l in V8_PRED.read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    assert len(rows) == 3
    meta = json.loads(V8_META.read_text(encoding="utf-8"))
    assert meta.get("run_status") == "running"
    assert [r["task_id"] for r in rows] == EXPECTED_IDS[:3]


def test_replay_task4_budget_failure_becomes_fourth_prediction():
    """Simulate v8 crash: 3 written + binding repair at used=10 → 4th failed row."""
    rows = [
        json.loads(l)
        for l in V8_PRED.read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    assert len(rows) == 3
    preds = {r["task_id"]: {"status": r.get("status")} for r in rows}
    exc = LlmCallBudgetExhausted(used=10, limit=10, stage="tool_binding_repair")
    fourth = structured_llm_budget_failure(exc)
    preds[EXPECTED_IDS[3]] = {
        "status": fourth["status"],
        "error_type": fourth["error_type"],
        "stopping_reason": fourth["stopping_reason"],
    }
    summary = summarize_prediction_outcomes(
        expected_ids=EXPECTED_IDS, predictions_by_id=preds
    )
    assert summary["processed_task_count"] == 4
    assert summary["all_predictions_written"] is True
    assert summary["failed_count"] >= 1
    # Batch finalization policy: all predictions → completed (task failure ≠ run failure).
    run_status = "completed" if summary["all_predictions_written"] else "aborted"
    assert run_status == "completed"
    assert fourth["error_type"] == "resource_limit_failure"
    assert fourth["stopping_reason"] == "llm_call_budget_exhausted"
    assert "tool_binding_repair" in fourth["error"]


def test_v8_fin_cf_003_had_zero_dependency_expansion():
    """Document the live-path defect that v1.4.1 disclosure prefix fixes."""
    row = json.loads(V8_PRED.read_text(encoding="utf-8").splitlines()[0])
    assert row["task_id"] == "fin_cf_003"
    assert int(row.get("dependency_auto_expansion_count") or 0) == 0
    step = (row.get("plan_state") or {}).get("steps", {}).get("s1_retrieve") or {}
    assert step.get("required_capability") == "financial.official_narrative_evidence"


def test_v8_exception_replan_exceeded_recorded():
    row = json.loads(V8_PRED.read_text(encoding="utf-8").splitlines()[0])
    # Pre-fix artifact may show used>limit; v1.4.1 forbids that going forward.
    assert int(row.get("exception_replan_calls") or 0) == 3
    assert row.get("stopping_reason") == "max_exception_replan_calls_per_task"
