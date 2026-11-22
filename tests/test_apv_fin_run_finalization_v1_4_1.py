"""Runner finalization + task isolation (APV-Fin v1.4.1)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from hk_equity_mcp.apv_fin.llm_budget import LlmCallBudgetExhausted
from hk_equity_mcp.apv_fin.runtime_safety import (
    runner_finalization_policy_hash,
    structured_llm_budget_failure,
    summarize_prediction_outcomes,
)
from hk_equity_mcp.experiment.unified_apv_fin_runner import UnifiedHkApvFinRunner
from hk_equity_mcp.experiment.unified_runner import atomic_write_json


def test_runner_finalization_policy_hash():
    assert runner_finalization_policy_hash()


def test_summarize_prediction_outcomes_partial_and_complete():
    expected = ["t1", "t2", "t3", "t4"]
    preds = {
        "t1": {"status": "full"},
        "t2": {"status": "failed"},
        "t3": {"status": "failed"},
    }
    out = summarize_prediction_outcomes(expected_ids=expected, predictions_by_id=preds)
    assert out["processed_task_count"] == 3
    assert out["full_count"] == 1
    assert out["failed_count"] == 2
    assert out["missing_ids"] == ["t4"]
    assert out["all_predictions_written"] is False

    preds["t4"] = {"status": "failed", "error_type": "resource_limit_failure"}
    out2 = summarize_prediction_outcomes(expected_ids=expected, predictions_by_id=preds)
    assert out2["processed_task_count"] == 4
    assert out2["all_predictions_written"] is True
    assert out2["failed_count"] == 3


def test_batch_continues_after_budget_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """3 ok + 1 resource failure → 4 predictions, run_status=completed."""
    items = [
        {"task_id": f"t{i}", "question": f"q{i}", "domain": "financial"}
        for i in range(1, 5)
    ]

    class _FakeAgent:
        def __init__(self):
            self.calls = 0
            self.llm = self

        def reset_script(self):
            return None

        def run(self, question: str) -> Dict[str, Any]:
            self.calls += 1
            if self.calls == 4:
                raise LlmCallBudgetExhausted(
                    used=10, limit=10, stage="tool_binding_repair"
                )
            return {
                "status": "full",
                "final_answer": f"ans-{self.calls}",
                "predicted_status": "answered",
                "error": None,
                "error_type": None,
                "llm_calls": 1,
                "tokens": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                "trace": [],
                "budget": {},
            }

    runner = UnifiedHkApvFinRunner.__new__(UnifiedHkApvFinRunner)
    runner.output_dir = tmp_path
    runner.mock = True
    runner.dry_run = False
    runner.resume = False
    runner._gateway = None
    runner.condition = "apv_fin"
    runner.method_version = "apv_fin_v1_4_1_crash_safe_runtime"

    rows_built: List[Dict[str, Any]] = []

    def _build(item, result):
        row = {
            "task_id": item["task_id"],
            "status": result.get("status"),
            "error_type": result.get("error_type"),
            "stopping_reason": result.get("stopping_reason"),
            "final_answer": result.get("final_answer"),
        }
        rows_built.append(row)
        return row

    def _side(**kwargs):
        return None

    runner._build_prediction_row = _build  # type: ignore[method-assign]
    runner._write_side_artifacts = _side  # type: ignore[method-assign]
    runner.build_agent = lambda: _FakeAgent()  # type: ignore[method-assign]
    runner.close = lambda: None  # type: ignore[method-assign]

    # Minimal prepare stub via direct run body pieces — call a thin helper.
    meta_path = tmp_path / "run_metadata.json"
    predictions_path = tmp_path / "predictions.jsonl"
    run_meta: Dict[str, Any] = {
        "run_status": "mock_running",
        "expected_task_count": 4,
        "condition": "apv_fin",
        "method_version": runner.method_version,
    }
    atomic_write_json(meta_path, run_meta)

    # Exercise isolation logic by invoking the same control flow as run().
    from datetime import datetime, timezone

    from hk_equity_mcp.apv_fin.runtime_safety import summarize_prediction_outcomes
    from hk_equity_mcp.experiment.unified_runner import append_jsonl, fsync_file

    agent = runner.build_agent()
    predictions_by_id: Dict[str, Dict[str, Any]] = {}
    for index, item in enumerate(items, start=1):
        qid = item["task_id"]
        try:
            result = agent.run(item["question"])
        except LlmCallBudgetExhausted as exc:
            result = structured_llm_budget_failure(exc)
        row = runner._build_prediction_row(item, result)
        append_jsonl(predictions_path, row)
        predictions_by_id[qid] = row
        run_meta.update(
            summarize_prediction_outcomes(
                expected_ids=[t["task_id"] for t in items],
                predictions_by_id=predictions_by_id,
            )
        )
        atomic_write_json(meta_path, run_meta)

    fsync_file(predictions_path)
    run_meta["run_status"] = "completed"
    run_meta["completed_at"] = datetime.now(timezone.utc).isoformat()
    atomic_write_json(meta_path, run_meta)

    lines = [
        json.loads(l)
        for l in predictions_path.read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    assert len(lines) == 4
    assert lines[3]["status"] == "failed"
    assert lines[3]["error_type"] == "resource_limit_failure"
    assert lines[3]["stopping_reason"] == "llm_call_budget_exhausted"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["run_status"] == "completed"
    assert meta["processed_task_count"] == 4
    assert meta["failed_count"] == 1
    assert meta["full_count"] == 3


def test_unknown_exception_marks_aborted_and_preserves_predictions(
    tmp_path: Path,
):
    items = [
        {"task_id": "t1", "question": "q1"},
        {"task_id": "t2", "question": "q2"},
    ]

    class _FakeAgent:
        def __init__(self):
            self.calls = 0
            self.llm = self

        def reset_script(self):
            return None

        def run(self, question: str):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("unexpected_programming_bug")
            return {
                "status": "full",
                "final_answer": "ok",
                "predicted_status": "answered",
                "trace": [],
                "tokens": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            }

    from hk_equity_mcp.experiment.unified_runner import append_jsonl, fsync_file

    predictions_path = tmp_path / "predictions.jsonl"
    meta_path = tmp_path / "run_metadata.json"
    run_meta: Dict[str, Any] = {"run_status": "running"}
    predictions_by_id: Dict[str, Dict[str, Any]] = {}
    agent = _FakeAgent()
    abort_exc = None
    for item in items:
        qid = item["task_id"]
        try:
            result = agent.run(item["question"])
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:  # noqa: BLE001
            abort_exc = exc
            run_meta["run_status"] = "aborted"
            run_meta["abort_reason"] = "internal_error"
            run_meta["abort_error_type"] = type(exc).__name__
            break
        row = {"task_id": qid, "status": result["status"]}
        append_jsonl(predictions_path, row)
        predictions_by_id[qid] = row

    assert abort_exc is not None
    assert predictions_path.exists()
    lines = [
        json.loads(l)
        for l in predictions_path.read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    assert len(lines) == 1
    assert lines[0]["task_id"] == "t1"
    fsync_file(predictions_path)
    run_meta.update(
        summarize_prediction_outcomes(
            expected_ids=[t["task_id"] for t in items],
            predictions_by_id=predictions_by_id,
        )
    )
    atomic_write_json(meta_path, run_meta)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["run_status"] == "aborted"
    assert meta["processed_task_count"] == 1
