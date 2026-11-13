"""Offline tests for pe_fin_replan_etom_run_finalization_v2 (no live API)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import patch

import pytest

from hk_equity_mcp.pe_fin_replan_etom.run_finalization import (
    RUN_FINALIZATION_CONTRACT_VERSION,
    decide_run_status,
    engineering_smoke_gate,
    prediction_accounting,
    run_finalization_contract_hash,
    should_skip_on_resume,
)

ROOT = Path(__file__).resolve().parents[1]
SMOKE_V6 = ROOT / "outputs" / "pe_fin_replan_etom_smoke5_v6"
AUDIT = SMOKE_V6 / "OFFLINE_ENGINEERING_AUDIT_v1"

V6_RAW = {
    "predictions.jsonl": (
        "e2c464027001187d1df16039e0dade635fd594ca2658c7b5429ef9746140a0f7",
        76612,
        5,
    ),
    "run_metadata.json": (
        "f0d835d6e9c5ecb27fc7269700ea56ee214439ccc898cdf23d04df781a1ed109",
        127910,
        None,
    ),
    "error_summary.json": (
        "c2f970878482044c6b0f06e3d56c1942def8f3ddeb15800430b51b4ece73d179",
        345,
        None,
    ),
}


def _row(tid: str, status: str, **extra: Any) -> Dict[str, Any]:
    out = {
        "task_id": tid,
        "status": status,
        "exact_match": False,
        "mcp_invocations": 0,
        "routing_record_count": 0,
        "real_tool_executions": 0,
        "error_type": extra.pop("error_type", None),
        "stopping_reason": extra.pop("stopping_reason", None),
    }
    out.update(extra)
    return out


def test_01_contract_version_stable():
    assert RUN_FINALIZATION_CONTRACT_VERSION == "pe_fin_replan_etom_run_finalization_v2"
    assert len(run_finalization_contract_hash()) == 64


def test_02_five_rows_three_completed_two_failed_is_run_completed():
    ids = [f"L{i}-q{i}" for i in range(1, 6)]
    rows = [
        _row(ids[0], "completed"),
        _row(ids[1], "failed", error_type="replanner_grounding_failure"),
        _row(ids[2], "completed"),
        _row(ids[3], "failed", error_type="replanner_grounding_failure"),
        _row(ids[4], "completed"),
    ]
    assert decide_run_status(selected_task_ids=ids, rows=rows) == "completed"
    acct = prediction_accounting(selected_task_ids=ids, rows=rows)
    assert acct["processed_task_count"] == 5
    assert acct["completed_prediction_count"] == 3
    assert acct["failed_prediction_count"] == 2
    assert acct["missing_task_ids"] == []
    assert acct["all_selected_tasks_processed"] is True


def test_03_four_of_five_rows_is_partial():
    ids = [f"T{i}" for i in range(5)]
    rows = [_row(ids[i], "completed") for i in range(4)]
    assert decide_run_status(selected_task_ids=ids, rows=rows) == "partial"
    acct = prediction_accounting(selected_task_ids=ids, rows=rows)
    assert acct["missing_task_ids"] == [ids[4]]
    assert acct["all_selected_tasks_processed"] is False


def test_04_aborted_flags():
    ids = ["a", "b"]
    rows = [_row("a", "completed")]
    assert (
        decide_run_status(selected_task_ids=ids, rows=rows, aborted=True) == "aborted"
    )
    assert (
        decide_run_status(
            selected_task_ids=ids, rows=rows, aborted=False, finalization_ok=False
        )
        == "aborted"
    )


def test_05_controlled_grounding_failure_row_failed_run_completed():
    ids = ["L2-q766", "L4-q57", "L1-q556", "L3-q108", "L5-q68"]
    rows = [
        _row(
            "L2-q766",
            "failed",
            error_type="replanner_grounding_failure",
            stopping_reason="ungrounded_replanner_response",
        ),
        _row(
            "L4-q57",
            "failed",
            error_type="replanner_grounding_failure",
            stopping_reason="ungrounded_replanner_response",
        ),
        _row("L1-q556", "completed"),
        _row("L3-q108", "completed"),
        _row("L5-q68", "completed"),
    ]
    assert decide_run_status(selected_task_ids=ids, rows=rows) == "completed"
    gate = engineering_smoke_gate(
        accounting=prediction_accounting(selected_task_ids=ids, rows=rows),
        run_status="completed",
        routing_record_count_total=11,
        mcp_invocations_total=11,
        real_tool_executions_total=0,
        live_pass_claimed=False,
        artifacts_complete=True,
        runner_crashed=False,
    )
    assert gate["engineering_smoke_valid"] is True
    assert gate["requires_all_tasks_status_completed"] is False


def test_06_resume_skips_controlled_failed_row():
    assert should_skip_on_resume(_row("L2-q766", "failed")) is True
    assert should_skip_on_resume(_row("L1-q556", "completed")) is True
    assert should_skip_on_resume(None) is False
    assert should_skip_on_resume({"task_id": "x", "status": "running"}) is False


def test_07_formal100_fixture_100_rows_with_failures_completed():
    ids = [f"L{(i % 5) + 1}-q{i}" for i in range(100)]
    rows: List[Dict[str, Any]] = []
    for i, tid in enumerate(ids):
        status = "failed" if i % 7 == 0 else "completed"
        rows.append(_row(tid, status))
    assert len(ids) == 100
    assert decide_run_status(selected_task_ids=ids, rows=rows) == "completed"
    acct = prediction_accounting(selected_task_ids=ids, rows=rows)
    assert acct["processed_task_count"] == 100
    assert acct["failed_prediction_count"] > 0
    assert acct["all_selected_tasks_processed"] is True


def test_08_smoke_v6_raw_unchanged_and_offline_audit():
    for name, (want, nbytes, rows) in V6_RAW.items():
        path = SMOKE_V6 / name
        data = path.read_bytes()
        assert len(data) == nbytes
        assert hashlib.sha256(data).hexdigest() == want
        if rows is not None:
            assert sum(1 for line in data.decode().splitlines() if line.strip()) == rows
    text = (AUDIT / "AUDIT.md").read_text(encoding="utf-8")
    assert "valid_engineering_smoke_with_controlled_task_failures" in text
    assert "offline_corrected_engineering_run_status" in text
    assert "`completed`" in text
    assert "No raw mutation" in text or "no_raw_mutation" in text
    summary = json.loads((AUDIT / "audit_summary.json").read_text(encoding="utf-8"))
    assert summary["marker"] == "valid_engineering_smoke_with_controlled_task_failures"
    assert summary["method_exact_match"] == "2/5"
    assert summary["task_failures"] == 2
    assert summary["raw_run_status"] == "partial"
    assert summary["offline_corrected_engineering_run_status"] == "completed"
    assert summary["smoke_v7_required"] is False
    # Raw still records old partial semantics; do not rewrite.
    meta = json.loads((SMOKE_V6 / "run_metadata.json").read_text(encoding="utf-8"))
    assert meta["run_status"] == "partial"


def _raw_available() -> bool:
    return (ROOT / "data" / "etom" / "raw" / "data" / "mcp_registry.json").exists()


@pytest.mark.skipif(not _raw_available(), reason="pinned ETOM raw checkout not present")
def test_09_runner_mock_smoke_completed_with_mixed_outcomes(tmp_path):
    from hk_equity_mcp.experiment.unified_pe_fin_replan_etom_runner import (
        UnifiedPeFinReplanEtomRunner,
    )

    out = tmp_path / "mock_smoke"
    runner = UnifiedPeFinReplanEtomRunner(
        ROOT, out, mode="mock", engineering_smoke5=True, formal100=False
    )
    # Force two structured failed prediction rows after agent returns by patching
    # only the row status mapping path: inject post-agent failures via _run_one wrap.
    real_run_one = runner._run_one

    def _wrap(*args, **kwargs):
        row = real_run_one(*args, **kwargs)
        tid = row["task_id"]
        if tid in {"L2-q766", "L4-q57"}:
            row = dict(row)
            row["status"] = "failed"
            row["error_type"] = "replanner_grounding_failure"
            row["stopping_reason"] = "ungrounded_replanner_response"
            row["finalized_by_replanner"] = False
            row["exact_match"] = False
        return row

    with patch.object(runner, "_run_one", side_effect=_wrap):
        compact = runner.execute()
    assert compact["run_status"] == "completed"
    assert compact["live_pass_claimed"] is False
    assert compact["completed"] == 3
    assert compact["failed"] == 2
    assert compact.get("engineering_smoke_valid") is True
    meta = json.loads((out / "run_metadata.json").read_text(encoding="utf-8"))
    assert meta["run_status"] == "completed"
    assert meta["processed_task_count"] == 5
    assert meta["failed_prediction_count"] == 2
    assert meta["all_selected_tasks_processed"] is True
    assert meta["run_finalization_contract_version"] == RUN_FINALIZATION_CONTRACT_VERSION
    assert set(meta["failed_task_ids"]) == {"L2-q766", "L4-q57"}
    # resume must not re-append controlled failed rows
    n0 = len((out / "predictions.jsonl").read_text().splitlines())
    resumed = UnifiedPeFinReplanEtomRunner(
        ROOT, out, mode="mock", engineering_smoke5=True, formal100=False, resume=True
    )
    resumed.execute()
    n1 = len((out / "predictions.jsonl").read_text().splitlines())
    assert n1 == n0 == 5


@pytest.mark.skipif(not _raw_available(), reason="pinned ETOM raw checkout not present")
def test_10_keyboardinterrupt_aborts_and_reraises(tmp_path):
    from hk_equity_mcp.experiment.unified_pe_fin_replan_etom_runner import (
        UnifiedPeFinReplanEtomRunner,
    )

    out = tmp_path / "abort"
    runner = UnifiedPeFinReplanEtomRunner(
        ROOT, out, mode="mock", engineering_smoke5=True, formal100=False
    )
    real_run_one = runner._run_one
    calls = {"n": 0}

    def _boom(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise KeyboardInterrupt()
        return real_run_one(*args, **kwargs)

    with patch.object(runner, "_run_one", side_effect=_boom):
        with pytest.raises(KeyboardInterrupt):
            runner.execute()
    meta = json.loads((out / "run_metadata.json").read_text(encoding="utf-8"))
    assert meta["run_status"] == "aborted"
    assert meta["live_pass_claimed"] is False
    assert meta["processed_task_count"] >= 1


@pytest.mark.skipif(not _raw_available(), reason="pinned ETOM raw checkout not present")
def test_11_unknown_runner_error_aborts_and_reraises(tmp_path):
    from hk_equity_mcp.experiment.unified_pe_fin_replan_etom_runner import (
        UnifiedPeFinReplanEtomRunner,
    )

    out = tmp_path / "err"
    runner = UnifiedPeFinReplanEtomRunner(
        ROOT, out, mode="mock", engineering_smoke5=True, formal100=False
    )

    def _boom(*_a, **_k):
        raise RuntimeError("unexpected_runner_fault")

    with patch.object(runner, "_run_one", side_effect=_boom):
        with pytest.raises(RuntimeError, match="unexpected_runner_fault"):
            runner.execute()
    meta = json.loads((out / "run_metadata.json").read_text(encoding="utf-8"))
    assert meta["run_status"] == "aborted"
    assert meta["live_pass_claimed"] is False


@pytest.mark.skipif(not _raw_available(), reason="pinned ETOM raw checkout not present")
def test_12_routing_only_invariants_on_mock(tmp_path):
    from hk_equity_mcp.experiment.unified_pe_fin_replan_etom_runner import (
        UnifiedPeFinReplanEtomRunner,
    )

    out = tmp_path / "inv"
    compact = UnifiedPeFinReplanEtomRunner(
        ROOT, out, mode="mock", engineering_smoke5=True, formal100=False
    ).execute()
    assert compact["real_tool_executions"] == 0
    assert compact["routing_records"] == compact["mcp_invocations"]
    assert compact["live_pass_claimed"] is False
    assert compact["run_status"] == "completed"
