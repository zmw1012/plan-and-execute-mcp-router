"""Offline evaluation/runtime hardening tests (no paid APIs / no Smoke re-run)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hk_equity_mcp.experiment.unified_evaluator import (
    compute_tool_execution_metrics,
    evaluate_task,
    extract_invoke_records,
    summarize_predictions,
)
from hk_equity_mcp.experiment.unified_postprocess import postprocess_run_dir
from hk_equity_mcp.experiment.unified_prompt import (
    UNIFIED_HK_OPERATION_VISIBLE_SYSTEM_PROMPT,
    unified_prompt_hash,
)
from hk_equity_mcp.real_mcp.command_resolve import (
    candidate_mcp_commands,
    resolve_mcp_executable,
)
from hk_equity_mcp.real_mcp.unified_catalog import build_unified_catalog

ROOT = Path(__file__).resolve().parents[1]
FROZEN_PROMPT_HASH = (
    "9674844a98c9f438ae02d19572f173a7cdc27e1c365ed27dd39ea7e0963eec1b"
)


def _invoke(oid: str, success: bool, *, protocol: bool | None = None) -> dict:
    nested = {
        "success": success,
        "protocol_success": protocol if protocol is not None else success,
        "execution_success": success,
    }
    return {
        "type": "tool_result",
        "name": "invoke_api",
        "arguments": {"operation_id": oid, "arguments": {}},
        "result": {
            "success": success,
            "error": None if success else "boom",
            "error_type": None if success else "underlying_tool_error",
            "result": nested,
        },
    }


def test_status_full_not_counted_as_error():
    rows = [
        {"task_id": "a", "status": "full", "error": None, "operation_f1": 1.0},
        {"task_id": "b", "status": "full", "error": "", "operation_f1": 0.5},
    ]
    summary = summarize_predictions(rows)
    assert summary["error_counts"] == {}
    assert summary["result_status_counts"] == {"full": 2}
    assert "full" not in summary["error_counts"]


def test_nonempty_error_counted():
    rows = [
        {"task_id": "a", "status": "failed", "error": "TimeoutError"},
        {"task_id": "b", "status": "full", "error": None},
        {"task_id": "c", "status": "failed", "error_type": "RuntimeError"},
    ]
    summary = summarize_predictions(rows)
    assert summary["error_counts"]["TimeoutError"] == 1
    assert summary["error_counts"]["RuntimeError"] == 1
    assert summary["result_status_counts"]["full"] == 1
    assert summary["result_status_counts"]["failed"] == 2


def test_recovered_after_tool_error_true():
    result = {
        "status": "full",
        "predicted_status": "answered",
        "final_answer": "ok",
        "trace": [_invoke("op.a", False), _invoke("op.b", True)],
        "tokens": {},
    }
    metrics = evaluate_task(item={"gold_operations": ["op.b"]}, result=result)
    assert metrics["recovered_after_tool_error"] is True
    assert metrics["all_tool_invocations_successful"] is False
    assert metrics["tool_execution_success"] is False
    assert metrics["successful_invocation_count"] == 1
    assert metrics["failed_invocation_count"] == 1
    assert metrics["final_operation_success_by_operation_id"] == {
        "op.a": False,
        "op.b": True,
    }


def test_all_invocations_successful_true():
    result = {
        "status": "full",
        "predicted_status": "answered",
        "final_answer": "ok",
        "trace": [_invoke("op.a", True), _invoke("op.b", True)],
        "tokens": {},
    }
    metrics = evaluate_task(item={"gold_operations": ["op.a", "op.b"]}, result=result)
    assert metrics["all_tool_invocations_successful"] is True
    assert metrics["tool_execution_success"] is True
    assert metrics["recovered_after_tool_error"] is False
    assert metrics["successful_invocation_count"] == 2
    assert metrics["failed_invocation_count"] == 0


def test_refusal_without_invoke_not_execution_failure():
    result = {
        "status": "full",
        "predicted_status": "unavailable",
        "final_answer": "cannot answer",
        "trace": [],
        "tokens": {},
    }
    metrics = evaluate_task(
        item={
            "gold_operations": [],
            "answerability_class": "intentionally_infeasible",
            "gold_status": "unavailable",
        },
        result=result,
    )
    assert metrics["all_tool_invocations_successful"] is True
    assert metrics["tool_execution_success"] is True
    assert metrics["successful_invocation_count"] == 0
    assert metrics["failed_invocation_count"] == 0
    assert metrics["recovered_after_tool_error"] is False
    assert metrics["refusal_correctness"] == 1.0


def test_sibling_mcp_executable_resolution(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("AKTOOLS_MCP_COMMAND", raising=False)
    # Fake sibling layout under an isolated root.
    root = tmp_path / "hk-equity-mcp-agent-hk-unified"
    sibling = tmp_path / "hk-equity-mcp-agent"
    bin_dir = sibling / ".venv-aktools-discovery" / "bin"
    bin_dir.mkdir(parents=True)
    exe = bin_dir / "aktools-mcp"
    exe.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    exe.chmod(0o755)

    resolved = resolve_mcp_executable(
        env_var="AKTOOLS_MCP_COMMAND",
        binary_name="aktools-mcp",
        venv_dirname=".venv-aktools-discovery",
        root=root,
        require=True,
    )
    assert resolved == str(exe)


def test_explicit_env_command_preferred(tmp_path: Path, monkeypatch):
    root = tmp_path / "hk-equity-mcp-agent-hk-unified"
    sibling = tmp_path / "hk-equity-mcp-agent"
    (sibling / ".venv-aktools-discovery" / "bin").mkdir(parents=True)
    sibling_exe = sibling / ".venv-aktools-discovery" / "bin" / "aktools-mcp"
    sibling_exe.write_text("#!/bin/sh\n", encoding="utf-8")
    sibling_exe.chmod(0o755)

    env_exe = tmp_path / "custom-aktools-mcp"
    env_exe.write_text("#!/bin/sh\n", encoding="utf-8")
    env_exe.chmod(0o755)
    monkeypatch.setenv("AKTOOLS_MCP_COMMAND", str(env_exe))

    resolved = resolve_mcp_executable(
        env_var="AKTOOLS_MCP_COMMAND",
        binary_name="aktools-mcp",
        venv_dirname=".venv-aktools-discovery",
        root=root,
        require=True,
    )
    assert resolved == str(env_exe)

    cands = candidate_mcp_commands(
        env_var="AKTOOLS_MCP_COMMAND",
        binary_name="aktools-mcp",
        venv_dirname=".venv-aktools-discovery",
        root=root,
    )
    assert cands[0] == str(env_exe)


def test_prompt_catalog_dataset_hashes_unchanged():
    assert unified_prompt_hash() == FROZEN_PROMPT_HASH
    assert "gold_operations" not in UNIFIED_HK_OPERATION_VISIBLE_SYSTEM_PROMPT
    bundle = build_unified_catalog(root=ROOT)
    assert bundle.stats.unique_operation_ids == 42
    assert bundle.stats.namespaced_entries == 54
    assert bundle.stats.runtime_exposed_unique == 39
    # Frozen dataset pointers still present and sized.
    from hk_equity_mcp.experiment.unified_dataset import load_split_tasks

    assert len(load_split_tasks(ROOT, "development")) == 70
    assert len(load_split_tasks(ROOT, "heldout")) == 85


def test_offline_postprocess_smoke_dir(tmp_path: Path):
    src = ROOT / "outputs" / "hk_unified_real_smoke8_v1"
    if not (src / "predictions.jsonl").exists():
        pytest.skip("real smoke8 predictions not present in this worktree")

    # Copy only predictions + metadata (do not mutate original outputs in-place
    # during unit test; postprocess of the real dir is a separate manual step).
    import shutil

    dst = tmp_path / "smoke_copy"
    dst.mkdir()
    shutil.copy2(src / "predictions.jsonl", dst / "predictions.jsonl")
    if (src / "run_metadata.json").exists():
        shutil.copy2(src / "run_metadata.json", dst / "run_metadata.json")
    before_pred = (dst / "predictions.jsonl").read_bytes()
    before_meta = (
        (dst / "run_metadata.json").read_bytes()
        if (dst / "run_metadata.json").exists()
        else None
    )

    result = postprocess_run_dir(dst)
    assert result["n_tasks"] == 8
    assert result["error_counts"] == {}
    assert result["result_status_counts"].get("full") == 8
    assert (dst / "predictions.jsonl").read_bytes() == before_pred
    if before_meta is not None:
        assert (dst / "run_metadata.json").read_bytes() == before_meta
    assert (dst / "evaluation_metadata.json").exists()
    assert (dst / "error_summary.json").exists()
    err = json.loads((dst / "error_summary.json").read_text(encoding="utf-8"))
    assert "full" not in err["error_counts"]
    assert err["result_status_counts"]["full"] == 8

    # recovered flag should light up for tasks that failed then succeeded.
    metrics_csv = (dst / "per_task_metrics.csv").read_text(encoding="utf-8")
    assert "recovered_after_tool_error" in metrics_csv.splitlines()[0]
    assert "all_tool_invocations_successful" in metrics_csv.splitlines()[0]


def test_compute_tool_metrics_protocol_rate():
    records = extract_invoke_records(
        [_invoke("a", False, protocol=False), _invoke("b", True, protocol=True)]
    )
    out = compute_tool_execution_metrics(
        records, result={"status": "full", "final_answer": "x"}
    )
    assert out["protocol_success_rate"] == 0.5
    assert out["recovered_after_tool_error"] is True
