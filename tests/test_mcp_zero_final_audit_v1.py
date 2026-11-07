"""Offline MCP-Zero final-audit determinism and raw-SHA protection."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from hk_equity_mcp.experiment.mcp_zero_final_audit import (
    EXPECTED_METHOD_HASH,
    EXPECTED_PROMPT_HASH,
    fingerprint_raw_artifacts,
    run_mcp_zero_final_audit,
    sha256_file,
)
from hk_equity_mcp.mcp_zero import method_identity_sha256
from hk_equity_mcp.mcp_zero.prompt import mcp_zero_prompt_hash

ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "outputs" / "hk_unified_mcp_zero_exploratory155_v2"


@pytest.fixture(scope="module")
def require_run():
    if not (RUN_DIR / "predictions.jsonl").exists():
        pytest.skip("MCP-Zero exploratory155_v2 run directory missing")


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
    # Core metric artifacts deterministic across reruns.
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
    out = tmp_path / "audit"
    result = run_mcp_zero_final_audit(
        RUN_DIR, root=ROOT, audit_outdir=out, docs_mirror_dir=None
    )
    assert result["manual_required"] == ["fin_cf_022"]
    queue = (out / "manual_review_queue.csv").read_text(encoding="utf-8")
    assert "fin_cf_022" in queue
    assert "human_confirmed_answer_correct" in queue
    grades = json.loads((out / "human_grades.json").read_text(encoding="utf-8"))
    assert grades["tasks"]["fin_cf_022"]["answer_correct"] is None
    strict = json.loads((out / "strict_metrics.json").read_text(encoding="utf-8"))
    assert strict["n"] == 155
    # Failed tasks cannot contribute to strict correct count beyond auto.
    assert strict["strict_answer_accuracy"] <= strict["auto_graded_accuracy"] + 1e-12
    audit_csv = (out / "per_task_audit.csv").read_text(encoding="utf-8")
    assert "mcp_zero_primary_error" in audit_csv
    assert result["failed_n"] == 25


def test_predictions_sha_matches_known_pre_audit_value(require_run):
    # Guard against accidental mutation of the formal run.
    assert sha256_file(RUN_DIR / "predictions.jsonl") == (
        "f35f30aa712df505025f39d3b4d5043da2786439531ace9a7f5e6df92d353d1a"
    )
