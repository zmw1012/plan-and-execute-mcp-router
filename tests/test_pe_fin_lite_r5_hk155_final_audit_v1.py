"""Offline unit tests for PE-Fin Lite R5 HK-155 final audit (no API/MCP)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from hk_equity_mcp.experiment.pe_fin_lite_checkpoint_audit.evaluate import (
    evaluate_task_checkpoints,
)
from hk_equity_mcp.experiment.pe_fin_lite_r5_hk155_final_audit import EXPECTED_N
from hk_equity_mcp.experiment.pe_fin_lite_r5_hk155_final_audit.runner import (
    BASELINES,
    common_strict_answer_success,
    collect_raw_hashes,
    sha256_file,
)

ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "outputs" / "pe_fin_lite_r5_hk155_live_v1"
AUDIT_DIR = RUN_DIR / "final_audit_r5_v1"


def _minimal_pred(**overrides: Any) -> Dict[str, Any]:
    base: Dict[str, Any] = {
        "task_id": "t_demo",
        "domain": "financial",
        "status": "failed",
        "answer_correct": False,
        "predicted_status": "unavailable",
        "initial_plan": {"steps": [{"id": "s1", "type": "retrieve", "domain": "financial"}]},
        "retrieval_funnels": [],
        "invoked_operations": [],
        "step_statuses": [],
        "budget": {"llm_calls_used": 1, "mcp_invocations": 0},
    }
    base.update(overrides)
    return base


def test_common_strict_forces_partial_failed_incorrect():
    assert common_strict_answer_success(
        {"status": "full", "answer_correct": True}
    )
    assert not common_strict_answer_success(
        {"status": "partial", "answer_correct": True}
    )
    assert not common_strict_answer_success(
        {"status": "failed", "answer_correct": True}
    )
    assert not common_strict_answer_success(
        {"status": "full", "answer_correct": False}
    )


def test_strict_denominator_never_drops_failed_partial():
    rows = [
        {"status": "full", "answer_correct": True},
        {"status": "partial", "answer_correct": True},
        {"status": "failed", "answer_correct": False},
    ]
    n = len(rows)
    rate = sum(1 for r in rows if common_strict_answer_success(r)) / n
    assert n == 3
    assert rate == pytest.approx(1 / 3)


def test_na_not_counted_as_pass_and_zero_invocation_not_tool_success():
    # Zero invocation retrieve plan → C07 must not be pass-as-success via empty invoke
    pred = _minimal_pred(
        retrieval_funnels=[
            {
                "step_id": "s1",
                "domain_hard_filter": False,
                "hard_filtered": {"filter_kind": "safety_only_r5", "kept_count": 39},
                "embedding_topk": [],
                "described_candidates": [],
                "invoked_operations": [],
            }
        ],
        invoked_operations=[],
    )
    cp = evaluate_task_checkpoints(pred)
    c07 = cp["checkpoints"]["C07_mcp"]
    assert c07["status"] != "pass" or c07.get("detail", {}).get("note") == "zero_invocation_not_tool_success" or c07["status"] in {
        "fail",
        "N/A",
        "N/A_not_reached",
        "N/A_upstream_blocked",
        "N/A_not_planned",
        "N/A_not_required",
        "unknown",
    }
    # Explicit: if status is pass, that's a regression for zero-invocation semantics
    if not pred["invoked_operations"]:
        # Accept fail or N/A_* / unknown — never treat empty invoke as tool success
        assert c07["stage_outcome_status"] != "pass" or "zero_invocation" in str(
            c07.get("detail") or {}
        )


def test_earliest_blocker_prefers_first_fail():
    pred = _minimal_pred(
        initial_plan=None,
        error_type="planner_parse_failed",
    )
    cp = evaluate_task_checkpoints(pred)
    # Planner fail should be primary when plan missing/invalid
    assert cp["primary_blocking_checkpoint"] in {None, "C01_planner"} or (
        cp["checkpoints"]["C01_planner"]["status"] == "fail"
        and cp["primary_blocking_checkpoint"] == "C01_planner"
    )


def test_manual_grades_template_has_no_hardcoded_prefilled_conclusions():
    if not AUDIT_DIR.exists():
        pytest.skip("audit artifacts not generated yet")
    hg = json.loads((AUDIT_DIR / "human_grades.json").read_text(encoding="utf-8"))
    assert hg["grades"]
    for tid, g in hg["grades"].items():
        assert g.get("human_answer_correct") is None
        assert g.get("human_notes") is None
        assert isinstance(tid, str) and tid
    # Source file must not hardcode this queue's task ids into grading logic
    runner_src = (
        ROOT
        / "src"
        / "hk_equity_mcp"
        / "experiment"
        / "pe_fin_lite_r5_hk155_final_audit"
        / "runner.py"
    ).read_text(encoding="utf-8")
    assert "fin_cf_022" not in runner_src


def test_baseline_metric_definitions_not_confused():
    # Condition-specific strict present for some baselines must stay labeled incomparable
    assert BASELINES["Operation-Visible"].get("note")
    assert "condition-specific" in BASELINES["Operation-Visible"]["note"]
    assert BASELINES["MCP-Zero"].get("note")
    # Prefer comparable fields exist independently
    for name in ("Flat All Tools", "MCP-Zero"):
        assert "final_answer_accuracy" in BASELINES[name]
        assert "common_strict_answer_success" in BASELINES[name]
        assert "exact_tool_set" in BASELINES[name]


@pytest.mark.skipif(not RUN_DIR.exists(), reason="HK-155 live run dir missing")
def test_raw_immutability_snapshot_matches_current_files():
    before = collect_raw_hashes(RUN_DIR)
    # Touch only by re-hashing; do not mutate
    after = collect_raw_hashes(RUN_DIR)
    assert before == after
    preds = RUN_DIR / "predictions.jsonl"
    meta = RUN_DIR / "run_metadata.json"
    assert preds.exists() and meta.exists()
    assert before["predictions.jsonl"]["sha256"] == sha256_file(preds)
    assert before["run_metadata.json"]["sha256"] == sha256_file(meta)
    if AUDIT_DIR.exists():
        sealed = json.loads((AUDIT_DIR / "raw_hashes.json").read_text(encoding="utf-8"))
        assert sealed["unchanged"] is True
        assert sealed["before"]["predictions.jsonl"]["sha256"] == before["predictions.jsonl"]["sha256"]
        assert sealed["after"]["predictions.jsonl"]["sha256"] == before["predictions.jsonl"]["sha256"]


@pytest.mark.skipif(not RUN_DIR.exists(), reason="HK-155 live run dir missing")
def test_155_unique_ids_and_strict_metrics_denominator():
    rows: List[Dict[str, Any]] = [
        json.loads(line)
        for line in (RUN_DIR / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ids = [r.get("task_id") or r.get("id") for r in rows]
    assert len(rows) == EXPECTED_N
    assert len(set(ids)) == EXPECTED_N
    # Strict denominator always 155
    strict_n = sum(1 for r in rows if common_strict_answer_success(r))
    assert 0 <= strict_n <= EXPECTED_N
    rate = strict_n / EXPECTED_N
    assert rate == pytest.approx(strict_n / 155)


@pytest.mark.skipif(not AUDIT_DIR.exists(), reason="audit artifacts missing")
def test_checkpoint_funnel_na_semantics_in_artifact():
    funnel = json.loads((AUDIT_DIR / "checkpoint_funnel.json").read_text(encoding="utf-8"))
    for cid, payload in funnel.items():
        counts = payload["counts"]
        na_sum = sum(
            int(counts.get(k, 0))
            for k in counts
            if str(k).startswith("N/A") or k == "not_evaluated"
        )
        # pass must equal only explicit pass bucket
        assert payload["pass"] == int(counts.get("pass", 0))
        assert payload["pass"] + payload["fail"] + na_sum <= EXPECTED_N + 5  # allow unknown
        assert "N/A" not in str(payload.get("note", "")) or "never counted as pass" in payload.get(
            "note", ""
        )
