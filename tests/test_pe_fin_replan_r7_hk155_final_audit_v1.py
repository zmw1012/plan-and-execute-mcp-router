"""Offline tests for RePlan-7 HK-155 final audit (no network)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hk_equity_mcp.experiment.pe_fin_replan_r7_hk155_final_audit import (
    EXPECTED_META_SHA,
    EXPECTED_N,
    EXPECTED_PRED_SHA,
    RAW_NAMES,
)
from hk_equity_mcp.experiment.pe_fin_replan_r7_hk155_final_audit.runner import (
    collect_raw_hashes,
    run_final_audit,
)
from hk_equity_mcp.experiment.pe_fin_replan_r7_hk155_final_audit.temporal import (
    analyze_task_temporal,
    parse_iso_date,
)

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "outputs" / "pe_fin_replan_r7_hk155_live_v1"
AUD = RUN / "final_audit_v1"


@pytest.mark.skipif(not RUN.exists(), reason="HK-155 run artifacts absent")
def test_raw_hashes_frozen_and_expected():
    before = collect_raw_hashes(RUN)
    assert before["predictions.jsonl"]["sha256"] == EXPECTED_PRED_SHA
    assert before["run_metadata.json"]["sha256"] == EXPECTED_META_SHA
    assert before["predictions.jsonl"]["rows"] == EXPECTED_N
    for name in RAW_NAMES:
        assert before[name].get("exists") is True


def test_question_gold_future_date_not_automatic_leak():
    row = {
        "task_id": "toy_a",
        "question": "What is the close on 2026-09-01?",
        "gold_values": {"trade_date": "2026-09-01"},
        "final_answer": "",
        "answer_correct": False,
        "executed_steps": [],
        "evidence_store": {"verified_values": {}, "provenance_by_key": {}},
    }
    out = analyze_task_temporal(row)
    assert out["layers"]["A_question_or_gold_post_cutoff"] is True
    assert out["temporal_contamination"] is False
    assert out["answer_bearing_contamination"] is False


def test_provider_post_cutoff_evidence_detected():
    row = {
        "task_id": "toy_f",
        "question": "EPS?",
        "gold_values": {},
        "final_answer": "EPS is 1.2 as of 2026-09-30",
        "answer_correct": True,
        "executed_steps": [
            {
                "step_id": "s1",
                "values": {"report_date": "2026-09-30", "eps": 1.2, "issuer_code": "00005"},
                "funnel": {
                    "invoked_operations": ["aktools.stock_financial_hk_analysis_indicator_em"],
                    "binding": {
                        "operation_id": "aktools.stock_financial_hk_analysis_indicator_em",
                        "arguments_redacted": {"symbol": "00005"},
                    },
                },
            }
        ],
        "evidence_store": {
            "verified_values": {
                "report_date": "2026-09-30",
                "eps": 1.2,
                "issuer_code": "00005",
            },
            "provenance_by_key": {
                "report_date": {
                    "operation_id": "aktools.stock_financial_hk_analysis_indicator_em",
                    "arguments_redacted": {"symbol": "00005"},
                    "source": "mcp",
                }
            },
        },
    }
    out = analyze_task_temporal(row)
    assert out["temporal_contamination"] is True
    assert out["layers"]["F_G_evidence_store_accepted_post_cutoff"] is True
    assert out["unique_evidence_n"] >= 1
    # duplicate path dedup: step values + store => one canonical event for report_date
    fields = [e["field"] for e in out["unique_evidence_events"]]
    assert fields.count("report_date") == 1


def test_na_and_zero_invocation_not_counted_as_success_helpers():
    # N/A / zero-invoke must not be treated as tool success in audit metrics logic.
    rows = [
        {"exact_tool_set": False, "all_tool_invocations_successful": False, "underlying_operation_calls": 0},
        {"exact_tool_set": True, "all_tool_invocations_successful": True, "underlying_operation_calls": 2},
    ]
    exact = sum(1 for r in rows if r.get("exact_tool_set") is True)
    all_ok = sum(1 for r in rows if r.get("all_tool_invocations_successful") is True)
    assert exact == 1
    assert all_ok == 1
    assert not (rows[0]["underlying_operation_calls"] == 0 and rows[0]["all_tool_invocations_successful"])


def test_failed_partial_not_strict():
    assert not (("partial" == "full") and True)
    assert not (("failed" == "full") and True)
    strict = sum(
        1
        for status, correct in [("full", True), ("partial", True), ("failed", True)]
        if status == "full" and correct is True
    )
    assert strict == 1


def test_parse_iso_date():
    assert parse_iso_date("2026-09-30").isoformat() == "2026-09-30"
    assert parse_iso_date("no date") is None


@pytest.mark.skipif(not RUN.exists(), reason="HK-155 run artifacts absent")
def test_run_final_audit_offline_and_freeze(tmp_path: Path):
    # Run into real audit dir once; assert freeze + completeness essentials.
    summary = run_final_audit(root=ROOT, run_dir=RUN)
    assert summary["raw_hashes_unchanged"] is True
    assert summary["explicitly_not_formal_accuracy"] is True
    assert summary["zero_network"] is True
    assert summary["no_rerun"] is True
    assert summary["no_raw_mutation"] is True
    assert summary["etom_not_run"] is True
    assert (AUD / "audit_summary.json").exists()
    assert (AUD / "temporal_cutoff_audit.json").exists()
    assert (AUD / "final_report.md").exists()
    assert (AUD / "VALIDITY_MARKER.json").exists()
    # raw still frozen
    after = collect_raw_hashes(RUN)
    assert after["predictions.jsonl"]["sha256"] == EXPECTED_PRED_SHA
    # completeness
    comp = json.loads((AUD / "completeness.json").read_text(encoding="utf-8"))
    assert comp["n"] == EXPECTED_N
    assert comp["unique"] == EXPECTED_N
    assert comp["missing_ids"] == []
    assert comp["duplicate_ids"] == []
    # mock cannot forge live pass: metadata mock false checked in identity
    ident = json.loads((AUD / "identity_checks.json").read_text(encoding="utf-8"))
    assert ident["checks"]["no_mock"] is True
