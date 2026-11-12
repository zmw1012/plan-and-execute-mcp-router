"""Offline tests: benchmark cutoff enforcement for disclosure list/download."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from hk_equity_mcp.pe_fin_lite.benchmark_cutoff_enforcement import (
    BENCHMARK_CUTOFF_ENFORCEMENT_VERSION,
    benchmark_cutoff_enforcement_hash,
    clamp_disclosure_date_window,
    effective_to_date,
    enforce_disclosure_invoke_arguments,
    filter_rows_by_benchmark_cutoff,
    reject_post_cutoff_evidence_values,
)
from hk_equity_mcp.pe_fin_lite.disclosure_workflow import (
    build_stage_arguments,
    enrich_disclosure_grounding,
    filter_list_rows,
)
from hk_equity_mcp.pe_fin_lite.temporal_policy import DEFAULT_DATASET_CUTOFF

ROOT = Path(__file__).resolve().parents[1]
RUN2 = ROOT / "outputs" / "pe_fin_replan_diagnostic20_live_v1_4_1_run2"
AUD = ROOT / "outputs" / "pe_fin_replan_diagnostic20_live_v1_4_1_run2_audit"

CUTOFF = "2026-08-12"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_frozen_run2_raw_hashes_unchanged():
    frozen = json.loads((AUD / "raw_artifact_sha256.json").read_text(encoding="utf-8"))
    for name, expected in frozen.items():
        assert _sha(RUN2 / name) == expected
    marker = (RUN2 / "INVALIDATED.md").read_text(encoding="utf-8")
    assert (
        "invalidated_engineering_diagnostic_due_to_disclosure_broadening_benchmark_cutoff_leakage"
        in marker
    )
    assert "pe_fin_replan_diagnostic20_live_v1_4_1_run3" in marker


def test_requested_to_date_gt_cutoff_clamped():
    assert effective_to_date("2026-11-12", cutoff=CUTOFF) == CUTOFF
    args, notes = clamp_disclosure_date_window(
        {"stock_code": "09988", "start_date": "2026-07-01", "end_date": "2026-11-12"},
        cutoff=CUTOFF,
    )
    assert args["end_date"] == CUTOFF
    assert any("clamped" in n for n in notes)


def test_missing_to_date_uses_cutoff_not_today():
    assert effective_to_date(None, cutoff=CUTOFF) == CUTOFF
    args, notes = clamp_disclosure_date_window(
        {"stock_code": "09988", "start_date": "2026-07-01"},
        cutoff=CUTOFF,
    )
    assert args["end_date"] == CUTOFF
    assert any("cutoff" in n for n in notes)


def test_broad_fallback_cannot_use_system_date():
    # Missing end → cutoff (not today). Missing start → explicit 1999-04-01
    # (provider-compatible lower bound) with end still capped.
    args, notes = clamp_disclosure_date_window({}, cutoff=CUTOFF)
    assert args["end_date"] == CUTOFF
    assert args["start_date"] == "1999-04-01"
    assert any("1999" in n or "cutoff" in n for n in notes)
    out, _, block = enforce_disclosure_invoke_arguments(
        "hkexnews.list_hkexnews_announcements",
        {"stock_code": "09988", "title_keyword": "announcement"},
        cutoff=CUTOFF,
        grounding={},
    )
    assert block is None
    assert out["end_date"] == CUTOFF
    assert out["start_date"] == "1999-04-01"
    # Existing scoped from_date must not be widened to 1999 by enforcement.
    out2, _, block2 = enforce_disclosure_invoke_arguments(
        "hkexnews.list_hkexnews_announcements",
        {
            "stock_code": "09988",
            "start_date": "2026-07-01",
            "end_date": "2026-08-12",
            "title_keyword": "announcement",
        },
        cutoff=CUTOFF,
        grounding={"from_date": "2026-07-01", "to_date": "2026-08-12"},
    )
    assert block2 is None
    assert out2["start_date"] == "2026-07-01"


def test_returned_row_after_cutoff_rejected():
    rows = [
        {"title": "OK", "release_date": "2026-08-10", "stock_code": "09988"},
        {"title": "LEAK", "release_date": "2026-10-14", "stock_code": "09988"},
    ]
    kept, rejected = filter_rows_by_benchmark_cutoff(rows, cutoff=CUTOFF)
    assert len(kept) == 1 and kept[0]["title"] == "OK"
    assert rejected and rejected[0]["reason"] == "beyond_benchmark_cutoff"
    filtered = filter_list_rows(
        rows,
        grounding={
            "stock_code": "09988",
            "from_date": "2026-07-01",
            "to_date": "2026-08-12",
            "benchmark_cutoff": CUTOFF,
        },
        step={"scope_constraints": {"entity": {"code": "09988"}}},
        require_title_match=False,
    )
    assert [r["title"] for r in filtered] == ["OK"]


def test_post_cutoff_document_cannot_download():
    _, _, block = enforce_disclosure_invoke_arguments(
        "hkexnews.download_hkexnews_announcement",
        {
            "stock_code": "09988",
            "title_keyword": "announcement",
        },
        cutoff=CUTOFF,
    )
    assert block == "download_missing_release_date_or_key_under_cutoff_policy"

    _, _, block2 = enforce_disclosure_invoke_arguments(
        "hkexnews.download_hkexnews_announcement",
        {
            "stock_code": "09988",
            "release_date": "2026-10-14",
            "title_keyword": "OVERSEAS",
        },
        cutoff=CUTOFF,
    )
    assert block2 == "download_release_date_beyond_benchmark_cutoff"

    _, _, block3 = enforce_disclosure_invoke_arguments(
        "hkexnews.download_hkexnews_announcement",
        {
            "stock_code": "09988",
            "local_key": "09988_13150_2026-10-14_abcdef12",
        },
        cutoff=CUTOFF,
    )
    assert block3 == "download_local_key_beyond_benchmark_cutoff"


def test_pre_cutoff_valid_disclosure_remains_usable():
    args, _, block = enforce_disclosure_invoke_arguments(
        "hkexnews.list_hkexnews_announcements",
        {
            "stock_code": "09988",
            "start_date": "2026-07-01",
            "end_date": "2026-08-12",
            "title_keyword": "DATE OF BOARD MEETING",
        },
        cutoff=CUTOFF,
        grounding={"from_date": "2026-07-01", "to_date": "2026-08-12"},
    )
    assert block is None
    assert args["end_date"] == CUTOFF
    assert args["start_date"] == "2026-07-01"

    args2, _, block2 = enforce_disclosure_invoke_arguments(
        "hkexnews.download_hkexnews_announcement",
        {
            "stock_code": "09988",
            "release_date": "2026-08-07",
            "local_key": "09988_13150_2026-08-07_df274199",
        },
        cutoff=CUTOFF,
    )
    assert block2 is None
    vals, reason = reject_post_cutoff_evidence_values(
        {
            "title": "DATE OF BOARD MEETING",
            "release_date": "2026-08-07",
            "local_key": "09988_13150_2026-08-07_df274199",
        },
        cutoff=CUTOFF,
    )
    assert reason is None
    assert vals["release_date"] == "2026-08-07"


def test_replan_generated_out_of_range_args_clamped_or_rejected():
    # Simulated RePlan binder args with system-date end.
    args, notes, block = enforce_disclosure_invoke_arguments(
        "hkexnews.list_hkexnews_announcements",
        {
            "stock_code": "00700",
            "start_date": "2026-07-01",
            "end_date": "2026-11-12",
            "title_keyword": "Monthly Return",
        },
        cutoff=CUTOFF,
        grounding={"from_date": "2026-07-01", "to_date": "2026-08-12"},
    )
    assert block is None
    assert args["end_date"] == CUTOFF
    assert any("clamped" in n for n in notes)

    # Undated download fallback must be rejected (news_dev_015 shape).
    _, _, block2 = enforce_disclosure_invoke_arguments(
        "hkexnews.download_hkexnews_announcement",
        {"stock_code": "09988", "title_keyword": "announcement"},
        cutoff=CUTOFF,
        grounding={"from_date": "2026-07-01", "to_date": "2026-08-12"},
    )
    assert block2 is not None


def test_post_cutoff_evidence_stripped():
    vals, reason = reject_post_cutoff_evidence_values(
        {
            "title": "OVERSEAS REGULATORY ANNOUNCEMENT",
            "release_date": "2026-10-14",
            "issuer_code": "09988",
        },
        cutoff=CUTOFF,
    )
    assert reason == "beyond_benchmark_cutoff"
    assert "release_date" not in vals
    assert "title" not in vals
    assert vals.get("issuer_code") == "09988"


def test_enrich_and_build_stage_args_honor_cutoff():
    g = enrich_disclosure_grounding(
        {"stock_code": "09988"},
        step={"objective": "list announcements"},
        question="window 2026-07-01..2026-08-12",
        cutoff=CUTOFF,
    )
    assert g["benchmark_cutoff"] == CUTOFF
    assert g["to_date"] == CUTOFF
    # Missing from_date stays unset in grounding (filter has no lower bound).
    assert not g.get("from_date")

    args = build_stage_arguments(
        stage="list",
        contract={
            "json_schema": {
                "properties": {
                    "stock_code": {"type": "string"},
                    "start_date": {"type": "string"},
                    "end_date": {"type": "string"},
                },
                "required": ["stock_code"],
            }
        },
        chain_state={},
        step={"scope_constraints": {"from_date": "2026-07-01", "to_date": "2026-11-12"}},
        grounding={
            "stock_code": "09988",
            "from_date": "2026-07-01",
            "to_date": "2026-11-12",
            "benchmark_cutoff": CUTOFF,
        },
    )
    assert args["end_date"] == CUTOFF
    assert args["start_date"] == "2026-07-01"

    # Unscoped build still emits explicit start/end (no today default).
    args2 = build_stage_arguments(
        stage="list",
        contract={
            "json_schema": {
                "properties": {
                    "stock_code": {"type": "string"},
                    "start_date": {"type": "string"},
                    "end_date": {"type": "string"},
                },
                "required": ["stock_code"],
            }
        },
        chain_state={},
        step={},
        grounding={"stock_code": "09988", "benchmark_cutoff": CUTOFF},
    )
    assert args2["end_date"] == CUTOFF
    assert args2["start_date"] == "1999-04-01"


def test_enforcement_identity_version():
    assert BENCHMARK_CUTOFF_ENFORCEMENT_VERSION.endswith("_v1")
    assert len(benchmark_cutoff_enforcement_hash()) == 64
    assert DEFAULT_DATASET_CUTOFF == CUTOFF
