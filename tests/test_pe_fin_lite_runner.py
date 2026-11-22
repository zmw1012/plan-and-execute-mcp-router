"""PE-Fin Lite runner / manifest / isolation tests (offline)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from hk_equity_mcp.experiment.unified_evaluator import EVALUATOR_VERSION
from hk_equity_mcp.experiment.unified_pe_fin_lite_runner import (
    UnifiedHkPeFinLiteRunner,
    assert_output_dir_safe,
    load_diagnostic12_manifest,
    load_diagnostic12_tasks,
    load_smoke5_manifest,
    load_smoke5_tasks,
)
from hk_equity_mcp.experiment.unified_apv_fin_runner import (
    load_diagnostic12_manifest as load_apv_d12,
)

ROOT = Path(__file__).resolve().parents[1]


def test_evaluator_version_unchanged():
    assert EVALUATOR_VERSION == "hk_unified_evaluator_v1.2"


def test_diagnostic12_identity_and_order_matches_apv():
    pe = load_diagnostic12_manifest(ROOT)
    apv = load_apv_d12(ROOT)
    assert pe["ordered_task_ids"] == apv["ordered_task_ids"]
    assert pe["ordered_id_hash"] == apv["ordered_id_hash"]
    assert pe["ordered_task_ids"] == [
        "fin_cf_003",
        "fin_cf_011",
        "fin_cf_020",
        "fin_cf_023",
        "fin_ht_016",
        "xd_dev_002",
        "xd_ht_007",
        "xd_ht_014",
        "mkt_dev_014",
        "fin_cf_024",
        "mkt_dev_001",
        "news_ht_004",
    ]
    tasks = load_diagnostic12_tasks(ROOT)
    assert [t["task_id"] for t in tasks] == pe["ordered_task_ids"]
    # gold not mutated vs exploratory source
    from hk_equity_mcp.experiment.unified_dataset import load_split_tasks

    all_tasks = {t["task_id"]: t for t in load_split_tasks(ROOT, "exploratory_all")}
    for t in tasks:
        src = all_tasks[t["task_id"]]
        assert t["question"] == src["question"]
        assert t.get("reference_answer") == src.get("reference_answer")
        assert t.get("gold_operations") == src.get("gold_operations")


def test_smoke5_ids_and_hash():
    man = load_smoke5_manifest(ROOT)
    assert man["ordered_task_ids"] == [
        "fin_cf_003",
        "fin_cf_023",
        "xd_dev_002",
        "mkt_dev_014",
        "news_ht_004",
    ]
    expected = hashlib.sha256(
        ("\n".join(man["ordered_task_ids"]) + "\n").encode()
    ).hexdigest()
    assert man["ordered_id_hash"] == expected
    tasks = load_smoke5_tasks(ROOT)
    assert [t["task_id"] for t in tasks] == man["ordered_task_ids"]


def test_dry_run_smoke5_and_diagnostic12(tmp_path: Path):
    out5 = tmp_path / "smoke5_dry"
    meta5 = UnifiedHkPeFinLiteRunner(
        output_dir=out5,
        diagnostic12=False,
        smoke5=True,
        dry_run=True,
        allow_dirty=True,
        confirm_retire_heldout=True,
        root=ROOT,
    ).run()
    assert meta5["smoke5"] is True
    assert meta5["question_count"] == 5
    assert meta5["condition"] == "pe_fin_lite"
    assert meta5["architecture"] == "plan_first_step_level_retrieval_v1"
    assert meta5["monitor_enabled"] is False
    assert meta5["dry_run"] is True

    out12 = tmp_path / "d12_dry"
    meta12 = UnifiedHkPeFinLiteRunner(
        output_dir=out12,
        diagnostic12=True,
        dry_run=True,
        allow_dirty=True,
        confirm_retire_heldout=True,
        root=ROOT,
    ).run()
    assert meta12["diagnostic12"] is True
    assert meta12["question_count"] == 12
    assert meta12["diagnostic_manifest_hash"] == load_diagnostic12_manifest(ROOT)[
        "manifest_hash"
    ]


def test_output_dir_isolation_dry_to_live(tmp_path: Path):
    out = tmp_path / "iso"
    out.mkdir()
    dry_meta = {
        "condition": "pe_fin_lite",
        "method_version": "pe_fin_lite_v1",
        "architecture": "plan_first_step_level_retrieval_v1",
        "dry_run": True,
        "mock": False,
        "subset_kind": "smoke5",
    }
    (out / "run_metadata.json").write_text(json.dumps(dry_meta), encoding="utf-8")
    (out / "predictions.jsonl").write_text("", encoding="utf-8")
    live_meta = dict(dry_meta)
    live_meta["dry_run"] = False
    with pytest.raises(RuntimeError, match="dry-run"):
        assert_output_dir_safe(
            output_dir=out,
            dry_run=False,
            mock=False,
            resume=True,
            run_meta=live_meta,
        )


def test_output_dir_isolation_mock_to_live_and_subset(tmp_path: Path):
    out = tmp_path / "iso2"
    out.mkdir()
    mock_meta = {
        "condition": "pe_fin_lite",
        "method_version": "pe_fin_lite_v1",
        "architecture": "plan_first_step_level_retrieval_v1",
        "dry_run": False,
        "mock": True,
        "subset_kind": "smoke5",
    }
    (out / "run_metadata.json").write_text(json.dumps(mock_meta), encoding="utf-8")
    live_meta = dict(mock_meta)
    live_meta["mock"] = False
    with pytest.raises(RuntimeError, match="mock"):
        assert_output_dir_safe(
            output_dir=out,
            dry_run=False,
            mock=False,
            resume=True,
            run_meta=live_meta,
        )
    other = dict(mock_meta)
    other["subset_kind"] = "diagnostic12"
    with pytest.raises(RuntimeError, match="subset_kind"):
        assert_output_dir_safe(
            output_dir=out,
            dry_run=False,
            mock=True,
            resume=True,
            run_meta=other,
        )


def test_mock_smoke5_offline(tmp_path: Path):
    out = tmp_path / "smoke5_mock"
    meta = UnifiedHkPeFinLiteRunner(
        output_dir=out,
        smoke5=True,
        mock=True,
        allow_dirty=True,
        confirm_retire_heldout=True,
        root=ROOT,
    ).run()
    assert meta["run_status"] == "completed"
    assert meta["processed_task_count"] == 5
    preds = (out / "predictions.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(preds) == 5
    # No monitor / replan fields populated as control loops
    row = json.loads(preds[0])
    assert row["condition"] == "pe_fin_lite"
    assert row.get("monitor_decisions") == []
    assert row.get("plan_revisions") == []
    assert (out / "pe_fin_lite_metrics.json").exists()
    assert (out / "retrieval_analysis.json").exists()
    assert (out / "cost_summary.json").exists()
    assert (out / "error_summary.json").exists()
    assert (out / "grouped_aggregates.json").exists()
    assert (out / "grading_summary.json").exists()


def test_no_monitor_replan_in_package_imports():
    import hk_equity_mcp.pe_fin_lite.agent as agent_mod
    src = Path(agent_mod.__file__).read_text(encoding="utf-8")
    assert "run_monitor" not in src
    assert "exception_replan" not in src
    assert "capability_requirement_compiler" not in src
