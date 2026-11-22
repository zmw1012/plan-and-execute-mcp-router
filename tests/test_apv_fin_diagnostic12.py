"""Diagnostic-12 manifest validation (offline; no Agent gold mutation)."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from hk_equity_mcp.experiment.unified_apv_fin_runner import (
    UnifiedHkApvFinRunner,
    load_diagnostic12_manifest,
    load_diagnostic12_tasks,
)
from hk_equity_mcp.experiment.unified_dataset import load_split_tasks

ROOT = Path(__file__).resolve().parents[1]


def test_diagnostic12_distribution_and_hashes():
    man = load_diagnostic12_manifest(ROOT)
    ids = man["ordered_task_ids"]
    assert len(ids) == 12
    assert len(set(ids)) == 12
    counts = Counter(t["selection_bucket"] for t in man["tasks"])
    assert counts == {
        "financial_multi_step_evidence_chain": 3,
        "financial_calculation_period_basis": 2,
        "cross_domain_sequential_dependency": 2,
        "cross_domain_parallel_mixed": 1,
        "refusal_clarification_temporal_gap": 2,
        "market_easy_regression_sentinel": 1,
        "news_easy_regression_sentinel": 1,
    }
    companies = {t["company"] for t in man["tasks"]}
    assert {"Tencent", "HSBC", "Alibaba"}.issubset(companies)
    assert all(t.get("copies_question_or_gold") is False for t in man["tasks"])

    # ordered id hash
    expected_id_hash = hashlib.sha256(("\n".join(ids) + "\n").encode()).hexdigest()
    assert man["ordered_id_hash"] == expected_id_hash

    # manifest hash excludes hash fields
    body = {
        k: v
        for k, v in man.items()
        if k not in {"manifest_hash", "ordered_id_hash", "bucket_counts"}
    }
    # Recompute using same canonicalization as builder (without bucket_counts).
    # The stored manifest_hash was computed before adding ordered_id_hash/manifest_hash/bucket_counts.
    rebuilt = {
        "manifest_id": man["manifest_id"],
        "version": man["version"],
        "description": man["description"],
        "parent_dataset": man["parent_dataset"],
        "parent_manifest": man["parent_manifest"],
        "baseline_audit": man["baseline_audit"],
        "selection_policy": man["selection_policy"],
        "ordered_task_ids": man["ordered_task_ids"],
        "tasks": man["tasks"],
    }
    canon = json.dumps(rebuilt, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    assert man["manifest_hash"] == hashlib.sha256(canon.encode()).hexdigest()


def test_diagnostic12_pointer_only_loads_frozen_questions():
    tasks = load_diagnostic12_tasks(ROOT)
    assert [t["task_id"] for t in tasks] == load_diagnostic12_manifest(ROOT)[
        "ordered_task_ids"
    ]
    all_tasks = {t["task_id"]: t for t in load_split_tasks(ROOT, "exploratory_all")}
    for t in tasks:
        src = all_tasks[t["task_id"]]
        assert t["question"] == src["question"]
        assert t.get("reference_answer") == src.get("reference_answer")
        assert t.get("gold_operations") == src.get("gold_operations")


def test_diagnostic12_dry_run(tmp_path: Path):
    out = tmp_path / "d12_dry"
    meta = UnifiedHkApvFinRunner(
        output_dir=out,
        split="exploratory_all",
        diagnostic12=True,
        dry_run=True,
        allow_dirty=True,
        confirm_retire_heldout=True,
        root=ROOT,
    ).run()
    assert meta["diagnostic12"] is True
    assert meta["question_count"] == 12
    assert meta["diagnostic_manifest_hash"] == load_diagnostic12_manifest(ROOT)[
        "manifest_hash"
    ]
