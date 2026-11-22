"""Offline tests for HK-155 exploratory_all mode (no paid APIs)."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

from hk_equity_mcp.experiment.unified_aggregates import build_grouped_aggregates
from hk_equity_mcp.experiment.unified_dataset import load_split_tasks
from hk_equity_mcp.experiment.unified_exploratory import (
    RETIREMENT_POLICY_VERSION,
    assert_exploratory_unlocked,
    build_retirement_metadata,
    load_exploratory_manifest,
    load_exploratory_run_order,
    load_exploratory_tasks,
    load_exploratory_tasks_in_locked_order,
)
from hk_equity_mcp.experiment.unified_grading import grade_task
from hk_equity_mcp.experiment.unified_prompt import unified_prompt_hash
from hk_equity_mcp.experiment.unified_runner import (
    UnifiedHkOperationVisibleRunner,
    assert_resume_compatible_unified,
)
from hk_equity_mcp.real_mcp.unified_catalog import build_unified_catalog

ROOT = Path(__file__).resolve().parents[1]
FROZEN_PROMPT_HASH = (
    "9674844a98c9f438ae02d19572f173a7cdc27e1c365ed27dd39ea7e0963eec1b"
)
SOURCE_FILES = [
    "benchmarks/questions/financial_real_mcp_coverage_v1_3.jsonl",
    "benchmarks/heldout/financial_test_30_v1_3.jsonl",
    "benchmarks/questions/market_development_15_v1_1.jsonl",
    "benchmarks/heldout/market_test_20_v1_1.jsonl",
    "benchmarks/questions/news_development_15_v1.jsonl",
    "benchmarks/heldout/news_test_20_v1.jsonl",
    "benchmarks/questions/cross_domain_development_10_v1.jsonl",
    "benchmarks/heldout/cross_domain_test_15_v1.jsonl",
]


def test_exploratory_requires_confirm():
    with pytest.raises(RuntimeError, match="confirm-retire-heldout"):
        assert_exploratory_unlocked(split="exploratory_all", confirm_retire_heldout=False)
    ok = assert_exploratory_unlocked(
        split="exploratory_all", confirm_retire_heldout=True
    )
    assert ok["allowed"] is True


def test_dry_run_mock_do_not_retire(tmp_path: Path):
    for dry, mock in ((True, False), (False, True)):
        meta = build_retirement_metadata(
            dry_run=dry,
            mock=mock,
            confirm_retire_heldout=True,
            git_commit="abc",
            prompt_hash="p",
            catalog_fingerprint="c",
            manifest_hash="m",
            run_order_hash="r",
            timestamp="t0",
        )
        assert meta["retirement_effective"] is False
        assert meta["heldout_status"] == "pending_real_exploratory_run"
        assert meta["unseen_final_claim_allowed"] is False


def test_real_run_retirement_metadata_fields():
    meta = build_retirement_metadata(
        dry_run=False,
        mock=False,
        confirm_retire_heldout=True,
        git_commit="deadbeef",
        prompt_hash=FROZEN_PROMPT_HASH,
        catalog_fingerprint="fp",
        manifest_hash="mh",
        run_order_hash="rh",
        timestamp="2026-11-04T00:00:00+00:00",
    )
    assert meta["heldout_status"] == "retired_into_exploratory_development"
    assert meta["retirement_effective"] is True
    assert meta["benchmark_role"] == "method_development_and_difficulty_audit"
    assert meta["unseen_final_claim_allowed"] is False
    first = meta["first_real_exploratory_run"]
    assert first["git_commit"] == "deadbeef"
    assert first["system_prompt_hash"] == FROZEN_PROMPT_HASH
    assert first["manifest_hash"] == "mh"
    # Preserve first real on later calls.
    meta2 = build_retirement_metadata(
        dry_run=False,
        mock=False,
        confirm_retire_heldout=True,
        git_commit="later",
        prompt_hash=FROZEN_PROMPT_HASH,
        catalog_fingerprint="fp",
        manifest_hash="mh",
        run_order_hash="rh",
        timestamp="later-ts",
        existing_first_real=first,
    )
    assert meta2["first_real_exploratory_run"] == first


def test_155_counts_and_distributions():
    tasks = load_exploratory_tasks(ROOT)
    assert len(tasks) == 155
    assert len({t["task_id"] for t in tasks}) == 155
    assert Counter(t["domain"] for t in tasks) == {
        "financial": 60,
        "market": 35,
        "news": 35,
        "cross_domain": 25,
    }
    assert Counter(t["source_split"] for t in tasks) == {
        "development": 70,
        "previously_reserved": 85,
    }
    for t in tasks:
        assert t["dataset_split"] == "exploratory_all"
        assert t["benchmark_role"] == "method_development_and_difficulty_audit"
        assert t["previously_reserved"] is (t["source_split"] == "previously_reserved")


def test_locked_order_coverage_and_hash_stable():
    order = load_exploratory_run_order(ROOT)
    man = load_exploratory_manifest(ROOT)
    tasks = load_exploratory_tasks_in_locked_order(ROOT)
    assert len(order["task_ids"]) == 155
    assert len(set(order["task_ids"])) == 155
    assert [t["task_id"] for t in tasks] == order["task_ids"]
    expected = hashlib.sha256(
        json.dumps(
            {
                "seed": order["seed"],
                "policy": order["policy"],
                "task_ids": order["task_ids"],
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    assert order["run_order_hash"] == expected
    assert man["run_order_hash"] == order["run_order_hash"]
    # Interleave starts with domain cycle.
    assert [t["domain"] for t in tasks[:4]] == [
        "financial",
        "market",
        "news",
        "cross_domain",
    ]


def test_max_tasks_resume_identity_compatible():
    base = {
        "condition": "operation_visible",
        "execution_backend": "real_mcp",
        "generation_model": "deepseek-v4.1-flash",
        "embedding_model": "text-embedding-3-small",
        "dataset_split": "exploratory_all",
        "system_prompt_hash": FROZEN_PROMPT_HASH,
        "temperature": 0.0,
        "max_tokens": 1024,
        "catalog_fingerprint": "fp",
        "exploratory_manifest_hash": "mh",
        "run_order_hash": "rh",
        "retirement_policy_version": RETIREMENT_POLICY_VERSION,
        "max_tasks": 20,
    }
    expanded = dict(base)
    expanded["max_tasks"] = 50
    # max_tasks is intentionally excluded from identity.
    assert_resume_compatible_unified(base, expanded)
    bad = dict(expanded)
    bad["run_order_hash"] = "other"
    with pytest.raises(RuntimeError, match="run_order_hash"):
        assert_resume_compatible_unified(base, bad)


def test_source_jsonl_unchanged_hashes():
    man = load_exploratory_manifest(ROOT)
    by_path = {s["path"]: s["sha256"] for s in man["sources"]}
    for rel in SOURCE_FILES:
        path = ROOT / rel
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        assert by_path[rel] == digest
        # Ensure no exploratory fields leaked into frozen files.
        text = data.decode("utf-8")
        assert "exploratory_all" not in text
        assert "previously_reserved_items_cease" not in text


def test_deterministic_numeric_and_unit_grading():
    item = {
        "task_id": "t1",
        "answerability_class": "structured_answerable",
        "gold_status": "answered",
        "gold_values": [{"value": 751766, "unit": "CNY million", "tolerance_abs": 0.5}],
    }
    ok = grade_task(
        item,
        {
            "task_id": "t1",
            "predicted_status": "answered",
            "final_answer": "Revenue is 751766 CNY million.",
        },
    )
    assert ok["auto_grade_status"] == "graded"
    assert ok["answer_correct"] is True

    bad_unit = grade_task(
        item,
        {
            "task_id": "t1",
            "predicted_status": "answered",
            "final_answer": "Revenue is 751766 HKD million.",
        },
    )
    assert bad_unit["answer_correct"] is False
    assert "unit" in bad_unit["reason"]


def test_refusal_fabricated_numeric_wrong():
    item = {
        "task_id": "t2",
        "answerability_class": "intentionally_infeasible",
        "gold_status": "unavailable",
        "gold_values": None,
    }
    ok = grade_task(
        item,
        {
            "task_id": "t2",
            "predicted_status": "unavailable",
            "final_answer": "Unavailable under catalog constraints.",
        },
    )
    assert ok["answer_correct"] is True
    bad = grade_task(
        item,
        {
            "task_id": "t2",
            "predicted_status": "unavailable",
            "final_answer": "Unavailable, but revenue was about 751766 CNY million.",
        },
    )
    assert bad["answer_correct"] is False
    assert "fabricated" in bad["reason"]


def test_manual_required_not_auto_correct():
    item = {
        "task_id": "t3",
        "answerability_class": "evidence_answerable",
        "gold_status": "answered",
        "gold_values": [{"note": "narrative only"}],
    }
    out = grade_task(
        item,
        {
            "task_id": "t3",
            "predicted_status": "answered",
            "final_answer": "A long evidence summary.",
        },
    )
    assert out["auto_grade_status"] == "manual_required"
    assert out["answer_correct"] is None


def test_grouped_aggregate_totals():
    rows = []
    for i, (dom, src) in enumerate(
        [("financial", "development")] * 2
        + [("market", "previously_reserved")] * 3
    ):
        rows.append(
            {
                "task_id": f"id{i}",
                "domain": dom,
                "domain_combination": dom,
                "source_split": src,
                "previously_reserved": src == "previously_reserved",
                "project_difficulty": "D1",
                "company": "Tencent",
                "answerability_class": "structured_answerable",
                "status": "full",
                "exact_tool_set": True,
                "operation_f1": 1.0,
                "all_tool_invocations_successful": True,
            }
        )
    grouped = build_grouped_aggregates(rows)
    assert grouped["overall"]["task_count"] == 5
    assert grouped["domain"]["financial"]["task_count"] == 2
    assert grouped["domain"]["market"]["task_count"] == 3
    assert grouped["development_origin"]["task_count"] == 2
    assert grouped["previously_reserved_origin"]["task_count"] == 3
    assert grouped["origin_count_check"]["ok"] is True


def test_prompt_catalog_frozen():
    assert unified_prompt_hash() == FROZEN_PROMPT_HASH
    bundle = build_unified_catalog(root=ROOT)
    assert bundle.stats.unique_operation_ids == 42
    assert bundle.stats.runtime_exposed_unique == 39


def test_exploratory_dry_run_cli_path(tmp_path: Path):
    out = tmp_path / "exp_dry"
    with pytest.raises(RuntimeError, match="confirm-retire-heldout"):
        UnifiedHkOperationVisibleRunner(
            output_dir=out,
            split="exploratory_all",
            dry_run=True,
            allow_dirty=True,
            confirm_retire_heldout=False,
            root=ROOT,
        ).run()

    meta = UnifiedHkOperationVisibleRunner(
        output_dir=out,
        split="exploratory_all",
        dry_run=True,
        allow_dirty=True,
        confirm_retire_heldout=True,
        max_tasks=20,
        root=ROOT,
    ).run()
    assert meta["question_count"] == 20
    assert meta["dataset_split"] == "exploratory_all"
    assert meta["retirement"]["retirement_effective"] is False
    assert meta["retirement"]["heldout_status"] == "pending_real_exploratory_run"
    assert meta["exploratory_manifest_hash"]
    assert meta["run_order_hash"]
    assert meta["retirement_policy_version"] == RETIREMENT_POLICY_VERSION
    # Locked-order prefix
    ordered = load_exploratory_run_order(ROOT)["task_ids"][:20]
    assert meta["question_ids"] == ordered


def test_exploratory_mock_prefix_resume(tmp_path: Path):
    out = tmp_path / "exp_mock"
    r1 = UnifiedHkOperationVisibleRunner(
        output_dir=out,
        split="exploratory_all",
        mock=True,
        allow_dirty=True,
        confirm_retire_heldout=True,
        max_tasks=2,
        root=ROOT,
    ).run()
    assert r1["retirement"]["retirement_effective"] is False
    assert r1["question_count"] == 2
    preds = [
        json.loads(l)
        for l in (out / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    assert len(preds) == 2
    assert all(p.get("source_split") in {"development", "previously_reserved"} for p in preds)

    r2 = UnifiedHkOperationVisibleRunner(
        output_dir=out,
        split="exploratory_all",
        mock=True,
        allow_dirty=True,
        confirm_retire_heldout=True,
        max_tasks=5,
        resume=True,
        root=ROOT,
    ).run()
    assert r2["question_count"] == 5
    preds2 = [
        json.loads(l)
        for l in (out / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    assert len(preds2) == 5
    assert (out / "grouped_aggregates.json").exists()
    assert (out / "grading_summary.json").exists()


def test_load_split_tasks_exploratory_uses_locked_order():
    tasks = load_split_tasks(ROOT, "exploratory_all")
    order = load_exploratory_run_order(ROOT)["task_ids"]
    assert [t["task_id"] for t in tasks] == order
