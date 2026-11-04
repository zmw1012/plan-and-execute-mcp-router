"""Offline tests for unified HK Operation-Visible baseline (no paid APIs)."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from hk_equity_mcp.experiment.unified_dataset import (
    assert_heldout_unlocked,
    load_manifest,
    load_smoke_tasks,
    load_split_tasks,
)
from hk_equity_mcp.experiment.unified_evaluator import (
    evaluate_task,
    tool_prf,
)
from hk_equity_mcp.experiment.unified_prompt import (
    UNIFIED_HK_OPERATION_VISIBLE_SYSTEM_PROMPT,
    unified_prompt_hash,
)
from hk_equity_mcp.experiment.unified_runner import UnifiedHkOperationVisibleRunner
from hk_equity_mcp.experiment.unified_schema import (
    BANNED_PROMPT_SUBSTRINGS,
    MANUAL_GRADING_FIELDS,
    PER_TASK_METRIC_FIELDS,
    PREDICTION_FIELDS,
)
from hk_equity_mcp.operation_resolver.retrieval import DeterministicHashEmbeddingBackend
from hk_equity_mcp.operation_resolver.server import META_TOOLS, OperationResolverServer
from hk_equity_mcp.real_mcp.market_gateway import BLOCKED_MARKET_OPS
from hk_equity_mcp.real_mcp.unified_catalog import build_unified_catalog
from hk_equity_mcp.real_mcp.unified_gateway import (
    ScriptedMockTransport,
    UnifiedRealMcpGateway,
)

ROOT = Path(__file__).resolve().parents[1]


def test_dataset_counts():
    man = load_manifest(ROOT)
    assert man["default_split"] == "development"
    dev = load_split_tasks(ROOT, "development")
    held = load_split_tasks(ROOT, "heldout")
    assert len(dev) == 70
    assert len(held) == 85
    assert Counter(t["domain"] for t in dev) == {
        "financial": 30,
        "market": 15,
        "news": 15,
        "cross_domain": 10,
    }
    assert Counter(t["domain"] for t in held) == {
        "financial": 30,
        "market": 20,
        "news": 20,
        "cross_domain": 15,
    }


def test_unique_runtime_operations_and_blocked_not_exposed():
    bundle = build_unified_catalog(root=ROOT)
    stats = bundle.stats
    assert stats.unique_operation_ids == 42, (
        f"expected 42 unique operation_ids, got {stats.unique_operation_ids}"
    )
    assert stats.namespaced_entries == 54
    assert len(bundle.registry) == stats.runtime_exposed_unique
    # No duplicate registration.
    assert len(bundle.registry) == len(set(bundle.registry))
    for blocked in stats.blocked_operation_ids:
        assert blocked not in bundle.registry
        assert bundle.records[blocked].runtime_exposed is False
    for oid in BLOCKED_MARKET_OPS:
        assert oid not in bundle.registry
    # runtime_exposed=false never in retrieval catalog
    for oid, rec in bundle.records.items():
        if not rec.runtime_exposed:
            assert oid not in bundle.registry
    # formal_gold vs runtime separation: distractors may be runtime but not gold
    for oid, rec in bundle.records.items():
        if rec.is_distractor:
            assert rec.formal_gold_eligible is False
        if rec.is_blocked:
            assert rec.formal_gold_eligible is False
            assert rec.runtime_exposed is False
    assert stats.runtime_exposed_unique == 39
    assert stats.blocked_or_rejected_unique == 3


def test_gold_and_reference_not_in_prompt():
    prompt = UNIFIED_HK_OPERATION_VISIBLE_SYSTEM_PROMPT
    for banned in BANNED_PROMPT_SUBSTRINGS:
        assert banned not in prompt
    # No domain-specific answer recipes / pre-named gold tools for issuers.
    for needle in ("fin_cf_", "mkt_dev_", "news_dev_", "xd_dev_", "Tencent revenue"):
        assert needle not in prompt


def test_unified_prompt_hash_stable_across_domains():
    h1 = unified_prompt_hash()
    h2 = unified_prompt_hash()
    assert h1 == h2
    assert len(h1) == 64
    # Same prompt object used for all domains (no per-domain variant).
    assert "Financial, Market, News, and Cross-domain" in UNIFIED_HK_OPERATION_VISIBLE_SYSTEM_PROMPT


def test_smoke_8_distribution():
    smoke = json.loads(
        (ROOT / "benchmarks/smoke/hk_unified_dev_smoke_8_v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert smoke["dataset_split"] == "development"
    assert len(smoke["tasks"]) == 8
    assert Counter(t["domain"] for t in smoke["tasks"]) == {
        "financial": 2,
        "market": 2,
        "news": 2,
        "cross_domain": 2,
    }
    tasks = load_smoke_tasks(ROOT)
    assert [t["task_id"] for t in tasks] == [t["task_id"] for t in smoke["tasks"]]
    companies = {t.get("company") for t in tasks}
    assert {"Tencent", "HSBC", "Alibaba"}.issubset(companies)
    roles = {t["role"] for t in smoke["tasks"]}
    assert "single_tool_query" in roles
    assert "multi_step_sequential_dependency" in roles
    assert "unanswerable_or_refuse" in roles
    assert "exact_l4_cross_domain" in roles
    xd007 = next(t for t in tasks if t["task_id"] == "xd_dev_007")
    assert xd007.get("exact_etom_alignment") is True
    assert xd007.get("exact_etom_level") == "L4"


def test_heldout_lock():
    ok = assert_heldout_unlocked(split="development", confirm_heldout_final=False)
    assert ok["allowed"] is True
    with pytest.raises(RuntimeError, match="Held-out is locked"):
        assert_heldout_unlocked(split="heldout", confirm_heldout_final=False)
    unlocked = assert_heldout_unlocked(
        split="heldout", confirm_heldout_final=True
    )
    assert unlocked["allowed"] is True


def test_dry_run_shows_heldout_lock(tmp_path: Path):
    out = tmp_path / "dry"
    runner = UnifiedHkOperationVisibleRunner(
        output_dir=out,
        split="development",
        dry_run=True,
        allow_dirty=True,
        root=ROOT,
    )
    meta = runner.run()
    assert meta["dry_run"] is True
    assert meta["heldout_lock"]["allowed"] is True
    assert meta["system_prompt_hash"] == unified_prompt_hash()
    assert (out / "run_metadata.json").exists()

    with pytest.raises(RuntimeError, match="Held-out is locked"):
        UnifiedHkOperationVisibleRunner(
            output_dir=tmp_path / "held_dry",
            split="heldout",
            dry_run=True,
            allow_dirty=True,
            confirm_heldout_final=False,
            root=ROOT,
        ).run()


def test_fail_closed_unknown_and_blocked():
    gateway = UnifiedRealMcpGateway(
        catalog_bundle=build_unified_catalog(root=ROOT),
        mock_transport=ScriptedMockTransport(),
    )
    gateway.open()
    unknown = gateway.invoke("totally.unknown.op", {})
    assert unknown.success is False
    assert unknown.error_type == "unknown_operation"
    blocked = gateway.invoke("ftshare.ft_v1_report_announcements_list", {})
    assert blocked.success is False
    assert blocked.error_type in {"blocked_operation", "unknown_operation"}
    # Blocked must not be in runtime registry either.
    assert "ftshare.ft_v1_report_announcements_list" not in gateway.registry
    gateway.close()


def test_unified_gateway_mock_dispatch(tmp_path: Path):
    transport = ScriptedMockTransport()
    bundle = build_unified_catalog(root=ROOT)
    gateway = UnifiedRealMcpGateway(catalog_bundle=bundle, mock_transport=transport)
    gateway.open()
    oid = "aktools.stock_financial_hk_analysis_indicator_em"
    assert oid in bundle.registry
    result = gateway.invoke(oid, {"symbol": "0700"})
    assert result.success is True
    assert transport.calls[-1]["operation_id"] == oid
    # Resolver must not keep FinancialServer when gateway is attached.
    resolver = OperationResolverServer(
        embedding_backend=DeterministicHashEmbeddingBackend(),
        registry=bundle.registry,
        real_mcp_gateway=gateway,
        cache_dir=tmp_path / "emb_cache",
    )
    assert resolver.financial is None
    assert set(resolver.list_meta_tools()) == set(META_TOOLS)
    out = resolver.invoke_api(oid, {"symbol": "0700"})
    assert out["success"] is True
    gateway.close()


def test_resume_skips_completed(tmp_path: Path):
    out = tmp_path / "resume_run"
    # First mock smoke run with max_tasks=2
    r1 = UnifiedHkOperationVisibleRunner(
        output_dir=out,
        split="development",
        smoke=True,
        max_tasks=2,
        mock=True,
        allow_dirty=True,
        root=ROOT,
    )
    meta1 = r1.run()
    assert meta1["question_count"] == 2
    preds = [
        json.loads(line)
        for line in (out / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(preds) == 2
    done_ids = {p["task_id"] for p in preds}

    # Resume full smoke-8: should skip the first two.
    r2 = UnifiedHkOperationVisibleRunner(
        output_dir=out,
        split="development",
        smoke=True,
        resume=True,
        mock=True,
        allow_dirty=True,
        root=ROOT,
    )
    meta2 = r2.run()
    assert meta2["question_count"] == 8
    preds2 = [
        json.loads(line)
        for line in (out / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(preds2) == 8
    assert done_ids.issubset({p["task_id"] for p in preds2})
    # Required artifacts
    for name in (
        "predictions.jsonl",
        "per_task_metrics.csv",
        "manual_grading.csv",
        "cost_summary.json",
        "run_metadata.json",
        "retrieval_analysis.json",
        "error_summary.json",
    ):
        assert (out / name).exists(), name


def test_output_schema_fields(tmp_path: Path):
    out = tmp_path / "schema_run"
    UnifiedHkOperationVisibleRunner(
        output_dir=out,
        split="development",
        smoke=True,
        max_tasks=1,
        mock=True,
        allow_dirty=True,
        root=ROOT,
    ).run()
    row = json.loads((out / "predictions.jsonl").read_text(encoding="utf-8").splitlines()[0])
    for field in PREDICTION_FIELDS:
        assert field in row, field
    # Gold only present post-run for evaluator (not in prompt).
    assert "gold_operations" in row
    header = (out / "manual_grading.csv").read_text(encoding="utf-8").splitlines()[0]
    for field in MANUAL_GRADING_FIELDS:
        assert field in header.split(",")
    metric_header = (out / "per_task_metrics.csv").read_text(encoding="utf-8").splitlines()[0]
    for field in PER_TASK_METRIC_FIELDS:
        assert field in metric_header.split(",")


def test_evaluator_unit():
    p, r, f = tool_prf({"a", "b"}, {"a", "c"})
    assert p == 0.5 and r == 0.5
    item = {
        "gold_operations": ["a"],
        "answerability_class": "intentionally_infeasible",
        "gold_status": "unavailable",
        "dependency_graph": [],
    }
    result = {
        "trace": [
            {
                "type": "tool_result",
                "name": "invoke_api",
                "arguments": {"operation_id": "a", "arguments": {}},
                "result": {"success": True},
            }
        ],
        "predicted_status": "unavailable",
        "final_answer": {"evidence": []},
        "tokens": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
    }
    metrics = evaluate_task(item=item, result=result)
    assert metrics["exact_tool_set"] is True
    assert metrics["refusal_correctness"] == 1.0


def test_cross_domain_field_dependency_preserved():
    tasks = load_split_tasks(ROOT, "development")
    exact = [
        t
        for t in tasks
        if t.get("domain") == "cross_domain" and t.get("exact_etom_alignment") is True
    ]
    assert exact
    for t in exact:
        assert t.get("dependency_graph")
        assert any(
            e.get("producer_output_field") == "release_date"
            and e.get("consumer_input_field") in {"start_date", "end_date"}
            for e in t["dependency_graph"]
        )


def test_no_financial_server_fallback_on_unknown(tmp_path: Path):
    bundle = build_unified_catalog(root=ROOT)
    gateway = UnifiedRealMcpGateway(
        catalog_bundle=bundle, mock_transport=ScriptedMockTransport()
    )
    resolver = OperationResolverServer(
        embedding_backend=DeterministicHashEmbeddingBackend(),
        registry=bundle.registry,
        real_mcp_gateway=gateway,
        cache_dir=tmp_path / "emb_cache",
    )
    assert resolver.financial is None
    out = resolver.invoke_api("financial.get_financial_metric", {"ticker": "0700.HK"})
    assert out["success"] is False
    assert out["error_type"] == "unknown_operation"
