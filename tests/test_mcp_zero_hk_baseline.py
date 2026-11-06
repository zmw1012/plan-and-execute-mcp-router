"""Offline HK-155 MCP-Zero baseline tests (no paid / network APIs)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hk_equity_mcp.experiment.unified_evaluator import EVALUATOR_VERSION
from hk_equity_mcp.experiment.unified_mcp_zero_runner import (
    MCP_ZERO_RESUME_IDENTITY_KEYS,
    UnifiedHkMcpZeroRunner,
    assert_resume_compatible_mcp_zero,
    load_mcp_zero_config,
)
from hk_equity_mcp.experiment.unified_schema import (
    BANNED_PROMPT_SUBSTRINGS,
    MANUAL_GRADING_FIELDS,
    PER_TASK_METRIC_FIELDS,
    PREDICTION_FIELDS,
)
from hk_equity_mcp.mcp_zero.metadata import build_hk_metadata
from hk_equity_mcp.mcp_zero.prompt import (
    HK_MCP_ZERO_SYSTEM_PROMPT,
    mcp_zero_prompt_hash,
)
from hk_equity_mcp.real_mcp.market_gateway import BLOCKED_MARKET_OPS
from hk_equity_mcp.real_mcp.unified_catalog import build_unified_catalog

ROOT = Path(__file__).resolve().parents[1]


def test_config_models_are_experiment_not_cursor():
    cfg = load_mcp_zero_config(root=ROOT)
    assert cfg["generation_model"] == "deepseek-v4.1-flash"
    assert cfg["embedding_model"] == "text-embedding-3-small"
    assert cfg["condition"] == "mcp_zero"
    assert cfg["retrieval"]["server_top_k"] == 3
    assert cfg["retrieval"]["tool_top_k"] == 5
    # Experiment model slots must never point at Cursor coding models.
    model_slots = f"{cfg['generation_model']} {cfg['embedding_model']}".lower()
    for banned in ("claude", "sonnet", "anthropic", "cursor"):
        assert banned not in model_slots


def test_prompt_no_gold_leakage_and_stable_hash():
    prompt = HK_MCP_ZERO_SYSTEM_PROMPT
    for banned in BANNED_PROMPT_SUBSTRINGS:
        assert banned not in prompt
    for needle in ("fin_cf_", "mkt_dev_", "news_dev_", "xd_dev_", "Tencent revenue"):
        assert needle not in prompt
    assert "claude" not in prompt.lower()
    assert mcp_zero_prompt_hash() == mcp_zero_prompt_hash()
    assert len(mcp_zero_prompt_hash()) == 64


def test_blocked_operations_not_in_mcp_zero_catalog():
    bundle = build_unified_catalog(root=ROOT)
    catalog = build_hk_metadata(bundle)
    assert bundle.stats.unique_operation_ids == 42
    assert bundle.stats.runtime_exposed_unique == 39
    assert len(catalog) == 39
    for blocked in bundle.stats.blocked_operation_ids:
        assert catalog.tool(blocked) is None
    for oid in BLOCKED_MARKET_OPS:
        assert catalog.tool(oid) is None
    # Registry invariant mirrored.
    for oid, rec in bundle.records.items():
        if rec.is_blocked or not rec.runtime_exposed:
            assert catalog.tool(oid) is None


def test_dry_run_no_api(tmp_path: Path):
    out = tmp_path / "dry"
    meta = UnifiedHkMcpZeroRunner(
        output_dir=out,
        split="development",
        dry_run=True,
        allow_dirty=True,
        root=ROOT,
    ).run()
    assert meta["dry_run"] is True
    assert meta["condition"] == "mcp_zero"
    assert meta["generation_model"] == "deepseek-v4.1-flash"
    assert meta["embedding_model"] == "text-embedding-3-small"
    assert meta["server_top_k"] == 3
    assert meta["tool_top_k"] == 5
    assert meta["catalog_stats"]["unique_operation_ids"] == 42
    assert meta["catalog_stats"]["runtime_exposed_unique"] == 39
    assert meta["catalog_stats"]["mcp_zero_tool_count"] == 39
    assert meta["system_prompt_hash"] == mcp_zero_prompt_hash()
    assert (out / "run_metadata.json").exists()
    # No predictions written on dry-run.
    assert not (out / "predictions.jsonl").exists()


def test_mock_smoke_artifacts_and_no_real_api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # Guard: any accidental OpenAI-compatible client construction should fail.
    def _boom(*_a, **_k):
        raise AssertionError("real API client must not be constructed in --mock")

    monkeypatch.setattr(
        "hk_equity_mcp.experiment.unified_mcp_zero_runner.OpenAICompatibleChatClient",
        _boom,
    )
    monkeypatch.setattr(
        "hk_equity_mcp.experiment.unified_mcp_zero_runner.OpenAICompatibleEmbeddingBackend",
        _boom,
    )

    out = tmp_path / "mock_smoke"
    meta = UnifiedHkMcpZeroRunner(
        output_dir=out,
        split="development",
        smoke=True,
        max_tasks=1,
        mock=True,
        allow_dirty=True,
        root=ROOT,
    ).run()
    assert meta["mock"] is True
    assert meta["run_status"] == "completed"
    assert meta["generation_model"] == "deepseek-v4.1-flash"
    row = json.loads((out / "predictions.jsonl").read_text(encoding="utf-8").splitlines()[0])
    for field in PREDICTION_FIELDS:
        assert field in row, field
    assert row["condition"] == "mcp_zero"
    assert row.get("active_tool_request")
    assert row.get("termination_reason")
    assert "gold_operations" in row  # evaluator post-hoc only
    # Trace must not imply network MCP.
    for event in row.get("trace") or []:
        if event.get("name") == "invoke_api":
            result = event.get("result") or {}
            assert result.get("transport") in {None, "mock"}
            assert result.get("real_mcp") in {None, False}
    for name in (
        "predictions.jsonl",
        "per_task_metrics.csv",
        "manual_grading.csv",
        "cost_summary.json",
        "run_metadata.json",
        "retrieval_analysis.json",
        "error_summary.json",
        "grouped_aggregates.json",
    ):
        assert (out / name).exists(), name
    # Evaluator identity remains v1.2.
    err = json.loads((out / "error_summary.json").read_text(encoding="utf-8"))
    assert err["evaluator"]["evaluator_version"] == EVALUATOR_VERSION


def test_resume_identity(tmp_path: Path):
    out = tmp_path / "resume"
    r1 = UnifiedHkMcpZeroRunner(
        output_dir=out,
        split="development",
        smoke=True,
        max_tasks=1,
        mock=True,
        allow_dirty=True,
        root=ROOT,
    )
    meta1 = r1.run()
    preds = (out / "predictions.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(preds) == 1

    r2 = UnifiedHkMcpZeroRunner(
        output_dir=out,
        split="development",
        smoke=True,
        max_tasks=2,
        resume=True,
        mock=True,
        allow_dirty=True,
        root=ROOT,
    )
    meta2 = r2.run()
    assert meta2["question_count"] == 2
    preds2 = (out / "predictions.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(preds2) == 2

    # Identity mismatch must fail closed.
    bad = dict(meta1)
    bad["server_top_k"] = 99
    with pytest.raises(RuntimeError, match="identity mismatch"):
        assert_resume_compatible_mcp_zero(meta1, bad)
    assert "server_top_k" in MCP_ZERO_RESUME_IDENTITY_KEYS


def test_output_schema_csv_headers(tmp_path: Path):
    out = tmp_path / "schema"
    UnifiedHkMcpZeroRunner(
        output_dir=out,
        split="development",
        smoke=True,
        max_tasks=1,
        mock=True,
        allow_dirty=True,
        root=ROOT,
    ).run()
    header = (out / "manual_grading.csv").read_text(encoding="utf-8").splitlines()[0]
    for field in MANUAL_GRADING_FIELDS:
        assert field in header.split(",")
    metric_header = (out / "per_task_metrics.csv").read_text(encoding="utf-8").splitlines()[0]
    for field in PER_TASK_METRIC_FIELDS:
        assert field in metric_header.split(",")


def test_agent_never_selects_blocked_even_if_requested(tmp_path: Path):
    """Gateway fail-closed for blocked ops; catalog excludes them from retrieval."""
    from hk_equity_mcp.mcp_zero.agent import McpZeroAgent
    from hk_equity_mcp.mcp_zero.embedding import CountingEmbedder, build_embedding_table
    from hk_equity_mcp.mcp_zero.router import HierarchicalRouter
    from hk_equity_mcp.operation_resolver.retrieval import DeterministicHashEmbeddingBackend
    from hk_equity_mcp.real_mcp.unified_gateway import (
        ScriptedMockTransport,
        UnifiedRealMcpGateway,
    )
    from hk_equity_mcp.agent.llm_client import ScriptedLLMClient

    bundle = build_unified_catalog(root=ROOT)
    catalog = build_hk_metadata(bundle)
    blocked = next(iter(bundle.stats.blocked_operation_ids))
    assert catalog.tool(blocked) is None

    gateway = UnifiedRealMcpGateway(
        catalog_bundle=bundle, mock_transport=ScriptedMockTransport()
    )
    gateway.open()
    embedder = CountingEmbedder(DeterministicHashEmbeddingBackend())
    table = build_embedding_table(catalog, embedder, cache_dir=tmp_path / "emb")
    embedder.calls = 0
    embedder.tokens = 0
    router = HierarchicalRouter(catalog, table, embedder, server_top_k=3, tool_top_k=5)
    # Force invoke of blocked id without retrieval.
    llm = ScriptedLLMClient(
        [
            json.dumps(
                {
                    "thought": "bad",
                    "tool_calls": [
                        {
                            "name": "invoke_api",
                            "arguments": {
                                "operation_id": blocked,
                                "arguments": {},
                            },
                        }
                    ],
                    "final_answer": None,
                }
            ),
            json.dumps(
                {
                    "thought": "give up",
                    "tool_calls": [],
                    "final_answer": {
                        "status": "unavailable",
                        "final_answer": "blocked",
                        "reasoning_brief": "x",
                        "evidence": [],
                    },
                }
            ),
        ]
    )
    agent = McpZeroAgent(llm=llm, router=router, gateway=gateway, max_llm_turns=4)
    result = agent.run("test question with no gold")
    invoke_events = [
        e
        for e in result["trace"]
        if e.get("type") == "tool_result" and e.get("name") == "invoke_api"
    ]
    assert invoke_events
    assert invoke_events[0]["result"]["success"] is False
    assert invoke_events[0]["result"]["error_type"] == "not_retrieved"
    gateway.close()
