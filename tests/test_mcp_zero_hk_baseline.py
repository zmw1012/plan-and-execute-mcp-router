"""Offline HK-155 MCP-Zero baseline tests (no paid / network APIs)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hk_equity_mcp.experiment.unified_evaluator import EVALUATOR_VERSION
from hk_equity_mcp.experiment.unified_mcp_zero_runner import (
    MAX_TASKS_SEMANTICS,
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
    assert cfg["generation_provider"] == "openai_compatible"
    assert cfg["generation_base_url_host"] == "api.zhizengzeng.com"
    assert cfg["condition"] == "mcp_zero"
    assert cfg["retrieval"]["server_top_k"] == 3
    assert cfg["retrieval"]["tool_top_k"] == 5
    assert "aliyun" not in json.dumps(cfg).lower()
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


def test_method_identity_is_frozen_before_results():
    from hk_equity_mcp.mcp_zero import (
        ABLATION_POLICY,
        OFFICIAL_TOP_SERVERS,
        OFFICIAL_TOP_TOOLS,
        assert_frozen_config,
        frozen_method_identity,
        frozen_runtime_env_identity,
        method_identity_sha256,
    )

    ident = frozen_method_identity()
    assert ident["server_top_k"] == 3
    assert ident["tool_top_k"] == 5
    assert ident["paper_official_server_top_k"] == OFFICIAL_TOP_SERVERS == 5
    assert ident["paper_official_tool_top_k"] == OFFICIAL_TOP_TOOLS == 3
    assert ident["topk_provenance"] == "etom_pinned_adaptation"
    assert ident["frozen_before_hk155_real_results"] is True
    assert "must not replace" in ABLATION_POLICY
    assert ident["generation_model"] == "deepseek-v4.1-flash"
    assert ident["embedding_model"] == "text-embedding-3-small"
    assert ident["temperature"] == 0.0
    assert ident["max_tokens"] == 1024
    assert ident["max_llm_turns"] == 8
    assert ident["evaluator"] == EVALUATOR_VERSION == "hk_unified_evaluator_v1.2"
    assert ident["tie_break"] == ["-score", "server_id", "tool_name", "operation_id"]
    # Provider/host are runtime env metadata, not routing method-card fields.
    assert "generation_provider" not in ident
    assert "generation_base_url_host" not in ident
    assert method_identity_sha256() == (
        "77cc3b07a4c2b9af00ad3f6bfb7cba59c82084d94b7592f2c820f46f1d3e8d3f"
    )
    runtime = frozen_runtime_env_identity()
    assert runtime["generation_provider"] == "openai_compatible"
    assert runtime["generation_base_url_host"] == "api.zhizengzeng.com"
    assert_frozen_config(load_mcp_zero_config(root=ROOT))
    fidelity = (ROOT / "docs" / "mcp_zero_fidelity_audit.md").read_text(encoding="utf-8")
    assert "Adapted default" in fidelity
    assert "not the paper's original exact configuration" in fidelity
    assert "independent ablation" in fidelity
    assert "Official exact configuration: server_top_k=3" not in fidelity


def test_generation_provider_profile_gate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Relay openai_compatible profile passes; aliyun / model mismatch refuse.

    Never prints or asserts on raw API key values.
    """
    from hk_equity_mcp.common import env_loader

    relay = tmp_path / ".env.relay-v41"
    relay.write_text(
        "\n".join(
            [
                "LLM_PROVIDER=openai_compatible",
                "LLM_BASE_URL=https://api.zhizengzeng.com/v1",
                "LLM_MODEL=deepseek-v4.1-flash",
                "LLM_API_KEY=test-key-not-for-live-use",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    aliyun = tmp_path / ".env.aliyun-mismatch"
    aliyun.write_text(
        "\n".join(
            [
                "LLM_PROVIDER=aliyun",
                "LLM_BASE_URL=https://token-plan.cn-beijing.maas.aliyuncs.com/v1",
                "LLM_MODEL=deepseek-v4.1-flash",
                "LLM_API_KEY=test-key-not-for-live-use",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    bad_model = tmp_path / ".env.bad-model"
    bad_model.write_text(
        "\n".join(
            [
                "LLM_PROVIDER=openai_compatible",
                "LLM_BASE_URL=https://api.zhizengzeng.com/v1",
                "LLM_MODEL=deepseek-flash",
                "LLM_API_KEY=test-key-not-for-live-use",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    emb = tmp_path / ".env.embedding-relay"
    emb.write_text(
        "\n".join(
            [
                "EMBEDDING_BASE_URL=https://api.zhizengzeng.com/v1",
                "EMBEDDING_MODEL=text-embedding-3-small",
                "EMBEDDING_API_KEY=test-embedding-key-not-for-live-use",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    def _clear_gen_env():
        for key in (
            "LLM_PROVIDER",
            "LLM_BASE_URL",
            "LLM_MODEL",
            "LLM_API_KEY",
            "EMBEDDING_BASE_URL",
            "EMBEDDING_MODEL",
            "EMBEDDING_API_KEY",
        ):
            monkeypatch.delenv(key, raising=False)

    _clear_gen_env()
    meta = env_loader.load_generation_profile(ROOT, str(relay))
    assert meta["provider"] == "openai_compatible"
    assert meta["model"] == "deepseek-v4.1-flash"
    assert "api_key" not in meta
    assert "test-key" not in json.dumps(meta)

    _clear_gen_env()
    ok = UnifiedHkMcpZeroRunner(
        output_dir=tmp_path / "relay_dry",
        split="development",
        dry_run=True,
        allow_dirty=True,
        env_file=str(relay),
        embedding_env_file=str(emb),
        root=ROOT,
    ).run()
    assert ok["generation_provider"] == "openai_compatible"
    assert ok["generation_base_url_host"] == "api.zhizengzeng.com"
    assert ok["generation_model"] == "deepseek-v4.1-flash"
    assert ok["method_identity_sha256"] == (
        "77cc3b07a4c2b9af00ad3f6bfb7cba59c82084d94b7592f2c820f46f1d3e8d3f"
    )
    assert ok["server_top_k"] == 3
    assert ok["tool_top_k"] == 5
    dumped = json.dumps(ok)
    assert "test-key" not in dumped
    assert "test-key-not-for-live-use" not in dumped
    assert "LLM_API_KEY" not in dumped

    _clear_gen_env()
    with pytest.raises(RuntimeError, match="generation provider mismatch"):
        UnifiedHkMcpZeroRunner(
            output_dir=tmp_path / "aliyun_dry",
            split="development",
            dry_run=True,
            allow_dirty=True,
            env_file=str(aliyun),
            embedding_env_file=str(emb),
            root=ROOT,
        ).run()

    _clear_gen_env()
    with pytest.raises(RuntimeError, match="generation model mismatch"):
        UnifiedHkMcpZeroRunner(
            output_dir=tmp_path / "model_dry",
            split="development",
            dry_run=True,
            allow_dirty=True,
            env_file=str(bad_model),
            embedding_env_file=str(emb),
            root=ROOT,
        ).run()

    assert "generation_provider" in MCP_ZERO_RESUME_IDENTITY_KEYS
    assert "generation_base_url_host" in MCP_ZERO_RESUME_IDENTITY_KEYS
    assert "generation_model" in MCP_ZERO_RESUME_IDENTITY_KEYS


def test_embedding_cache_identity_and_namespace(tmp_path: Path):
    from hk_equity_mcp.mcp_zero.embedding import (
        CACHE_NAMESPACE,
        CacheNamespaceError,
        IncompleteEmbeddingTableError,
        build_embedding_table,
        embedding_cache_identity,
        load_complete_table,
        table_paths,
    )
    from hk_equity_mcp.mcp_zero.embedding import CountingEmbedder
    from hk_equity_mcp.operation_resolver.retrieval import DeterministicHashEmbeddingBackend

    bundle = build_unified_catalog(root=ROOT)
    catalog = build_hk_metadata(bundle)
    metadata_hash = catalog.fingerprint()
    catalog_fp = "unified-catalog-fingerprint"
    base = dict(
        embedding_model="text-embedding-3-small",
        catalog_fingerprint=catalog_fp,
        metadata_hash=metadata_hash,
        score_mode="official_combined",
        server_top_k=3,
        tool_top_k=5,
        similarity_threshold=0.0,
        use_summary=True,
    )
    ident = embedding_cache_identity(**base)
    other = embedding_cache_identity(**{**base, "server_top_k": 5, "tool_top_k": 3})
    cache = tmp_path / CACHE_NAMESPACE
    path_a, _, digest_a = table_paths(cache, ident)
    path_b, _, digest_b = table_paths(cache, other)
    assert path_a != path_b
    assert digest_a != digest_b
    assert ident["namespace"] == CACHE_NAMESPACE
    assert "operation_embeddings" not in str(path_a)
    with pytest.raises(CacheNamespaceError):
        table_paths(tmp_path / "operation_embeddings", ident)

    embedder = CountingEmbedder(
        DeterministicHashEmbeddingBackend(), model_name="text-embedding-3-small"
    )
    missing = tmp_path / "missing"
    with pytest.raises(IncompleteEmbeddingTableError):
        load_complete_table(
            catalog,
            model_name="text-embedding-3-small",
            cache_dir=missing,
            use_summary=True,
            catalog_fingerprint=catalog_fp,
        )

    table = build_embedding_table(
        catalog,
        embedder,
        cache_dir=cache,
        use_summary=True,
        catalog_fingerprint=catalog_fp,
        batch_size=8,
    )
    assert table.complete is True
    assert table.cache_reused is False
    assert table.index_action == "build"
    assert table.build_calls > 0
    assert table.build_tokens > 0
    assert table.build_latency_ms >= 0
    first_calls = table.build_calls

    reused = build_embedding_table(
        catalog,
        embedder,
        cache_dir=cache,
        use_summary=True,
        catalog_fingerprint=catalog_fp,
        batch_size=8,
    )
    assert reused.cache_reused is True
    assert reused.index_action == "reuse"
    assert reused.build_calls == 0
    assert embedder.calls == first_calls

    # Partial resume must not re-embed keys already checkpointed.
    partial_dir = tmp_path / "partial_case"
    embedder2 = CountingEmbedder(
        DeterministicHashEmbeddingBackend(), model_name="text-embedding-3-small"
    )
    from hk_equity_mcp.mcp_zero.embedding import _units

    units = _units(catalog, True)
    final, partial, _ = table_paths(partial_dir, ident)
    partial_dir.mkdir()
    kept = units[0]
    vec = embedder2.embed([kept[1]])[0]
    partial.write_text(
        json.dumps({"k": kept[0], "v": vec}) + "\n{not-json",
        encoding="utf-8",
    )
    embedder2.calls = 0
    embedder2.tokens = 0
    resumed = build_embedding_table(
        catalog,
        embedder2,
        cache_dir=partial_dir,
        use_summary=True,
        catalog_fingerprint=catalog_fp,
        batch_size=1000,
    )
    assert resumed.index_action == "resume_partial"
    assert resumed.complete is True
    assert resumed.build_calls == 1
    assert final.exists()
    assert not partial.exists()

    foreign = tmp_path / "foreign"
    foreign.mkdir()
    foreign_final, _, _ = table_paths(foreign, ident)
    foreign_final.write_text(
        json.dumps({"namespace": "etom_catalog", "complete": True, "vectors": {}}),
        encoding="utf-8",
    )
    with pytest.raises(CacheNamespaceError):
        build_embedding_table(
            catalog,
            CountingEmbedder(
                DeterministicHashEmbeddingBackend(),
                model_name="text-embedding-3-small",
            ),
            cache_dir=foreign,
            catalog_fingerprint=catalog_fp,
        )


def test_resume_prefix_and_max_tasks_semantics(tmp_path: Path):
    assert "max_tasks" not in MCP_ZERO_RESUME_IDENTITY_KEYS
    assert "max_tasks_semantics" in MCP_ZERO_RESUME_IDENTITY_KEYS
    out = tmp_path / "prefix"
    meta = UnifiedHkMcpZeroRunner(
        output_dir=out,
        split="development",
        smoke=True,
        max_tasks=1,
        mock=True,
        allow_dirty=True,
        root=ROOT,
    ).run()
    assert meta["max_tasks_semantics"] == MAX_TASKS_SEMANTICS
    assert meta["max_tasks_in_resume_identity"] is False
    assert meta["max_tasks"] == 1
    index = meta["embedding_index"]
    assert index["complete"] is True
    assert index["namespace"] == "mcp_zero_hk_v1"
    assert index["cache_reused"] is False
    assert "operation_embeddings" not in index["cache_path"]

    extended = dict(meta)
    extended["question_ids"] = list(meta["question_ids"]) + ["later-task"]
    extended["max_tasks"] = 2
    assert_resume_compatible_mcp_zero(meta, extended)

    shrunk = dict(meta)
    shrunk["question_ids"] = []
    with pytest.raises(RuntimeError, match="cumulative prefix"):
        assert_resume_compatible_mcp_zero(meta, shrunk)

    meta2 = UnifiedHkMcpZeroRunner(
        output_dir=out,
        split="development",
        smoke=True,
        max_tasks=2,
        resume=True,
        mock=True,
        allow_dirty=True,
        root=ROOT,
    ).run()
    assert meta2["question_count"] == 2
    assert meta2["embedding_index"]["cache_reused"] is True
    assert meta2["embedding_index"]["index_action"] == "reuse"
    preds = (out / "predictions.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(preds) == 2
    cost = json.loads((out / "cost_summary.json").read_text(encoding="utf-8"))
    assert "embedding_index_cache_reused" in cost
    assert cost["embedding_index_cache_reused"] is True
