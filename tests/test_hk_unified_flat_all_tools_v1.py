"""Offline tests for HK-155 Flat All Tools baseline (no paid APIs / no network)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from hk_equity_mcp.agent.flat_all_tools import FlatAllToolsAgent
from hk_equity_mcp.agent.llm_client import ScriptedLLMClient
from hk_equity_mcp.experiment.unified_evaluator import (
    EVALUATOR_VERSION,
    evaluate_task,
    extract_invoke_records,
    extract_predicted_operations,
    evaluator_identity,
)
from hk_equity_mcp.experiment.unified_flat_action_mapping import (
    SAFE_ACTION_NAME_RE,
    build_flat_action_mapping,
)
from hk_equity_mcp.experiment.unified_flat_metrics import (
    RETRIEVAL_NA,
    apply_flat_retrieval_na,
    evaluate_flat_task,
)
from hk_equity_mcp.experiment.unified_flat_prompt import (
    build_flat_all_tools_system_prompt,
    unified_flat_prompt_hash,
)
from hk_equity_mcp.experiment.unified_flat_runner import (
    DEFAULT_GENERATION_MODEL,
    DEFAULT_GENERATION_PROVIDER,
    EXPECTED_GENERATION_HOST,
    RESUME_IDENTITY_KEYS,
    UnifiedHkFlatAllToolsRunner,
    assert_resume_compatible_flat,
)
from hk_equity_mcp.experiment.unified_flat_schema_bundle import (
    build_flat_schema_bundle,
    estimate_tokens,
)
from hk_equity_mcp.experiment.unified_prompt import UNIFIED_HK_OPERATION_VISIBLE_SYSTEM_PROMPT
from hk_equity_mcp.experiment.unified_schema import BANNED_PROMPT_SUBSTRINGS
from hk_equity_mcp.operation_resolver.schemas import OperationSpec, ParameterSpec
from hk_equity_mcp.real_mcp.unified_catalog import build_unified_catalog
from hk_equity_mcp.real_mcp.unified_gateway import (
    ScriptedMockTransport,
    UnifiedRealMcpGateway,
)

ROOT = Path(__file__).resolve().parents[1]

FOUR_DOMAIN_IDS = (
    "fin_cf_001",
    "mkt_dev_014",
    "news_dev_006",
    "xd_dev_007",
)


@pytest.fixture(scope="module")
def catalog_bundle():
    return build_unified_catalog(root=ROOT)


@pytest.fixture(scope="module")
def schema_bundle(catalog_bundle):
    return build_flat_schema_bundle(catalog_bundle)


def test_catalog_42_unique_39_exposed_3_blocked(catalog_bundle, schema_bundle):
    stats = catalog_bundle.stats
    assert stats.unique_operation_ids == 42
    assert stats.runtime_exposed_unique == 39
    assert stats.blocked_or_rejected_unique == 3
    assert len(schema_bundle.runtime_operation_ids) == 39
    assert len(schema_bundle.entries) == 39
    for oid in stats.blocked_operation_ids:
        assert oid not in schema_bundle.runtime_operation_ids
        assert oid not in catalog_bundle.registry
    # All 39 exposed; none blocked.
    exposed_names = {e["action_name"] for e in schema_bundle.entries}
    assert len(exposed_names) == 39
    for oid in stats.blocked_operation_ids:
        assert oid not in {e["operation_id"] for e in schema_bundle.entries}


def test_operation_id_dedup(catalog_bundle, schema_bundle):
    ids = schema_bundle.runtime_operation_ids
    assert len(ids) == len(set(ids))
    assert len(catalog_bundle.registry) == len(set(catalog_bundle.registry))


def test_action_mapping_reversible_no_collision(schema_bundle):
    mapping = schema_bundle.mapping
    assert len(mapping.action_to_operation) == 39
    assert len(mapping.operation_to_action) == 39
    for action, oid in mapping.action_to_operation.items():
        assert SAFE_ACTION_NAME_RE.match(action)
        assert mapping.operation_to_action[oid] == action
        assert mapping.resolve(action) == oid
        assert mapping.resolve(oid) == oid
    # Fingerprint stable.
    rebuilt = build_flat_action_mapping(schema_bundle.runtime_operation_ids)
    assert rebuilt.fingerprint == mapping.fingerprint
    assert rebuilt.action_to_operation == mapping.action_to_operation


def test_schema_missing_fail_closed(catalog_bundle):
    # Clone registry and strip schema usability for one op.
    broken = dict(catalog_bundle.registry)
    oid = sorted(broken.keys())[0]
    op = broken[oid]
    broken[oid] = OperationSpec(
        operation_id=op.operation_id,
        service_id=op.service_id,
        name="",
        summary="",
        description="",
        action_terms=[],
        parameters=[],
    )

    class _Bundle:
        pass

    fake = _Bundle()
    fake.stats = catalog_bundle.stats
    fake.registry = broken
    fake.records = catalog_bundle.records
    fake.source_catalogs = catalog_bundle.source_catalogs
    with pytest.raises(RuntimeError, match="fail-closed"):
        build_flat_schema_bundle(fake)  # type: ignore[arg-type]


def test_model_direct_select_dispatches_to_gateway(schema_bundle, catalog_bundle):
    oid = "aktools.stock_financial_hk_analysis_indicator_em"
    action = schema_bundle.mapping.action_name(oid)
    transport = ScriptedMockTransport()
    gateway = UnifiedRealMcpGateway(
        catalog_bundle=catalog_bundle, mock_transport=transport
    )
    gateway.open()
    prompt = build_flat_all_tools_system_prompt(schema_bundle)
    turns = [
        json.dumps(
            {
                "thought": "call directly",
                "tool_calls": [{"name": action, "arguments": {"symbol": "0700"}}],
                "final_answer": None,
            }
        ),
        json.dumps(
            {
                "thought": "done",
                "tool_calls": [],
                "final_answer": {
                    "status": "answered",
                    "final_answer": "ok",
                    "reasoning_brief": "mock",
                    "evidence": [{"operation_id": oid, "fields": ["mock"], "note": ""}],
                },
            }
        ),
    ]
    agent = FlatAllToolsAgent(
        ScriptedLLMClient(turns),
        gateway=gateway,
        schema_bundle=schema_bundle,
        system_prompt=prompt,
    )
    assert len(agent.exposed_tools()) == 39
    assert oid in agent.exposed_operation_ids()
    for blocked in schema_bundle.blocked_operation_ids:
        assert blocked not in agent.exposed_operation_ids()
    result = agent.run("What is Tencent revenue?")
    gateway.close()
    assert result["embedding_calls"] == 0
    assert result["retrieval_calls"] == 0
    assert result["resolver_calls"] == {
        "find_operations": 0,
        "describe_operation": 0,
    }
    assert result["selected_operations"] == [oid]
    assert result["invoked_operations"] == [oid]
    assert result["successful_invoked_operations"] == [oid]
    assert result["invalid_selections"] == []
    assert transport.calls[-1]["operation_id"] == oid
    # Evaluator sees original operation_id.
    preds = extract_predicted_operations(result["trace"])
    assert preds == [oid]
    invokes = extract_invoke_records(result["trace"])
    assert len(invokes) == 1
    assert invokes[0]["operation_id"] == oid
    assert invokes[0]["success"] is True


def test_selected_invoked_successful_invalid_separation(
    schema_bundle, catalog_bundle
):
    oid = "aktools.stock_financial_hk_analysis_indicator_em"
    action = schema_bundle.mapping.action_name(oid)
    gateway = UnifiedRealMcpGateway(
        catalog_bundle=catalog_bundle, mock_transport=ScriptedMockTransport()
    )
    gateway.open()
    prompt = build_flat_all_tools_system_prompt(schema_bundle)
    turns = [
        json.dumps(
            {
                "thought": "mix valid + invalid",
                "tool_calls": [
                    {"name": "totally_unknown_tool", "arguments": {}},
                    {"name": "find_operations", "arguments": {"query": "x"}},
                    {"name": action, "arguments": {"symbol": "0700"}},
                ],
                "final_answer": None,
            }
        ),
        json.dumps(
            {
                "thought": "done",
                "tool_calls": [],
                "final_answer": {
                    "status": "answered",
                    "final_answer": "ok",
                    "reasoning_brief": "mock",
                    "evidence": [],
                },
            }
        ),
    ]
    agent = FlatAllToolsAgent(
        ScriptedLLMClient(turns),
        gateway=gateway,
        schema_bundle=schema_bundle,
        system_prompt=prompt,
    )
    result = agent.run("Q?")
    gateway.close()
    assert oid in result["selected_operations"]
    assert result["invoked_operations"] == [oid]
    assert result["successful_invoked_operations"] == [oid]
    assert len(result["invalid_selections"]) >= 2
    types = {row["error_type"] for row in result["invalid_selections"]}
    assert "unknown_operation" in types
    assert "meta_tool_not_allowed" in types


def test_no_embedding_retrieval_find_describe(schema_bundle, catalog_bundle):
    prompt = build_flat_all_tools_system_prompt(schema_bundle)
    assert "find_operations" in prompt  # mentioned as forbidden
    assert "NO find_operations" in prompt or "There is NO find_operations" in prompt
    gateway = UnifiedRealMcpGateway(
        catalog_bundle=catalog_bundle, mock_transport=ScriptedMockTransport()
    )
    gateway.open()
    agent = FlatAllToolsAgent(
        ScriptedLLMClient(
            [
                json.dumps(
                    {
                        "thought": "refuse",
                        "tool_calls": [],
                        "final_answer": {
                            "status": "unavailable",
                            "final_answer": "n/a",
                            "reasoning_brief": "test",
                            "evidence": [],
                        },
                    }
                )
            ]
        ),
        gateway=gateway,
        schema_bundle=schema_bundle,
        system_prompt=prompt,
    )
    out = agent.run("Q?")
    gateway.close()
    assert out["embedding_calls"] == 0
    assert out["embedding_tokens"] == 0
    assert out["retrieval_calls"] == 0
    assert out["resolver_calls"]["find_operations"] == 0
    assert out["resolver_calls"]["describe_operation"] == 0


def test_gold_reference_not_in_prompt(schema_bundle):
    prompt = build_flat_all_tools_system_prompt(schema_bundle)
    for banned in BANNED_PROMPT_SUBSTRINGS:
        assert banned not in prompt
    for needle in (
        "fin_cf_001",
        "mkt_dev_014",
        "news_dev_006",
        "xd_dev_007",
        "reference_answer",
        "Tencent revenue",
        "gold_operations",
    ):
        assert needle not in prompt
    # Independent from Operation-Visible prompt.
    assert prompt != UNIFIED_HK_OPERATION_VISIBLE_SYSTEM_PROMPT
    h = unified_flat_prompt_hash(prompt)
    assert h == unified_flat_prompt_hash(prompt)
    assert len(h) == 64


def test_schema_bundle_hash_char_token(schema_bundle):
    assert len(schema_bundle.schema_bundle_sha256) == 64
    assert schema_bundle.serialized_schema_char_count > 0
    assert schema_bundle.token_estimate == estimate_tokens(
        schema_bundle.serialized_schema_char_count
    )
    # All 39 schemas present with complete input schema.
    for entry in schema_bundle.entries:
        assert entry["operation_id"]
        assert entry["action_name"]
        assert entry["description"]
        assert isinstance(entry["input_json_schema"], dict)
        assert entry["input_json_schema"].get("type") == "object"


def test_resume_identity_and_output_dir_guard(tmp_path: Path):
    out = tmp_path / "flat_mock"
    r1 = UnifiedHkFlatAllToolsRunner(
        output_dir=out,
        split="development",
        ids=list(FOUR_DOMAIN_IDS[:1]),
        mock=True,
        allow_dirty=True,
        root=ROOT,
    )
    meta1 = r1.run()
    assert meta1["condition"] == "flat"
    assert meta1["embedding_model"] is None
    assert meta1["retrieval_metrics"] == RETRIEVAL_NA
    assert (out / "predictions.jsonl").exists()

    # Without resume: refuse overwrite.
    with pytest.raises(RuntimeError, match="already exists"):
        UnifiedHkFlatAllToolsRunner(
            output_dir=out,
            split="development",
            ids=list(FOUR_DOMAIN_IDS[:1]),
            mock=True,
            allow_dirty=True,
            root=ROOT,
        ).run()

    # Mode mix: dry-run into mock dir refused.
    with pytest.raises(RuntimeError, match="mode mismatch"):
        UnifiedHkFlatAllToolsRunner(
            output_dir=out,
            split="development",
            dry_run=True,
            allow_dirty=True,
            root=ROOT,
        ).run()

    # Resume identity: max_tasks not in keys.
    assert "max_tasks" not in RESUME_IDENTITY_KEYS
    r2 = UnifiedHkFlatAllToolsRunner(
        output_dir=out,
        split="development",
        ids=list(FOUR_DOMAIN_IDS[:2]),
        resume=True,
        mock=True,
        allow_dirty=True,
        root=ROOT,
    )
    meta2 = r2.run()
    assert meta2["question_count"] == 2
    preds = [
        json.loads(line)
        for line in (out / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(preds) == 2

    # Tampered prompt hash fails resume.
    existing = json.loads((out / "run_metadata.json").read_text(encoding="utf-8"))
    current = dict(existing)
    current["system_prompt_hash"] = "0" * 64
    with pytest.raises(RuntimeError, match="identity mismatch"):
        assert_resume_compatible_flat(existing, current)


def test_four_domain_mock(tmp_path: Path):
    out = tmp_path / "flat_four_domain"
    runner = UnifiedHkFlatAllToolsRunner(
        output_dir=out,
        split="development",
        ids=list(FOUR_DOMAIN_IDS),
        mock=True,
        allow_dirty=True,
        root=ROOT,
    )
    meta = runner.run()
    assert meta["question_count"] == 4
    assert set(meta["domain_counts"]) >= {"financial", "market", "news", "cross_domain"}
    preds = [
        json.loads(line)
        for line in (out / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [p["task_id"] for p in preds] == list(FOUR_DOMAIN_IDS)
    for p in preds:
        assert p["condition"] == "flat"
        assert p["embedding_calls"] == 0
        assert p["retrieval_calls"] == 0
        assert p["resolver_calls"]["find_operations"] == 0
        assert p["resolver_calls"]["describe_operation"] == 0
        assert p["retrieval_metrics"] == RETRIEVAL_NA
        assert p["retrieval_metrics"] != 1.0
    for name in (
        "predictions.jsonl",
        "per_task_metrics.csv",
        "manual_grading.csv",
        "cost_summary.json",
        "run_metadata.json",
        "error_summary.json",
        "grouped_aggregates.json",
        "retrieval_analysis.json",
        "evaluation_metadata.json",
    ):
        assert (out / name).exists(), name
    cost = json.loads((out / "cost_summary.json").read_text(encoding="utf-8"))
    assert cost["total_embedding_calls"] == 0
    assert cost["total_retrieval_calls"] == 0
    assert cost["total_find_operations"] == 0


def test_evaluator_v1_2_compatible_with_ov_and_flat():
    assert EVALUATOR_VERSION == "hk_unified_evaluator_v1.2"
    # OV-shaped invoke_api trace still works.
    ov_trace = [
        {
            "type": "tool_result",
            "name": "invoke_api",
            "arguments": {
                "operation_id": "aktools.stock_financial_hk_analysis_indicator_em",
                "arguments": {"symbol": "0700"},
            },
            "result": {
                "success": True,
                "result": {"protocol_success": True, "execution_success": True},
            },
        }
    ]
    assert extract_predicted_operations(ov_trace) == [
        "aktools.stock_financial_hk_analysis_indicator_em"
    ]
    # Flat-shaped direct call.
    flat_trace = [
        {
            "type": "tool_result",
            "name": "aktools_stock_financial_hk_analysis_indicator_em",
            "operation_id": "aktools.stock_financial_hk_analysis_indicator_em",
            "arguments": {"symbol": "0700"},
            "result": {
                "success": True,
                "operation_id": "aktools.stock_financial_hk_analysis_indicator_em",
                "result": {"protocol_success": True, "execution_success": True},
            },
        }
    ]
    assert extract_predicted_operations(flat_trace) == [
        "aktools.stock_financial_hk_analysis_indicator_em"
    ]
    item = {
        "gold_operations": ["aktools.stock_financial_hk_analysis_indicator_em"],
        "gold_status": "answered",
        "answerability_class": "structured_answerable",
    }
    metrics = evaluate_task(
        item=item,
        result={
            "trace": flat_trace,
            "predicted_status": "answered",
            "final_answer": "x",
            "status": "full",
            "tokens": {},
        },
    )
    assert metrics["exact_tool_set"] is True
    flat_metrics = apply_flat_retrieval_na(metrics)
    assert flat_metrics["retrieval_metrics"] == RETRIEVAL_NA
    assert flat_metrics["retrieved_candidates"] == RETRIEVAL_NA
    assert flat_metrics["retrieval_metrics"] != 1.0
    assert flat_metrics["exact_tool_set"] is True  # tool metrics still apply
    assert evaluator_identity()["evaluator_version"] == EVALUATOR_VERSION


def test_dry_run_exploratory(tmp_path: Path):
    out = tmp_path / "flat_dry"
    meta = UnifiedHkFlatAllToolsRunner(
        output_dir=out,
        split="exploratory_all",
        confirm_retire_heldout=True,
        dry_run=True,
        allow_dirty=True,
        root=ROOT,
    ).run()
    assert meta["dry_run"] is True
    assert meta["question_count"] == 155
    assert meta["catalog_stats"]["runtime_exposed_unique"] == 39
    assert meta["catalog_stats"]["unique_operation_ids"] == 42
    assert meta["catalog_stats"]["blocked_or_rejected_unique"] == 3
    assert meta["embedding_model"] is None
    assert meta["retrieval_metrics"] == RETRIEVAL_NA
    assert (out / "run_metadata.json").exists()
    # No predictions for dry-run.
    assert not (out / "predictions.jsonl").exists()


def test_schema_invalid_fail_closed(schema_bundle, catalog_bundle):
    oid = "aktools.stock_financial_hk_analysis_indicator_em"
    action = schema_bundle.mapping.action_name(oid)
    gateway = UnifiedRealMcpGateway(
        catalog_bundle=catalog_bundle, mock_transport=ScriptedMockTransport()
    )
    gateway.open()
    prompt = build_flat_all_tools_system_prompt(schema_bundle)
    agent = FlatAllToolsAgent(
        ScriptedLLMClient(
            [
                json.dumps(
                    {
                        "thought": "bad args",
                        "tool_calls": [
                            {"name": action, "arguments": {"not_a_real_param": 1}}
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
                            "final_answer": "invalid",
                            "reasoning_brief": "schema",
                            "evidence": [],
                        },
                    }
                ),
            ]
        ),
        gateway=gateway,
        schema_bundle=schema_bundle,
        system_prompt=prompt,
    )
    result = agent.run("Q?")
    gateway.close()
    assert result["invoked_operations"] == []
    assert result["successful_invoked_operations"] == []
    assert any(
        row.get("error_type") == "validation_error"
        for row in result["invalid_selections"]
    )


def _write_gen_env(path: Path, *, provider: str, model: str, host: str) -> Path:
    path.write_text(
        "\n".join(
            [
                f"LLM_PROVIDER={provider}",
                f"LLM_BASE_URL=https://{host}/v1",
                "LLM_API_KEY=test-key-not-for-network",
                f"LLM_MODEL={model}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


def test_provider_defaults_are_openai_compatible_relay():
    assert DEFAULT_GENERATION_PROVIDER == "openai_compatible"
    assert DEFAULT_GENERATION_MODEL == "deepseek-v4.1-flash"
    assert EXPECTED_GENERATION_HOST == "api.zhizengzeng.com"
    for key in (
        "generation_provider",
        "generation_base_url_host",
        "provider_host",
        "generation_model",
        "system_prompt_hash",
        "schema_bundle_sha256",
        "action_mapping_fingerprint",
        "catalog_fingerprint",
        "exploratory_manifest_hash",
        "run_order_hash",
        "temperature",
        "max_tokens",
        "max_llm_turns",
    ):
        assert key in RESUME_IDENTITY_KEYS


def test_openai_compatible_correct_host_model_accepted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    for key in ("LLM_PROVIDER", "LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL", "ENV_FILE"):
        monkeypatch.delenv(key, raising=False)
    env = _write_gen_env(
        tmp_path / ".env.relay-v41",
        provider="openai_compatible",
        model="deepseek-v4.1-flash",
        host="api.zhizengzeng.com",
    )
    meta = UnifiedHkFlatAllToolsRunner(
        output_dir=tmp_path / "relay_ok",
        split="development",
        dry_run=True,
        allow_dirty=True,
        env_file=str(env),
        root=ROOT,
    ).run()
    assert meta["generation_provider"] == "openai_compatible"
    assert meta["generation_model"] == "deepseek-v4.1-flash"
    assert meta["generation_base_url_host"] == "api.zhizengzeng.com"
    assert meta["provider_host"] == "api.zhizengzeng.com"
    assert meta["expected_generation_host"] == "api.zhizengzeng.com"
    blob = json.dumps(meta, ensure_ascii=False)
    assert "test-key-not-for-network" not in blob
    assert "API_KEY" not in blob


def test_aliyun_provider_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    for key in ("LLM_PROVIDER", "LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL", "ENV_FILE"):
        monkeypatch.delenv(key, raising=False)
    env = _write_gen_env(
        tmp_path / ".env.aliyun",
        provider="aliyun",
        model="deepseek-v4.1-flash",
        host="token-plan.cn-beijing.maas.aliyuncs.com",
    )
    with pytest.raises(RuntimeError, match="provider mismatch"):
        UnifiedHkFlatAllToolsRunner(
            output_dir=tmp_path / "aliyun_reject",
            split="development",
            dry_run=True,
            allow_dirty=True,
            env_file=str(env),
            root=ROOT,
        ).run()


def test_wrong_host_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    for key in ("LLM_PROVIDER", "LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL", "ENV_FILE"):
        monkeypatch.delenv(key, raising=False)
    env = _write_gen_env(
        tmp_path / ".env.bad-host",
        provider="openai_compatible",
        model="deepseek-v4.1-flash",
        host="token-plan.cn-beijing.maas.aliyuncs.com",
    )
    with pytest.raises(RuntimeError, match="host mismatch"):
        UnifiedHkFlatAllToolsRunner(
            output_dir=tmp_path / "bad_host",
            split="development",
            dry_run=True,
            allow_dirty=True,
            env_file=str(env),
            root=ROOT,
        ).run()


def test_wrong_model_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    for key in ("LLM_PROVIDER", "LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL", "ENV_FILE"):
        monkeypatch.delenv(key, raising=False)
    env = _write_gen_env(
        tmp_path / ".env.bad-model",
        provider="openai_compatible",
        model="deepseek-flash",
        host="api.zhizengzeng.com",
    )
    with pytest.raises(RuntimeError, match="model mismatch"):
        UnifiedHkFlatAllToolsRunner(
            output_dir=tmp_path / "bad_model",
            split="development",
            dry_run=True,
            allow_dirty=True,
            env_file=str(env),
            root=ROOT,
        ).run()


def test_provider_host_model_in_resume_identity():
    base = {
        "condition": "flat",
        "execution_backend": "real_mcp",
        "generation_model": "deepseek-v4.1-flash",
        "generation_provider": "openai_compatible",
        "generation_base_url_host": "api.zhizengzeng.com",
        "provider_host": "api.zhizengzeng.com",
        "dataset_split": "exploratory_all",
        "system_prompt_hash": "a" * 64,
        "schema_bundle_sha256": "b" * 64,
        "action_mapping_fingerprint": "c" * 64,
        "catalog_fingerprint": "fp",
        "temperature": 0.0,
        "max_tokens": 1024,
        "max_llm_turns": 8,
        "exploratory_manifest_hash": "mh",
        "run_order_hash": "rh",
        "retirement_policy_version": "v1",
        "max_tasks": 20,
    }
    ok = dict(base)
    ok["max_tasks"] = 50  # not part of resume identity
    assert_resume_compatible_flat(base, ok)

    bad_provider = dict(base)
    bad_provider["generation_provider"] = "aliyun"
    with pytest.raises(RuntimeError, match="generation_provider"):
        assert_resume_compatible_flat(base, bad_provider)

    bad_host = dict(base)
    bad_host["generation_base_url_host"] = "token-plan.cn-beijing.maas.aliyuncs.com"
    bad_host["provider_host"] = "token-plan.cn-beijing.maas.aliyuncs.com"
    with pytest.raises(RuntimeError, match="generation_base_url_host|provider_host"):
        assert_resume_compatible_flat(base, bad_host)

    bad_model = dict(base)
    bad_model["generation_model"] = "other-model"
    with pytest.raises(RuntimeError, match="generation_model"):
        assert_resume_compatible_flat(base, bad_model)
