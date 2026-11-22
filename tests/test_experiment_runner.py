"""Tests for env profiles, dry-run, and unified experiment conditions."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

import pytest

from hk_equity_mcp.agent.flat import FlatAgent
from hk_equity_mcp.agent.llm_client import ScriptedLLMClient
from hk_equity_mcp.agent.mcp_free import MCPFreeAgent
from hk_equity_mcp.common.env_loader import (
    load_embedding_profile,
    load_env_file,
    load_generation_profile,
)
from hk_equity_mcp.experiment.runner import FinancialExperimentRunner, build_prediction_row
from hk_equity_mcp.experiment.schema import MANUAL_GRADING_FIELDS, PREDICTION_FIELDS
from hk_equity_mcp.operation_resolver.retrieval import OpenAICompatibleEmbeddingBackend


def _write_env(path: Path, mapping: Dict[str, str]) -> None:
    path.write_text(
        "\n".join(f"{k}={v}" for k, v in mapping.items()) + "\n",
        encoding="utf-8",
    )


def _clear_gen_env(monkeypatch):
    for key in (
        "LLM_PROVIDER",
        "LLM_BASE_URL",
        "LLM_API_KEY",
        "LLM_MODEL",
        "EMBEDDING_BASE_URL",
        "EMBEDDING_API_KEY",
        "EMBEDDING_MODEL",
        "ENV_FILE",
        "EMBEDDING_ENV_FILE",
    ):
        monkeypatch.delenv(key, raising=False)


def test_dual_env_profile_loader_shell_wins(tmp_path: Path, monkeypatch):
    _clear_gen_env(monkeypatch)
    gen = tmp_path / ".env.aliyun"
    emb = tmp_path / ".env.embedding-relay"
    _write_env(
        gen,
        {
            "LLM_PROVIDER": "from-file",
            "LLM_BASE_URL": "https://gen.example/v1",
            "LLM_API_KEY": "file-gen-key",
            "LLM_MODEL": "deepseek-v4.1-flash",
        },
    )
    _write_env(
        emb,
        {
            "EMBEDDING_BASE_URL": "https://emb.example/v1",
            "EMBEDDING_API_KEY": "file-emb-key",
            "EMBEDDING_MODEL": "text-embedding-3-small",
        },
    )
    monkeypatch.setenv("LLM_PROVIDER", "from-shell")
    meta = load_generation_profile(tmp_path, str(gen))
    assert meta["provider"] == "from-shell"
    assert meta["model"] == "deepseek-v4.1-flash"
    assert "api_key" not in meta
    emb_meta = load_embedding_profile(tmp_path, str(emb))
    assert emb_meta["model"] == "text-embedding-3-small"
    assert "api_key" not in emb_meta


def test_load_env_file_missing_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_env_file(tmp_path / "missing.env")


def test_operation_visible_missing_embedding_config_errors(monkeypatch, tmp_path: Path):
    _clear_gen_env(monkeypatch)
    gen = tmp_path / ".env.aliyun"
    _write_env(
        gen,
        {
            "LLM_PROVIDER": "aliyun",
            "LLM_BASE_URL": "https://gen.example/v1",
            "LLM_API_KEY": "gen-key",
            "LLM_MODEL": "deepseek-v4.1-flash",
        },
    )
    load_generation_profile(tmp_path, str(gen))
    with pytest.raises(RuntimeError, match="EMBEDDING_BASE_URL"):
        OpenAICompatibleEmbeddingBackend()


def test_mcp_free_and_flat_do_not_init_embedding(monkeypatch):
    # Ensure constructing these agents never requires embedding env.
    _clear_gen_env(monkeypatch)
    llm = ScriptedLLMClient(
        [
            json.dumps(
                {
                    "status": "answered",
                    "final_answer": "x",
                    "reasoning_brief": "y",
                }
            )
        ]
    )
    free = MCPFreeAgent(llm)
    out = free.run("Q?")
    assert out["provider_metadata"]["embedding_model"] is None
    assert out["underlying_operation_calls"] == 0
    assert out["resolver_calls"] == {
        "find_operations": 0,
        "describe_operation": 0,
    }

    flat_llm = ScriptedLLMClient(
        [
            json.dumps(
                {
                    "thought": "done",
                    "tool_calls": [],
                    "final_answer": {
                        "status": "unavailable",
                        "final_answer": "no tools used",
                        "reasoning_brief": "",
                    },
                }
            )
        ]
    )
    flat = FlatAgent(flat_llm)
    flat_out = flat.run("Q?")
    assert flat_out["provider_metadata"]["embedding_model"] is None
    assert flat_out["exposed_tools"] == [
        "get_financial_metric",
        "get_filing_summary",
    ]


def test_prediction_schema_consistent_across_conditions():
    base = {
        "final_answer": "a",
        "predicted_status": "answered",
        "trace": [],
        "llm_calls": 1,
        "resolver_calls": {"find_operations": 0, "describe_operation": 0},
        "underlying_operation_calls": 0,
        "tokens": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        "latency_ms": 1.0,
        "provider_metadata": {
            "llm_provider": "mock",
            "llm_model": "mock",
            "embedding_model": None,
        },
        "error": None,
    }
    item = {"id": "q1", "question": "Q"}
    rows = [
        build_prediction_row(item=item, condition=cond, result=dict(base))
        for cond in ("mcp_free", "flat", "operation_visible")
    ]
    for row in rows:
        assert set(PREDICTION_FIELDS) <= set(row.keys())
        blob = json.dumps(row)
        assert "api_key" not in blob.lower()
        assert "gen-key" not in blob
    assert set(MANUAL_GRADING_FIELDS) == {
        "id",
        "condition",
        "company",
        "task_type",
        "question",
        "gold_status",
        "reference_answer",
        "predicted_status",
        "predicted_answer",
        "correct_0_or_1",
        "error_type",
        "notes",
    }


def test_dry_run_no_network(tmp_path: Path, monkeypatch):
    _clear_gen_env(monkeypatch)
    gen = tmp_path / ".env.aliyun"
    emb = tmp_path / ".env.embedding-relay"
    _write_env(
        gen,
        {
            "LLM_PROVIDER": "aliyun",
            "LLM_BASE_URL": "https://gen.example/v1",
            "LLM_API_KEY": "secret-gen-key-should-not-appear",
            "LLM_MODEL": "deepseek-v4.1-flash",
        },
    )
    _write_env(
        emb,
        {
            "EMBEDDING_BASE_URL": "https://emb.example/v1",
            "EMBEDDING_API_KEY": "secret-emb-key-should-not-appear",
            "EMBEDDING_MODEL": "text-embedding-3-small",
        },
    )
    questions = tmp_path / "q.jsonl"
    questions.write_text(
        json.dumps(
            {
                "id": "fin_cf_001",
                "company": "Tencent",
                "task_type": "direct_retrieval",
                "question": "What was revenue?",
                "gold_status": "answered",
                "reference_answer": "hidden from model",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "out"

    # Patch urllib to fail if anything tries network.
    import urllib.request

    def _blocked(*args, **kwargs):
        raise AssertionError("dry-run must not access network")

    monkeypatch.setattr(urllib.request, "urlopen", _blocked)

    for condition in ("mcp_free", "flat", "operation_visible"):
        runner = FinancialExperimentRunner(
            condition=condition,
            questions_path=questions,
            output_dir=out_dir / condition,
            dry_run=True,
            env_file=str(gen),
            embedding_env_file=str(emb),
            root=tmp_path,
        )
        meta = runner.run()
        assert meta["dry_run"] is True
        assert meta["generation_model"] == "deepseek-v4.1-flash"
        if condition == "operation_visible":
            assert meta["embedding_model"] == "text-embedding-3-small"
            assert meta["catalog_size"] == 2
        else:
            assert meta["embedding_model"] is None
        blob = json.dumps(meta)
        assert "secret-gen-key" not in blob
        assert "secret-emb-key" not in blob
        assert "api_key" not in blob.lower()


def test_mcp_free_does_not_see_reference_answer():
    captured: List[List[Dict[str, str]]] = []

    class CaptureLLM(ScriptedLLMClient):
        def complete(self, messages, *, temperature=0.0, max_tokens=1024):
            captured.append(messages)
            return super().complete(
                messages, temperature=temperature, max_tokens=max_tokens
            )

    llm = CaptureLLM(
        [
            json.dumps(
                {
                    "status": "answered",
                    "final_answer": "guess",
                    "reasoning_brief": "",
                }
            )
        ]
    )
    MCPFreeAgent(llm).run("Public question only")
    blob = json.dumps(captured)
    assert "reference_answer" not in blob
    assert "gold" not in blob.lower()


def test_flat_prompt_includes_full_contracts():
    from hk_equity_mcp.experiment.prompts import build_flat_system_prompt

    prompt = build_flat_system_prompt()
    assert "metric_type" in prompt
    assert '"reported"' in prompt and '"adjusted"' in prompt
    assert "period" in prompt
    assert "report_type" in prompt
    assert "annual" in prompt and "interim" in prompt
    assert "constraints:" in prompt
    assert "Ambiguous period" in prompt or "Do not mix currencies" in prompt
    assert "invocation_guidance:" in prompt
    assert "get_financial_metric" in prompt
    assert "get_filing_summary" in prompt
    # Flat still documents direct tool names, not meta-tools as the only API.
    assert "find_operations" not in prompt.split("Full tool contracts")[0]


def test_select_ids_keeps_benchmark_order_and_validates():
    from hk_equity_mcp.experiment.runner import select_questions_by_ids

    items = [
        {"id": "fin_cf_001", "question": "a"},
        {"id": "fin_cf_017", "question": "b"},
        {"id": "fin_cf_021", "question": "c"},
        {"id": "fin_cf_030", "question": "d"},
    ]
    selected = select_questions_by_ids(
        items, ["fin_cf_021", "fin_cf_001", "fin_cf_017"]
    )
    assert [x["id"] for x in selected] == [
        "fin_cf_001",
        "fin_cf_017",
        "fin_cf_021",
    ]
    with pytest.raises(ValueError, match="Unknown question id"):
        select_questions_by_ids(items, ["fin_cf_001", "missing"])


def test_resume_identity_mismatch_refused():
    from hk_equity_mcp.experiment.runner import assert_resume_compatible

    existing = {
        "condition": "flat",
        "generation_model": "deepseek-v4.1-flash",
        "questions_file": "/q.jsonl",
        "system_prompt_hash": "aaa",
        "temperature": 0.0,
        "max_tokens": 1024,
    }
    current = dict(existing)
    current["system_prompt_hash"] = "bbb"
    with pytest.raises(RuntimeError, match="Resume refused"):
        assert_resume_compatible(existing, current)


def test_paid_run_refuses_dirty_tree(tmp_path: Path, monkeypatch):
    _clear_gen_env(monkeypatch)
    gen = tmp_path / ".env.aliyun"
    _write_env(
        gen,
        {
            "LLM_PROVIDER": "aliyun",
            "LLM_BASE_URL": "https://token-plan.cn-beijing.maas.aliyuncs.com/v1",
            "LLM_API_KEY": "gen-key",
            "LLM_MODEL": "deepseek-v4.1-flash",
        },
    )
    questions = tmp_path / "q.jsonl"
    questions.write_text(
        json.dumps({"id": "fin_cf_001", "question": "Q"}) + "\n", encoding="utf-8"
    )
    monkeypatch.setattr(
        "hk_equity_mcp.experiment.runner.git_dirty", lambda root: True
    )
    runner = FinancialExperimentRunner(
        condition="mcp_free",
        questions_path=questions,
        output_dir=tmp_path / "out",
        dry_run=False,
        allow_dirty=False,
        env_file=str(gen),
        root=tmp_path,
    )
    with pytest.raises(RuntimeError, match="dirty"):
        runner.run()


def test_dry_run_allows_dirty_and_records_hosts(tmp_path: Path, monkeypatch):
    _clear_gen_env(monkeypatch)
    gen = tmp_path / ".env.aliyun"
    emb = tmp_path / ".env.embedding-relay"
    _write_env(
        gen,
        {
            "LLM_PROVIDER": "aliyun",
            "LLM_BASE_URL": "https://token-plan.cn-beijing.maas.aliyuncs.com/v1",
            "LLM_API_KEY": "gen-key",
            "LLM_MODEL": "deepseek-v4.1-flash",
        },
    )
    _write_env(
        emb,
        {
            "EMBEDDING_BASE_URL": "https://api.zhizengzeng.com/v1",
            "EMBEDDING_API_KEY": "emb-key",
            "EMBEDDING_MODEL": "text-embedding-3-small",
        },
    )
    questions = tmp_path / "q.jsonl"
    questions.write_text(
        "\n".join(
            json.dumps({"id": i, "question": i})
            for i in ("fin_cf_001", "fin_cf_017", "fin_cf_021", "fin_cf_030")
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "hk_equity_mcp.experiment.runner.git_dirty", lambda root: True
    )
    runner = FinancialExperimentRunner(
        condition="operation_visible",
        questions_path=questions,
        output_dir=tmp_path / "out",
        ids=["fin_cf_021", "fin_cf_001", "fin_cf_017"],
        dry_run=True,
        env_file=str(gen),
        embedding_env_file=str(emb),
        root=tmp_path,
    )
    meta = runner.run()
    assert meta["git_dirty"] is True
    assert meta["json_mode"] is True
    assert meta["generation_base_url_host"] == "token-plan.cn-beijing.maas.aliyuncs.com"
    assert meta["embedding_base_url_host"] == "api.zhizengzeng.com"
    assert meta["question_ids"] == ["fin_cf_001", "fin_cf_017", "fin_cf_021"]
    assert meta["question_count"] == 3
    blob = json.dumps(meta)
    assert "gen-key" not in blob and "emb-key" not in blob


def test_json_mode_enabled_on_chat_client(monkeypatch):
    from hk_equity_mcp.agent.llm_client import OpenAICompatibleChatClient

    captured = {}

    class FakeResp:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps(
                {
                    "choices": [{"message": {"content": "{\"ok\": true}"}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                }
            ).encode("utf-8")

    def fake_urlopen(req, timeout=0):
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return FakeResp()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = OpenAICompatibleChatClient(
        base_url="https://example.invalid/v1",
        api_key="x",
        model="deepseek-v4.1-flash",
        provider="aliyun",
        json_mode=True,
    )
    client.complete([{"role": "user", "content": "hi"}])
    assert captured["body"]["response_format"] == {"type": "json_object"}
