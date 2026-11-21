"""Offline tests for HK-155 MCP-Free baseline (no paid APIs / no network)."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import pytest

from hk_equity_mcp.agent.llm_client import ScriptedLLMClient
from hk_equity_mcp.agent.mcp_free import MCPFreeAgent
from hk_equity_mcp.experiment.unified_dataset import load_split_tasks
from hk_equity_mcp.experiment.unified_evaluator import EVALUATOR_VERSION, evaluator_identity
from hk_equity_mcp.experiment.unified_exploratory import (
    RETIREMENT_POLICY_VERSION,
    load_exploratory_manifest,
    load_exploratory_run_order,
    load_exploratory_tasks_in_locked_order,
)
from hk_equity_mcp.experiment.unified_grading import (
    GRADING_VERSION,
    GRADING_VERSION_V1_2,
    GRADER_VERSION,
    GRADER_VERSION_V1_2,
    HISTORICAL_GRADING_VERSIONS,
    resolve_grading_identity,
)
from hk_equity_mcp.experiment.unified_mcp_free_metrics import (
    TOOL_NA,
    apply_mcp_free_tool_na,
    evaluate_mcp_free_task,
)
from hk_equity_mcp.experiment.unified_mcp_free_prompt import (
    UNIFIED_HK_MCP_FREE_SYSTEM_PROMPT,
    unified_mcp_free_prompt_hash,
)
from hk_equity_mcp.experiment.unified_mcp_free_runner import (
    RESUME_IDENTITY_KEYS,
    UnifiedHkMcpFreeRunner,
    assert_resume_compatible_mcp_free,
    questions_identity_hash,
)
from hk_equity_mcp.experiment.unified_schema import BANNED_PROMPT_SUBSTRINGS

ROOT = Path(__file__).resolve().parents[1]
OV_META = (
    ROOT.parent
    / "hk-equity-mcp-agent-hk-unified"
    / "outputs"
    / "hk_unified_exploratory155_operation_visible_v1"
    / "run_metadata.json"
)


def _ov_meta() -> Dict[str, Any]:
    if not OV_META.exists():
        pytest.skip(f"Operation-Visible metadata missing: {OV_META}")
    return json.loads(OV_META.read_text(encoding="utf-8"))


def test_155_identity_matches_operation_visible():
    tasks = load_exploratory_tasks_in_locked_order(ROOT)
    assert len(tasks) == 155
    assert Counter(t["domain"] for t in tasks) == {
        "financial": 60,
        "market": 35,
        "news": 35,
        "cross_domain": 25,
    }
    ov = _ov_meta()
    assert [t["task_id"] for t in tasks] == ov["question_ids"]
    assert Counter(t["domain"] for t in tasks) == Counter(
        {k: int(v) for k, v in (ov.get("domain_counts") or {}).items()}
    )


def test_question_text_hash_matches_operation_visible_ids_and_text():
    tasks = load_exploratory_tasks_in_locked_order(ROOT)
    ov = _ov_meta()
    assert [t["task_id"] for t in tasks] == ov["question_ids"]
    # Question text hash must be stable and cover all 155 questions.
    h1 = questions_identity_hash(tasks)
    h2 = questions_identity_hash(tasks)
    assert h1 == h2
    assert len(h1) == 64
    # Mutating one question changes the hash.
    mutated = [dict(t) for t in tasks]
    mutated[0] = dict(mutated[0])
    mutated[0]["question"] = mutated[0]["question"] + " [tamper]"
    assert questions_identity_hash(mutated) != h1


def test_run_order_matches_operation_visible():
    order = load_exploratory_run_order(ROOT)
    man = load_exploratory_manifest(ROOT)
    ov = _ov_meta()
    assert order["run_order_hash"] == ov["run_order_hash"]
    assert man["manifest_hash"] == ov["exploratory_manifest_hash"]
    assert order["task_ids"] == ov["question_ids"]


def test_prompt_frozen_no_gold_leak_and_hash_stable():
    prompt = UNIFIED_HK_MCP_FREE_SYSTEM_PROMPT
    for banned in BANNED_PROMPT_SUBSTRINGS:
        assert banned not in prompt
    lowered = prompt.lower()
    for needle in (
        "gold_values",
        "reference_answer",
        "find_operations",
        "describe_operation",
        "invoke_api",
        "answerability_class",
        "dataset cutoff",
        "fin_cf_",
        "tencent revenue",
        "required domain",
        "gold_operations",
        "gold tools",
    ):
        assert needle not in lowered
    h = unified_mcp_free_prompt_hash()
    assert h == unified_mcp_free_prompt_hash()
    assert len(h) == 64
    assert "no external tools" in prompt.lower()
    assert "model-internal knowledge" in prompt.lower()


def test_no_catalog_embedding_mcp_in_agent(monkeypatch):
    calls = {"complete": 0}

    class CountingLLM(ScriptedLLMClient):
        def complete(self, messages, *, temperature=0.0, max_tokens=1024):
            calls["complete"] += 1
            # Ensure prompt/messages never include gold.
            blob = json.dumps(messages, ensure_ascii=False)
            assert "gold_operations" not in blob
            assert "reference_answer" not in blob
            assert "gold_values" not in blob
            return super().complete(
                messages, temperature=temperature, max_tokens=max_tokens
            )

    answer = {
        "status": "unavailable",
        "final_answer": "Cannot verify exact figures without tools.",
        "reasoning_brief": "closed book",
    }
    agent = MCPFreeAgent(
        CountingLLM([json.dumps(answer)]),
        system_prompt=UNIFIED_HK_MCP_FREE_SYSTEM_PROMPT,
    )
    assert agent.exposed_tools() == []
    result = agent.run("What was Tencent FY2024 revenue?")
    assert result["llm_calls"] == 1
    assert result["embedding_calls"] == 0
    assert result["real_mcp_execution_count"] == 0
    assert result["underlying_operation_calls"] == 0
    assert result["resolver_calls"] == {
        "find_operations": 0,
        "describe_operation": 0,
    }
    assert result["exposed_tools"] == []
    assert result["catalog_exposed"] is False
    assert calls["complete"] == 1


def test_limited_parse_repair_only():
    bad = "not-json"
    good = json.dumps(
        {
            "status": "unavailable",
            "final_answer": "repaired",
            "reasoning_brief": "ok",
        }
    )
    agent = MCPFreeAgent(
        ScriptedLLMClient([bad, good]),
        system_prompt=UNIFIED_HK_MCP_FREE_SYSTEM_PROMPT,
        max_parse_repairs=1,
    )
    result = agent.run("Q?")
    assert result["llm_calls"] == 2
    assert result["parse_repair_count"] == 1
    assert result["predicted_status"] == "unavailable"
    assert result["final_answer"] == "repaired"
    assert result["real_mcp_execution_count"] == 0


def test_no_tool_metrics_not_counted_as_failure():
    item = {
        "task_id": "fin_cf_001",
        "gold_operations": ["aktools.some_op"],
        "gold_status": "answered",
        "answerability_class": "structured_answerable",
        "dependency_graph": [],
    }
    result = {
        "final_answer": "unknown",
        "predicted_status": "unavailable",
        "status": "full",
        "trace": [],
        "tokens": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        "parse_repair_count": 0,
    }
    metrics = evaluate_mcp_free_task(item=item, result=result)
    assert metrics["tool_execution_success"] == TOOL_NA
    assert metrics["all_tool_invocations_successful"] == TOOL_NA
    assert metrics["exact_tool_set"] == TOOL_NA
    assert metrics["operation_f1"] == TOOL_NA
    assert metrics["real_mcp_execution_count"] == 0
    assert metrics["tool_metrics_applicable"] is False
    # Stock vacuous-true path would have been True; MCP-Free must not use that.
    stockish = apply_mcp_free_tool_na(
        {
            "all_tool_invocations_successful": True,
            "tool_execution_success": True,
            "exact_tool_set": False,
        }
    )
    assert stockish["tool_execution_success"] == TOOL_NA


def test_evaluator_v1_2_reusable():
    """Evaluator stays at v1.2; shared grader advanced to v1.2.1 with historical pin."""
    assert EVALUATOR_VERSION == "hk_unified_evaluator_v1.2"
    # Current integrated identity (do not silently alias back to v1.2).
    assert GRADING_VERSION == "hk_unified_grading_v1.2.1"
    assert GRADER_VERSION == "deterministic_v1.2.1"
    # Frozen MCP-Free HK-155 audits remain reproducible under the historical pin.
    assert HISTORICAL_GRADING_VERSIONS["mcp_free_hk155"] == GRADING_VERSION_V1_2
    hist = resolve_grading_identity(GRADING_VERSION_V1_2)
    assert hist["grading_version"] == GRADING_VERSION_V1_2
    assert hist["grader_version"] == GRADER_VERSION_V1_2
    ident = evaluator_identity()
    assert ident["evaluator_version"] == EVALUATOR_VERSION


def test_resume_identity_excludes_max_tasks(tmp_path: Path):
    qh = "a" * 64
    base = {
        "condition": "mcp_free",
        "execution_backend": "none",
        "generation_provider": "openai_compatible",
        "generation_model": "deepseek-v4.1-flash",
        "generation_base_url_host": "api.zhizengzeng.com",
        "provider_host": "api.zhizengzeng.com",
        "dataset_split": "exploratory_all",
        "system_prompt_hash": unified_mcp_free_prompt_hash(),
        "questions_identity_hash": qh,
        "temperature": 0.0,
        "max_tokens": 1024,
        "catalog_fingerprint": None,
        "exploratory_manifest_hash": "mh",
        "run_order_hash": "rh",
        "retirement_policy_version": RETIREMENT_POLICY_VERSION,
        "max_tasks": 20,
    }
    expanded = dict(base)
    expanded["max_tasks"] = 50
    assert "max_tasks" not in RESUME_IDENTITY_KEYS
    assert "questions_identity_hash" in RESUME_IDENTITY_KEYS
    for key in (
        "generation_provider",
        "generation_model",
        "generation_base_url_host",
        "provider_host",
    ):
        assert key in RESUME_IDENTITY_KEYS
    assert_resume_compatible_mcp_free(base, expanded)
    bad = dict(expanded)
    bad["system_prompt_hash"] = "other"
    with pytest.raises(RuntimeError, match="system_prompt_hash"):
        assert_resume_compatible_mcp_free(base, bad)
    bad_q = dict(expanded)
    bad_q["questions_identity_hash"] = "b" * 64
    with pytest.raises(RuntimeError, match="questions_identity_hash"):
        assert_resume_compatible_mcp_free(base, bad_q)
    bad_provider = dict(expanded)
    bad_provider["generation_provider"] = "aliyun"
    with pytest.raises(RuntimeError, match="generation_provider"):
        assert_resume_compatible_mcp_free(base, bad_provider)
    bad_host = dict(expanded)
    bad_host["provider_host"] = "token-plan.cn-beijing.maas.aliyuncs.com"
    bad_host["generation_base_url_host"] = "token-plan.cn-beijing.maas.aliyuncs.com"
    with pytest.raises(RuntimeError, match="provider_host|generation_base_url_host"):
        assert_resume_compatible_mcp_free(base, bad_host)


def test_max_tasks_does_not_change_questions_identity_hash(tmp_path: Path):
    full = UnifiedHkMcpFreeRunner(
        output_dir=tmp_path / "full",
        split="exploratory_all",
        confirm_retire_heldout=True,
        dry_run=True,
        allow_dirty=True,
        root=ROOT,
    ).run()
    prefix = UnifiedHkMcpFreeRunner(
        output_dir=tmp_path / "prefix",
        split="exploratory_all",
        confirm_retire_heldout=True,
        dry_run=True,
        allow_dirty=True,
        max_tasks=20,
        root=ROOT,
    ).run()
    assert full["questions_identity_hash"] == prefix["questions_identity_hash"]
    assert full["question_count"] == 155
    assert prefix["question_count"] == 20


def test_refuse_live_on_dry_run_dir_and_no_delete_without_resume(tmp_path: Path):
    out = tmp_path / "shared"
    UnifiedHkMcpFreeRunner(
        output_dir=out,
        split="exploratory_all",
        confirm_retire_heldout=True,
        dry_run=True,
        allow_dirty=True,
        root=ROOT,
    ).run()
    with pytest.raises(RuntimeError, match="mode mismatch"):
        UnifiedHkMcpFreeRunner(
            output_dir=out,
            split="exploratory_all",
            confirm_retire_heldout=True,
            mock=True,
            allow_dirty=True,
            max_tasks=1,
            root=ROOT,
        ).run()
    with pytest.raises(RuntimeError, match="mode mismatch"):
        UnifiedHkMcpFreeRunner(
            output_dir=out,
            split="exploratory_all",
            confirm_retire_heldout=True,
            mock=True,
            allow_dirty=True,
            max_tasks=1,
            resume=True,
            root=ROOT,
        ).run()

    mock_out = tmp_path / "mock"
    UnifiedHkMcpFreeRunner(
        output_dir=mock_out,
        split="exploratory_all",
        confirm_retire_heldout=True,
        mock=True,
        allow_dirty=True,
        max_tasks=1,
        root=ROOT,
    ).run()
    preds = mock_out / "predictions.jsonl"
    assert preds.exists()
    before = preds.read_text(encoding="utf-8")
    with pytest.raises(RuntimeError, match="already exists"):
        UnifiedHkMcpFreeRunner(
            output_dir=mock_out,
            split="exploratory_all",
            confirm_retire_heldout=True,
            mock=True,
            allow_dirty=True,
            max_tasks=2,
            root=ROOT,
        ).run()
    assert preds.read_text(encoding="utf-8") == before


def test_dry_run_exploratory(tmp_path: Path):
    out = tmp_path / "dry"
    meta = UnifiedHkMcpFreeRunner(
        output_dir=out,
        split="exploratory_all",
        confirm_retire_heldout=True,
        dry_run=True,
        allow_dirty=True,
        root=ROOT,
    ).run()
    assert meta["condition"] == "mcp_free"
    assert meta["execution_backend"] == "none"
    assert meta["generation_provider"] == "openai_compatible"
    assert meta["generation_model"] == "deepseek-v4.1-flash"
    assert meta["question_count"] == 155
    assert meta["domain_counts"] == {
        "financial": 60,
        "market": 35,
        "news": 35,
        "cross_domain": 25,
    }
    assert meta["catalog_fingerprint"] is None
    assert meta["embedding_model"] is None
    assert meta["system_prompt_hash"] == unified_mcp_free_prompt_hash()
    ov = _ov_meta()
    assert meta["question_ids"] == ov["question_ids"]
    assert meta["exploratory_manifest_hash"] == ov["exploratory_manifest_hash"]
    assert meta["run_order_hash"] == ov["run_order_hash"]
    assert (out / "run_metadata.json").exists()


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


def test_openai_compatible_relay_profile_accepted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    for key in (
        "LLM_PROVIDER",
        "LLM_BASE_URL",
        "LLM_API_KEY",
        "LLM_MODEL",
        "ENV_FILE",
    ):
        monkeypatch.delenv(key, raising=False)
    env = _write_gen_env(
        tmp_path / ".env.relay-v41",
        provider="openai_compatible",
        model="deepseek-v4.1-flash",
        host="api.zhizengzeng.com",
    )
    meta = UnifiedHkMcpFreeRunner(
        output_dir=tmp_path / "relay_ok",
        split="exploratory_all",
        confirm_retire_heldout=True,
        dry_run=True,
        allow_dirty=True,
        env_file=str(env),
        root=ROOT,
    ).run()
    assert meta["generation_provider"] == "openai_compatible"
    assert meta["generation_model"] == "deepseek-v4.1-flash"
    assert meta["generation_base_url_host"] == "api.zhizengzeng.com"
    assert meta["provider_host"] == "api.zhizengzeng.com"
    blob = json.dumps(meta, ensure_ascii=False)
    assert "test-key-not-for-network" not in blob
    assert "API_KEY" not in blob


def test_aliyun_profile_rejected_under_openai_compatible_config(
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
        UnifiedHkMcpFreeRunner(
            output_dir=tmp_path / "aliyun_reject",
            split="exploratory_all",
            confirm_retire_heldout=True,
            dry_run=True,
            allow_dirty=True,
            env_file=str(env),
            root=ROOT,
        ).run()


def test_missing_llm_provider_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    for key in ("LLM_PROVIDER", "LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL", "ENV_FILE"):
        monkeypatch.delenv(key, raising=False)
    env = tmp_path / ".env.missing-provider"
    env.write_text(
        "\n".join(
            [
                "LLM_BASE_URL=https://api.zhizengzeng.com/v1",
                "LLM_API_KEY=test-key-not-for-network",
                "LLM_MODEL=deepseek-v4.1-flash",
                "",
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="LLM_PROVIDER"):
        UnifiedHkMcpFreeRunner(
            output_dir=tmp_path / "missing_provider",
            split="exploratory_all",
            confirm_retire_heldout=True,
            dry_run=True,
            allow_dirty=True,
            env_file=str(env),
            root=ROOT,
        ).run()


def test_model_mismatch_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    for key in ("LLM_PROVIDER", "LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL", "ENV_FILE"):
        monkeypatch.delenv(key, raising=False)
    env = _write_gen_env(
        tmp_path / ".env.bad-model",
        provider="openai_compatible",
        model="deepseek-flash",
        host="api.zhizengzeng.com",
    )
    with pytest.raises(RuntimeError, match="model mismatch"):
        UnifiedHkMcpFreeRunner(
            output_dir=tmp_path / "bad_model",
            split="exploratory_all",
            confirm_retire_heldout=True,
            dry_run=True,
            allow_dirty=True,
            env_file=str(env),
            root=ROOT,
        ).run()


def test_mock_prefix_and_resume(tmp_path: Path):
    out = tmp_path / "mock"
    r1 = UnifiedHkMcpFreeRunner(
        output_dir=out,
        split="exploratory_all",
        confirm_retire_heldout=True,
        mock=True,
        allow_dirty=True,
        max_tasks=2,
        root=ROOT,
    )
    meta1 = r1.run()
    assert meta1["condition"] == "mcp_free"
    preds = (out / "predictions.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(preds) == 2
    for line in preds:
        row = json.loads(line)
        assert row["embedding_calls"] == 0
        assert row["real_mcp_execution_count"] == 0
        assert row["resolver_calls"]["find_operations"] == 0
        assert row["tool_execution_success"] == TOOL_NA
        # Gold must not appear in trace messages.
        trace_blob = json.dumps(row.get("trace") or [], ensure_ascii=False)
        assert "reference_answer" not in trace_blob
        assert "gold_values" not in trace_blob

    r2 = UnifiedHkMcpFreeRunner(
        output_dir=out,
        split="exploratory_all",
        confirm_retire_heldout=True,
        mock=True,
        allow_dirty=True,
        max_tasks=3,
        resume=True,
        root=ROOT,
    )
    meta2 = r2.run()
    preds2 = (out / "predictions.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(preds2) == 3
    assert meta2["system_prompt_hash"] == meta1["system_prompt_hash"]
    assert (out / "grouped_aggregates.json").exists()
    assert (out / "cost_summary.json").exists()
    cost = json.loads((out / "cost_summary.json").read_text(encoding="utf-8"))
    assert cost["total_real_mcp_executions"] == 0
    assert cost["total_embedding_calls"] == 0


def test_load_split_tasks_exploratory_same_as_ov_order():
    tasks = load_split_tasks(ROOT, "exploratory_all")
    order = load_exploratory_run_order(ROOT)["task_ids"]
    assert [t["task_id"] for t in tasks] == order


def test_etom_empty_route_offline(tmp_path: Path):
    import importlib.util
    import subprocess
    import sys

    script = ROOT / "scripts" / "evaluate_etom_empty_route_control.py"
    etom_root = ROOT.parent / "hk-equity-mcp-agent-etom"
    lock = etom_root / "benchmarks" / "etom" / "formal_100_v1.lock.json"
    if not lock.exists():
        pytest.skip("ETOM lock missing")
    out = tmp_path / "etom_empty"
    # Run as subprocess to prove no network dependency beyond local files.
    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--lock",
            str(lock),
            "--etom-root",
            str(etom_root),
            "--output-dir",
            str(out),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr
    summary = json.loads((out / "empty_route_summary.json").read_text(encoding="utf-8"))
    assert summary["control_name"] == "ETOM Empty-Route Control"
    assert summary["not_an_mcp_free_agent"] is True
    assert summary["network_calls"] == 0
    assert summary["llm_calls"] == 0
    assert summary["embedding_calls"] == 0
    assert summary["mcp_calls"] == 0
    assert summary["expectation_ok"] is True
    for lv in (1, 2, 3, 4):
        assert summary["per_level"][f"L{lv}"]["exact_match"] == 0.0
    assert summary["per_level"]["L5"]["exact_match"] == 1.0
    # importlib unused guard silence
    _ = importlib.util
