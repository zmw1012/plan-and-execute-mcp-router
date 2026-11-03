"""Tests for Real-MCP experiment runner wiring (no paid LLM)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import pytest

from hk_equity_mcp.experiment.prompts import OPERATION_VISIBLE_SYSTEM_PROMPT
from hk_equity_mcp.experiment.runner import (
    FinancialExperimentRunner,
    catalog_size_for,
    normalize_execution_backend,
    system_prompt_for,
)
from hk_equity_mcp.operation_resolver.retrieval import DeterministicHashEmbeddingBackend
from hk_equity_mcp.operation_resolver.server import OperationResolverServer
from hk_equity_mcp.real_mcp.catalog import load_real_mcp_registry
from hk_equity_mcp.real_mcp.readiness import require_real_mcp_ready
from hk_equity_mcp.real_mcp.types import RealMcpResult


ROOT = Path(__file__).resolve().parents[1]


def test_normalize_backends():
    assert normalize_execution_backend("mcp_free", None) == "none"
    assert normalize_execution_backend("operation_visible", None) == "controlled_local"
    assert normalize_execution_backend("operation_visible", "real_mcp") == "real_mcp"
    with pytest.raises(ValueError):
        normalize_execution_backend("mcp_free", "real_mcp")
    with pytest.raises(ValueError):
        normalize_execution_backend("flat", "real_mcp")


def test_catalog_size_real_vs_local():
    assert catalog_size_for("operation_visible", "controlled_local") == 2
    assert catalog_size_for("operation_visible", "real_mcp") == 21
    assert catalog_size_for("mcp_free", "none") == 0


def test_coverage_l4_cross_server_and_dataset_role():
    path = ROOT / "benchmarks" / "questions" / "financial_real_mcp_coverage_v1.jsonl"
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    cross = ["fin_cf_005", "fin_cf_006", "fin_cf_021", "fin_cf_027", "fin_cf_030"]
    by_id = {r["id"]: r for r in rows}
    for qid in cross:
        assert by_id[qid]["etom_level"] == "L4"
        assert set(by_id[qid]["gold_server_ids"]) == {"aktools", "mcp_hkexnews"}
    assert sum(1 for r in rows if r["etom_level"] == "L4") == 5
    assert sum(1 for r in rows if r["etom_level"] == "L3") == 4
    for r in rows:
        assert r["dataset_split"] == "development"
        assert r["benchmark_role"] == "method_development_and_error_analysis"


def test_prompt_has_no_gold_leakage_fields():
    prompt = OPERATION_VISIBLE_SYSTEM_PROMPT
    for banned in (
        "reference_answer",
        "gold_operations",
        "expected_tool_path",
        "answerability_class",
        "known_data_risks",
        "gold_values",
        "fin_cf_001",
    ):
        assert banned not in prompt
    assert system_prompt_for("operation_visible") == prompt


class _SpyGateway:
    def __init__(self):
        self.calls = []
        self.closed = False
        self.registry = load_real_mcp_registry()

    def open(self):
        return self

    def close(self):
        self.closed = True

    def invoke(self, operation_id: str, arguments: Optional[Dict[str, Any]] = None):
        self.calls.append((operation_id, dict(arguments or {})))
        return RealMcpResult(
            success=True,
            server_id="aktools",
            operation_id=operation_id,
            arguments=arguments or {},
            normalized={"ok": True},
            real_mcp=True,
            protocol_success=True,
            execution_success=True,
            transport="stdio",
        )


def test_resolver_real_mcp_uses_gateway_not_financial_server(tmp_path: Path):
    gateway = _SpyGateway()
    resolver = OperationResolverServer(
        embedding_backend=DeterministicHashEmbeddingBackend(),
        registry=load_real_mcp_registry(),
        cache_dir=tmp_path / "e",
        real_mcp_gateway=gateway,
    )
    assert resolver.financial is None
    assert len(resolver.registry) == 21
    out = resolver.invoke_api(
        "aktools.stock_financial_hk_analysis_indicator_em",
        {"symbol": "00700", "indicator": "年度"},
    )
    assert out["success"] is True
    assert out["result"]["real_mcp"] is True
    assert gateway.calls
    assert gateway.calls[0][0] == "aktools.stock_financial_hk_analysis_indicator_em"


def test_runner_metadata_local_vs_real(tmp_path: Path, monkeypatch):
    questions = ROOT / "benchmarks" / "questions" / "financial_mcp_free_30_v1.jsonl"
    gen = tmp_path / ".env.aliyun"
    emb = tmp_path / ".env.embedding-relay"
    gen.write_text(
        "\n".join(
            [
                "LLM_PROVIDER=test",
                "LLM_BASE_URL=https://gen.example/v1",
                "LLM_API_KEY=secret-gen",
                "LLM_MODEL=dummy-model",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    emb.write_text(
        "\n".join(
            [
                "EMBEDDING_BASE_URL=https://emb.example/v1",
                "EMBEDDING_API_KEY=secret-emb",
                "EMBEDDING_MODEL=text-embedding-3-small",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
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

    local = FinancialExperimentRunner(
        condition="operation_visible",
        execution_backend="controlled_local",
        questions_path=questions,
        output_dir=tmp_path / "local_out",
        ids=["fin_cf_001"],
        dry_run=True,
        allow_dirty=True,
        env_file=str(gen),
        embedding_env_file=str(emb),
        root=ROOT,
    )
    local_meta = local.prepare()["run_metadata"]
    assert local_meta["execution_backend"] == "controlled_local"
    assert local_meta["real_mcp"] is False
    assert local_meta["catalog_entry_count"] == 2
    assert "api_key" not in json.dumps(local_meta).lower()
    assert "secret-gen" not in json.dumps(local_meta)
    assert "reference_answer" not in local_meta["system_prompt"]

    # Real-MCP dry-run fail-closed when readiness missing OR succeed when ready.
    real = FinancialExperimentRunner(
        condition="operation_visible",
        execution_backend="real_mcp",
        questions_path=questions,
        output_dir=tmp_path / "real_out",
        ids=["fin_cf_001"],
        dry_run=True,
        allow_dirty=True,
        env_file=str(gen),
        embedding_env_file=str(emb),
        root=ROOT,
    )
    try:
        require_real_mcp_ready(ROOT / "configs" / "operations" / "financial_real_mcp.yaml")
        ready = True
    except RuntimeError:
        ready = False
    if ready:
        meta = real.prepare()["run_metadata"]
        assert meta["execution_backend"] == "real_mcp"
        assert meta["real_mcp"] is True
        assert meta["catalog_entry_count"] == 21
        assert meta["catalog_id"] == "financial_real_mcp_v1"
        assert meta["catalog_sha256"]
        assert set(meta["server_ids"]) == {"aktools", "mcp_hkexnews"}
        assert "secret" not in json.dumps(meta).lower() or "secret-gen" not in json.dumps(
            meta
        )
    else:
        with pytest.raises(RuntimeError, match="fail-closed"):
            real.prepare()


def test_require_real_mcp_ready_missing_catalog(tmp_path: Path):
    with pytest.raises(RuntimeError, match="fail-closed"):
        require_real_mcp_ready(tmp_path / "missing.yaml")
