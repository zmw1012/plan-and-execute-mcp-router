"""APV-Fin runner / agent offline tests (no real APIs)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hk_equity_mcp.agent.llm_client import ScriptedLLMClient
from hk_equity_mcp.apv_fin.agent import ApvFinAgent
from hk_equity_mcp.apv_fin.metadata import load_and_validate_metadata
from hk_equity_mcp.apv_fin.prompts import (
    APV_FIN_SYSTEM_PROMPT,
    apv_fin_prompt_hash,
    assert_prompts_clean,
)
from hk_equity_mcp.apv_fin.scheduler import (
    BudgetTracker,
    execute_compute_step,
    execute_retrieve_step,
)
from hk_equity_mcp.apv_fin.evidence_ledger import EvidenceLedger
from hk_equity_mcp.apv_fin.plan_state import PlanStep
from hk_equity_mcp.experiment.unified_apv_fin_runner import (
    UnifiedHkApvFinRunner,
    _mock_apv_scripted_llm,
)
from hk_equity_mcp.experiment.unified_schema import BANNED_PROMPT_SUBSTRINGS
from hk_equity_mcp.operation_resolver.retrieval import DeterministicHashEmbeddingBackend
from hk_equity_mcp.operation_resolver.server import OperationResolverServer
from hk_equity_mcp.real_mcp.unified_catalog import build_unified_catalog
from hk_equity_mcp.real_mcp.unified_gateway import (
    ScriptedMockTransport,
    UnifiedRealMcpGateway,
)

ROOT = Path(__file__).resolve().parents[1]


def test_prompts_clean_and_hash_stable():
    assert_prompts_clean()
    for banned in BANNED_PROMPT_SUBSTRINGS:
        assert banned not in APV_FIN_SYSTEM_PROMPT
    for needle in ("Tencent", "HSBC", "Alibaba", "Pop Mart", "Xiaomi", "gold_operations"):
        assert needle not in APV_FIN_SYSTEM_PROMPT
    assert apv_fin_prompt_hash() == apv_fin_prompt_hash()
    assert len(apv_fin_prompt_hash()) == 64


def test_retrieve_uses_resolver_compute_does_not_call_mcp(tmp_path: Path):
    bundle = build_unified_catalog(root=ROOT)
    transport = ScriptedMockTransport()
    gateway = UnifiedRealMcpGateway(catalog_bundle=bundle, mock_transport=transport)
    gateway.open()
    resolver = OperationResolverServer(
        embedding_backend=DeterministicHashEmbeddingBackend(),
        registry=bundle.registry,
        real_mcp_gateway=gateway,
        cache_dir=tmp_path / "emb",
    )
    meta = load_and_validate_metadata(catalog=bundle, root=ROOT)
    ledger = EvidenceLedger()
    budget = BudgetTracker({})
    step = PlanStep(
        step_id="r1",
        step_type="retrieve",
        objective="HK equity close price history",
        depends_on=[],
        mandatory=True,
        success_criteria="got rows",
        retrieval_query_hint="stock_hk_hist close",
    )
    out = execute_retrieve_step(
        step=step,
        task_spec={
            "goal": "close price",
            "metrics": ["close"],
            "periods": ["2026-07-31"],
            "cutoff_date": "2026-08-12",
        },
        resolver=resolver,
        metadata_by_id=meta,
        runtime_operation_ids=bundle.stats.runtime_operation_ids,
        ledger=ledger,
        budget=budget,
        invoke_arguments={
            "symbol": "09992",
            "adjust": "",
            "period": "daily",
            "start_date": "20260701",
            "end_date": "20260731",
        },
        select_operation_id="aktools.stock_hk_hist",
        allow_forced_select=True,
    )
    assert out.get("ok") is True, out.get("error")
    assert out["mcp_called"] is True
    assert transport.calls, "retrieve must invoke gateway"
    # Unseen issuer code 09992 must not fail due to whitelist.
    assert transport.calls[-1]["arguments"]["symbol"] == "09992"

    compute_step = PlanStep(
        step_id="c1",
        step_type="compute",
        objective="simple return",
        depends_on=[],
        mandatory=True,
        success_criteria="numeric",
    )
    before = len(transport.calls)
    cout = execute_compute_step(
        step=compute_step,
        ledger=ledger,
        scripted_payload={
            "schema_version": "apv_fin_compute_v1",
            "expression": "end / start - 1",
            "operands": {"end": 12.0, "start": 10.0},
            "unit": "decimal_return",
            "notes": None,
        },
    )
    assert cout["ok"] is True
    assert cout["mcp_called"] is False
    assert cout["llm_called"] is False
    assert len(transport.calls) == before
    gateway.close()


def test_dry_run_and_mock_smoke(tmp_path: Path):
    dry = tmp_path / "dry"
    meta = UnifiedHkApvFinRunner(
        output_dir=dry,
        split="development",
        smoke=True,
        dry_run=True,
        allow_dirty=True,
        root=ROOT,
    ).run()
    assert meta["dry_run"] is True
    assert meta["condition"] == "apv_fin"
    assert meta["system_prompt_hash"] == apv_fin_prompt_hash()
    assert meta["operation_metadata_coverage"] == 39
    assert (dry / "run_metadata.json").exists()

    mock_out = tmp_path / "mock_smoke"
    meta2 = UnifiedHkApvFinRunner(
        output_dir=mock_out,
        split="development",
        smoke=True,
        max_tasks=2,
        mock=True,
        allow_dirty=True,
        root=ROOT,
    ).run()
    assert meta2["mock"] is True
    preds = [
        json.loads(line)
        for line in (mock_out / "predictions.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    assert len(preds) == 2
    for row in preds:
        assert row["condition"] == "apv_fin"
        assert "task_spec" in row
        assert "plan_state" in row
        assert "evidence_ledger" in row
        # No gold leakage into model-facing question path is enforced by runner;
        # ensure prediction question equals item question only (already stored).
        assert "reference_answer" not in json.dumps(row.get("trace") or [])


def test_resume_identity(tmp_path: Path):
    out = tmp_path / "resume"
    r1 = UnifiedHkApvFinRunner(
        output_dir=out,
        split="development",
        smoke=True,
        max_tasks=1,
        mock=True,
        allow_dirty=True,
        root=ROOT,
    )
    r1.run()
    r2 = UnifiedHkApvFinRunner(
        output_dir=out,
        split="development",
        smoke=True,
        max_tasks=2,
        resume=True,
        mock=True,
        allow_dirty=True,
        root=ROOT,
    )
    meta = r2.run()
    assert meta["run_status"] == "completed"
    preds = [
        json.loads(line)
        for line in (out / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(preds) == 2


def test_mock_scripted_llm_cycles():
    llm = _mock_apv_scripted_llm()
    # More than one script length should cycle without raising.
    for _ in range(10):
        out = llm.complete([{"role": "user", "content": "q"}])
        assert out["content"]
        json.loads(out["content"])
