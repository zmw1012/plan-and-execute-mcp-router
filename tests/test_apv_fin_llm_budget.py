"""APV-Fin global per-task LLM call budget (offline; no real APIs)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from hk_equity_mcp.agent.llm_client import ScriptedLLMClient
from hk_equity_mcp.apv_fin.agent import ApvFinAgent
from hk_equity_mcp.apv_fin.evidence_ledger import EvidenceLedger
from hk_equity_mcp.apv_fin.llm_budget import BudgetedLLMClient, LlmCallBudgetExhausted
from hk_equity_mcp.apv_fin.metadata import load_and_validate_metadata
from hk_equity_mcp.apv_fin.plan_state import PlanStep
from hk_equity_mcp.apv_fin.planner import run_initial_planner
from hk_equity_mcp.apv_fin.scheduler import BudgetTracker, execute_compute_step
from hk_equity_mcp.experiment.unified_apv_fin_runner import (
    UnifiedHkApvFinRunner,
    assert_output_dir_safe,
    load_apv_fin_config,
)
from hk_equity_mcp.operation_resolver.retrieval import DeterministicHashEmbeddingBackend
from hk_equity_mcp.operation_resolver.server import OperationResolverServer
from hk_equity_mcp.real_mcp.unified_catalog import build_unified_catalog
from hk_equity_mcp.real_mcp.unified_gateway import (
    ScriptedMockTransport,
    UnifiedRealMcpGateway,
)

ROOT = Path(__file__).resolve().parents[1]


def _plan() -> dict:
    return {
        "schema_version": "apv_fin_plan_v1",
        "task_spec": {
            "goal": "g",
            "entities": [],
            "periods": [],
            "metrics": ["m"],
            "constraints": [],
            "ambiguities": [],
            "answerability_hypothesis": "answerable",
            "answerability_reason": "Required evidence is available from supported sources.",
            "cutoff_date": "2026-08-12",
            "currency_unit_hints": [],
            "metric_basis_hints": [],
        },
        "initial_plan": {
            "steps": [
                {
                    "step_id": "s1",
                    "step_type": "answer",
                    "objective": "answer",
                    "depends_on": [],
                    "mandatory": True,
                    "success_criteria": "done",
                    "retrieval_query_hint": None,
                    "notes": None,
                }
            ],
            "planning_notes": None,
        },
    }


def _final() -> dict:
    return {
        "schema_version": "apv_fin_final_v1",
        "status": "answered",
        "final_answer": "ok",
        "reasoning_brief": "brief",
        "evidence_ids": [],
        "clarification_question": None,
        "refusal_reason": None,
        "currency": None,
        "unit": None,
        "metric_basis": None,
        "period": None,
        "entity": None,
    }


def test_config_has_llm_call_cap_unchanged_other_limits():
    cfg = load_apv_fin_config(root=ROOT)
    assert cfg["limits"]["max_total_llm_calls_per_task"] == 10
    assert cfg["limits"]["max_plan_revisions"] == 2
    assert cfg["limits"]["max_total_mcp_invocations"] == 8
    assert cfg["limits"]["max_exception_replan_calls_per_task"] == 2
    assert cfg["limits"]["max_llm_monitor_rounds"] == 2


def test_budgeted_client_counts_all_helper_and_repair_calls():
    budget = BudgetTracker({"max_total_llm_calls_per_task": 10})
    plan = json.dumps(_plan())
    # First content invalid → planner repair; both must count.
    inner = ScriptedLLMClient(["not-json", plan])
    llm = BudgetedLLMClient(inner, budget)
    out = run_initial_planner(llm, "q")
    assert out["ok"] is True
    assert budget.llm_calls == 2
    assert budget.calls_by_stage.get("planner") == 1
    assert budget.calls_by_stage.get("planner_repair") == 1


def test_tenth_call_allowed_eleventh_blocked():
    budget = BudgetTracker({"max_total_llm_calls_per_task": 10})
    inner = ScriptedLLMClient(["{}"] * 20)
    llm = BudgetedLLMClient(inner, budget)
    for i in range(10):
        llm.complete([{"role": "user", "content": "x"}], stage=f"s{i}")
    assert budget.llm_calls == 10
    assert budget.llm_calls_remaining == 0
    assert budget.llm_budget_exhausted is True
    with pytest.raises(LlmCallBudgetExhausted) as ei:
        llm.complete([{"role": "user", "content": "blocked"}], stage="s11")
    assert ei.value.error_type == "resource_limit_failure"
    assert ei.value.used == 10
    assert budget.llm_calls == 10  # 11th not recorded


def test_compute_does_not_increment_llm_calls():
    budget = BudgetTracker({"max_total_llm_calls_per_task": 10})
    before = budget.llm_calls
    step = PlanStep(
        step_id="c1",
        step_type="compute",
        objective="ret",
        depends_on=[],
        mandatory=True,
        success_criteria="num",
        notes=json.dumps(
            {
                "schema_version": "apv_fin_compute_v1",
                "expression": "end / start - 1",
                "operands": {"end": 12.0, "start": 10.0},
                "unit": "decimal_return",
                "notes": None,
            }
        ),
    )
    out = execute_compute_step(step=step, ledger=EvidenceLedger())
    assert out["ok"] is True
    assert out["llm_called"] is False
    assert budget.llm_calls == before


def test_compute_missing_operands_unresolved_no_llm():
    out = execute_compute_step(
        step=PlanStep(
            step_id="c1",
            step_type="compute",
            objective="need formula",
            depends_on=[],
            mandatory=True,
            success_criteria="num",
        ),
        ledger=EvidenceLedger(),
    )
    assert out["ok"] is False
    assert out["unresolved"] is True
    assert out["error"] == "compute_missing_expression_operands"
    assert out["llm_called"] is False


def test_agent_budget_failure_keeps_trace(tmp_path: Path):
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
    # Cap=1: planner call succeeds; any later generation (final) is blocked.
    plan = json.dumps(_plan())
    finals = json.dumps(_final())
    llm = ScriptedLLMClient([plan, finals, finals])
    agent = ApvFinAgent(
        llm=llm,
        resolver=resolver,
        config={
            "allow_mock_hooks": True,
            "limits": {"max_total_llm_calls_per_task": 1, "max_idle_monitor_rounds": 1},
        },
        metadata_by_id=load_and_validate_metadata(catalog=bundle, root=ROOT),
        runtime_operation_ids=list(bundle.stats.runtime_operation_ids),
        live_mode=False,
    )
    result = agent.run("budget probe")
    assert result["error_type"] == "resource_limit_failure"
    assert result["llm_budget_exhausted"] is True
    assert result["llm_call_budget"] == 1
    assert result["llm_calls_used"] == 1
    assert result["trace"], "trace must be preserved"
    assert any(t.get("type") == "planner" for t in result["trace"]) or any(
        t.get("type") == "llm_budget" for t in result["trace"]
    )
    assert result.get("plan_state") or result.get("initial_plan") is not None or any(
        t.get("type") == "planner" for t in result["trace"]
    )
    gateway.close()


def test_resume_identity_rejects_llm_cap_mismatch(tmp_path: Path):
    out = tmp_path / "resume_cap"
    meta = {
        "condition": "apv_fin",
        "method_version": "apv_fin_v1_1_deterministic_controller",
        "dry_run": False,
        "mock": True,
        "subset_kind": "smoke8",
        "max_total_llm_calls_per_task": 10,
        "limits": {"max_total_llm_calls_per_task": 10},
    }
    out.mkdir()
    (out / "run_metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    (out / "predictions.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="max_total_llm_calls_per_task"):
        assert_output_dir_safe(
            output_dir=out,
            dry_run=False,
            mock=True,
            resume=True,
            run_meta={**meta, "max_total_llm_calls_per_task": 20},
        )


def test_dry_run_records_llm_cap(tmp_path: Path):
    meta = UnifiedHkApvFinRunner(
        output_dir=tmp_path / "dry",
        split="development",
        smoke=True,
        dry_run=True,
        allow_dirty=True,
        root=ROOT,
    ).run()
    assert meta["max_total_llm_calls_per_task"] == 10
    assert meta["limits"]["max_total_llm_calls_per_task"] == 10
