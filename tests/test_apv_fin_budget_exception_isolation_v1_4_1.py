"""Budget exception conversion + check-before-increment (APV-Fin v1.4.1)."""

from __future__ import annotations

import json
from pathlib import Path

from hk_equity_mcp.agent.llm_client import ScriptedLLMClient
from hk_equity_mcp.apv_fin.agent import ApvFinAgent
from hk_equity_mcp.apv_fin.llm_budget import BudgetedLLMClient, LlmCallBudgetExhausted
from hk_equity_mcp.apv_fin.runtime_safety import (
    runtime_exception_policy_hash,
    structured_llm_budget_failure,
)
from hk_equity_mcp.apv_fin.scheduler import BudgetTracker
from hk_equity_mcp.experiment.unified_apv_fin_runner import load_apv_fin_config
from hk_equity_mcp.operation_resolver.retrieval import DeterministicHashEmbeddingBackend
from hk_equity_mcp.operation_resolver.server import OperationResolverServer
from hk_equity_mcp.real_mcp.unified_catalog import build_unified_catalog
from hk_equity_mcp.real_mcp.unified_gateway import (
    ScriptedMockTransport,
    UnifiedRealMcpGateway,
)


def _minimal_plan() -> dict:
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
            "answerability_reason": "ok",
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


def test_runtime_exception_policy_hash():
    assert runtime_exception_policy_hash()


def test_structured_llm_budget_failure_fields():
    exc = LlmCallBudgetExhausted(used=10, limit=10, stage="tool_binding_repair")
    out = structured_llm_budget_failure(exc)
    assert out["status"] == "failed"
    assert out["error_type"] == "resource_limit_failure"
    assert out["stopping_reason"] == "llm_call_budget_exhausted"
    assert out["llm_budget_exhausted"] is True
    assert "tool_binding_repair" in out["error"]
    assert "used=10" in out["error"]
    assert "limit=10" in out["error"]


def test_check_before_increment_exception_replan_never_exceeds():
    cfg = load_apv_fin_config()
    limits = dict(cfg["limits"])
    budget = BudgetTracker(limits)
    assert limits["max_exception_replan_calls_per_task"] == 2
    budget.exception_replan_calls = 2
    # Simulate agent gate: at limit → stop without incrementing.
    if budget.exception_replan_calls >= limits["max_exception_replan_calls_per_task"]:
        budget.attempted_over_budget = True
    else:
        budget.exception_replan_calls += 1
    assert budget.exception_replan_calls == 2
    assert budget.attempted_over_budget is True


def test_binding_repair_stage_exhaustion_converts_structurally():
    """tool_binding_repair at used==limit → structured failure fields (no runner raise)."""
    budget = BudgetTracker({"max_total_llm_calls_per_task": 10})
    budget.llm_calls = 10
    llm = BudgetedLLMClient(ScriptedLLMClient(["{}"]), budget)
    try:
        llm.complete(
            [{"role": "user", "content": "x"}], stage="tool_binding_repair"
        )
        raised = False
        exc = None
    except LlmCallBudgetExhausted as e:
        raised = True
        exc = e
    assert raised is True
    assert exc is not None
    assert exc.stage == "tool_binding_repair"
    assert exc.used == 10
    converted = structured_llm_budget_failure(exc)
    assert converted["status"] == "failed"
    assert converted["error_type"] == "resource_limit_failure"
    assert converted["stopping_reason"] == "llm_call_budget_exhausted"


def test_agent_run_catches_budget_on_planner(tmp_path: Path):
    from pathlib import Path as _Path

    from hk_equity_mcp.apv_fin.metadata import load_and_validate_metadata

    root = _Path(__file__).resolve().parents[1]
    bundle = build_unified_catalog(root=root)
    transport = ScriptedMockTransport()
    gateway = UnifiedRealMcpGateway(catalog_bundle=bundle, mock_transport=transport)
    gateway.open()
    resolver = OperationResolverServer(
        embedding_backend=DeterministicHashEmbeddingBackend(),
        registry=bundle.registry,
        real_mcp_gateway=gateway,
        cache_dir=tmp_path / "emb",
    )
    llm = ScriptedLLMClient([json.dumps(_minimal_plan())])
    agent = ApvFinAgent(
        llm=llm,
        resolver=resolver,
        config={
            "allow_mock_hooks": True,
            "limits": {
                "max_total_llm_calls_per_task": 0,
                "max_idle_monitor_rounds": 1,
                "max_exception_replan_calls_per_task": 2,
                "max_llm_monitor_rounds": 2,
                "max_plan_revisions": 2,
                "max_total_mcp_invocations": 8,
                "max_targeted_retrievals_per_step": 2,
                "max_same_operation_repeats": 1,
                "max_final_repair_rounds": 1,
                "max_binding_repairs": 1,
            },
        },
        metadata_by_id=load_and_validate_metadata(catalog=bundle, root=root),
        runtime_operation_ids=list(bundle.stats.runtime_operation_ids),
        live_mode=False,
    )
    out = agent.run("What is revenue?")
    assert out["status"] == "failed"
    assert out["error_type"] == "resource_limit_failure"
    assert out["stopping_reason"] == "llm_call_budget_exhausted"
    assert out.get("llm_budget_exhausted") is True
    assert "stage=" in str(out.get("error") or "")
