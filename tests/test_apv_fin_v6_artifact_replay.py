"""Offline deterministic replay of anonymized real Smoke-4 v6 planner payloads."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from artifact_gates import hk_artifact_root, require_hk_artifact

from hk_equity_mcp.apv_fin.capability_contract import load_and_validate_capability_contracts
from hk_equity_mcp.apv_fin.capability_filter import filter_by_capability_contract
from hk_equity_mcp.apv_fin.capability_requirement_compiler import (
    compile_capability_requirement,
    compile_retrieve_step,
)
from hk_equity_mcp.apv_fin.schemas import (
    SchemaValidationError,
    assert_monitor_patch_retrieve_contract,
)
from hk_equity_mcp.experiment.unified_apv_fin_metrics import (
    aggregate_apv_fin_metrics,
    strict_answer_success,
    tool_execution_success_flag,
)

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = hk_artifact_root()
FIXTURE = ROOT / "tests" / "fixtures" / "apv_fin_v6_planner_payloads.json"
V6_PRED = ARTIFACT_ROOT / "outputs" / "apv_fin_real_smoke4_v6" / "predictions.jsonl"
V6_INVALIDATED = ARTIFACT_ROOT / "outputs" / "apv_fin_real_smoke4_v6" / "INVALIDATED.md"


def _payloads():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _by_category(cat: str):
    for row in _payloads():
        if row["category"] == cat:
            return row
    raise KeyError(cat)


def test_v6_invalidated_marker_present():
    require_hk_artifact("outputs/apv_fin_real_smoke4_v6/INVALIDATED.md")
    text = V6_INVALIDATED.read_text(encoding="utf-8")
    assert (
        "invalidated_engineering_smoke_due_to_capability_vocabulary_and_contract_compilation_mismatch"
        in text
    )
    # Raw predictions preserved.
    rows = [json.loads(l) for l in V6_PRED.read_text().splitlines() if l.strip()]
    assert len(rows) == 4
    assert all(r.get("status") == "failed" for r in rows)


def test_replay_uses_category_fixtures_not_task_branches():
    # Category labels only — no task-identifier if/elif routing for behavior.
    src_lines = [
        ln
        for ln in Path(__file__).read_text(encoding="utf-8").splitlines()
        if not ln.lstrip().startswith("#") and "assert" not in ln
    ]
    joined = "\n".join(src_lines)
    assert "if task_id" not in joined
    assert "elif task_id" not in joined
    cats = {r["category"] for r in _payloads()}
    assert cats == {
        "narrative_non_ifrs",
        "financial_margin",
        "cross_domain_eps_volume",
        "market_latest_close",
    }


def test_A_market_latest_close_replay():
    row = _by_category("market_latest_close")
    step = row["retrieve_steps"][0]
    compiled = compile_retrieve_step(step, task_spec=row["task_spec"])
    assert compiled["compilation"]["ok"] is True
    out_step = compiled["step"]
    assert out_step["required_capability"] == "market.latest_close"
    assert set(out_step["required_outputs"]) >= {"trade_date", "close"}

    contracts = load_and_validate_capability_contracts()
    filt = filter_by_capability_contract(
        candidates=[
            {
                "operation_id": "aktools.stock_financial_hk_analysis_indicator_em",
                "base_score": 0.95,
                "compatibility_tier": "C",
                "final_rank": 1,
            }
        ],
        contracts=contracts,
        required_capability=out_step["required_capability"],
        required_outputs=out_step["required_outputs"],
        runtime_operation_ids=list(contracts.keys()),
    )
    ids = {r["operation_id"] for r in filt["binder_candidates"]}
    assert "aktools.stock_hk_hist" in ids or "aktools.stock_hk_daily" in ids
    assert not ids.issubset(
        {
            "aktools.stock_financial_hk_analysis_indicator_em",
            "aktools.stock_hk_financial_indicator_em",
        }
    )


def test_B_market_trading_volume_replay():
    row = _by_category("cross_domain_eps_volume")
    volume_step = next(
        s
        for s in row["retrieve_steps"]
        if "volume" in str(s.get("required_capability") or "").lower()
        or "volume" in " ".join(s.get("required_outputs") or []).lower()
        or "trading_volume" in " ".join(s.get("required_outputs") or [])
    )
    compiled = compile_retrieve_step(volume_step, task_spec=row["task_spec"])
    assert compiled["step"]["required_capability"] == "market.daily_ohlcv"
    assert set(compiled["step"]["required_outputs"]) >= {"trade_date", "volume"}


def test_C_financial_margin_replay():
    row = _by_category("financial_margin")
    step = row["retrieve_steps"][0]
    compiled = compile_retrieve_step(step, task_spec=row["task_spec"])
    outs = compiled["step"]["required_outputs"]
    assert "attributable_profit" in outs
    assert "revenue" in outs
    mapping = compiled["step"]["semantic_to_canonical_output_map"]
    assert (
        mapping.get("net_income_attributable_to_ordinary_shareholders")
        == "attributable_profit"
    )


def test_D_non_ifrs_narrative_replay():
    row = _by_category("narrative_non_ifrs")
    step = row["retrieve_steps"][0]
    compiled = compile_retrieve_step(step, task_spec=row["task_spec"])
    assert (
        compiled["step"]["required_capability"]
        == "financial.official_narrative_evidence"
    )
    assert "non_ifrs_profit_attributable_to_equity_holders" not in compiled["step"][
        "required_outputs"
    ]
    assert set(compiled["step"]["required_outputs"]) >= {"snippet", "local_key"}


def test_E_monitor_incomplete_patches_rejected():
    row = _by_category("financial_margin")
    for incomplete in row.get("incomplete_monitor_new_steps") or []:
        payload = {
            "schema_version": "apv_fin_monitor_v1",
            "decision": "replan",
            "replan_action": "retry_current_capability",
            "decision_reason": "missing outputs",
            "ready_step_ids": [],
            "patch": {
                "completed_plan_step_ids": [],
                "updated_plan_steps": [],
                "new_plan_steps": [incomplete],
                "removed_plan_step_ids": [],
            },
        }
        with pytest.raises(SchemaValidationError):
            assert_monitor_patch_retrieve_contract(payload)


def test_metrics_taxonomy_failed_not_strict_and_null_tools():
    # Synthetic rows mirroring v6 failure taxonomy after fix.
    rows = [
        {
            "status": "failed",
            "answer_correct": True,  # poisoned auto-grade
            "auto_grade_status": "graded",
            "predicted_status": "unavailable",
            "all_tool_invocations_successful": None,
            "successful_invocation_count": 0,
            "failed_invocation_count": 0,
            "real_mcp_execution_count": 0,
            "stopping_reason": "capability_contract_no_candidate",
            "error": "capability_contract_no_candidate",
            "plan_state": {"steps": {}},
        }
    ]
    assert strict_answer_success(rows[0]) is False
    assert tool_execution_success_flag(rows[0]) is None
    agg = aggregate_apv_fin_metrics(rows)
    assert agg["common_strict_answer_success"] == 0.0
    assert "descriptive_auto_grade_accuracy" in agg


def test_empty_binder_skips_llm_binding_path():
    from hk_equity_mcp.apv_fin.evidence_ledger import EvidenceLedger
    from hk_equity_mcp.apv_fin.plan_state import PlanStep
    from hk_equity_mcp.apv_fin.scheduler import BudgetTracker, execute_retrieve_step

    class _DummyResolver:
        def find_operations(self, query, top_k=20):
            return {
                "candidates": [
                    {
                        "operation_id": "aktools.stock_financial_hk_analysis_indicator_em",
                        "score": 0.9,
                    }
                ]
            }

        def describe_operation(self, oid):
            return {"success": False}

    step = PlanStep.from_dict(
        {
            "step_id": "s1",
            "step_type": "retrieve",
            "objective": "latest close",
            "depends_on": [],
            "mandatory": True,
            "success_criteria": "close",
            "required_capability": "market.latest_close",
            "required_outputs": ["trade_date", "close"],
            "argument_slots": {"symbol": "00700"},
            "source_preference": [],
            "fallback_capabilities": [],
            "compute_spec": None,
        },
        allow_runtime=False,
    )
    contracts = load_and_validate_capability_contracts()
    budget = BudgetTracker({})
    # Restrict runtime to a financial-only op so market injection cannot fill pool.
    out = execute_retrieve_step(
        step=step,
        task_spec={"goal": "close", "metrics": ["close"], "periods": []},
        resolver=_DummyResolver(),
        metadata_by_id={
            "aktools.stock_financial_hk_analysis_indicator_em": {
                "operation_id": "aktools.stock_financial_hk_analysis_indicator_em",
                "domain": "financial",
            }
        },
        runtime_operation_ids=["aktools.stock_financial_hk_analysis_indicator_em"],
        ledger=EvidenceLedger(),
        budget=budget,
        top_k=5,
        llm=None,
        question="What is the latest close?",
        offline_tool_binding=True,
        contracts=contracts,
    )
    assert out["error"] == "capability_contract_no_candidate"
    assert out.get("llm_binding_skipped") is True
    assert budget.llm_binding_count == 0
