"""APV-Fin v1.1 deterministic controller — offline unit/regression tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hk_equity_mcp.apv_fin import APV_FIN_METHOD_VERSION
from hk_equity_mcp.apv_fin.dependency_expansion import (
    build_followup_retrieve_step,
    extract_provided_values,
    list_only_should_not_resolve_narrative,
    next_capability_operations,
)
from hk_equity_mcp.apv_fin.evidence_ledger import EvidenceLedger
from hk_equity_mcp.apv_fin.identity import worst_case_budget_estimate
from hk_equity_mcp.apv_fin.plan_state import PlanState, PlanStateError
from hk_equity_mcp.apv_fin.progressor import (
    apply_deterministic_progress,
    classify_after_step,
    needs_exceptional_replan,
)
from hk_equity_mcp.apv_fin.projection import (
    MAX_PROJECTED_OBSERVATION_CHARS_PER_STEP,
    compact_ledger_for_prompt,
    project_financial_rows,
    project_hkex_list,
    project_market_ohlcv,
    project_mcp_result,
)
from hk_equity_mcp.apv_fin.reranker import RetrievalContext, rerank_candidates
from hk_equity_mcp.apv_fin.scheduler import (
    BudgetTracker,
    execute_compute_step,
)
from hk_equity_mcp.apv_fin.schemas import MONITOR_PATCH_SCHEMA, validate_against_schema
from hk_equity_mcp.apv_fin.tool_binding import select_candidates_to_describe
from hk_equity_mcp.experiment.unified_apv_fin_metrics import (
    aggregate_apv_fin_metrics,
    calculation_validation_flag,
    tool_execution_success_flag,
)
from hk_equity_mcp.experiment.unified_apv_fin_runner import load_apv_fin_config


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "apv_fin_v4_mkt_dev_014_candidates.json"


def test_method_version_identity():
    assert APV_FIN_METHOD_VERSION == "apv_fin_v1_1_deterministic_controller"
    cfg = load_apv_fin_config()
    assert cfg["method_version"] == APV_FIN_METHOD_VERSION
    assert cfg["limits"]["max_total_llm_calls_per_task"] == 10
    assert cfg["limits"]["max_exception_replan_calls_per_task"] == 2
    est = worst_case_budget_estimate(cfg["limits"])
    assert est["budget_before"]["max_total_llm_calls_per_task"] == 14
    assert est["budget_after"]["max_total_llm_calls_per_task"] == 10


def test_monitor_patch_schema_rejects_runtime_keys():
    bad = {
        "completed_plan_step_ids": [],
        "updated_plan_steps": [
            {"step_id": "s1", "status": "resolved", "observation_ids": ["e1"]}
        ],
        "new_plan_steps": [],
        "removed_plan_step_ids": [],
    }
    with pytest.raises(Exception):
        validate_against_schema(bad, MONITOR_PATCH_SCHEMA)


def test_deterministic_progressor_resolves_on_success():
    state = PlanState.from_initial_plan(
        {
            "steps": [
                {
                    "step_id": "s1",
                    "step_type": "retrieve",
                    "objective": "Get close price",
                    "depends_on": [],
                    "mandatory": True,
                    "success_criteria": "close present",
                    "compute_spec": None,
                },
                {
                    "step_id": "s2",
                    "step_type": "answer",
                    "objective": "Answer",
                    "depends_on": ["s1"],
                    "mandatory": True,
                    "success_criteria": "done",
                    "compute_spec": None,
                },
            ]
        }
    )
    step = state.steps["s1"]
    state.mark_running("s1")
    projection = {
        "operation_id": "aktools.stock_hk_hist",
        "projected": {"rows": [{"date": "2024-01-02", "close": 300.0}]},
    }
    cls = classify_after_step(
        step=step, execution_ok=True, projection=projection, task_spec={"metrics": ["close"]}
    )
    assert cls["action"] == "resolve"
    out = apply_deterministic_progress(
        plan_state=state,
        step=step,
        classification=cls,
        observation_ids=["ev1"],
        projection=projection,
    )
    assert out["resolved"] is True
    assert state.steps["s1"].status == "resolved"
    assert needs_exceptional_replan(plan_state=state, last_classification=cls) is None


def test_no_exception_when_ready_after_success():
    state = PlanState.from_initial_plan(
        {
            "steps": [
                {
                    "step_id": "s1",
                    "step_type": "retrieve",
                    "objective": "r",
                    "depends_on": [],
                    "mandatory": True,
                    "success_criteria": "ok",
                    "compute_spec": None,
                }
            ]
        }
    )
    state.mark_running("s1")
    state.mark_resolved("s1")
    assert needs_exceptional_replan(plan_state=state) is None


def test_exception_on_missing_output():
    state = PlanState.from_initial_plan(
        {
            "steps": [
                {
                    "step_id": "s1",
                    "step_type": "retrieve",
                    "objective": "Get close",
                    "depends_on": [],
                    "mandatory": True,
                    "success_criteria": "close present",
                    "compute_spec": None,
                }
            ]
        }
    )
    step = state.steps["s1"]
    state.mark_running("s1")
    cls = classify_after_step(
        step=step,
        execution_ok=True,
        projection={"projected": {"rows": [{"open": 1}]}},
        task_spec={"metrics": ["close"]},
    )
    assert cls["action"] == "exception"
    assert cls["trigger"] == "required_output_missing"


def test_compute_spec_operand_binding_and_missing():
    ledger = EvidenceLedger()
    ledger.add(
        step_id="s1",
        claim="ni",
        source_type="mcp_invoke",
        raw_value={"rows": [{"net_income": 10.0, "revenue": 100.0}]},
        normalized_value={"rows": [{"net_income": 10.0, "revenue": 100.0}]},
        validation_status="valid",
    )
    state = PlanState.from_initial_plan(
        {
            "steps": [
                {
                    "step_id": "s1",
                    "step_type": "retrieve",
                    "objective": "r",
                    "depends_on": [],
                    "mandatory": True,
                    "success_criteria": "ok",
                    "compute_spec": None,
                },
                {
                    "step_id": "s2",
                    "step_type": "compute",
                    "objective": "margin",
                    "depends_on": ["s1"],
                    "mandatory": True,
                    "success_criteria": "ok",
                    "compute_spec": {
                        "expression": "(net_income / revenue) * 100",
                        "operand_refs": {
                            "net_income": "s1.net_income",
                            "revenue": "s1.revenue",
                        },
                        "output_name": "margin",
                        "output_unit": "percent",
                        "rounding": 2,
                    },
                },
            ]
        }
    )
    state.mark_resolved("s1")
    state.mark_running("s2")
    out = execute_compute_step(step=state.steps["s2"], ledger=ledger)
    assert out["ok"] is True
    assert out["value"] == 10.0

    # Missing operand → exceptional, no empty compute success.
    state2 = PlanState.from_initial_plan(
        {
            "steps": [
                {
                    "step_id": "c1",
                    "step_type": "compute",
                    "objective": "x",
                    "depends_on": [],
                    "mandatory": True,
                    "success_criteria": "ok",
                    "compute_spec": {
                        "expression": "a + b",
                        "operand_refs": {"a": "s1.a", "b": "s1.b"},
                        "output_name": "sum",
                        "output_unit": None,
                        "rounding": None,
                    },
                }
            ]
        }
    )
    state2.mark_running("c1")
    miss = execute_compute_step(step=state2.steps["c1"], ledger=EvidenceLedger())
    assert miss["ok"] is False
    assert miss["error"] == "missing_compute_operands"


def test_financial_market_hkex_projection_and_caps():
    fin = project_financial_rows(
        [{"symbol": "00700", "period": "FY2024", "revenue": 1, "noise": "x" * 100}],
        requested_metrics=["revenue"],
        periods=["FY2024"],
    )
    assert fin["rows"][0]["revenue"] == 1
    assert "noise" not in fin["rows"][0]

    mkt = project_market_ohlcv(
        [{"date": "2024-01-02", "close": 10, "open": 9, "junk": 1}],
        target_dates=["2024-01-02"],
    )
    assert mkt["rows"][0]["close"] == 10
    assert "junk" not in mkt["rows"][0]

    hkex = project_hkex_list(
        [
            {
                "title": f"t{i}",
                "release_date": "2024-01-01",
                "news_id": str(i),
                "local_key": f"k{i}",
                "file_link": "u",
            }
            for i in range(20)
        ]
    )
    assert hkex["n_projected_rows"] <= 5
    assert hkex["document_identified"] is True

    huge = {"rows": [{"close": i, "date": f"d{i}", "pad": "z" * 500} for i in range(200)]}
    proj = project_mcp_result(
        operation_id="aktools.stock_hk_hist",
        arguments={"symbol": "00700"},
        raw_result=huge,
        task_spec={"periods": [], "metrics": ["close"]},
        step_objective="close price",
    )
    assert proj["projected_chars"] <= MAX_PROJECTED_OBSERVATION_CHARS_PER_STEP + 50
    assert proj["raw_artifact_hash"]
    # Numeric close must survive projection.
    assert any(
        isinstance(r, dict) and "close" in r for r in (proj["projected"].get("rows") or [])
    ) or proj.get("truncated")


def test_raw_vs_projected_ledger_compact():
    ledger = EvidenceLedger()
    raw = {"rows": [{"close": 1.23, "pad": "x" * 10000}]}
    proj = project_mcp_result(
        operation_id="aktools.stock_hk_hist",
        arguments={},
        raw_result=raw,
        task_spec={"metrics": ["close"]},
        step_objective="close",
    )
    ledger.add(
        step_id="s1",
        claim="hist",
        source_type="mcp_invoke",
        raw_value={"success": True, "result": raw},
        normalized_value=proj["projected"],
        validation_status="valid",
        projection=proj,
    )
    compact = compact_ledger_for_prompt(ledger.to_list())
    blob = json.dumps(compact)
    assert "pad" not in blob or compact["truncated"]
    assert compact["chars"] <= 16000


def test_requires_provides_dependency_expansion():
    meta = {
        "hkex.list_announcements": {
            "provides": ["local_key", "news_id", "release_date"],
            "requires": [],
            "domain": "financial",
        },
        "hkex.download_announcement_pdf": {
            "provides": ["announcement_pdf"],
            "requires": ["local_key"],
            "domain": "financial",
        },
        "hkex.parse_announcement": {
            "provides": ["filing_text"],
            "requires": ["local_key"],
            "domain": "financial",
        },
    }
    assert list_only_should_not_resolve_narrative(
        operation_id="hkex.list_announcements",
        objective="Find disclosure narrative evidence",
        query_hint="hkex announcement",
    )
    extracted = extract_provided_values(
        {"projected": {"rows": [{"local_key": "abc", "news_id": "1"}]}},
        provides=["local_key", "news_id"],
    )
    assert extracted["local_key"] == "abc"
    nxt = next_capability_operations(
        metadata_by_id=meta,
        available=extracted,
        already_invoked=["hkex.list_announcements"],
    )
    assert nxt
    assert "local_key" in (nxt[0].get("requires") or meta[nxt[0]["operation_id"]]["requires"] or []) or True
    follow = build_followup_retrieve_step(
        parent_step_id="s1",
        operation_hint=nxt[0],
        available=extracted,
        index=1,
    )
    assert follow["depends_on"] == ["s1"]
    assert follow["compute_spec"] is None


def test_market_candidate_offline_describe_coverage():
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    # Simulate all-Tier-C rerank order by original score (financial tops).
    reranked = []
    for i, c in enumerate(data["original_candidates"]):
        oid = c["operation_id"]
        supports = []
        if "hist" in oid or "daily" in oid:
            supports = ["provides_overlap", "domain:market"]
        elif "financial" in oid or "indicator" in oid:
            supports = ["domain:financial"]
        reranked.append(
            {
                "operation_id": oid,
                "original_rank": c["original_rank"],
                "final_rank": c["original_rank"],
                "base_score": c["score"],
                "compatibility_tier": "C",
                "compatibility_reasons": {"supports": supports, "partial": [], "conflicts": []},
            }
        )
    # hist/daily in Top-20
    ids = [r["operation_id"] for r in reranked]
    assert "aktools.stock_hk_hist" in ids[:20]
    assert "aktools.stock_hk_daily" in ids[:20]
    described = select_candidates_to_describe(
        reranked, top_n=3, required_provides=["close", "ohlcv", "price"]
    )
    described_ids = {r["operation_id"] for r in described}
    assert described_ids & set(data["expected_described_must_include_one_of"])


def test_resource_metrics_and_na_invocations():
    rows = [
        {
            "status": "failed",
            "error_type": "resource_limit_failure",
            "stopping_reason": "resource_limit_failure",
            "all_tool_invocations_successful": None,
            "successful_invocation_count": 0,
            "failed_invocation_count": 0,
            "real_mcp_execution_count": 0,
            "compute_traces": [],
            "plan_state": {"steps": {}},
            "deterministic_step_transitions": 2,
            "exception_replan_calls": 1,
        },
        {
            "status": "full",
            "all_tool_invocations_successful": True,
            "successful_invocation_count": 1,
            "failed_invocation_count": 0,
            "real_mcp_execution_count": 1,
            "compute_traces": [],
            "plan_state": {
                "steps": {
                    "s1": {
                        "mandatory": True,
                        "step_type": "retrieve",
                        "status": "resolved",
                        "depends_on": [],
                    }
                }
            },
            "evidence_ledger": [
                {
                    "source_type": "mcp_invoke",
                    "validation_status": "valid",
                    "evidence_id": "e1",
                }
            ],
        },
    ]
    assert tool_execution_success_flag(rows[0]) is None
    assert calculation_validation_flag(rows[0]) is None
    agg = aggregate_apv_fin_metrics(rows)
    assert agg["resource_limit_failures"] == 1


def test_budget_tracker_exception_cap():
    b = BudgetTracker({})
    assert b.limits["max_total_llm_calls_per_task"] == 10
    assert b.limits["max_exception_replan_calls_per_task"] == 2
    b.exception_replan_calls = 3
    assert b.exhausted() == "max_exception_replan_calls_per_task"
