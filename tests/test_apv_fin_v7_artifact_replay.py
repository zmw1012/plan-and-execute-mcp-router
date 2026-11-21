"""Offline replay of real Smoke-4 v7 Gateway envelopes (structure preserved)."""

from __future__ import annotations

from artifact_gates import hk_artifact_root, require_hk_artifact

import json
from pathlib import Path

from hk_equity_mcp.apv_fin.dependency_expansion import (
    disclosure_prefix_needed,
    rewrite_step_to_disclosure_list,
)
from hk_equity_mcp.apv_fin.output_adapters import (
    adapt_mcp_result,
    required_outputs_present,
)
from hk_equity_mcp.apv_fin.plan_state import PlanState
from hk_equity_mcp.apv_fin.progressor import needs_exceptional_replan
from hk_equity_mcp.apv_fin.result_envelope import normalize_gateway_envelope

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = hk_artifact_root()
FIX = ROOT / "tests" / "fixtures" / "apv_fin_v7_envelopes"
V7_INVALIDATED = ARTIFACT_ROOT / "outputs" / "apv_fin_real_smoke4_v7" / "INVALIDATED.md"
V7_PRED = ARTIFACT_ROOT / "outputs" / "apv_fin_real_smoke4_v7" / "predictions.jsonl"


def _load(name: str):
    return json.loads((FIX / name).read_text(encoding="utf-8"))


def test_v7_invalidated_marker():
    require_hk_artifact("outputs/apv_fin_real_smoke4_v7/INVALIDATED.md")
    text = V7_INVALIDATED.read_text(encoding="utf-8")
    assert (
        "invalidated_engineering_smoke_due_to_gateway_result_adapter_and_workflow_progression_defects"
        in text
    )
    rows = [json.loads(l) for l in V7_PRED.read_text().splitlines() if l.strip()]
    assert len(rows) == 4


def test_A_alibaba_financial_report_revenue_and_pagination_needed():
    data = _load("financial_report_long_table.json")
    env = normalize_gateway_envelope(data["invoke"])
    assert env["rows"], "must read rows from nested envelope"
    adapted = adapt_mcp_result(
        capability="financial.reported_metric_bundle",
        raw_result=data["invoke"],
        task_spec={"periods": ["2026-03-31", "FY2026"]},
    )
    assert adapted["metrics"].get("revenue") is not None
    # attributable_profit absent on truncated first page → pagination-needed, not MCP fail
    assert adapted.get("pagination_needed") is True
    assert "attributable_profit" not in (adapted.get("metrics") or {})
    assert env.get("execution_success") is not False


def test_B_hsbc_analysis_indicator_eps_satisfied():
    data = _load("financial_analysis_indicator.json")
    adapted = adapt_mcp_result(
        capability="financial.reported_metric_bundle",
        raw_result=data["invoke"],
        task_spec={"periods": list(data.get("task_spec_periods") or [])},
    )
    check = required_outputs_present(
        adapted, ["basic_eps", "currency", "report_date"]
    )
    assert check["ok"] is True
    assert adapted["basic_eps"] is not None
    assert adapted["currency"] is not None
    assert adapted["report_date"] is not None


def test_C_tencent_market_chinese_fields_cutoff():
    data = _load("market_ohlcv_chinese.json")
    adapted = adapt_mcp_result(
        capability="market.latest_close",
        raw_result=data["invoke"],
        arguments={"symbol": "00700"},
        task_spec={"cutoff_date": data.get("cutoff_date") or "2026-08-12"},
    )
    assert adapted["trade_date"] is not None
    assert adapted["close"] is not None
    assert adapted["volume"] is not None
    assert adapted["trade_date"] <= "2026-08-12"
    check = required_outputs_present(
        adapted, ["trade_date", "close", "volume"]
    )
    assert check["ok"] is True


def test_D_disclosure_prefix_before_binder():
    assert disclosure_prefix_needed(
        required_capability="financial.official_narrative_evidence",
        argument_slots={},
        available_deps={},
        contracts={},
    )
    rewritten = rewrite_step_to_disclosure_list(
        {
            "required_capability": "financial.official_narrative_evidence",
            "required_outputs": ["snippet", "local_key"],
            "argument_slots": {},
            "notes": None,
        }
    )
    assert rewritten["required_capability"] == "disclosure.list"
    assert "local_key" in rewritten["required_outputs"]
    assert "disclosure_prefix_for:financial.official_narrative_evidence" in (
        rewritten.get("notes") or ""
    )


def test_E_cross_domain_parallel_ready_continues():
    state = PlanState.from_initial_plan(
        {
            "steps": [
                {
                    "step_id": "s1_eps",
                    "step_type": "retrieve",
                    "objective": "eps",
                    "depends_on": [],
                    "mandatory": True,
                    "success_criteria": "eps",
                    "required_capability": "financial.reported_metric",
                    "required_outputs": ["basic_eps"],
                    "argument_slots": {},
                    "compute_spec": None,
                },
                {
                    "step_id": "s2_volume",
                    "step_type": "retrieve",
                    "objective": "volume",
                    "depends_on": [],
                    "mandatory": True,
                    "success_criteria": "volume",
                    "required_capability": "market.daily_ohlcv",
                    "required_outputs": ["volume", "trade_date"],
                    "argument_slots": {},
                    "compute_spec": None,
                },
            ]
        }
    )
    state.mark_running("s1_eps")
    state.mark_failed("s1_eps", "required_outputs_missing")
    state._recompute_ready()
    assert any(s.step_id == "s2_volume" for s in state.ready_steps())
    # Exceptional replan deferred while independent branch remains ready.
    assert (
        needs_exceptional_replan(
            plan_state=state,
            last_classification={
                "action": "exception",
                "trigger": "required_output_missing",
            },
        )
        is None
    )


def test_replay_fixtures_have_no_task_id_branching_in_module():
    src = Path(__file__).read_text(encoding="utf-8")
    body = "\n".join(
        ln
        for ln in src.splitlines()
        if not ln.lstrip().startswith("#") and "assert" not in ln
    )
    assert "if task_id" not in body
    assert "elif task_id" not in body
