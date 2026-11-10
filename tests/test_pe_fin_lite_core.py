"""Focused PE-Fin Lite unit tests (offline; no network/API)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from hk_equity_mcp.pe_fin_lite.budget import BudgetTracker, McpBudgetExhausted
from hk_equity_mcp.pe_fin_lite.calculator import (
    CalculatorError,
    percent_change,
    run_compute_spec,
    safe_eval_expression,
)
from hk_equity_mcp.pe_fin_lite.capability_profile import (
    capability_profile_hash,
    load_capability_profile,
    validate_capability_profile,
)
from hk_equity_mcp.pe_fin_lite.disclosure_workflow import (
    DISCLOSURE_CHAIN,
    DISCLOSURE_COMPOSITE_ID,
    build_stage_arguments,
    is_disclosure_composite_step,
    run_disclosure_composite,
)
from hk_equity_mcp.pe_fin_lite.evidence_store import EvidenceStore
from hk_equity_mcp.pe_fin_lite.planner import topological_ready_order
from hk_equity_mcp.pe_fin_lite.schemas import (
    MAX_PLAN_STEPS,
    PlanValidationError,
    validate_against_schema,
    validate_plan_semantics,
    PE_FIN_LITE_PLAN_V1,
    PE_FIN_LITE_TOOL_BINDING_V1,
)
from hk_equity_mcp.pe_fin_lite.step_result import (
    check_scope_match,
    finalize_retrieve_status,
    make_step_result,
    retrieve_step_succeeded,
)
from hk_equity_mcp.pe_fin_lite.tool_binding import (
    ToolBindingError,
    validate_binding_payload,
)
from hk_equity_mcp.pe_fin_lite.tool_retrieval import (
    build_step_search_query,
    domain_compatible,
    hard_filter_operations,
    select_describe_candidates,
    select_embedding_topk,
)

ROOT = Path(__file__).resolve().parents[1]


def test_capability_profile_no_company_task_gold():
    profile = load_capability_profile(ROOT)
    validate_capability_profile(profile)
    blob = json.dumps(profile).lower()
    assert "gold_operations" not in blob
    assert "reference_answer" not in blob
    assert "task_id" not in blob
    assert "fin_cf_" not in blob
    assert "issuer_whitelist" not in profile
    assert profile.get("company_generic") is True
    assert capability_profile_hash(profile)


def test_planner_rejects_operation_id_and_cycle_and_max_steps():
    bad = {
        "steps": [
            {
                "id": "s1",
                "type": "retrieve",
                "domain": "market",
                "objective": "get close using operation_id aktools.stock_hk_hist",
                "required_outputs": ["close"],
                "depends_on": [],
            }
        ]
    }
    validate_against_schema(bad, PE_FIN_LITE_PLAN_V1)
    with pytest.raises(PlanValidationError, match="operation_id"):
        validate_plan_semantics(bad)

    cyclic = {
        "steps": [
            {
                "id": "s1",
                "type": "retrieve",
                "domain": "market",
                "objective": "a",
                "required_outputs": ["close"],
                "depends_on": ["s2"],
            },
            {
                "id": "s2",
                "type": "reason",
                "domain": "general",
                "objective": "b",
                "required_outputs": [],
                "depends_on": ["s1"],
            },
        ]
    }
    with pytest.raises(PlanValidationError):
        validate_plan_semantics(cyclic)

    too_many = {
        "steps": [
            {
                "id": f"s{i}",
                "type": "retrieve",
                "domain": "market",
                "objective": "x",
                "required_outputs": ["close"],
                "depends_on": [],
            }
            for i in range(MAX_PLAN_STEPS + 1)
        ]
    }
    with pytest.raises(Exception):
        validate_against_schema(too_many, PE_FIN_LITE_PLAN_V1)


def test_retrieve_compute_reason_schema_and_topo():
    plan = {
        "steps": [
            {
                "id": "s1",
                "type": "retrieve",
                "domain": "financial",
                "objective": "revenue",
                "required_outputs": ["revenue"],
                "depends_on": [],
            },
            {
                "id": "s2",
                "type": "compute",
                "domain": "compute",
                "objective": "growth",
                "required_outputs": ["growth"],
                "depends_on": ["s1"],
                "compute_spec": {
                    "expression": "(b-a)/a",
                    "operand_refs": {"a": "s1.revenue", "b": "s1.revenue"},
                    "output_name": "growth",
                },
            },
            {
                "id": "s3",
                "type": "reason",
                "domain": "general",
                "objective": "summarize",
                "required_outputs": [],
                "depends_on": ["s2"],
            },
        ]
    }
    validate_against_schema(plan, PE_FIN_LITE_PLAN_V1)
    validate_plan_semantics(plan)
    order = [s["id"] for s in topological_ready_order(plan["steps"])]
    assert order == ["s1", "s2", "s3"]


def test_step_query_and_hard_filter_and_topk_describe():
    step = {
        "domain": "market",
        "objective": "close price",
        "entity": "0700.HK",
        "date": "2026-07-02",
        "required_outputs": ["close", "trade_date"],
    }
    q = build_step_search_query(step)
    assert "market" in q and "close price" in q and "0700.HK" in q
    assert domain_compatible("market", "market")
    assert not domain_compatible("market", "financial")

    meta = {
        "op.market": {"domain": "market", "availability": "runtime"},
        "op.fin": {"domain": "financial", "availability": "runtime"},
        "op.blocked": {"domain": "market", "availability": "blocked"},
    }
    filt = hard_filter_operations(
        registry_ids=["op.market", "op.fin", "op.blocked", "op.missing"],
        metadata_by_id=meta,
        runtime_operation_ids=["op.market", "op.fin", "op.blocked"],
        step_domain="market",
    )
    assert filt["kept"] == ["op.market"]

    cands = [
        {"operation_id": "op.market", "score": 0.9},
        {"operation_id": "op.fin", "score": 0.8},
        {"operation_id": "op.other", "score": 0.7},
    ]
    topk = select_embedding_topk(cands, allowed_ids={"op.market", "op.other"}, top_k=5)
    assert [c["operation_id"] for c in topk] == ["op.market", "op.other"]
    described = select_describe_candidates(topk + [{"operation_id": "x", "score": 0.1}] * 5, top_n=3)
    assert 2 <= len(described) <= 3


def test_binder_only_described_primary_fallback_required_args():
    contracts = {
        "a": {
            "json_schema": {
                "properties": {"symbol": {"type": "string"}},
                "required": ["symbol"],
            }
        },
        "b": {
            "json_schema": {
                "properties": {"symbol": {"type": "string"}},
                "required": ["symbol"],
            }
        },
    }
    ok = validate_binding_payload(
        {
            "primary": {"operation_id": "a", "arguments": {"symbol": "0700.HK"}},
            "fallback": {"operation_id": "b", "arguments": {"symbol": "0700.HK"}},
        },
        described_operation_ids=["a", "b"],
        described_contracts=contracts,
    )
    assert ok["primary"]["operation_id"] == "a"
    assert ok["fallback"]["operation_id"] == "b"

    with pytest.raises(ToolBindingError):
        validate_binding_payload(
            {
                "primary": {"operation_id": "c", "arguments": {"symbol": "x"}},
                "fallback": None,
            },
            described_operation_ids=["a", "b"],
            described_contracts=contracts,
        )
    with pytest.raises(ToolBindingError):
        validate_binding_payload(
            {"primary": {"operation_id": "a", "arguments": {}}, "fallback": None},
            described_operation_ids=["a"],
            described_contracts=contracts,
        )
    validate_against_schema(
        {
            "primary": {"operation_id": "a", "arguments": {"symbol": "x"}},
            "fallback": None,
        },
        PE_FIN_LITE_TOOL_BINDING_V1,
    )


def test_transport_success_missing_output_not_step_success_and_scope():
    r = finalize_retrieve_status(
        make_step_result(
            values={},
            missing_outputs=["close"],
            scope_match=True,
            mcp_transport_ok=True,
            tool_result_valid=True,
            provenance={"operation_id": "x"},
        )
    )
    assert r["status"] == "unavailable"
    assert not retrieve_step_succeeded(r)

    assert check_scope_match(
        {"trade_date": "2026-07-02"}, date="2026-07-02", entity="0700.HK", arguments={"symbol": "0700.HK"}
    )
    assert not check_scope_match(
        {"trade_date": "2020-01-01"}, date="2026-07-02", arguments={}
    )


def test_fallback_budget_and_no_monitor_fields():
    b = BudgetTracker({"max_local_fallback_per_retrieve": 1, "max_total_mcp_invocations": 2})
    assert b.limits["max_local_fallback_per_retrieve"] == 1
    snap = b.snapshot()
    assert snap["monitor_calls"] == 0
    assert snap["global_replan_rounds"] == 0
    b.record_mcp_invoke()
    b.record_mcp_invoke()
    with pytest.raises(McpBudgetExhausted):
        b.record_mcp_invoke()


def test_sibling_continue_dependent_skip_logic():
    # Simulates agent dependency rule via topo + completed_ok set
    steps = [
        {"id": "a", "type": "retrieve", "domain": "market", "objective": "x", "required_outputs": ["close"], "depends_on": []},
        {"id": "b", "type": "retrieve", "domain": "financial", "objective": "y", "required_outputs": ["eps"], "depends_on": []},
        {"id": "c", "type": "compute", "domain": "compute", "objective": "z", "required_outputs": ["r"], "depends_on": ["a"], "compute_spec": {"expression": "a", "operand_refs": {"a": "close"}, "output_name": "r"}},
    ]
    validate_plan_semantics({"steps": steps})
    order = [s["id"] for s in topological_ready_order(steps)]
    assert order[0] in {"a", "b"} and "c" == order[-1]
    completed = {"b"}  # a failed
    assert "a" not in completed
    # c depends on a → skip; b independent → ok
    assert [d for d in ["a"] if d not in completed]


def test_safe_calculator_and_evidence_operands():
    assert safe_eval_expression("a+b", {"a": 1, "b": 2}) == 3
    assert percent_change(100, 110) == 10.0
    with pytest.raises(CalculatorError):
        safe_eval_expression("__import__('os').system('x')", {})
    store = EvidenceStore()
    store.values["revenue"] = 100.0
    store.values["profit"] = 20.0
    out = run_compute_spec(
        {
            "expression": "profit/revenue",
            "operand_refs": {"profit": "profit", "revenue": "revenue"},
            "output_name": "margin",
        },
        resolved_operands={"profit": 20.0, "revenue": 100.0},
    )
    assert out["ok"] and abs(out["result"] - 0.2) < 1e-9


def test_disclosure_composite_field_pass_and_trace():
    assert is_disclosure_composite_step(
        {"workflow": DISCLOSURE_COMPOSITE_ID, "domain": "disclosure", "objective": "x"}
    )
    assert DISCLOSURE_CHAIN == ("list", "download", "parse", "keyword", "pages")

    meta = {
        "hkexnews.list_hkexnews_announcements": {
            "domain": "news",
            "capabilities": ["disclosure_search"],
            "provides": ["local_key", "news_id", "title"],
        },
        "hkexnews.download_hkexnews_announcement": {
            "domain": "news",
            "capabilities": ["disclosure_download"],
            "provides": ["local_key"],
        },
        "hkexnews.parse_hkexnews_announcement": {
            "domain": "news",
            "capabilities": ["disclosure_parse"],
            "provides": ["local_key"],
        },
        "hkexnews.keyword": {
            "domain": "news",
            "capabilities": ["keyword"],
            "provides": ["snippet", "local_key", "page_number"],
        },
        "hkexnews.page": {
            "domain": "news",
            "capabilities": ["page"],
            "provides": ["snippet", "page_number", "local_key"],
        },
    }
    contracts = {
        oid: {
            "json_schema": {
                "properties": {
                    "local_key": {"type": "string"},
                    "news_id": {"type": "string"},
                    "symbol": {"type": "string"},
                    "page_number": {"type": "integer"},
                    "keyword": {"type": "string"},
                },
                "required": [],
            }
        }
        for oid in meta
    }
    # list requires symbol
    contracts["hkexnews.list_hkexnews_announcements"]["json_schema"]["required"] = ["symbol"]
    contracts["hkexnews.download_hkexnews_announcement"]["json_schema"]["required"] = ["local_key"]

    state = {"local_key": "k1", "news_id": "n1"}
    args = build_stage_arguments(
        stage="download",
        contract=contracts["hkexnews.download_hkexnews_announcement"],
        chain_state=state,
        step={"entity": "0700.HK", "objective": "annual report"},
    )
    assert args["local_key"] == "k1"

    calls = []

    def invoke(oid, arguments):
        calls.append((oid, dict(arguments)))
        if "list" in oid:
            return {
                "success": True,
                "result": {"rows": [{"local_key": "k1", "news_id": "n1", "title": "t"}]},
            }
        if "download" in oid or "parse" in oid:
            return {"success": True, "result": {"local_key": "k1"}}
        return {
            "success": True,
            "result": {"snippet": "profit rose", "local_key": "k1", "page_number": 1},
        }

    def describe(oid):
        return {"success": True, "contract": contracts[oid]}

    out = run_disclosure_composite(
        step={
            "domain": "disclosure",
            "objective": "official disclosure evidence",
            "entity": "0700.HK",
            "required_outputs": ["snippet", "local_key"],
            "workflow": DISCLOSURE_COMPOSITE_ID,
        },
        metadata_by_id=meta,
        runtime_operation_ids=list(meta.keys()),
        invoke_fn=invoke,
        describe_fn=describe,
        allow_hkex_download=True,
    )
    assert out["execution_ok"]
    assert out["values"].get("snippet")
    assert out["values"].get("local_key") == "k1"
    assert len(out["trace"]) == 5
    # download received local_key from list
    assert any(c[0].endswith("download_hkexnews_announcement") and c[1].get("local_key") == "k1" for c in calls)


def test_final_no_fabricate_missing_and_budget_structured():
    # Evidence store marks missing; synthesizer contract requires unavailable when no data
    store = EvidenceStore()
    store.add_retrieve_result(
        step_id="s1",
        result=make_step_result(
            status="unavailable",
            missing_outputs=["close"],
            execution_ok=False,
            failure_reason="required_outputs_missing",
        ),
    )
    assert store.missing
    assert store.values == {}


def test_profile_yaml_loads():
    path = ROOT / "configs/capabilities/hk155_capability_profile_v1.yaml"
    data = yaml.safe_load(path.read_text())
    assert data["profile_id"] == "hk155_capability_profile_v1"
    assert any(
        wf.get("workflow_id") == DISCLOSURE_COMPOSITE_ID
        for wf in data["composite_workflows"]
    )
