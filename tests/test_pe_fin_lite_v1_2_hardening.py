"""PE-Fin Lite v1.2 final pre-v3 engineering hardening (offline / mock only)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from hk_equity_mcp.pe_fin_lite import PE_FIN_LITE_METHOD_VERSION
from hk_equity_mcp.pe_fin_lite.disclosure_workflow import (
    DISCLOSURE_COMPOSITE_ID,
    build_stage_arguments,
    derive_period_date_window,
    enrich_disclosure_grounding,
    filter_list_rows,
    needs_document_text,
    rank_list_rows,
    run_disclosure_composite,
    select_list_row,
)
from hk_equity_mcp.pe_fin_lite.evidence_store import EvidenceStore
from hk_equity_mcp.pe_fin_lite.executor import execute_reason_step
from hk_equity_mcp.pe_fin_lite.identity import protocol_identity_hashes
from hk_equity_mcp.pe_fin_lite.tool_binding import binder_protocol_hash, run_tool_binder
from hk_equity_mcp.real_mcp.adapters.hkexnews import HkexnewsAdapter

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = (
    ROOT
    / "tests"
    / "fixtures"
    / "apv_fin_v7_envelopes"
    / "disclosure_list_hkex.json"
)


def _valid_reason_json() -> str:
    return json.dumps(
        {
            "summary": "Structured summary after repair.",
            "claims": ["claim-a"],
            "gaps": [],
        }
    )


def test_reason_direct_success():
    store = EvidenceStore()
    llm = MagicMock()
    llm.complete.return_value = {"content": _valid_reason_json(), "usage": {"total_tokens": 2}}
    out = execute_reason_step(
        step={"id": "s3", "type": "reason", "objective": "summarize", "depends_on": []},
        store=store,
        llm=llm,
        question="q",
    )
    assert out["reason_parse_failed"] is False
    assert out["runtime_recovered"] is False
    assert out["repair_count"] == 0
    assert out["step_result"]["status"] == "completed"
    assert out["initial_parse_error"] is None
    assert llm.complete.call_count == 1


def test_reason_initial_fail_valid_repair_clears_sticky_flag():
    store = EvidenceStore()
    llm = MagicMock()
    llm.complete.side_effect = [
        {"content": "NOT_JSON{{{", "usage": {"total_tokens": 1}},
        {"content": _valid_reason_json(), "usage": {"total_tokens": 2}},
    ]
    out = execute_reason_step(
        step={"id": "s3", "type": "reason", "objective": "summarize", "depends_on": []},
        store=store,
        llm=llm,
        question="q",
    )
    assert out["reason_parse_failed"] is False
    assert out["runtime_recovered"] is True
    assert out["repair_count"] == 1
    assert out["step_result"]["status"] == "completed"
    assert out["initial_parse_error"]
    assert "Structured summary" in str(out["reason_artifact"]["summary"])
    assert llm.complete.call_count == 2


def test_reason_initial_fail_invalid_repair_stage_failed():
    store = EvidenceStore()
    llm = MagicMock()
    llm.complete.side_effect = [
        {"content": "NOT_JSON{{{", "usage": {"total_tokens": 1}},
        {"content": "still broken", "usage": {"total_tokens": 1}},
    ]
    out = execute_reason_step(
        step={"id": "s3", "type": "reason", "objective": "summarize", "depends_on": []},
        store=store,
        llm=llm,
        question="q",
    )
    assert out["reason_parse_failed"] is True
    assert out["runtime_recovered"] is False
    assert out["repair_count"] == 1
    assert out["step_result"]["status"] == "failed"


def _binder_candidates() -> tuple[list, dict]:
    candidates = [{"operation_id": "market.get_daily"}]
    contracts = {
        "market.get_daily": {
            "json_schema": {
                "properties": {
                    "stock_code": {"type": "string"},
                    "trade_date": {"type": "string"},
                },
                "required": ["stock_code", "trade_date"],
            }
        }
    }
    return candidates, contracts


def _valid_binding_json() -> str:
    return json.dumps(
        {
            "primary": {
                "operation_id": "market.get_daily",
                "arguments": {"stock_code": "00700", "trade_date": "2026-08-12"},
            },
            "fallback": None,
            "notes": "ok",
        }
    )


def test_binder_empty_content_one_repair_success():
    candidates, contracts = _binder_candidates()
    llm = MagicMock()
    llm.complete.side_effect = [
        {
            "content": "",
            "usage": {"total_tokens": 1},
            "finish_reason": "stop",
            "content_was_null": False,
        },
        {
            "content": _valid_binding_json(),
            "usage": {"total_tokens": 3},
            "finish_reason": "stop",
        },
    ]
    out = run_tool_binder(
        llm=llm,
        question="close?",
        step={"id": "s1", "type": "retrieve", "objective": "close", "depends_on": []},
        described_candidates=candidates,
        described_contracts=contracts,
        evidence={},
    )
    assert out["ok"] is True
    assert out["repair_count"] == 1
    assert out["runtime_recovered"] is True
    assert out["initial_parse_error"] == "empty content"
    obs = out["call_observability"]
    assert len(obs) == 2
    assert obs[0]["content_present"] is False
    assert obs[0]["content_length"] == 0
    assert "finish_reason" in obs[0]
    assert obs[1]["parse_result"] == "ok"


def test_binder_null_then_invalid_repair_structured_failure():
    candidates, contracts = _binder_candidates()
    llm = MagicMock()
    llm.complete.side_effect = [
        {"content": None, "usage": {}, "finish_reason": "length", "content_was_null": True},
        {"content": "{not-json", "usage": {}, "finish_reason": "stop"},
    ]
    out = run_tool_binder(
        llm=llm,
        question="close?",
        step={"id": "s1", "type": "retrieve", "objective": "close", "depends_on": []},
        described_candidates=candidates,
        described_contracts=contracts,
        evidence={},
    )
    assert out["ok"] is False
    assert out["repair_count"] == 1
    assert out["binding"] is None
    assert out["error"]
    assert len(out["call_observability"]) == 2
    assert llm.complete.call_count == 2  # no infinite repair


def test_binder_direct_success_no_repair():
    candidates, contracts = _binder_candidates()
    llm = MagicMock()
    llm.complete.return_value = {
        "content": _valid_binding_json(),
        "usage": {"total_tokens": 2},
        "finish_reason": "stop",
    }
    out = run_tool_binder(
        llm=llm,
        question="close?",
        step={"id": "s1", "type": "retrieve", "objective": "close", "depends_on": []},
        described_candidates=candidates,
        described_contracts=contracts,
        evidence={},
    )
    assert out["ok"] is True
    assert out["repair_count"] == 0
    assert out["runtime_recovered"] is False
    assert llm.complete.call_count == 1


def test_period_window_and_list_argument_propagation():
    from_d, to_d, rule = derive_period_date_window("FY2025", cutoff="2026-08-12")
    assert from_d == "2025-01-01"
    assert to_d == "2026-08-12"
    assert rule and "FY" in rule
    g = enrich_disclosure_grounding(
        {
            "stock_code": "00700",
            "period": "FY2025",
            "title_keyword": "non-IFRS net income",
        },
        step={"objective": "non-IFRS net income from annual results", "period": "FY2025"},
        question="Issuer 00700 FY2025 non-IFRS net income",
        cutoff="2026-08-12",
    )
    assert g["from_date"] == "2025-01-01"
    assert g["to_date"] == "2026-08-12"
    assert g["title_is_metric_phrase"] is True
    assert g["list_title_keyword"] and "result" in g["list_title_keyword"].lower()
    args = build_stage_arguments(
        stage="list",
        contract={
            "json_schema": {
                "properties": {
                    "stock_code": {"type": "string"},
                    "start_date": {"type": "string"},
                    "end_date": {"type": "string"},
                    "title_keyword": {"type": "string"},
                },
                "required": ["stock_code"],
            }
        },
        chain_state={},
        step={"objective": "results", "scope_constraints": {}},
        grounding=g,
    )
    assert args["stock_code"] == "00700"
    assert args["start_date"] == "2025-01-01"
    assert args["end_date"] == "2026-08-12"
    assert "non-ifrs" not in str(args.get("title_keyword") or "").lower()
    assert "result" in str(args.get("title_keyword") or "").lower()


def test_explicit_dates_propagated_for_agm():
    g = enrich_disclosure_grounding(
        {
            "stock_code": "09988",
            "from_date": "2026-07-01",
            "to_date": "2026-08-12",
            "title_keyword": "AGM notice",
        },
        step={"objective": "AGM notice", "scope_constraints": {}},
        question="09988 AGM notice",
    )
    assert g["document_type_intent"] == "agm_notice"
    args = build_stage_arguments(
        stage="list",
        contract={
            "json_schema": {
                "properties": {
                    "stock_code": {"type": "string"},
                    "from_date": {"type": "string"},
                    "to_date": {"type": "string"},
                    "title_keyword": {"type": "string"},
                },
                "required": ["stock_code"],
            }
        },
        chain_state={},
        step={"objective": "AGM notice", "scope_constraints": {}},
        grounding=g,
    )
    assert args["from_date"] == "2026-07-01"
    assert args["to_date"] == "2026-08-12"
    assert args.get("title_keyword")


def test_cutoff_rejects_post_cutoff_tender_offer():
    rows = [
        {
            "title": "HSBC HOLDINGS PLC - TENDER OFFERS",
            "release_date": "2026-08-13",
            "news_id": "1",
            "local_key": "00005_tender",
            "issuer_code": "00005",
        },
        {
            "title": "2025 ANNUAL RESULTS",
            "release_date": "2026-02-19",
            "news_id": "2",
            "local_key": "00005_results",
            "issuer_code": "00005",
        },
    ]
    g = enrich_disclosure_grounding(
        {
            "stock_code": "00005",
            "period": "FY2025",
            "title_keyword": "results",
        },
        step={"objective": "FY2025 results", "scope_constraints": {}},
        question="00005 FY2025 results",
        cutoff="2026-08-12",
    )
    filtered = filter_list_rows(rows, grounding=g, step={"scope_constraints": {}})
    assert all(r["release_date"] <= "2026-08-12" for r in filtered)
    selected = select_list_row(rows, grounding=g, step={"scope_constraints": {}})
    assert selected is not None
    assert "RESULTS" in str(selected.get("title") or "").upper()
    assert "TENDER" not in str(selected.get("title") or "").upper()
    ranked = rank_list_rows(rows, grounding=g, step={"scope_constraints": {}})
    assert any(
        r.get("reason") == "beyond_benchmark_cutoff" for r in ranked["rejected"]
    )


def test_deterministic_title_broadening_local():
    rows = [
        {
            "title": "ANNUAL RESULTS FOR THE YEAR ENDED 31 DECEMBER 2025",
            "release_date": "2026-03-19",
            "news_id": "9",
            "local_key": "00700_ar",
            "issuer_code": "00700",
        }
    ]
    meta = {
        "hkexnews.list_hkexnews_announcements": {
            "domain": "news",
            "capabilities": ["disclosure_search", "list"],
            "provides": ["local_key", "news_id", "title", "release_date"],
        }
    }
    contracts = {
        "hkexnews.list_hkexnews_announcements": {
            "json_schema": {
                "properties": {
                    "stock_code": {"type": "string"},
                    "start_date": {"type": "string"},
                    "end_date": {"type": "string"},
                    "title_keyword": {"type": "string"},
                },
                "required": ["stock_code"],
            }
        }
    }
    calls: List[Dict[str, Any]] = []

    def invoke(oid: str, args: Dict[str, Any]) -> Dict[str, Any]:
        calls.append(dict(args))
        # First call may include title; return rows regardless (local broaden path).
        return {"success": True, "result": {"announcements": rows}}

    def describe(oid: str) -> Dict[str, Any]:
        return {"contract": contracts[oid]}

    out = run_disclosure_composite(
        step={
            "domain": "disclosure",
            "workflow": DISCLOSURE_COMPOSITE_ID,
            "objective": "non-IFRS net income",
            "required_outputs": ["news_id", "title", "release_date", "local_key"],
            "depends_on": [],
            "scope_constraints": {
                "entity": {"code": "00700"},
                "period": "FY2025",
                "title_keyword": "non-IFRS",
            },
        },
        metadata_by_id=meta,
        runtime_operation_ids=list(meta),
        invoke_fn=invoke,
        describe_fn=describe,
        question="00700 FY2025 non-IFRS",
        allow_hkex_download=False,
    )
    assert out["status"] == "completed"
    assert out["values"]["local_key"] == "00700_ar"
    assert out["stopped_after_list"] is True
    assert len(calls) == 1  # local broaden, no second MCP when rows present
    assert "start_date" in calls[0] or "from_date" in calls[0] or "end_date" in calls[0]


def test_shortest_prefix_no_download_when_list_sufficient():
    disc = json.loads(FIXTURE.read_text(encoding="utf-8"))
    meta = {
        "hkexnews.list_hkexnews_announcements": {
            "domain": "news",
            "capabilities": ["list"],
            "provides": ["local_key", "news_id", "title", "release_date"],
        },
        "hkexnews.download_hkexnews_announcement": {
            "domain": "news",
            "capabilities": ["download"],
            "provides": ["local_key"],
        },
    }
    contracts = {
        "hkexnews.list_hkexnews_announcements": {
            "json_schema": {
                "properties": {
                    "stock_code": {"type": "string"},
                    "start_date": {"type": "string"},
                    "end_date": {"type": "string"},
                    "title_keyword": {"type": "string"},
                },
                "required": ["stock_code"],
            }
        },
        "hkexnews.download_hkexnews_announcement": {
            "json_schema": {
                "properties": {"stock_code": {"type": "string"}, "local_key": {"type": "string"}},
                "required": ["stock_code"],
            }
        },
    }
    called: List[str] = []

    def invoke(oid: str, args: Dict[str, Any]) -> Dict[str, Any]:
        called.append(oid)
        if "list" in oid:
            return {"success": True, "result": disc["invoke"]["result"]}
        raise AssertionError(f"download not allowed: {oid}")

    def describe(oid: str) -> Dict[str, Any]:
        return {"contract": contracts[oid]}

    assert needs_document_text(["news_id", "title", "release_date"]) is False
    out = run_disclosure_composite(
        step={
            "domain": "disclosure",
            "workflow": DISCLOSURE_COMPOSITE_ID,
            "objective": "Next Day Disclosure Return",
            "required_outputs": ["news_id", "title", "release_date", "local_key"],
            "depends_on": [],
            "scope_constraints": {
                "entity": {"code": "00700"},
                "from_date": "2026-07-01",
                "to_date": "2026-08-12",
                "release_date": "2026-07-09",
                "title_keyword": "Next Day Disclosure Return",
            },
        },
        metadata_by_id=meta,
        runtime_operation_ids=list(meta),
        invoke_fn=invoke,
        describe_fn=describe,
        question="00700 Next Day Disclosure Return 2026-07-09",
        allow_hkex_download=True,
    )
    assert out["stopped_after_list"] is True
    assert all("download" not in c for c in called)


def test_per_task_download_budget_resets():
    adapter = HkexnewsAdapter(allow_download=True, persistent=False)
    adapter._download_count = 1
    assert adapter._download_count >= adapter._download_budget_per_task
    adapter.reset_download_budget()
    assert adapter._download_count == 0
    # First download after reset is not exhausted.
    blocked = adapter._download_count >= adapter._download_budget_per_task
    assert blocked is False


def test_method_version_and_protocol_hashes():
    assert PE_FIN_LITE_METHOD_VERSION == "pe_fin_lite_v1.2"
    hashes = protocol_identity_hashes()
    assert len(hashes["disclosure_workflow_hash"]) == 64
    assert len(hashes["binder_protocol_hash"]) == 64
    assert len(hashes["reason_protocol_hash"]) == 64
    assert len(binder_protocol_hash()) == 64
    assert hashes["metrics_version"] == "pe_fin_lite_metrics_v3"


def test_xd_dev_002_eps_limitation_documented_not_patched():
    """Research limitation: operation completeness ≠ metric-basis/unit correctness."""
    # Guard: no HSBC / gold-specific EPS patch files introduced this round.
    bad = list(ROOT.glob("**/hsbc*eps*patch*")) + list(ROOT.glob("**/xd_dev_002*fix*"))
    assert bad == []
