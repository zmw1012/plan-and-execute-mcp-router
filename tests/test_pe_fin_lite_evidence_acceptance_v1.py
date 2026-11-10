"""Offline tests: provenance, required-output acceptance, partial merge, disclosure prefix, runtime-owned schema."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from hk_equity_mcp.pe_fin_lite.disclosure_workflow import (
    needs_document_text,
    run_disclosure_composite,
    select_list_row,
)
from hk_equity_mcp.pe_fin_lite.evidence_provenance import (
    apply_deterministic_derivations,
    derive_fiscal_period_from_annual_report_date,
    evidence_acceptance_hash,
    grounded_context_from_arguments,
    make_field_provenance,
    merge_field_maps,
    observed_field_map,
    provenance_contract_hash,
    required_outputs_satisfaction,
    scopes_compatible,
    values_from_field_map,
)
from hk_equity_mcp.pe_fin_lite.executor import (
    _build_accepted_field_map,
    merge_partial_retrieve_results,
)
from hk_equity_mcp.pe_fin_lite.identity import FORMAL_OUTPUT_DIRS, METRICS_VERSION
from hk_equity_mcp.pe_fin_lite.schemas import (
    SCHEMA_VERSIONS,
    PlanValidationError,
    parse_and_validate,
    strip_runtime_owned_fields,
)
from hk_equity_mcp.pe_fin_lite.step_result import retrieve_step_succeeded

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "apv_fin_v7_envelopes"


def test_runtime_owned_schema_version_variants():
    base_steps = [
        {
            "id": "s1",
            "type": "retrieve",
            "domain": "market",
            "objective": "Refuse latest close beyond cutoff",
            "required_outputs": ["close"],
            "depends_on": [],
        }
    ]
    # missing schema_version
    ok, _, err = parse_and_validate(json.dumps({"steps": base_steps}), "plan")
    assert ok is not None and err is None
    assert ok["schema_version"] == SCHEMA_VERSIONS["plan"]

    # stale plan_v1 / pe_fin_lite_plan_v1 / bare plan_v1 label
    for ver in ("plan_v1", "pe_fin_lite_plan_v1", "pe_fin_lite_plan_v2"):
        ok, _, err = parse_and_validate(
            json.dumps({"schema_version": ver, "steps": base_steps}), "plan"
        )
        assert ok is not None, err
        assert ok["schema_version"] == SCHEMA_VERSIONS["plan"]

    # semantic invalid still rejected
    bad, _, err = parse_and_validate(
        json.dumps(
            {
                "schema_version": "plan_v1",
                "steps": [
                    {
                        "id": "s1",
                        "type": "retrieve",
                        "domain": "market",
                        "objective": "x with operation_id foo.bar",
                        "required_outputs": ["close"],
                        "depends_on": [],
                    }
                ],
            }
        ),
        "plan",
    )
    assert bad is None
    assert "operation_id" in str(err)

    # unknown property still fail-closed
    bad2, _, err2 = parse_and_validate(
        json.dumps({"steps": base_steps, "gold_answer": "1"}), "plan"
    )
    assert bad2 is None
    assert "unexpected" in str(err2).lower() or "gold" in str(err2).lower() or err2

    stripped = strip_runtime_owned_fields(
        {"schema_version": "plan_v1", "method_version": "x", "steps": []}, "plan"
    )
    assert "schema_version" not in stripped
    assert "method_version" not in stripped


def test_fiscal_period_derived_from_annual_report_date():
    derived = derive_fiscal_period_from_annual_report_date(
        report_date="2026-03-31", indicator="年度"
    )
    assert derived is not None
    assert derived["value"] == "FY2026"
    assert derived["provenance_class"] == "derived"

    fmap = observed_field_map(
        {
            "revenue": 1,
            "attributable_profit": 2,
            "report_date": "2026-03-31",
            "currency": "人民币",
        }
    )
    out = apply_deterministic_derivations(
        fmap,
        arguments={"indicator": "年度", "symbol": "09988"},
        step={"scope_constraints": {"period": "FY2026"}},
    )
    assert out["fiscal_period"]["value"] == "FY2026"
    sat = required_outputs_satisfaction(
        out,
        ["revenue", "attributable_profit", "report_date", "fiscal_period", "currency"],
    )
    assert sat["ok"] is True


def test_market_grounded_issuer_and_adjustment():
    accepted = _build_accepted_field_map(
        projected_values={"trade_date": "2026-06-30", "volume": 13123569},
        required_outputs=["issuer_code", "trade_date", "volume", "adjustment"],
        step={
            "domain": "market",
            "scope_constraints": {"adjustment": "", "trade_date": "2026-06-30"},
        },
        arguments={"symbol": "00005", "adjust": "", "start_date": "20260630"},
        operation_id="aktools.stock_hk_hist",
        question_grounding={"stock_code": "00005"},
    )
    assert accepted["values"]["trade_date"]
    assert accepted["values"]["volume"] == 13123569
    assert accepted["values"]["issuer_code"] == "00005"
    assert accepted["values"]["adjustment"] == ""
    assert accepted["missing_outputs"] == []
    assert accepted["field_map"]["issuer_code"]["provenance_class"] == "grounded_input"
    assert accepted["field_map"]["volume"]["provenance_class"] == "observed"


def test_answer_bearing_cannot_be_grounded_alone():
    grounded = grounded_context_from_arguments(
        arguments={"symbol": "00700"},
        step={},
    )
    # Inject fake grounded revenue — must not satisfy.
    grounded["revenue"] = make_field_provenance(
        value=123, provenance_class="grounded_input", source="question"
    )
    sat = required_outputs_satisfaction(grounded, ["revenue", "issuer_code"])
    assert "revenue" in sat["missing"]
    assert "issuer_code" in sat["present"]


def test_partial_merge_same_scope_and_conflicts():
    primary = {
        "values": {"basic_eps": 1.2, "currency": "USD"},
        "field_map": observed_field_map({"basic_eps": 1.2, "currency": "USD"}),
        "missing_outputs": ["report_date", "company_name"],
        "satisfaction": {
            "present": ["basic_eps", "currency"],
            "missing": ["report_date", "company_name"],
        },
        "scope_fingerprint": {
            "entity": "00005",
            "period": "FY2025",
            "metric_basis": "reported",
            "trade_date": "",
            "adjustment": None,
        },
        "gateway_success": True,
        "mcp_transport_ok": True,
        "tool_result_valid": True,
        "scope_match": True,
        "normalization_success": True,
        "provenance": {"operation_id": "op.primary"},
    }
    fallback = {
        "values": {"report_date": "2025-12-31", "company_name": "HSBC"},
        "field_map": observed_field_map(
            {"report_date": "2025-12-31", "company_name": "HSBC"}
        ),
        "missing_outputs": ["basic_eps", "currency"],
        "satisfaction": {
            "present": ["report_date", "company_name"],
            "missing": ["basic_eps", "currency"],
        },
        "scope_fingerprint": dict(primary["scope_fingerprint"]),
        "gateway_success": True,
        "mcp_transport_ok": True,
        "tool_result_valid": True,
        "scope_match": True,
        "normalization_success": True,
        "provenance": {"operation_id": "op.fallback"},
    }
    merged = merge_partial_retrieve_results(primary, fallback)
    assert merged["values"]["basic_eps"] == 1.2
    assert merged["values"]["report_date"] == "2025-12-31"
    assert merged["values"]["company_name"] == "HSBC"
    assert merged.get("merge_conflicts") == []

    # period mismatch forbids merge
    bad_scope = dict(fallback)
    bad_scope["scope_fingerprint"] = {
        **primary["scope_fingerprint"],
        "period": "FY2024",
    }
    blocked = merge_partial_retrieve_results(primary, bad_scope)
    assert blocked["failure_reason"] == "partial_merge_scope_mismatch"

    # conflicting value forbids merge
    conflict_fb = dict(fallback)
    conflict_fb["field_map"] = observed_field_map(
        {"basic_eps": 9.9, "report_date": "2025-12-31", "company_name": "HSBC"}
    )
    conflict_fb["values"] = values_from_field_map(conflict_fb["field_map"])
    conflicted = merge_partial_retrieve_results(primary, conflict_fb)
    assert conflicted["failure_reason"] == "partial_merge_value_conflict"
    assert conflicted.get("merge_conflicts")


def test_disclosure_shortest_prefix_stops_after_list():
    rows = [
        {
            "news_id": "2026070900123",
            "title": "Next Day Disclosure Return",
            "release_date": "2026-07-09",
            "issuer_code": "00700",
            "local_key": "hkex:00700:2026070900123",
        }
    ]
    selected = select_list_row(
        rows,
        grounding={
            "stock_code": "00700",
            "from_date": "2026-07-01",
            "to_date": "2026-08-12",
            "title_keyword": "Next Day Disclosure Return",
        },
        step={
            "scope_constraints": {
                "from_date": "2026-07-01",
                "to_date": "2026-08-12",
                "title_keyword": "Next Day Disclosure Return",
            }
        },
    )
    assert selected and selected["news_id"] == "2026070900123"
    assert needs_document_text(["news_id", "title", "release_date", "local_key"]) is False
    assert needs_document_text(["snippet", "page_number"]) is True

    meta = {
        "hkexnews.list_hkexnews_announcements": {
            "domain": "news",
            "capabilities": ["disclosure_search", "list"],
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
                    "from_date": {"type": "string"},
                    "to_date": {"type": "string"},
                },
                "required": ["stock_code"],
            }
        },
        "hkexnews.download_hkexnews_announcement": {
            "json_schema": {
                "properties": {"local_key": {"type": "string"}},
                "required": ["local_key"],
            }
        },
    }
    calls: List[str] = []

    def invoke(oid: str, args: Dict[str, Any]) -> Dict[str, Any]:
        calls.append(oid)
        if "list" in oid:
            return {"success": True, "result": {"rows": rows}}
        raise AssertionError("download must not be called when list satisfies outputs")

    def describe(oid: str) -> Dict[str, Any]:
        return {"contract": contracts[oid]}

    out = run_disclosure_composite(
        step={
            "domain": "disclosure",
            "workflow": "retrieve_official_disclosure_evidence",
            "objective": "Find Next Day Disclosure Return",
            "required_outputs": ["news_id", "title", "release_date", "local_key"],
            "depends_on": [],
            "scope_constraints": {
                "entity": {"code": "00700"},
                "from_date": "2026-07-01",
                "to_date": "2026-08-12",
                "title_keyword": "Next Day Disclosure Return",
            },
        },
        metadata_by_id=meta,
        runtime_operation_ids=list(meta),
        invoke_fn=invoke,
        describe_fn=describe,
        question="For Tencent (00700) in 2026-07-01..2026-08-12 titled 'Next Day Disclosure Return'",
        allow_hkex_download=True,
    )
    assert out["stopped_after_list"] is True
    assert out["status"] == "completed"
    assert out["values"]["news_id"] == "2026070900123"
    assert "download" not in "".join(calls)


def test_disclosure_passes_local_key_to_download_parse():
    meta = {
        "hkexnews.list_hkexnews_announcements": {
            "domain": "news",
            "capabilities": ["list"],
            "provides": ["local_key", "news_id", "title"],
        },
        "hkexnews.download_hkexnews_announcement": {
            "domain": "news",
            "capabilities": ["download"],
            "provides": ["local_key"],
        },
        "hkexnews.parse_hkexnews_announcement": {
            "domain": "news",
            "capabilities": ["parse"],
            "provides": ["local_key"],
        },
        "hkexnews.keyword_search_hkexnews_announcement": {
            "domain": "news",
            "capabilities": ["keyword"],
            "provides": ["snippet", "page_number", "local_key"],
        },
    }
    contracts = {
        "hkexnews.list_hkexnews_announcements": {
            "json_schema": {
                "properties": {"stock_code": {"type": "string"}},
                "required": ["stock_code"],
            }
        },
        "hkexnews.download_hkexnews_announcement": {
            "json_schema": {
                "properties": {"local_key": {"type": "string"}},
                "required": ["local_key"],
            }
        },
        "hkexnews.parse_hkexnews_announcement": {
            "json_schema": {
                "properties": {"local_key": {"type": "string"}},
                "required": ["local_key"],
            }
        },
        "hkexnews.keyword_search_hkexnews_announcement": {
            "json_schema": {
                "properties": {
                    "local_key": {"type": "string"},
                    "keyword": {"type": "string"},
                },
                "required": ["local_key"],
            }
        },
    }
    seen_args: Dict[str, Dict[str, Any]] = {}

    def invoke(oid: str, args: Dict[str, Any]) -> Dict[str, Any]:
        seen_args[oid] = dict(args)
        if "list" in oid:
            return {
                "success": True,
                "result": {
                    "rows": [
                        {
                            "news_id": "n1",
                            "title": "ANNUAL RESULTS",
                            "release_date": "2026-03-19",
                            "local_key": "lk-annual-1",
                            "issuer_code": "00700",
                        }
                    ]
                },
            }
        if "download" in oid:
            assert args.get("local_key") == "lk-annual-1"
            return {"success": True, "result": {"local_key": "lk-annual-1"}}
        if "parse" in oid:
            assert args.get("local_key") == "lk-annual-1"
            return {"success": True, "result": {"local_key": "lk-annual-1"}}
        if "keyword" in oid:
            assert args.get("local_key") == "lk-annual-1"
            return {
                "success": True,
                "result": {
                    "snippet": "non-IFRS profit attributable",
                    "page_number": 2,
                    "local_key": "lk-annual-1",
                },
            }
        return {"success": False}

    def describe(oid: str) -> Dict[str, Any]:
        return {"contract": contracts[oid]}

    out = run_disclosure_composite(
        step={
            "domain": "disclosure",
            "workflow": "retrieve_official_disclosure_evidence",
            "objective": "annual results non-IFRS profit",
            "required_outputs": [
                "issuer_code",
                "local_key",
                "snippet",
                "page_number",
            ],
            "depends_on": [],
            "scope_constraints": {
                "entity": {"code": "00700"},
                "title_keyword": "annual results",
            },
        },
        metadata_by_id=meta,
        runtime_operation_ids=list(meta),
        invoke_fn=invoke,
        describe_fn=describe,
        question="Tencent (00700) annual results",
        allow_hkex_download=True,
    )
    assert "lk-annual-1" == out["values"].get("local_key")
    assert out["values"].get("snippet")
    assert out["chain_state"].get("local_key") == "lk-annual-1"
    assert "parse" in str(out.get("trace"))


def test_provenance_and_metrics_identities():
    assert provenance_contract_hash()
    assert evidence_acceptance_hash()
    assert METRICS_VERSION == "pe_fin_lite_metrics_v3"
    assert FORMAL_OUTPUT_DIRS["smoke5_live"].endswith("v3")


def test_scopes_compatible_helper():
    a = {"entity": "00700", "period": "FY2025", "metric_basis": "reported", "trade_date": ""}
    b = {"entity": "700", "period": "FY2025", "metric_basis": "reported", "trade_date": ""}
    assert scopes_compatible(a, b)
    c = {**b, "period": "FY2024"}
    assert not scopes_compatible(a, c)
