"""Offline Disclosure workflow fixture tests (no network / no LLM)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from hk_equity_mcp.pe_fin_lite.disclosure_workflow import (
    DISCLOSURE_COMPOSITE_ID,
    _collect_rows,
    _row_to_values,
    build_stage_arguments,
    disclosure_workflow_hash,
    filter_list_rows,
    needs_document_text,
    run_disclosure_composite,
    select_list_row,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = (
    ROOT
    / "tests"
    / "fixtures"
    / "apv_fin_v7_envelopes"
    / "disclosure_list_hkex.json"
)
RECORDED_SAMPLE = (
    ROOT
    / "docs"
    / "news_mcp_discovery_v1"
    / "hkexnews_list_window_20260701_20260812.json"
)


def _meta_chain() -> Dict[str, Dict[str, Any]]:
    return {
        "hkexnews.list_hkexnews_announcements": {
            "domain": "news",
            "capabilities": ["disclosure_search", "list"],
            "provides": [
                "local_key",
                "news_id",
                "title",
                "release_date",
                "file_link",
            ],
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
        "hkexnews.keyword_search": {
            "domain": "news",
            "capabilities": ["keyword"],
            "provides": ["snippet", "page_number", "local_key"],
        },
        "hkexnews.get_announcement_pages": {
            "domain": "news",
            "capabilities": ["page"],
            "provides": ["snippet", "page_number", "local_key"],
        },
    }


def _contracts() -> Dict[str, Dict[str, Any]]:
    return {
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
                "properties": {
                    "stock_code": {"type": "string"},
                    "local_key": {"type": "string"},
                    "news_id": {"type": "string"},
                    "release_date": {"type": "string"},
                    "title_keyword": {"type": "string"},
                },
                "required": ["stock_code"],
            }
        },
        "hkexnews.parse_hkexnews_announcement": {
            "json_schema": {
                "properties": {"local_key": {"type": "string"}},
                "required": ["local_key"],
            }
        },
        "hkexnews.keyword_search": {
            "json_schema": {
                "properties": {
                    "local_key": {"type": "string"},
                    "keywords": {"type": "array"},
                },
                "required": ["local_key", "keywords"],
            }
        },
        "hkexnews.get_announcement_pages": {
            "json_schema": {
                "properties": {
                    "local_key": {"type": "string"},
                    "start_page": {"type": "integer"},
                    "page_count": {"type": "integer"},
                },
                "required": ["local_key", "start_page"],
            }
        },
    }


def test_collect_rows_from_real_normalized_data_envelope():
    disc = json.loads(FIXTURE.read_text(encoding="utf-8"))
    invoke = disc["invoke"]
    rows = _collect_rows(invoke["result"])
    assert len(rows) >= 3
    assert any(r.get("local_key") for r in rows)
    assert any(r.get("file_link") for r in rows)
    vals = _row_to_values(rows[0], stock_code="00700")
    assert vals.get("local_key")
    assert vals.get("source_url") or vals.get("file_link")
    assert vals.get("issuer_code") == "00700"


def test_collect_rows_from_recorded_discovery_announcements():
    if not RECORDED_SAMPLE.exists():
        pytest.skip("recorded discovery sample missing")
    data = json.loads(RECORDED_SAMPLE.read_text(encoding="utf-8"))
    issuers = data.get("issuers") or {}
    first = next(iter(issuers.values()))
    rows = _collect_rows({"announcements": first.get("announcements")})
    assert rows
    assert rows[0].get("local_key")
    assert rows[0].get("news_id")
    assert rows[0].get("file_link")


def test_list_only_news_id_stops_without_download():
    disc = json.loads(FIXTURE.read_text(encoding="utf-8"))
    meta = _meta_chain()
    contracts = _contracts()
    calls: List[str] = []

    def invoke(oid: str, args: Dict[str, Any]) -> Dict[str, Any]:
        calls.append(oid)
        if "list" in oid:
            return {
                "success": True,
                "result": disc["invoke"]["result"],
            }
        raise AssertionError(f"unexpected stage call: {oid}")

    def describe(oid: str) -> Dict[str, Any]:
        return {"contract": contracts[oid]}

    assert needs_document_text(["news_id", "title", "release_date", "local_key"]) is False
    out = run_disclosure_composite(
        step={
            "domain": "disclosure",
            "workflow": DISCLOSURE_COMPOSITE_ID,
            "objective": "Next Day Disclosure Return news_id",
            "required_outputs": [
                "news_id",
                "title",
                "release_date",
                "local_key",
                "source_url",
            ],
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
        question="Issuer 00700 Next Day Disclosure Return on 2026-07-09",
        allow_hkex_download=True,
    )
    assert out["stopped_after_list"] is True
    assert out["status"] == "completed"
    assert out["values"]["news_id"] == "12239449"
    assert out["values"]["local_key"] == "00700_50100_2026-07-09_d80c8407"
    assert out["values"].get("source_url") or out["values"].get("file_link")
    assert all("download" not in c for c in calls)


def test_exact_date_and_title_filtering():
    disc = json.loads(FIXTURE.read_text(encoding="utf-8"))
    rows = _collect_rows(disc["invoke"]["result"])
    filtered = filter_list_rows(
        rows,
        grounding={"stock_code": "00700"},
        step={
            "scope_constraints": {
                "release_date": "2026-07-09",
                "title_keyword": "Next Day Disclosure Return",
            }
        },
    )
    assert len(filtered) == 1
    assert filtered[0]["news_id"] == "12239449"
    selected = select_list_row(
        rows,
        grounding={"stock_code": "00700"},
        step={
            "scope_constraints": {
                "title_keyword": "ANNUAL RESULTS",
            }
        },
    )
    assert selected is not None
    assert "ANNUAL RESULTS" in str(selected.get("title") or "").upper()


def test_complete_document_chain_passes_local_key_and_page():
    disc = json.loads(FIXTURE.read_text(encoding="utf-8"))
    meta = _meta_chain()
    contracts = _contracts()
    seen: Dict[str, Dict[str, Any]] = {}

    def invoke(oid: str, args: Dict[str, Any]) -> Dict[str, Any]:
        seen[oid] = dict(args)
        if "list" in oid:
            return {"success": True, "result": disc["invoke"]["result"]}
        if "download" in oid:
            assert args.get("local_key") == "00700_ANNUAL_2026-03-19_abc123"
            assert args.get("news_id") == "12000001"
            return {
                "success": True,
                "result": {
                    "normalized": {
                        "status": "success",
                        "data": {"local_key": args["local_key"]},
                    }
                },
            }
        if "parse" in oid:
            assert args.get("local_key") == "00700_ANNUAL_2026-03-19_abc123"
            return {
                "success": True,
                "result": {"normalized": {"local_key": args["local_key"]}},
            }
        if "keyword" in oid:
            assert args.get("local_key") == "00700_ANNUAL_2026-03-19_abc123"
            assert isinstance(args.get("keywords"), list) and args["keywords"]
            return {
                "success": True,
                "result": {
                    "normalized": {
                        "snippet": "non-IFRS profit attributable to equity holders",
                        "page_number": 4,
                        "local_key": args["local_key"],
                    }
                },
            }
        if "pages" in oid or "get_announcement" in oid:
            assert args.get("local_key") == "00700_ANNUAL_2026-03-19_abc123"
            assert args.get("start_page") == 4
            return {
                "success": True,
                "result": {
                    "normalized": {
                        "snippet": "page body",
                        "page_number": 4,
                        "local_key": args["local_key"],
                    }
                },
            }
        return {"success": False, "error": f"unexpected {oid}"}

    def describe(oid: str) -> Dict[str, Any]:
        return {"contract": contracts[oid]}

    out = run_disclosure_composite(
        step={
            "domain": "disclosure",
            "workflow": DISCLOSURE_COMPOSITE_ID,
            "objective": "annual results attributable profit snippet",
            "required_outputs": [
                "issuer_code",
                "local_key",
                "snippet",
                "page_number",
                "source_url",
            ],
            "depends_on": [],
            "scope_constraints": {
                "entity": {"code": "00700"},
                "title_keyword": "ANNUAL RESULTS",
            },
        },
        metadata_by_id=meta,
        runtime_operation_ids=list(meta),
        invoke_fn=invoke,
        describe_fn=describe,
        question="Issuer 00700 annual results attributable profit",
        allow_hkex_download=True,
    )
    assert out["status"] == "completed"
    assert out["values"]["local_key"] == "00700_ANNUAL_2026-03-19_abc123"
    assert out["values"].get("snippet")
    assert out["values"].get("page_number") == 4
    assert "hkexnews.parse_hkexnews_announcement" in seen
    assert seen["hkexnews.keyword_search"]["local_key"] == "00700_ANNUAL_2026-03-19_abc123"
    assert seen["hkexnews.get_announcement_pages"]["start_page"] == 4


def test_later_stage_failure_preserves_list_evidence():
    disc = json.loads(FIXTURE.read_text(encoding="utf-8"))
    meta = _meta_chain()
    contracts = _contracts()

    def invoke(oid: str, args: Dict[str, Any]) -> Dict[str, Any]:
        if "list" in oid:
            return {"success": True, "result": disc["invoke"]["result"]}
        if "download" in oid:
            return {
                "success": False,
                "error": "Download budget exhausted (max 1 announcement per adapter session).",
            }
        raise AssertionError(oid)

    def describe(oid: str) -> Dict[str, Any]:
        return {"contract": contracts[oid]}

    out = run_disclosure_composite(
        step={
            "domain": "disclosure",
            "workflow": DISCLOSURE_COMPOSITE_ID,
            "objective": "annual results text",
            "required_outputs": ["local_key", "news_id", "snippet"],
            "depends_on": [],
            "scope_constraints": {
                "entity": {"code": "00700"},
                "title_keyword": "ANNUAL RESULTS",
            },
        },
        metadata_by_id=meta,
        runtime_operation_ids=list(meta),
        invoke_fn=invoke,
        describe_fn=describe,
        question="Issuer 00700 annual results",
        allow_hkex_download=True,
    )
    assert out["execution_ok"] is False
    assert "stage_failed:download" in str(out.get("error") or "")
    assert out["values"].get("local_key") == "00700_ANNUAL_2026-03-19_abc123"
    assert out["values"].get("news_id") == "12000001"
    assert out["list_evidence"].get("local_key")


def test_build_stage_arguments_page_and_keywords():
    args = build_stage_arguments(
        stage="pages",
        contract=_contracts()["hkexnews.get_announcement_pages"],
        chain_state={"local_key": "lk1", "page_number": 7},
        step={"objective": "read page", "scope_constraints": {}},
        grounding={"stock_code": "00005"},
    )
    assert args["local_key"] == "lk1"
    assert args["start_page"] == 7
    kw = build_stage_arguments(
        stage="keyword",
        contract=_contracts()["hkexnews.keyword_search"],
        chain_state={"local_key": "lk1"},
        step={
            "objective": "attributable profit",
            "scope_constraints": {"title_keyword": "annual results"},
        },
        grounding={"title_keyword": "annual results"},
    )
    assert kw["local_key"] == "lk1"
    assert kw["keywords"] == ["annual results"]


def test_disclosure_workflow_hash_stable_and_present():
    h1 = disclosure_workflow_hash()
    h2 = disclosure_workflow_hash()
    assert h1 == h2
    assert len(h1) == 64
