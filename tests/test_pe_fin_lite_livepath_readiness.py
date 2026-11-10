"""PE-Fin Lite live-path readiness tests (offline fixtures/fake transports only)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest

from hk_equity_mcp.pe_fin_lite.disclosure_workflow import (
    DISCLOSURE_COMPOSITE_ID,
    run_disclosure_composite,
)
from hk_equity_mcp.pe_fin_lite.envelope_projection import project_invoke_to_required_outputs
from hk_equity_mcp.pe_fin_lite.executor import _evaluate_invoke
from hk_equity_mcp.pe_fin_lite.step_result import retrieve_step_succeeded
from hk_equity_mcp.experiment.unified_pe_fin_lite_runner import (
    PE_RESUME_IDENTITY_KEYS,
    UnifiedHkPeFinLiteRunner,
)
from hk_equity_mcp.real_mcp.unified_catalog import build_unified_catalog

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "apv_fin_v7_envelopes"


def _as_resolver_invoke(outer: Dict[str, Any]) -> Dict[str, Any]:
    """Shape returned by OperationResolverServer.invoke_api wrapping gateway payload."""
    return {
        "success": bool(outer.get("success")),
        "operation_id": outer.get("operation_id"),
        "arguments": outer.get("arguments") or {},
        "result": outer,
        "error": outer.get("error"),
        "error_type": None if outer.get("success") else "underlying_tool_error",
    }


def test_market_fixture_projects_close_trade_date():
    outer = json.loads((FIXTURES / "market_ohlcv_chinese.json").read_text())["invoke"]
    inv = _as_resolver_invoke(outer)
    proj = project_invoke_to_required_outputs(
        invoke_result=inv,
        required_outputs=["close", "trade_date"],
        domain="market",
    )
    assert proj["missing_outputs"] == []
    assert proj["values"]["close"] is not None
    assert proj["values"]["trade_date"]
    step_res = _evaluate_invoke(
        invoke_result=inv,
        required_outputs=["close", "trade_date"],
        step={"domain": "market", "entity": "0700.HK", "date": None},
        operation_id=str(outer["operation_id"]),
        arguments=outer.get("arguments") or {},
    )
    assert retrieve_step_succeeded(step_res)


def test_financial_fixture_projects_eps_revenue_synonyms():
    outer = json.loads((FIXTURES / "financial_analysis_indicator.json").read_text())[
        "invoke"
    ]
    inv = _as_resolver_invoke(outer)
    proj = project_invoke_to_required_outputs(
        invoke_result=inv,
        required_outputs=["eps", "revenue"],
        domain="financial",
    )
    assert proj["missing_outputs"] == []
    assert proj["values"]["eps"] == 8.504848
    assert proj["values"]["revenue"] == 452823411200
    step_res = _evaluate_invoke(
        invoke_result=inv,
        required_outputs=["eps", "revenue"],
        step={"domain": "financial"},
        operation_id=str(outer["operation_id"]),
        arguments=outer.get("arguments") or {},
    )
    assert retrieve_step_succeeded(step_res)


def test_transport_success_missing_required_not_completed():
    outer = {
        "success": True,
        "operation_id": "aktools.stock_hk_hist",
        "arguments": {},
        "result": {
            "success": True,
            "raw": {"is_error": False, "content": [{"json": {"rows": [], "columns": []}}]},
        },
    }
    inv = _as_resolver_invoke(outer)
    step_res = _evaluate_invoke(
        invoke_result=inv,
        required_outputs=["close", "trade_date"],
        step={"domain": "market"},
        operation_id="aktools.stock_hk_hist",
        arguments={},
    )
    assert not retrieve_step_succeeded(step_res)
    assert step_res["status"] in {"unavailable", "empty", "failed"}
    assert "close" in (step_res.get("missing_outputs") or [])


def test_disclosure_allow_hkex_download_gate_and_exception_to_stepresult():
    meta = {
        "hkexnews.list_hkexnews_announcements": {
            "domain": "news",
            "capabilities": ["disclosure_search", "list"],
            "provides": ["local_key", "news_id", "title"],
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
                "properties": {"symbol": {"type": "string"}},
                "required": ["symbol"],
            }
        },
        "hkexnews.download_hkexnews_announcement": {
            "json_schema": {
                "properties": {"local_key": {"type": "string"}},
                "required": ["local_key"],
            }
        },
    }

    def describe(oid):
        return {"success": True, "contract": contracts[oid]}

    def invoke_ok(oid, args):
        if "list" in oid:
            return {
                "success": True,
                "result": {"rows": [{"local_key": "k1", "news_id": "n1", "title": "t"}]},
            }
        return {"success": True, "result": {"local_key": "k1"}}

    blocked = run_disclosure_composite(
        step={
            "domain": "disclosure",
            "objective": "official disclosure",
            "entity": "0700.HK",
            "required_outputs": ["local_key"],
            "workflow": DISCLOSURE_COMPOSITE_ID,
        },
        metadata_by_id=meta,
        runtime_operation_ids=list(meta.keys()),
        invoke_fn=invoke_ok,
        describe_fn=describe,
        allow_hkex_download=False,
    )
    assert blocked["error"] == "hkex_download_not_allowed"
    assert blocked["execution_ok"] is False
    assert blocked["status"] != "completed"

    def invoke_raises(oid, args):
        if "list" in oid:
            return {
                "success": True,
                "result": {"rows": [{"local_key": "k1", "news_id": "n1"}]},
            }
        raise RuntimeError("download exploded")

    crashed = run_disclosure_composite(
        step={
            "domain": "disclosure",
            "objective": "official disclosure",
            "entity": "0700.HK",
            "required_outputs": ["local_key", "snippet"],
            "workflow": DISCLOSURE_COMPOSITE_ID,
        },
        metadata_by_id=meta,
        runtime_operation_ids=list(meta.keys()),
        invoke_fn=invoke_raises,
        describe_fn=describe,
        allow_hkex_download=True,
    )
    assert crashed["execution_ok"] is False
    assert "stage_exception:download" in str(crashed.get("error") or "")


def test_live_mode_rejects_mock_binder_and_runtime_ops_count():
    from hk_equity_mcp.agent.llm_client import ScriptedLLMClient
    from hk_equity_mcp.pe_fin_lite.agent import PeFinLiteAgent
    from hk_equity_mcp.pe_fin_lite.capability_profile import load_capability_profile
    from hk_equity_mcp.operation_resolver.retrieval import DeterministicHashEmbeddingBackend
    from hk_equity_mcp.operation_resolver.server import OperationResolverServer

    bundle = build_unified_catalog(root=ROOT)
    assert len(bundle.stats.runtime_operation_ids) == 39
    # Operation documents must be richer than bare function names.
    sample_oid = sorted(bundle.registry.keys())[0]
    doc = bundle.registry[sample_oid].index_document()
    assert "summary:" in doc and "description:" in doc
    assert "required_parameters:" in doc
    resolver = OperationResolverServer(
        embedding_backend=DeterministicHashEmbeddingBackend(),
        registry=bundle.registry,
        ensure_docs=True,
    )
    profile = load_capability_profile(ROOT)
    with pytest.raises(RuntimeError, match="use_mock_binder forbidden"):
        PeFinLiteAgent(
            llm=ScriptedLLMClient(["{}"], model="x"),
            resolver=resolver,
            capability_profile=profile,
            config={"use_mock_binder": True, "limits": {}},
            runtime_operation_ids=list(bundle.stats.runtime_operation_ids),
            live_mode=True,
        )


def test_resume_identity_includes_index_and_abort_finalization(tmp_path: Path):
    assert "embedding_index_identity_sha256" in PE_RESUME_IDENTITY_KEYS
    assert "capability_profile_hash" in PE_RESUME_IDENTITY_KEYS
    assert "system_prompt_hash" in PE_RESUME_IDENTITY_KEYS
    assert "schema_hash" in PE_RESUME_IDENTITY_KEYS

    out = tmp_path / "abort_run"
    runner = UnifiedHkPeFinLiteRunner(
        output_dir=out,
        smoke5=True,
        mock=True,
        allow_dirty=True,
        confirm_retire_heldout=True,
        root=ROOT,
    )
    prepared = runner.prepare()
    meta = prepared["run_metadata"]
    assert meta.get("embedding_index_identity_sha256")
    assert len(meta["embedding_index_identity_sha256"]) == 64

    # Simulate abort finalization path without network.
    predictions_by_id = {
        "fin_cf_003": {
            "task_id": "fin_cf_003",
            "status": "failed",
            "predicted_status": "unavailable",
            "final_answer": "",
            "tokens": {},
        }
    }
    run_meta = dict(meta)
    run_meta["run_status"] = "running"
    out.mkdir(parents=True, exist_ok=True)
    pred_path = out / "predictions.jsonl"
    pred_path.write_text(
        json.dumps(predictions_by_id["fin_cf_003"]) + "\n", encoding="utf-8"
    )
    run_meta["run_status"] = "aborted"
    run_meta["abort_reason"] = "KeyboardInterrupt"
    runner._write_side_artifacts(
        items=prepared["items"],
        predictions_by_id=predictions_by_id,
        run_meta=run_meta,
    )
    assert (out / "cost_summary.json").exists()
    assert (out / "error_summary.json").exists()
    assert (out / "retrieval_analysis.json").exists()
    assert (out / "grouped_aggregates.json").exists()


def test_fallback_at_most_once_local(monkeypatch):
    from hk_equity_mcp.pe_fin_lite.budget import BudgetTracker
    from hk_equity_mcp.pe_fin_lite import executor as ex
    from hk_equity_mcp.pe_fin_lite.evidence_store import EvidenceStore

    calls = []

    class FakeResolver:
        registry = {"a": object(), "b": object()}

        def find_operations(self, query, top_k=5):
            return {
                "success": True,
                "candidates": [
                    {"operation_id": "a", "score": 0.9},
                    {"operation_id": "b", "score": 0.8},
                ],
            }

        def describe_operation(self, oid):
            return {
                "success": True,
                "contract": {
                    "json_schema": {
                        "properties": {"symbol": {"type": "string"}},
                        "required": ["symbol"],
                    },
                    "summary": oid,
                },
            }

        def invoke_api(self, oid, arguments=None):
            calls.append(oid)
            return {
                "success": True,
                "operation_id": oid,
                "arguments": arguments or {},
                "result": {
                    "success": True,
                    "raw": {"is_error": False, "content": [{"json": {"rows": []}}]},
                },
                "error_type": None,
                "error": None,
            }

    class FakeLLM:
        def complete(self, messages, *, temperature=0.0, max_tokens=1024, stage=None):
            return {
                "content": json.dumps(
                    {
                        "primary": {"operation_id": "a", "arguments": {"symbol": "0700"}},
                        "fallback": {"operation_id": "b", "arguments": {"symbol": "0700"}},
                    }
                ),
                "usage": {},
            }

    store = EvidenceStore()
    budget = BudgetTracker(
        {
            "max_total_mcp_invocations": 8,
            "retrieval_top_k": 5,
            "describe_top_n": 3,
            "max_local_fallback_per_retrieve": 1,
        }
    )
    meta = {
        "a": {"domain": "market", "availability": "runtime"},
        "b": {"domain": "market", "availability": "runtime"},
    }
    out = ex.execute_retrieve_step(
        step={
            "id": "s1",
            "type": "retrieve",
            "domain": "market",
            "objective": "close",
            "required_outputs": ["close"],
            "depends_on": [],
        },
        question="what is close?",
        resolver=FakeResolver(),
        metadata_by_id=meta,
        runtime_operation_ids=["a", "b"],
        store=store,
        budget=budget,
        llm=FakeLLM(),
        use_mock_binder=False,
    )
    assert calls == ["a", "b"]
    assert out["funnel"]["local_fallback"]["attempted"] is True
    assert budget.local_fallback_count == 1
    assert not retrieve_step_succeeded(out["step_result"])
