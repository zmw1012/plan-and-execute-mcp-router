"""Tests for Operation-Visible central ReAct agent (mock LLM + embeddings)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from hk_equity_mcp.agent.llm_client import ScriptedLLMClient
from hk_equity_mcp.agent.operation_visible import (
    OperationVisibleAgent,
    load_operation_visible_config,
)
from hk_equity_mcp.operation_resolver.retrieval import DeterministicHashEmbeddingBackend
from hk_equity_mcp.operation_resolver.schemas import OperationSpec, ParameterSpec
from hk_equity_mcp.operation_resolver.server import META_TOOLS, OperationResolverServer


def _turn(thought: str, tool_calls: List[Dict[str, Any]], final_answer=None) -> str:
    return json.dumps(
        {
            "thought": thought,
            "tool_calls": tool_calls,
            "final_answer": final_answer,
        },
        ensure_ascii=False,
    )


@pytest.fixture
def backend() -> DeterministicHashEmbeddingBackend:
    return DeterministicHashEmbeddingBackend()


def test_config_defaults_documented():
    cfg = load_operation_visible_config()
    assert cfg["architecture"] == "operation_visible"
    assert cfg["limits"]["max_llm_turns"] == 8
    assert cfg["limits"]["max_search_reformulations"] == 3
    assert cfg["limits"]["max_invocations"] == 6
    assert cfg["embedding"]["top_k_max"] == 20
    assert cfg["embedding"]["lexical_weight"] == 0.12


def test_exposed_tools_are_meta_only(backend, tmp_path: Path):
    agent = OperationVisibleAgent(
        llm=ScriptedLLMClient(
            [
                _turn(
                    "answer",
                    [],
                    final_answer="done",
                )
            ]
        ),
        embedding_backend=backend,
        resolver=OperationResolverServer(
            embedding_backend=backend, cache_dir=tmp_path / "c"
        ),
    )
    assert agent.exposed_tools() == list(META_TOOLS)
    assert "get_financial_metric" not in agent.exposed_tools()


def test_full_visible_trajectory_and_metrics(backend, tmp_path: Path):
    scripts = [
        _turn(
            "search",
            [
                {
                    "name": "find_operations",
                    "arguments": {"query": "get revenue financial metric", "top_k": 20},
                }
            ],
        ),
        _turn(
            "describe",
            [
                {
                    "name": "describe_operation",
                    "arguments": {"operation_id": "financial.get_financial_metric"},
                }
            ],
        ),
        _turn(
            "invoke",
            [
                {
                    "name": "invoke_api",
                    "arguments": {
                        "operation_id": "financial.get_financial_metric",
                        "arguments": {
                            "ticker": "0700.HK",
                            "metric": "revenue",
                            "period": "FY2025",
                            "report_type": "annual",
                        },
                    },
                }
            ],
        ),
        _turn(
            "done",
            [],
            final_answer={
                "status": "answered",
                "final_answer": "Tencent FY2025 revenue is 751766000000 CNY.",
                "reasoning_brief": "from invoke_api",
            },
        ),
    ]
    agent = OperationVisibleAgent(
        llm=ScriptedLLMClient(scripts, model="mock-llm"),
        embedding_backend=backend,
        resolver=OperationResolverServer(
            embedding_backend=backend, cache_dir=tmp_path / "c"
        ),
    )
    out = agent.run("What was Tencent FY2025 revenue?")
    assert out["architecture"] == "operation_visible"
    assert "simplified reimplementation adapted from Ghoshal et al. (2026)" in out[
        "fidelity_note"
    ]
    assert out["status"] == "full"
    assert out["predicted_status"] == "answered"
    assert "751766000000" in out["final_answer"]
    assert out["llm_calls"] == 4
    assert out["resolver_calls"]["find_operations"] == 1
    assert out["resolver_calls"]["describe_operation"] == 1
    assert out["underlying_operation_calls"] == 1
    assert out["tokens"]["total_tokens"] > 0
    assert out["provider_metadata"]["llm_provider"] == "mock"
    assert out["provider_metadata"]["llm_model"] == "mock-llm"
    assert out["provider_metadata"]["embedding_model"] == "mock-hash-embedding-v1"
    assert "api_key" not in json.dumps(out).lower()
    # Trace visibility: search candidates, describe contract, invoke args/result.
    names = [e.get("name") for e in out["trace"] if e.get("type") == "tool_result"]
    assert names == [
        "find_operations",
        "describe_operation",
        "invoke_api",
    ]
    find_evt = next(e for e in out["trace"] if e.get("name") == "find_operations")
    assert find_evt["result"]["candidates"]
    desc_evt = next(e for e in out["trace"] if e.get("name") == "describe_operation")
    assert desc_evt["result"]["contract"]["json_schema"]
    inv_evt = next(e for e in out["trace"] if e.get("name") == "invoke_api")
    assert inv_evt["arguments"]["arguments"]["ticker"] == "0700.HK"
    assert inv_evt["result"]["success"] is True


def test_search_reformulation_visible_in_trace(backend, tmp_path: Path):
    scripts = [
        _turn(
            "first search",
            [{"name": "find_operations", "arguments": {"query": "xyz unknown"}}],
        ),
        _turn(
            "reformulate",
            [
                {
                    "name": "find_operations",
                    "arguments": {"query": "get financial metric revenue"},
                }
            ],
        ),
        _turn("stop", [], final_answer="reformulated search done"),
    ]
    agent = OperationVisibleAgent(
        llm=ScriptedLLMClient(scripts),
        embedding_backend=backend,
        resolver=OperationResolverServer(
            embedding_backend=backend, cache_dir=tmp_path / "c"
        ),
    )
    out = agent.run("test reformulation")
    finds = [
        e
        for e in out["trace"]
        if e.get("type") == "tool_result" and e.get("name") == "find_operations"
    ]
    assert len(finds) == 2
    assert finds[0]["arguments"]["query"] == "xyz unknown"
    assert finds[1]["arguments"]["query"] == "get financial metric revenue"
    assert out["resolver_calls"]["find_operations"] == 2
    assert out["underlying_operation_calls"] == 0


def test_refuse_direct_underlying_tool_in_agent(backend, tmp_path: Path):
    scripts = [
        _turn(
            "bad call",
            [
                {
                    "name": "get_financial_metric",
                    "arguments": {"ticker": "0700.HK", "metric": "revenue"},
                }
            ],
        ),
        _turn("recover", [], final_answer="cannot call underlying tools directly"),
    ]
    agent = OperationVisibleAgent(
        llm=ScriptedLLMClient(scripts),
        embedding_backend=backend,
        resolver=OperationResolverServer(
            embedding_backend=backend, cache_dir=tmp_path / "c"
        ),
    )
    out = agent.run("bypass?")
    refused = next(
        e for e in out["trace"] if e.get("type") == "tool_result"
    )
    assert refused["name"] == "get_financial_metric"
    assert refused["result"]["success"] is False
    assert "Refused direct underlying tool" in refused["result"]["error"]
    assert out["underlying_operation_calls"] == 0


def test_independent_invokes_marked_parallel(backend, tmp_path: Path):
    scripts = [
        _turn(
            "parallel reads",
            [
                {
                    "name": "invoke_api",
                    "arguments": {
                        "operation_id": "financial.get_financial_metric",
                        "arguments": {
                            "ticker": "0700.HK",
                            "metric": "revenue",
                            "period": "FY2025",
                            "report_type": "annual",
                        },
                    },
                },
                {
                    "name": "invoke_api",
                    "arguments": {
                        "operation_id": "financial.get_filing_summary",
                        "arguments": {
                            "ticker": "0700.HK",
                            "period": "FY2025",
                            "report_type": "annual",
                        },
                    },
                },
            ],
        ),
        _turn("done", [], final_answer="both ok"),
    ]
    agent = OperationVisibleAgent(
        llm=ScriptedLLMClient(scripts),
        embedding_backend=backend,
        resolver=OperationResolverServer(
            embedding_backend=backend, cache_dir=tmp_path / "c"
        ),
    )
    out = agent.run("two independent reads")
    invokes = [
        e
        for e in out["trace"]
        if e.get("type") == "tool_result" and e.get("name") == "invoke_api"
    ]
    assert len(invokes) == 2
    assert all(e["parallel"] is True for e in invokes)
    assert out["underlying_operation_calls"] == 2
    assert all(e["result"]["success"] for e in invokes)


def test_dependent_invoke_stays_sequential(backend, tmp_path: Path):
    # Inject a dependent read-only op that must not parallelize with others.
    registry = {
        "financial.get_financial_metric": OperationSpec(
            operation_id="financial.get_financial_metric",
            service_id="financial",
            name="get_financial_metric",
            summary="metric",
            description="metric",
            action_terms=["metric"],
            parameters=[
                ParameterSpec("ticker", "string", required=True),
                ParameterSpec(
                    "metric",
                    "string",
                    required=True,
                    enum=["revenue", "net_profit", "eps"],
                ),
            ],
            read_only=True,
            depends_on=[],
            underlying_tool="get_financial_metric",
        ),
        "financial.dependent_metric": OperationSpec(
            operation_id="financial.dependent_metric",
            service_id="financial",
            name="dependent_metric",
            summary="depends on prior metric",
            description="depends",
            action_terms=["metric"],
            parameters=[
                ParameterSpec("ticker", "string", required=True),
                ParameterSpec(
                    "metric",
                    "string",
                    required=True,
                    enum=["revenue", "net_profit", "eps"],
                ),
            ],
            read_only=True,
            depends_on=["financial.get_financial_metric"],
            underlying_tool="get_financial_metric",
        ),
    }
    resolver = OperationResolverServer(
        embedding_backend=backend,
        registry=registry,
        cache_dir=tmp_path / "c",
    )
    scripts = [
        _turn(
            "mixed",
            [
                {
                    "name": "invoke_api",
                    "arguments": {
                        "operation_id": "financial.get_financial_metric",
                        "arguments": {"ticker": "0700.HK", "metric": "revenue"},
                    },
                },
                {
                    "name": "invoke_api",
                    "arguments": {
                        "operation_id": "financial.dependent_metric",
                        "arguments": {"ticker": "0700.HK", "metric": "revenue"},
                    },
                },
            ],
        ),
        _turn("done", [], final_answer="ordered"),
    ]
    agent = OperationVisibleAgent(
        llm=ScriptedLLMClient(scripts),
        embedding_backend=backend,
        resolver=resolver,
    )
    out = agent.run("dependent ordering")
    invokes = [
        e
        for e in out["trace"]
        if e.get("type") == "tool_result" and e.get("name") == "invoke_api"
    ]
    assert len(invokes) == 2
    # Independent one may be parallel=False when alone in the parallel set;
    # dependent must be sequential (parallel=False).
    dep = next(
        e
        for e in invokes
        if e["arguments"]["operation_id"] == "financial.dependent_metric"
    )
    assert dep["parallel"] is False
    # Preserve call order in trace results relative to original indices.
    ids = [e["arguments"]["operation_id"] for e in invokes]
    assert ids == [
        "financial.get_financial_metric",
        "financial.dependent_metric",
    ]


def test_validation_error_visible_to_loop(backend, tmp_path: Path):
    scripts = [
        _turn(
            "bad args",
            [
                {
                    "name": "invoke_api",
                    "arguments": {
                        "operation_id": "financial.get_financial_metric",
                        "arguments": {"ticker": "0700.HK"},
                    },
                }
            ],
        ),
        _turn(
            "recover",
            [],
            final_answer={
                "status": "clarification_needed",
                "final_answer": "need metric",
                "reasoning_brief": "",
            },
        ),
    ]
    agent = OperationVisibleAgent(
        llm=ScriptedLLMClient(scripts),
        embedding_backend=backend,
        resolver=OperationResolverServer(
            embedding_backend=backend, cache_dir=tmp_path / "c"
        ),
    )
    out = agent.run("missing metric")
    inv = next(e for e in out["trace"] if e.get("name") == "invoke_api")
    assert inv["result"]["success"] is False
    assert inv["result"]["error_type"] == "validation_error"


class SpyResolver:
    """Records real call_meta dispatch order."""

    def __init__(self, inner: OperationResolverServer):
        self.inner = inner
        self.registry = inner.registry
        self.retriever = inner.retriever
        self.order: List[str] = []

    def call_meta(self, tool_name: str, **kwargs: Any) -> Dict[str, Any]:
        self.order.append(tool_name)
        return self.inner.call_meta(tool_name, **kwargs)


def test_mixed_find_describe_invoke_real_dispatch_order(backend, tmp_path: Path):
    """Dispatch order must be find → describe → invoke (not invoke-first)."""
    scripts = [
        _turn(
            "mixed in one turn",
            [
                {
                    "name": "find_operations",
                    "arguments": {"query": "revenue metric", "top_k": 20},
                },
                {
                    "name": "describe_operation",
                    "arguments": {"operation_id": "financial.get_financial_metric"},
                },
                {
                    "name": "invoke_api",
                    "arguments": {
                        "operation_id": "financial.get_financial_metric",
                        "arguments": {
                            "ticker": "0700.HK",
                            "metric": "revenue",
                            "period": "FY2025",
                            "report_type": "annual",
                        },
                    },
                },
            ],
        ),
        _turn(
            "done",
            [],
            final_answer={
                "status": "answered",
                "final_answer": "ok",
                "reasoning_brief": "",
            },
        ),
    ]
    inner = OperationResolverServer(
        embedding_backend=backend, cache_dir=tmp_path / "c"
    )
    spy = SpyResolver(inner)
    agent = OperationVisibleAgent(
        llm=ScriptedLLMClient(scripts),
        embedding_backend=backend,
        resolver=spy,  # type: ignore[arg-type]
    )
    out = agent.run("order check")
    assert spy.order == [
        "find_operations",
        "describe_operation",
        "invoke_api",
    ]
    names = [e.get("name") for e in out["trace"] if e.get("type") == "tool_result"]
    assert names == [
        "find_operations",
        "describe_operation",
        "invoke_api",
    ]


def test_noncontiguous_invokes_do_not_jump_ahead(backend, tmp_path: Path):
    """invoke after describe must wait; only contiguous invokes may batch."""
    scripts = [
        _turn(
            "describe then two invokes",
            [
                {
                    "name": "describe_operation",
                    "arguments": {"operation_id": "financial.get_financial_metric"},
                },
                {
                    "name": "invoke_api",
                    "arguments": {
                        "operation_id": "financial.get_financial_metric",
                        "arguments": {
                            "ticker": "0700.HK",
                            "metric": "revenue",
                            "period": "FY2025",
                            "report_type": "annual",
                        },
                    },
                },
                {
                    "name": "invoke_api",
                    "arguments": {
                        "operation_id": "financial.get_filing_summary",
                        "arguments": {
                            "ticker": "0700.HK",
                            "period": "FY2025",
                            "report_type": "annual",
                        },
                    },
                },
            ],
        ),
        _turn(
            "done",
            [],
            final_answer={
                "status": "answered",
                "final_answer": "ok",
                "reasoning_brief": "",
            },
        ),
    ]
    inner = OperationResolverServer(
        embedding_backend=backend, cache_dir=tmp_path / "c"
    )
    spy = SpyResolver(inner)
    agent = OperationVisibleAgent(
        llm=ScriptedLLMClient(scripts),
        embedding_backend=backend,
        resolver=spy,  # type: ignore[arg-type]
    )
    out = agent.run("contiguous batch")
    assert spy.order[0] == "describe_operation"
    assert spy.order[1:] == [
        "invoke_api",
        "invoke_api",
    ]
    tool_events = [e for e in out["trace"] if e.get("type") == "tool_result"]
    assert tool_events[0]["name"] == "describe_operation"
    assert tool_events[0]["parallel"] is False
    assert tool_events[1]["parallel"] is True
    assert tool_events[2]["parallel"] is True
