"""Offline MCP-Zero protocol v2: ATR/JSON state machine + metrics (no paid APIs)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from hk_equity_mcp.agent.llm_client import OpenAICompatibleChatClient, ScriptedLLMClient
from hk_equity_mcp.experiment.unified_evaluator import evaluate_task, summarize_predictions
from hk_equity_mcp.experiment.unified_grading import grade_task
from hk_equity_mcp.experiment.unified_mcp_zero_runner import (
    UnifiedHkMcpZeroRunner,
    load_mcp_zero_config,
)
from hk_equity_mcp.mcp_zero import method_identity_sha256
from hk_equity_mcp.mcp_zero.active_request import (
    format_active_tool_request,
    parse_active_tool_request,
)
from hk_equity_mcp.mcp_zero.agent import McpZeroAgent, build_protocol_repair_message
from hk_equity_mcp.mcp_zero.embedding import CountingEmbedder, build_embedding_table
from hk_equity_mcp.mcp_zero.metadata import build_hk_metadata
from hk_equity_mcp.mcp_zero.prompt import (
    MCP_ZERO_PROMPT_VERSION,
    HK_MCP_ZERO_SYSTEM_PROMPT,
    mcp_zero_prompt_hash,
)
from hk_equity_mcp.mcp_zero.router import HierarchicalRouter
from hk_equity_mcp.mcp_zero.schemas import ExtensionRound, ToolHit
from hk_equity_mcp.operation_resolver.retrieval import DeterministicHashEmbeddingBackend
from hk_equity_mcp.real_mcp.types import RealMcpResult
from hk_equity_mcp.real_mcp.unified_catalog import build_unified_catalog
from hk_equity_mcp.real_mcp.unified_gateway import (
    ScriptedMockTransport,
    UnifiedRealMcpGateway,
)

ROOT = Path(__file__).resolve().parents[1]
FROZEN_METHOD_HASH = (
    "77cc3b07a4c2b9af00ad3f6bfb7cba59c82084d94b7592f2c820f46f1d3e8d3f"
)
OLD_PROMPT_HASH = (
    "3d75abf11dd9d1cdb7229c420df6b04b56a05a2b4102f924795a825b45755c28"
)
HKEX_LIST = "hkexnews.list_hkexnews_announcements"
HKEX_DOWNLOAD = "hkexnews.download_hkexnews_announcement"


def _tool_hit(oid: str, rank: int = 1) -> ToolHit:
    server_id, _, tool_name = oid.partition(".")
    return ToolHit(
        operation_id=oid,
        server_id=server_id or "hkexnews",
        tool_name=tool_name or oid,
        description=f"mock {oid}",
        server_score=0.9,
        tool_score=0.9,
        final_score=0.8,
        rank=rank,
    )


def _gateway(handler=None):
    bundle = build_unified_catalog(root=ROOT)
    gw = UnifiedRealMcpGateway(
        catalog_bundle=bundle,
        mock_transport=ScriptedMockTransport(handler=handler),
        allow_hkex_download=True,
    )
    gw.open()
    return bundle, gw


def _router(tmp_path: Path, bundle):
    catalog = build_hk_metadata(bundle)
    embedder = CountingEmbedder(DeterministicHashEmbeddingBackend())
    table = build_embedding_table(catalog, embedder, cache_dir=tmp_path / "emb")
    embedder.calls = 0
    embedder.tokens = 0
    return HierarchicalRouter(catalog, table, embedder, server_top_k=3, tool_top_k=5), embedder


def _mock_router(embedder, rounds: List[List[str]]):
    state = {"i": 0}

    class _R:
        def __init__(self):
            self.embedder = embedder

        def retrieve(self, request, **kwargs):
            idx = min(state["i"], len(rounds) - 1)
            state["i"] += 1
            oids = rounds[idx]
            hits = [_tool_hit(oid, rank=j + 1) for j, oid in enumerate(oids)]
            return ExtensionRound(
                round_index=idx,
                request=request,
                servers=[],
                tools=hits,
                new_operation_ids=list(oids),
                is_extension=idx > 0,
                stop_reason="ok",
                latency_ms=1.0,
                embedding_calls=1,
                embedding_tokens=8,
            )

        def format_tools_for_prompt(self, tools):
            return json.dumps([t.operation_id for t in tools])

    return _R()


def _invoke(oid: str) -> str:
    return json.dumps(
        {
            "thought": f"invoke {oid}",
            "tool_calls": [
                {
                    "name": "invoke_api",
                    "arguments": {"operation_id": oid, "arguments": {}},
                }
            ],
            "final_answer": None,
        }
    )


def _final(answer: str = "done", status: str = "answered") -> str:
    return json.dumps(
        {
            "thought": "finish",
            "tool_calls": [],
            "final_answer": {
                "status": status,
                "final_answer": answer,
                "reasoning_brief": "x",
                "evidence": [],
            },
        }
    )


def _empty() -> str:
    return json.dumps({"thought": "need tools", "tool_calls": [], "final_answer": None})


# ---------------------------------------------------------------------------
# Prompt / method hash rules
# ---------------------------------------------------------------------------


def test_prompt_version_bumped_and_hash_changed():
    assert MCP_ZERO_PROMPT_VERSION == "mcp_zero_protocol_v2"
    assert "Response modes" in HK_MCP_ZERO_SYSTEM_PROMPT
    assert "<tool_assistant>" in HK_MCP_ZERO_SYSTEM_PROMPT
    new_hash = mcp_zero_prompt_hash()
    assert len(new_hash) == 64
    assert new_hash != OLD_PROMPT_HASH
    # Routing method card unchanged.
    assert method_identity_sha256() == FROZEN_METHOD_HASH


def test_config_json_mode_false_for_mcp_zero():
    cfg = load_mcp_zero_config(root=ROOT)
    assert cfg.get("json_mode") is False


# ---------------------------------------------------------------------------
# Parser / ATR
# ---------------------------------------------------------------------------


def test_raw_atr_parses():
    text = format_active_tool_request("HK finance", "indicator lookup")
    req = parse_active_tool_request(text)
    assert req is not None
    assert req.server == "HK finance"
    assert req.tool == "indicator lookup"


def test_atr_wrapped_in_common_text_container():
    block = format_active_tool_request("AAA", "BBB")
    wrapped = json.dumps([{"type": "text", "text": block}])
    req = parse_active_tool_request(wrapped)
    assert req is not None
    assert req.server == "AAA"
    assert req.tool == "BBB"


def test_raw_atr_then_retrieval(tmp_path: Path):
    bundle, gateway = _gateway()
    router, emb = _router(tmp_path, bundle)
    atr = format_active_tool_request("HKEX news", "list announcements")
    llm = ScriptedLLMClient([atr, _final("after retrieve", status="unavailable")])
    fake = _mock_router(emb, [[HKEX_LIST, HKEX_DOWNLOAD]])
    agent = McpZeroAgent(llm=llm, router=fake, gateway=gateway, max_llm_turns=4)
    result = agent.run("q")
    assert result["active_tool_request"]
    assert len(result["retrieved_tools"]) == 2
    assert result["status"] == "full"
    assert result["terminal_contract_valid"] is True
    gateway.close()


def test_json_invoke_action(tmp_path: Path):
    bundle, gateway = _gateway()
    _, emb = _router(tmp_path, bundle)
    atr = format_active_tool_request("HKEX", "list")
    llm = ScriptedLLMClient([atr, _invoke(HKEX_LIST), _final("ok")])
    fake = _mock_router(emb, [[HKEX_LIST]])
    agent = McpZeroAgent(llm=llm, router=fake, gateway=gateway, max_llm_turns=5)
    result = agent.run("q")
    assert result["selected_operations"] == [HKEX_LIST]
    assert result["invoked_operations"] == [HKEX_LIST]
    assert result["status"] == "full"
    gateway.close()


def test_successful_invoke_then_json_final(tmp_path: Path):
    bundle, gateway = _gateway()
    _, emb = _router(tmp_path, bundle)
    atr = format_active_tool_request("HKEX", "list")
    llm = ScriptedLLMClient([atr, _invoke(HKEX_LIST), _final("start_page=4")])
    agent = McpZeroAgent(
        llm=llm, router=_mock_router(emb, [[HKEX_LIST]]), gateway=gateway, max_llm_turns=5
    )
    result = agent.run("q")
    assert result["status"] == "full"
    assert result["task_protocol_success"] is True
    assert "start_page=4" in (result["final_answer"] or "")
    gateway.close()


def test_invoke_then_extension_atr_second_retrieval(tmp_path: Path):
    bundle, gateway = _gateway()
    _, emb = _router(tmp_path, bundle)
    atr1 = format_active_tool_request("HKEX disclosure", "list announcements")
    atr2 = format_active_tool_request("HKEX disclosure", "download announcement PDF")
    llm = ScriptedLLMClient(
        [atr1, _invoke(HKEX_LIST), atr2, _invoke(HKEX_DOWNLOAD), _final("done")]
    )
    fake = _mock_router(emb, [[HKEX_LIST], [HKEX_DOWNLOAD]])
    agent = McpZeroAgent(
        llm=llm, router=fake, gateway=gateway, max_llm_turns=8, max_extension_rounds=3
    )
    result = agent.run("q")
    assert len(result["active_tool_request"]) == 2
    assert result["number_of_extension_rounds"] >= 1
    assert result["invoked_operations"] == [HKEX_LIST, HKEX_DOWNLOAD]
    assert result["status"] == "full"
    gateway.close()


def test_empty_action_repair_then_valid_atr(tmp_path: Path):
    bundle, gateway = _gateway()
    _, emb = _router(tmp_path, bundle)
    atr = format_active_tool_request("HKEX", "list")
    llm = ScriptedLLMClient([_empty(), atr, _final("unavailable", status="unavailable")])
    agent = McpZeroAgent(
        llm=llm, router=_mock_router(emb, [[HKEX_LIST]]), gateway=gateway, max_llm_turns=5
    )
    result = agent.run("q")
    assert result["protocol_repair_count"] == 1
    assert result["active_tool_request"]
    assert result["status"] == "full"
    msg = build_protocol_repair_message(available_operation_ids=[])
    assert "do NOT wrap in JSON" in msg or "Active Tool Request" in msg
    gateway.close()


def test_repeated_invalid_hits_max_turns_with_raw_trace(tmp_path: Path):
    bundle, gateway = _gateway()
    router, _ = _router(tmp_path, bundle)
    llm = ScriptedLLMClient([_empty() for _ in range(4)])
    agent = McpZeroAgent(llm=llm, router=router, gateway=gateway, max_llm_turns=4)
    result = agent.run("q")
    assert result["status"] == "failed"
    assert result["error_type"] == "max_llm_turns"
    assert result["task_protocol_success"] is False
    assert result["terminal_contract_valid"] is False
    assert result["llm_calls"] == 4
    assert result["total_turn_count"] == 4
    turns = [e for e in result["trace"] if e.get("type") == "llm_turn"]
    assert len(turns) == 4
    for t in turns:
        assert isinstance(t.get("raw_content"), str) or isinstance(t.get("content"), str)
        assert (t.get("raw_content") or t.get("content") or "") != "" or t.get(
            "content_note"
        )
    gateway.close()


class _NullMessageLLM(ScriptedLLMClient):
    def complete(self, messages, *, temperature=0.0, max_tokens=1024):
        message = None
        return message.get("content")  # type: ignore[union-attr]


def test_message_null_structured_failure_does_not_crash_runner(tmp_path: Path):
    bundle, gateway = _gateway()
    router, _ = _router(tmp_path, bundle)
    dbg = tmp_path / "dbg"
    agent = McpZeroAgent(
        llm=_NullMessageLLM([]),
        router=router,
        gateway=gateway,
        max_llm_turns=2,
        debug_dir=dbg,
        task_label="null_msg",
    )
    result = agent.run("q", task_label="null_msg")
    assert result["status"] == "failed"
    assert result["error_type"] == "AttributeError"
    assert result["failure_stage"]
    assert result["task_protocol_success"] is False
    gateway.close()


def test_openai_client_does_not_force_json_when_disabled(monkeypatch):
    client = OpenAICompatibleChatClient(
        api_key="test-key-not-live",
        base_url="https://example.invalid/v1",
        model="mock",
        provider="openai_compatible",
        json_mode=False,
    )
    captured: Dict[str, Any] = {}

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps(
                {
                    "choices": [
                        {
                            "message": {
                                "content": format_active_tool_request("S", "T")
                            }
                        }
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 2},
                }
            ).encode("utf-8")

    import urllib.request

    def _urlopen(req, timeout=0):
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return _Resp()

    monkeypatch.setattr(urllib.request, "urlopen", _urlopen)
    out = client.complete([{"role": "user", "content": "hi"}])
    assert "response_format" not in captured["body"]
    assert "<tool_assistant>" in out["content"]


# ---------------------------------------------------------------------------
# Grading / metrics
# ---------------------------------------------------------------------------


def test_failed_task_cannot_autograde_correct():
    item = {
        "task_id": "mkt_dev_014",
        "answerability_class": "should_refuse_or_clarify",
        "gold_status": "unavailable",
        "gold_values": [],
        "gold_operations": [],
    }
    row = {
        "task_id": "mkt_dev_014",
        "status": "failed",
        "predicted_status": "unavailable",
        "final_answer": "Reached max LLM turns without a valid final_answer.",
        "error": "max_llm_turns",
    }
    grade = grade_task(item, row)
    assert grade["answer_correct"] is False
    assert grade["reason"] == "status_failed"


def test_processed_full_failed_counts_and_invocation_conditioned_metrics():
    rows = [
        {
            "task_id": "a",
            "status": "failed",
            "all_tool_invocations_successful": True,  # vacuous
            "real_mcp_execution_count": 0,
            "successful_invocation_count": 0,
            "task_protocol_success": False,
            "terminal_contract_valid": False,
            "repair_count": 7,
            "valid_turn_count": 0,
            "total_turn_count": 8,
            "protocol_success_rate": None,
        },
        {
            "task_id": "b",
            "status": "failed",
            "all_tool_invocations_successful": True,
            "real_mcp_execution_count": 4,
            "successful_invocation_count": 4,
            "task_protocol_success": False,
            "terminal_contract_valid": False,
            "repair_count": 3,
            "valid_turn_count": 5,
            "total_turn_count": 8,
            "protocol_success_rate": 1.0,  # invoke-level, must not imply task success
        },
        {
            "task_id": "c",
            "status": "full",
            "all_tool_invocations_successful": True,
            "real_mcp_execution_count": 1,
            "successful_invocation_count": 1,
            "task_protocol_success": True,
            "terminal_contract_valid": True,
            "repair_count": 0,
            "valid_turn_count": 3,
            "total_turn_count": 3,
            "protocol_success_rate": 1.0,
        },
    ]
    summary = summarize_predictions(rows)
    assert summary["processed_count"] == 3
    assert summary["full_count"] == 1
    assert summary["failed_count"] == 2
    assert summary["completed_success_count"] == 1
    # Vacuous rate would be 1.0; conditioned rate uses only attempt tasks.
    assert summary["all_tool_invocations_successful_rate"] == 1.0
    assert summary["invocation_attempt_task_count"] == 2
    assert summary["invocation_conditioned_task_success_rate"] == 1.0
    assert summary["successful_invocations"] == 5
    assert summary["total_invocations"] == 5
    assert summary["zero_invocation_failed_task_count"] == 1
    assert summary["task_protocol_success_rate"] == round(1 / 3, 6)
    assert summary["terminal_contract_valid_rate"] == round(1 / 3, 6)


def test_evaluate_task_marks_failed_protocol_false():
    item = {"task_id": "x", "gold_operations": [], "dependency_graph": None}
    result = {
        "status": "failed",
        "error": "max_llm_turns",
        "trace": [],
        "protocol_repair_count": 8,
        "parse_repair_count": 0,
        "llm_calls": 8,
        "tokens": {},
        "final_answer_obj": {
            "status": "unavailable",
            "final_answer": "max turns",
            "reasoning_brief": "",
            "evidence": [],
        },
    }
    metrics = evaluate_task(item=item, result=result)
    assert metrics["task_protocol_success"] is False
    assert metrics["terminal_contract_valid"] is False
    assert metrics["repair_count"] == 8


# ---------------------------------------------------------------------------
# Four-domain mock must be status=full
# ---------------------------------------------------------------------------


def test_four_domain_mock_all_full(tmp_path: Path, monkeypatch):
    def _boom(*_a, **_k):
        raise AssertionError("real API client must not be constructed in --mock")

    monkeypatch.setattr(
        "hk_equity_mcp.experiment.unified_mcp_zero_runner.OpenAICompatibleChatClient",
        _boom,
    )
    monkeypatch.setattr(
        "hk_equity_mcp.experiment.unified_mcp_zero_runner.OpenAICompatibleEmbeddingBackend",
        _boom,
    )
    out = tmp_path / "mock4"
    meta = UnifiedHkMcpZeroRunner(
        output_dir=out,
        split="exploratory_all",
        confirm_retire_heldout=True,
        ids=["fin_cf_001", "mkt_dev_014", "news_dev_006", "xd_dev_007"],
        mock=True,
        allow_dirty=True,
        root=ROOT,
    ).run()
    assert meta["json_mode"] is False
    assert meta["system_prompt_hash"] == mcp_zero_prompt_hash()
    assert meta["method_card_hash"] == FROZEN_METHOD_HASH
    assert meta["method_identity_sha256"] == FROZEN_METHOD_HASH
    assert meta["processed_count"] == 4
    assert meta["full_count"] == 4
    assert meta["failed_count"] == 0
    assert meta["completed_success_count"] == 4
    assert meta.get("generation_provider")
    assert meta.get("generation_base_url_host") or meta.get("provider_host")
    assert meta.get("generation_model")
    assert meta.get("embedding_model")
    assert meta.get("catalog_fingerprint")
    rows = [
        json.loads(line)
        for line in (out / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 4
    by_id = {r["task_id"]: r for r in rows}
    for tid in ("fin_cf_001", "mkt_dev_014", "news_dev_006", "xd_dev_007"):
        assert by_id[tid]["status"] == "full", tid
        assert by_id[tid].get("task_protocol_success") is True
        assert by_id[tid].get("terminal_contract_valid") is True
