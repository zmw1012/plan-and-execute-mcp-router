"""Offline protocol / observability / metric hardening for MCP-Zero (no paid APIs)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from hk_equity_mcp.agent.llm_client import (
    OpenAICompatibleChatClient,
    ScriptedLLMClient,
)
from hk_equity_mcp.mcp_zero.active_request import (
    format_active_tool_request,
    parse_active_tool_request,
)
from hk_equity_mcp.mcp_zero.agent import build_protocol_repair_message, McpZeroAgent
from hk_equity_mcp.mcp_zero.embedding import CountingEmbedder, build_embedding_table
from hk_equity_mcp.mcp_zero.metadata import build_hk_metadata
from hk_equity_mcp.mcp_zero.prompt import mcp_zero_prompt_hash
from hk_equity_mcp.mcp_zero.router import HierarchicalRouter
from hk_equity_mcp.mcp_zero.schemas import ExtensionRound, ToolHit
from hk_equity_mcp.operation_resolver.retrieval import DeterministicHashEmbeddingBackend
from hk_equity_mcp.real_mcp.types import RealMcpResult
from hk_equity_mcp.real_mcp.unified_catalog import build_unified_catalog
from hk_equity_mcp.real_mcp.unified_gateway import (
    ScriptedMockTransport,
    UnifiedRealMcpGateway,
)
from hk_equity_mcp.mcp_zero import method_identity_sha256

ROOT = Path(__file__).resolve().parents[1]

NEWS_QUESTION = (
    "For Tencent (00700) Monthly Return released 2026-08-06, download and parse "
    "the HKEXnews PDF, then keyword_search with keywords=[\"Monthly\",\"Return\","
    "\"Issuer\"]. What start_page should be passed to get_announcement_pages from "
    "the top hit page_number?"
)

HKEX_CHAIN = [
    "hkexnews.list_hkexnews_announcements",
    "hkexnews.download_hkexnews_announcement",
    "hkexnews.parse_hkexnews_announcement",
    "hkexnews.keyword_search",
    "hkexnews.get_announcement_pages",
]

FROZEN_METHOD_HASH = (
    "77cc3b07a4c2b9af00ad3f6bfb7cba59c82084d94b7592f2c820f46f1d3e8d3f"
)


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


def _build_gateway(handler=None):
    bundle = build_unified_catalog(root=ROOT)
    gateway = UnifiedRealMcpGateway(
        catalog_bundle=bundle,
        mock_transport=ScriptedMockTransport(handler=handler),
        allow_hkex_download=True,
    )
    gateway.open()
    return bundle, gateway


def _build_router(tmp_path: Path, bundle):
    catalog = build_hk_metadata(bundle)
    embedder = CountingEmbedder(DeterministicHashEmbeddingBackend())
    table = build_embedding_table(catalog, embedder, cache_dir=tmp_path / "emb")
    embedder.calls = 0
    embedder.tokens = 0
    router = HierarchicalRouter(catalog, table, embedder, server_top_k=3, tool_top_k=5)
    return router, embedder


def _final_unavailable(msg: str = "unavailable") -> str:
    return json.dumps(
        {
            "thought": "done",
            "tool_calls": [],
            "final_answer": {
                "status": "unavailable",
                "final_answer": msg,
                "reasoning_brief": "x",
                "evidence": [],
            },
        }
    )


def _empty_action(thought: str = "I need a tool") -> str:
    return json.dumps(
        {"thought": thought, "tool_calls": [], "final_answer": None}
    )


def _invoke(oid: str, arguments: Optional[Dict[str, Any]] = None) -> str:
    return json.dumps(
        {
            "thought": f"call {oid}",
            "tool_calls": [
                {
                    "name": "invoke_api",
                    "arguments": {
                        "operation_id": oid,
                        "arguments": arguments or {},
                    },
                }
            ],
            "final_answer": None,
        }
    )


# ---------------------------------------------------------------------------
# 1. Active Tool Request parser
# ---------------------------------------------------------------------------


def test_parse_ordinary_block():
    text = format_active_tool_request("AAA", "BBB")
    req = parse_active_tool_request(text)
    assert req is not None
    assert req.server == "AAA"
    assert req.tool == "BBB"


def test_parse_block_embedded_in_json_string_with_escaped_newlines():
    inner = "<tool_assistant>\\nserver: AAA\\ntool: BBB\\n</tool_assistant>"
    payload = json.dumps({"thought": "need tools", "block": inner})
    # Simulate model content that embeds escaped newlines inside a string.
    content = (
        'I need tools.\\n'
        '<tool_assistant>\\nserver: AAA\\ntool: BBB\\n</tool_assistant>'
    )
    req = parse_active_tool_request(content)
    assert req is not None
    assert req.server == "AAA"
    assert req.tool == "BBB"
    assert "tool:" not in req.server
    assert "\n" not in req.server or "tool" not in req.server.lower()


def test_parse_server_tool_with_leading_spaces():
    text = (
        "<tool_assistant>\n"
        "  server:   finance MCP  \n"
        "  tool:   indicator lookup  \n"
        "</tool_assistant>"
    )
    req = parse_active_tool_request(text)
    assert req is not None
    assert req.server == "finance MCP"
    assert req.tool == "indicator lookup"


def test_parse_crlf_block():
    text = (
        "<tool_assistant>\r\n"
        "server: AAA\r\n"
        "tool: BBB\r\n"
        "</tool_assistant>"
    )
    req = parse_active_tool_request(text)
    assert req is not None
    assert req.server == "AAA"
    assert req.tool == "BBB"


def test_parse_multiline_tool_description():
    text = (
        "<tool_assistant>\n"
        "server: HKEX disclosure portal\n"
        "tool: list announcements then download PDF\n"
        "for issuer 00700 Monthly Return\n"
        "</tool_assistant>"
    )
    req = parse_active_tool_request(text)
    assert req is not None
    assert req.server == "HKEX disclosure portal"
    assert "list announcements" in req.tool
    assert "00700" in req.tool
    assert "tool:" not in req.server


def test_parse_missing_server_or_tool_fail_closed():
    assert parse_active_tool_request(
        "<tool_assistant>\ntool: only tool\n</tool_assistant>"
    ) is None
    assert parse_active_tool_request(
        "<tool_assistant>\nserver: only server\n</tool_assistant>"
    ) is None
    assert parse_active_tool_request("<tool_assistant>\n</tool_assistant>") is None


# ---------------------------------------------------------------------------
# 2. Empty-action protocol repair
# ---------------------------------------------------------------------------


def test_thought_only_json_triggers_protocol_repair(tmp_path: Path):
    bundle, gateway = _build_gateway()
    router, _ = _build_router(tmp_path, bundle)
    llm = ScriptedLLMClient(
        [
            json.dumps({"thought": "planning only", "tool_calls": [], "final_answer": None}),
            _final_unavailable("refuse"),
        ]
    )
    agent = McpZeroAgent(
        llm=llm, router=router, gateway=gateway, max_llm_turns=4, debug_dir=tmp_path / "dbg"
    )
    result = agent.run("q")
    assert result["protocol_repair_count"] >= 1
    assert result["status"] == "full"
    assert result["predicted_status"] == "unavailable"
    repair = build_protocol_repair_message(available_operation_ids=[])
    assert "Active Tool Request" in repair
    assert "<tool_assistant>" in repair
    gateway.close()


def test_empty_tool_calls_null_final_does_not_end_as_partial(tmp_path: Path):
    bundle, gateway = _build_gateway()
    router, _ = _build_router(tmp_path, bundle)
    llm = ScriptedLLMClient([_empty_action(), _final_unavailable("ok")])
    agent = McpZeroAgent(llm=llm, router=router, gateway=gateway, max_llm_turns=4)
    result = agent.run("q")
    assert result["status"] != "partial"
    assert result["protocol_repair_count"] == 1
    assert result["status"] == "full"
    gateway.close()


def test_valid_unavailable_final_answer_ends_cleanly(tmp_path: Path):
    bundle, gateway = _build_gateway()
    router, _ = _build_router(tmp_path, bundle)
    llm = ScriptedLLMClient([_final_unavailable("cannot answer")])
    agent = McpZeroAgent(llm=llm, router=router, gateway=gateway, max_llm_turns=3)
    result = agent.run("q")
    assert result["status"] == "full"
    assert result["protocol_repair_count"] == 0
    assert result["predicted_status"] == "unavailable"
    gateway.close()


def test_correct_active_tool_request_continues(tmp_path: Path):
    bundle, gateway = _build_gateway()
    router, _ = _build_router(tmp_path, bundle)
    atr = format_active_tool_request(
        "HK financial indicators MCP", "analysis indicator lookup"
    )
    llm = ScriptedLLMClient([atr, _final_unavailable("after retrieve")])
    agent = McpZeroAgent(llm=llm, router=router, gateway=gateway, max_llm_turns=4)
    result = agent.run("lookup revenue")
    assert result["active_tool_request"]
    assert len(result["retrieved_tools"]) > 0
    assert result["status"] == "full"
    gateway.close()


def test_continuous_invalid_actions_hit_max_turns(tmp_path: Path):
    bundle, gateway = _build_gateway()
    router, _ = _build_router(tmp_path, bundle)
    llm = ScriptedLLMClient([_empty_action(f"turn {i}") for i in range(3)])
    agent = McpZeroAgent(llm=llm, router=router, gateway=gateway, max_llm_turns=3)
    result = agent.run("q")
    assert result["status"] == "failed"
    assert result["error_type"] == "max_llm_turns"
    assert result["protocol_repair_count"] == 3
    assert result["llm_calls"] == 3
    gateway.close()


# ---------------------------------------------------------------------------
# 3. Exception observability + AttributeError fixture
# ---------------------------------------------------------------------------


class _AttributeErrorLLM(ScriptedLLMClient):
    def complete(self, messages, *, temperature=0.0, max_tokens=1024):
        # Mimic classic null-message AttributeError path.
        message = None
        return message.get("content")  # type: ignore[union-attr]


def test_attribute_error_diagnostics_preserved(tmp_path: Path):
    bundle, gateway = _build_gateway()
    router, _ = _build_router(tmp_path, bundle)
    dbg = tmp_path / "dbg"
    agent = McpZeroAgent(
        llm=_AttributeErrorLLM([]),
        router=router,
        gateway=gateway,
        max_llm_turns=2,
        debug_dir=dbg,
        task_label="diag_task",
    )
    result = agent.run("q", task_label="diag_task")
    assert result["status"] == "failed"
    assert result["error_type"] == "AttributeError"
    assert result["error_message"]
    assert result["failure_stage"] == "llm_complete"
    assert result["debug_traceback_file"]
    assert (dbg / result["debug_traceback_file"]).exists()
    tb_text = (dbg / result["debug_traceback_file"]).read_text(encoding="utf-8")
    assert "api_key" not in tb_text.lower() or "redact" in tb_text.lower()
    assert "Authorization" not in tb_text
    # Predictions-style fields must not embed absolute secret paths as values of secrets.
    assert "/.env" not in json.dumps(result)
    gateway.close()


def test_openai_compatible_null_message_does_not_attribute_error(monkeypatch):
    """Guard: choices[0].message is null must not raise AttributeError."""
    client = OpenAICompatibleChatClient(
        api_key="test-key-not-live",
        base_url="https://example.invalid/v1",
        model="mock",
        provider="openai_compatible",
    )

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps(
                {
                    "choices": [{"message": None, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 0},
                }
            ).encode("utf-8")

    import urllib.request

    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: _Resp())
    out = client.complete([{"role": "user", "content": "hi"}])
    assert out["content"] == ""
    assert out["usage"]["input_tokens"] == 1


# ---------------------------------------------------------------------------
# 4. retrieved / selected / invoked semantics
# ---------------------------------------------------------------------------


def test_retrieve_5_invoke_1_counts(tmp_path: Path):
    bundle, gateway = _build_gateway()
    router, _ = _build_router(tmp_path, bundle)
    # Use real catalog operation_ids so Gateway accepts the invoke.
    hits = [_tool_hit(oid, rank=i + 1) for i, oid in enumerate(HKEX_CHAIN)]
    chosen = hits[0].operation_id

    class _Router:
        embedder = router.embedder

        def retrieve(self, request, **kwargs):
            return ExtensionRound(
                round_index=0,
                request=request,
                servers=[],
                tools=hits,
                new_operation_ids=[h.operation_id for h in hits],
                is_extension=False,
                stop_reason="ok",
                latency_ms=1.0,
                embedding_calls=1,
                embedding_tokens=10,
            )

        def format_tools_for_prompt(self, tools):
            return json.dumps([t.operation_id for t in tools])

    atr = format_active_tool_request("HKEX news", "announcement list")
    llm = ScriptedLLMClient([atr, _invoke(chosen), _final_unavailable("done")])
    agent = McpZeroAgent(llm=llm, router=_Router(), gateway=gateway, max_llm_turns=5)
    result = agent.run("q")
    assert len(result["retrieved_tools"]) == 5
    assert result["selected_operations"] == [chosen]
    assert result["invoked_operations"] == [chosen]
    assert result["successful_invoked_operations"] == [chosen]
    gateway.close()


def test_retrieve_5_no_invoke_selected_zero(tmp_path: Path):
    bundle, gateway = _build_gateway()
    router, _ = _build_router(tmp_path, bundle)
    hits = [_tool_hit(oid, rank=i + 1) for i, oid in enumerate(HKEX_CHAIN)]

    class _Router:
        embedder = router.embedder

        def retrieve(self, request, **kwargs):
            return ExtensionRound(
                round_index=0,
                request=request,
                servers=[],
                tools=hits,
                new_operation_ids=[h.operation_id for h in hits],
                is_extension=False,
                stop_reason="ok",
                latency_ms=1.0,
                embedding_calls=1,
                embedding_tokens=10,
            )

        def format_tools_for_prompt(self, tools):
            return "tools"

    atr = format_active_tool_request("HKEX news", "announcement list")
    llm = ScriptedLLMClient([atr, _final_unavailable("no invoke")])
    agent = McpZeroAgent(llm=llm, router=_Router(), gateway=gateway, max_llm_turns=4)
    result = agent.run("q")
    assert len(result["retrieved_tools"]) == 5
    assert result["selected_operations"] == []
    assert result["invoked_operations"] == []
    gateway.close()


def test_select_unretrieved_records_invalid_not_invoked(tmp_path: Path):
    bundle, gateway = _build_gateway()
    router, _ = _build_router(tmp_path, bundle)
    llm = ScriptedLLMClient(
        [
            _invoke("hkexnews.list_hkexnews_announcements"),
            _final_unavailable("after refuse"),
        ]
    )
    agent = McpZeroAgent(llm=llm, router=router, gateway=gateway, max_llm_turns=4)
    result = agent.run("q")
    assert "hkexnews.list_hkexnews_announcements" in result["selected_operations"]
    assert "hkexnews.list_hkexnews_announcements" in result["invalid_selections"]
    assert result["invoked_operations"] == []
    gateway.close()


def test_gateway_failure_invoked_but_not_successful(tmp_path: Path):
    def handler(oid: str, args: Dict[str, Any]) -> RealMcpResult:
        return RealMcpResult(
            success=False,
            server_id="hkexnews",
            operation_id=oid,
            arguments=args,
            raw={},
            normalized={},
            evidence=[],
            error="boom",
            error_type="execution_error",
            protocol_success=True,
            execution_success=False,
            real_mcp=False,
            transport="mock",
        )

    bundle, gateway = _build_gateway(handler=handler)
    router, _ = _build_router(tmp_path, bundle)
    hits = [_tool_hit("hkexnews.list_hkexnews_announcements")]

    class _Router:
        embedder = router.embedder

        def retrieve(self, request, **kwargs):
            return ExtensionRound(
                round_index=0,
                request=request,
                servers=[],
                tools=hits,
                new_operation_ids=[hits[0].operation_id],
                is_extension=False,
                stop_reason="ok",
                latency_ms=1.0,
                embedding_calls=1,
                embedding_tokens=5,
            )

        def format_tools_for_prompt(self, tools):
            return "tools"

    atr = format_active_tool_request("HKEX", "list")
    oid = hits[0].operation_id
    llm = ScriptedLLMClient([atr, _invoke(oid), _final_unavailable("gw fail")])
    agent = McpZeroAgent(llm=llm, router=_Router(), gateway=gateway, max_llm_turns=5)
    result = agent.run("q")
    assert result["invoked_operations"] == [oid]
    assert result["successful_invoked_operations"] == []
    gateway.close()


# ---------------------------------------------------------------------------
# 5. Per-task embedding counters
# ---------------------------------------------------------------------------


def test_second_task_embedding_delta_excludes_first(tmp_path: Path):
    bundle, gateway = _build_gateway()
    router, embedder = _build_router(tmp_path, bundle)

    class _Router:
        def __init__(self):
            self.embedder = embedder
            self.n = 0

        def retrieve(self, request, **kwargs):
            # Simulate query-time embedding usage on the shared counter.
            self.embedder.calls += 2
            self.embedder.tokens += 20
            self.n += 1
            oid = HKEX_CHAIN[(self.n - 1) % len(HKEX_CHAIN)]
            return ExtensionRound(
                round_index=0,
                request=request,
                servers=[],
                tools=[_tool_hit(oid)],
                new_operation_ids=[oid],
                is_extension=False,
                stop_reason="ok",
                latency_ms=1.0,
                embedding_calls=2,
                embedding_tokens=20,
            )

        def format_tools_for_prompt(self, tools):
            return "t"

    fake = _Router()
    atr = format_active_tool_request("s", "t")
    llm1 = ScriptedLLMClient([atr, _final_unavailable("t1")])
    llm2 = ScriptedLLMClient([atr, _final_unavailable("t2")])
    agent1 = McpZeroAgent(llm=llm1, router=fake, gateway=gateway, max_llm_turns=3)
    agent2 = McpZeroAgent(llm=llm2, router=fake, gateway=gateway, max_llm_turns=3)
    r1 = agent1.run("q1", task_label="t1")
    r2 = agent2.run("q2", task_label="t2")
    assert r1["embedding_calls"] == 2
    assert r2["embedding_calls"] == 2
    assert r2["embedding_calls"] != r1["embedding_calls"] + r2["embedding_calls"]
    assert embedder.calls == 4  # shared total advanced, per-task deltas isolated
    gateway.close()


# ---------------------------------------------------------------------------
# 6. Offline news multi-step chain (no real APIs)
# ---------------------------------------------------------------------------


def test_offline_hkex_multistep_chain_no_attribute_error(tmp_path: Path):
    call_log: List[str] = []

    def handler(oid: str, args: Dict[str, Any]) -> RealMcpResult:
        call_log.append(oid)
        payloads = {
            "hkexnews.list_hkexnews_announcements": {
                "announcements": [
                    {
                        "title": "Monthly Return",
                        "stock_code": "00700",
                        "news_id": "N1",
                        "release_date": "2026-08-06",
                    }
                ]
            },
            "hkexnews.download_hkexnews_announcement": {
                "local_path": "mock.pdf",
                "news_id": "N1",
            },
            "hkexnews.parse_hkexnews_announcement": {
                "pages": [{"page_number": 1, "text": "Monthly Return Issuer"}],
                "page_count": 1,
            },
            "hkexnews.keyword_search": {
                "hits": [{"page_number": 1, "snippet": "Monthly Return Issuer"}]
            },
            "hkexnews.get_announcement_pages": {
                "pages": [{"page_number": 1, "text": "Monthly Return Issuer"}],
                "start_page": 1,
            },
        }
        body = payloads.get(oid, {"ok": True})
        return RealMcpResult(
            success=True,
            server_id="hkexnews",
            operation_id=oid,
            arguments=args,
            raw=body,
            normalized=body,
            evidence=[{"operation_id": oid, "note": "mock"}],
            protocol_success=True,
            execution_success=True,
            real_mcp=False,
            transport="mock",
        )

    bundle, gateway = _build_gateway(handler=handler)
    router, _ = _build_router(tmp_path, bundle)
    hits = [_tool_hit(oid, rank=i + 1) for i, oid in enumerate(HKEX_CHAIN)]

    class _Router:
        embedder = router.embedder

        def retrieve(self, request, **kwargs):
            return ExtensionRound(
                round_index=0,
                request=request,
                servers=[],
                tools=hits,
                new_operation_ids=list(HKEX_CHAIN),
                is_extension=False,
                stop_reason="ok",
                latency_ms=1.0,
                embedding_calls=2,
                embedding_tokens=16,
            )

        def format_tools_for_prompt(self, tools):
            return json.dumps([t.operation_id for t in tools])

    atr = format_active_tool_request(
        "HKEX disclosure / HKEXnews announcement MCP",
        "list download parse keyword search and get announcement pages for Monthly Return",
    )
    script = [atr]
    for oid in HKEX_CHAIN:
        script.append(_invoke(oid, {"stock_code": "00700"} if "list" in oid else {}))
    script.append(
        json.dumps(
            {
                "thought": "answer",
                "tool_calls": [],
                "final_answer": {
                    "status": "answered",
                    "final_answer": "start_page=1",
                    "reasoning_brief": "top hit page_number is 1",
                    "evidence": [
                        {
                            "operation_id": "hkexnews.keyword_search",
                            "fields": ["hits"],
                            "note": "page_number=1",
                        }
                    ],
                },
            }
        )
    )
    llm = ScriptedLLMClient(script)
    agent = McpZeroAgent(
        llm=llm,
        router=_Router(),
        gateway=gateway,
        max_llm_turns=8,
        max_invocations=6,
        debug_dir=tmp_path / "dbg",
    )
    result = agent.run(NEWS_QUESTION, task_label="offline_news_chain")
    assert result["error_type"] is None
    assert result["status"] == "full"
    assert call_log == HKEX_CHAIN
    assert result["selected_operations"] == HKEX_CHAIN
    assert result["invoked_operations"] == HKEX_CHAIN
    assert result["successful_invoked_operations"] == HKEX_CHAIN
    assert "news_dev_006" not in json.dumps(result.get("mcp_zero_trace") or {})
    gateway.close()


# ---------------------------------------------------------------------------
# Identity invariants for this hardening pack
# ---------------------------------------------------------------------------


def test_method_and_prompt_hash_unchanged_by_hardening():
    assert method_identity_sha256() == FROZEN_METHOD_HASH
    # Prompt text not modified in this pack (repair is runtime user feedback).
    assert len(mcp_zero_prompt_hash()) == 64
