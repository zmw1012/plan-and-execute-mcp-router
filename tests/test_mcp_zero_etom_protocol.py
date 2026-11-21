"""ETOM route parser / protocol tests (no task-id branches)."""

from __future__ import annotations

import json

from hk_equity_mcp.mcp_zero.active_request import parse_active_tool_request
from hk_equity_mcp.mcp_zero.agent import _parse_agent_json
from hk_equity_mcp.mcp_zero_etom.agent import EtomMcpZeroAgent, _is_valid_route_final
from hk_equity_mcp.mcp_zero_etom.mock_llm import EtomMockLLM, atr_text, final_json, invoke_json
from hk_equity_mcp.mcp_zero_etom.prompt import assert_etom_prompt_safe


def test_prompt_has_no_gold_markers():
    assert_etom_prompt_safe()


def test_raw_atr_parse():
    text = atr_text("search server", "keyword search")
    req = parse_active_tool_request(text, round_index=0, source="llm")
    assert req is not None
    assert "search" in req.server.lower()


def test_json_wrapped_route_and_fenced():
    payload = {
        "thought": "x",
        "tool_calls": [],
        "final_answer": {"status": "route_complete", "infeasible": False, "reasoning_brief": "ok"},
    }
    raw = json.dumps(payload)
    assert _is_valid_route_final(_parse_agent_json(raw)["final_answer"])
    fenced = "```json\n" + raw + "\n```"
    assert _is_valid_route_final(_parse_agent_json(fenced)["final_answer"])


def test_smart_quotes_and_trailing_comma_repair_paths():
    # trailing comma should fail json parse -> agent repair path (unit: parser raises)
    bad = '{"thought":"x","tool_calls":[],"final_answer":{"status":"route_complete","infeasible":false,},}'
    try:
        _parse_agent_json(bad)
        raised = False
    except Exception:
        raised = True
    assert raised
    smart = raw = (
        "{\u201cthought\u201d:\u201cx\u201d,\"tool_calls\":[],"
        "\"final_answer\":{\"status\":\"infeasible\",\"infeasible\":true,"
        "\"reasoning_brief\":\"no\"}}"
    )
    # smart quotes generally fail standard json; ensure no task-id branch exists in agent source
    import inspect
    src = inspect.getsource(EtomMcpZeroAgent)
    assert "L1-q" not in src
    assert "task_id ==" not in src


def test_empty_action_not_valid_final():
    assert _is_valid_route_final(None) is False
    assert _is_valid_route_final({"status": "route_complete"}) is True


def test_l5_rejection_final():
    assert _is_valid_route_final(
        {"status": "infeasible", "infeasible": True, "reasoning_brief": "x"}
    )


def test_mock_llm_emits_raw_content_sequence():
    llm = EtomMockLLM()
    llm.set_task("x", [atr_text("s", "t"), final_json(infeasible=True)])
    r1 = llm.complete([{"role": "user", "content": "q"}])
    r2 = llm.complete([{"role": "user", "content": "q"}])
    assert "<tool_assistant>" in r1["content"]
    assert "infeasible" in r2["content"]
