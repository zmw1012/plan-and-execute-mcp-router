"""Initial structured planner: one LLM call → task_spec + initial plan."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from hk_equity_mcp.agent.llm_client import LLMClient
from hk_equity_mcp.apv_fin.plan_state import PlanState, PlanStateError
from hk_equity_mcp.apv_fin.prompts import (
    APV_FIN_PLANNER_USER_TEMPLATE,
    APV_FIN_SYSTEM_PROMPT,
)
from hk_equity_mcp.apv_fin.schemas import assert_no_operation_id_in_plan
from hk_equity_mcp.apv_fin.structured_parse import parse_structured, repair_user_message


def run_initial_planner(
    llm: LLMClient,
    question: str,
    *,
    temperature: float = 0.0,
    max_tokens: int = 2048,
) -> Dict[str, Any]:
    messages = [
        {"role": "system", "content": APV_FIN_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": APV_FIN_PLANNER_USER_TEMPLATE.format(question=question),
        },
    ]
    resp = llm.complete(
        messages, temperature=temperature, max_tokens=max_tokens, stage="planner"
    )
    parsed = parse_structured(resp.get("content"), "apv_fin_plan_v1")
    raw_contents = [parsed["raw_content"]]
    if not parsed["ok"]:
        repair_msg = repair_user_message(
            "apv_fin_plan_v1", parsed.get("error") or "parse_failed"
        )
        messages.append({"role": "assistant", "content": str(resp.get("content") or "")})
        messages.append({"role": "user", "content": repair_msg})
        repair_resp = llm.complete(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stage="planner_repair",
        )
        parsed = parse_structured(
            resp.get("content"),
            "apv_fin_plan_v1",
            repair_content=repair_resp.get("content"),
        )
        raw_contents.append(parsed.get("repair_raw_content"))
        resp = {
            "content": repair_resp.get("content"),
            "raw_content": repair_resp.get("raw_content", repair_resp.get("content")),
            "usage": _merge_usage(resp.get("usage"), repair_resp.get("usage")),
            "latency_ms": float(resp.get("latency_ms") or 0)
            + float(repair_resp.get("latency_ms") or 0),
            "provider": repair_resp.get("provider"),
            "model": repair_resp.get("model"),
            "repair": True,
            "content_was_null": repair_resp.get("content_was_null"),
        }
        if not parsed["ok"]:
            return {
                "ok": False,
                "error": parsed.get("error") or "planner_parse_fail_closed",
                "fail_closed": True,
                "repair_count": int(parsed.get("repair_count") or 1),
                "llm": resp,
                "payload": None,
                "plan_state": None,
                "raw_contents": raw_contents,
            }

    payload = parsed["payload"]
    # Reject empty steps even if somehow schema drifted.
    steps = ((payload or {}).get("initial_plan") or {}).get("steps") or []
    if not steps:
        return {
            "ok": False,
            "error": "empty_steps_not_allowed",
            "fail_closed": True,
            "repair_count": int(parsed.get("repair_count") or 0),
            "llm": resp,
            "payload": payload,
            "plan_state": None,
            "raw_contents": raw_contents,
        }

    try:
        assert_no_operation_id_in_plan(payload)
        # Gold / reference must never appear in planner structured output.
        blob = json.dumps(payload, ensure_ascii=False).lower()
        for banned in (
            "gold_operations",
            "reference_answer",
            "acceptable_tool_sets",
            "expected_tool_path",
            "gold_values",
        ):
            if banned in blob:
                raise ValueError(f"planner_output_contains_banned_{banned}")
        plan_state = PlanState.from_initial_plan(payload["initial_plan"])
    except (PlanStateError, ValueError) as exc:
        return {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "fail_closed": True,
            "repair_count": int(parsed.get("repair_count") or 0),
            "llm": resp,
            "payload": payload,
            "plan_state": None,
            "raw_contents": raw_contents,
        }

    return {
        "ok": True,
        "error": None,
        "fail_closed": False,
        "repair_count": int(parsed.get("repair_count") or 0),
        "llm": resp,
        "payload": payload,
        "task_spec": payload["task_spec"],
        "initial_plan": payload["initial_plan"],
        "plan_state": plan_state,
        "raw_contents": raw_contents,
    }


def _merge_usage(a: Optional[Dict[str, Any]], b: Optional[Dict[str, Any]]) -> Dict[str, int]:
    a = a or {}
    b = b or {}
    return {
        "input_tokens": int(a.get("input_tokens") or 0) + int(b.get("input_tokens") or 0),
        "output_tokens": int(a.get("output_tokens") or 0)
        + int(b.get("output_tokens") or 0),
        "total_tokens": int(a.get("total_tokens") or 0) + int(b.get("total_tokens") or 0),
    }


def plan_to_public_json(plan_state: PlanState) -> str:
    return json.dumps(
        {
            "revision": plan_state.revision,
            "steps": [s.to_dict() for s in plan_state.steps.values()],
        },
        ensure_ascii=False,
        indent=2,
    )
