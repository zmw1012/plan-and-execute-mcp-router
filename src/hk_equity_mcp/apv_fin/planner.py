"""Initial structured planner: one LLM call → task_spec + initial plan."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional, Tuple

from hk_equity_mcp.agent.llm_client import LLMClient
from hk_equity_mcp.apv_fin.plan_state import PlanState, PlanStateError
from hk_equity_mcp.apv_fin.prompts import (
    APV_FIN_PLANNER_USER_TEMPLATE,
    APV_FIN_SYSTEM_PROMPT,
)
from hk_equity_mcp.apv_fin.schemas import (
    assert_no_operation_id_in_plan,
    format_repair_user_message,
    parse_and_validate,
)


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
    resp = llm.complete(messages, temperature=temperature, max_tokens=max_tokens)
    content = resp["content"]
    payload, repair_count, error = parse_and_validate(content, "apv_fin_plan_v1")
    if error is not None:
        repair_msg = format_repair_user_message("apv_fin_plan_v1", error)
        messages.append({"role": "assistant", "content": content})
        messages.append({"role": "user", "content": repair_msg})
        repair_resp = llm.complete(
            messages, temperature=temperature, max_tokens=max_tokens
        )
        payload, repair_count2, error2 = parse_and_validate(
            content,
            "apv_fin_plan_v1",
            repair_content=repair_resp["content"],
        )
        repair_count = max(repair_count, repair_count2)
        resp = {
            "content": repair_resp["content"],
            "usage": _merge_usage(resp.get("usage"), repair_resp.get("usage")),
            "latency_ms": float(resp.get("latency_ms") or 0)
            + float(repair_resp.get("latency_ms") or 0),
            "provider": repair_resp.get("provider"),
            "model": repair_resp.get("model"),
            "repair": True,
        }
        if error2 is not None:
            return {
                "ok": False,
                "error": error2,
                "fail_closed": True,
                "repair_count": repair_count,
                "llm": resp,
                "payload": None,
                "plan_state": None,
            }

    try:
        assert_no_operation_id_in_plan(payload)
        plan_state = PlanState.from_initial_plan(payload["initial_plan"])
    except (PlanStateError, ValueError) as exc:
        return {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "fail_closed": True,
            "repair_count": repair_count,
            "llm": resp,
            "payload": payload,
            "plan_state": None,
        }

    return {
        "ok": True,
        "error": None,
        "fail_closed": False,
        "repair_count": repair_count,
        "llm": resp,
        "payload": payload,
        "task_spec": payload["task_spec"],
        "initial_plan": payload["initial_plan"],
        "plan_state": plan_state,
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
        {"revision": plan_state.revision, "steps": [s.to_dict() for s in plan_state.steps.values()]},
        ensure_ascii=False,
        indent=2,
    )
