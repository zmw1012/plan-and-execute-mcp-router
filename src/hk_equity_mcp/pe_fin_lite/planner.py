"""One-shot Planner (+ at most one JSON repair). No operation_id output."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from hk_equity_mcp.agent.llm_client import LLMClient
from hk_equity_mcp.pe_fin_lite.prompts import (
    PE_FIN_LITE_SYSTEM_PROMPT,
    render_planner_user,
)
from hk_equity_mcp.pe_fin_lite.schemas import (
    PlanValidationError,
    SchemaValidationError,
    extract_json_object,
    format_repair_user_message,
    parse_and_validate,
    validate_plan_semantics,
)


def _merge_usage(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(a or {})
    for key, value in (b or {}).items():
        if isinstance(value, (int, float)) and isinstance(out.get(key), (int, float)):
            out[key] = out[key] + value
        else:
            out[key] = value
    return out


def run_planner(
    *,
    llm: LLMClient,
    question: str,
    capability_profile: Dict[str, Any],
    limits: Dict[str, Any],
    max_plan_steps: int = 6,
    temperature: float = 0.0,
    max_tokens: int = 2048,
) -> Dict[str, Any]:
    user = render_planner_user(
        question=question,
        capability_profile=capability_profile,
        limits={
            "max_plan_steps": max_plan_steps,
            "max_total_llm_calls_per_task": limits.get("max_total_llm_calls_per_task", 8),
            "max_total_mcp_invocations": limits.get("max_total_mcp_invocations", 8),
            "no_global_replan": True,
            "no_monitor": True,
        },
        max_plan_steps=max_plan_steps,
    )
    messages = [
        {"role": "system", "content": PE_FIN_LITE_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
    completion = llm.complete(
        messages, temperature=temperature, max_tokens=max_tokens, stage="planner"
    )
    usage = dict(completion.get("usage") or {})
    content = completion.get("content")
    parsed = parse_and_validate(str(content or ""), "plan")
    repair_count = 0
    if not parsed[0]:
        repair_msg = format_repair_user_message("plan", parsed[2] or "invalid plan")
        repair_completion = llm.complete(
            messages
            + [
                {"role": "assistant", "content": str(content or "")},
                {"role": "user", "content": repair_msg},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            stage="planner_repair",
        )
        usage = _merge_usage(usage, repair_completion.get("usage") or {})
        repair_count = 1
        parsed = parse_and_validate(
            str(content or ""),
            "plan",
            repair_content=str(repair_completion.get("content") or ""),
        )
        content_repair = repair_completion.get("content")
    else:
        content_repair = None

    payload, _, error = parsed
    if payload is None:
        return {
            "ok": False,
            "plan": None,
            "error": error or "planner_failed",
            "repair_count": repair_count,
            "usage": usage,
            "raw_content": content,
            "repair_raw_content": content_repair,
        }
    # Defense in depth
    try:
        validate_plan_semantics(payload)
    except PlanValidationError as exc:
        return {
            "ok": False,
            "plan": None,
            "error": str(exc),
            "repair_count": repair_count,
            "usage": usage,
            "raw_content": content,
            "repair_raw_content": content_repair,
        }
    return {
        "ok": True,
        "plan": payload,
        "error": None,
        "repair_count": repair_count,
        "usage": usage,
        "raw_content": content,
        "repair_raw_content": content_repair,
    }


def topological_ready_order(steps: list) -> list:
    """Return step dicts in a valid DAG execution order (Kahn)."""
    by_id = {str(s["id"]): s for s in steps}
    indeg = {sid: 0 for sid in by_id}
    children = {sid: [] for sid in by_id}
    for sid, step in by_id.items():
        for dep in step.get("depends_on") or []:
            if dep in by_id:
                indeg[sid] += 1
                children[dep].append(sid)
    queue = [sid for sid, d in indeg.items() if d == 0]
    order = []
    while queue:
        # Stable: original order among ready
        queue.sort(key=lambda x: list(by_id.keys()).index(x))
        sid = queue.pop(0)
        order.append(by_id[sid])
        for child in children[sid]:
            indeg[child] -= 1
            if indeg[child] == 0:
                queue.append(child)
    if len(order) != len(steps):
        raise PlanValidationError("plan DAG not fully orderable (cycle)")
    return order
