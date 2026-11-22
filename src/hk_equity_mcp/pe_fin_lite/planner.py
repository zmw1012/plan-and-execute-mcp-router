"""One-shot Planner (+ at most one JSON repair). No operation_id output."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from hk_equity_mcp.agent.llm_client import LLMClient, build_response_observability
from hk_equity_mcp.pe_fin_lite.prompts import (
    PE_FIN_LITE_SYSTEM_PROMPT,
    render_planner_user,
)
from hk_equity_mcp.pe_fin_lite.schemas import (
    PLANNER_MAX_TOKENS,
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


def _obs_from_completion(
    completion: Dict[str, Any],
    *,
    stage: str,
    max_tokens: int,
    parse_result: Optional[str] = None,
    schema_error: Optional[str] = None,
) -> Dict[str, Any]:
    base = dict(completion.get("response_observability") or {})
    if not base:
        usage = dict(completion.get("usage") or {})
        content = str(completion.get("content") or "")
        base = build_response_observability(
            stage=stage,
            content=content,
            usage=usage,
            latency_ms=float(completion.get("latency_ms") or 0.0),
            max_tokens_requested=max_tokens,
            provider=str(completion.get("provider") or ""),
            model=str(completion.get("model") or ""),
            host=completion.get("host"),
            finish_reason=completion.get("finish_reason"),
            message_meta={
                "message_was_null": completion.get("message_was_null"),
                "content_was_null": completion.get("content_was_null"),
                "message_keys": [],
                "reasoning_content_present": False,
                "reasoning_content_length": 0,
                "content_shape": None,
            },
        )
    base["stage"] = stage
    base["max_tokens_requested"] = int(max_tokens)
    if parse_result is not None:
        base["parse_result"] = parse_result
    if schema_error is not None:
        base["schema_error"] = schema_error
    # Never persist secrets / reasoning text.
    for banned in ("api_key", "authorization", "reasoning_content", "reasoning_text"):
        base.pop(banned, None)
    return base


def _planner_failure(
    *,
    error: str,
    repair_count: int,
    usage: Dict[str, Any],
    content: Any,
    content_repair: Any,
    error_type: str = "planner_schema_or_semantics",
    call_observability: Optional[List[Dict[str, Any]]] = None,
    max_tokens: int = PLANNER_MAX_TOKENS,
) -> Dict[str, Any]:
    return {
        "ok": False,
        "plan": None,
        "error": error,
        "error_type": error_type,
        "stopping_reason": (
            "planner_invalid_after_repair" if repair_count else "planner_invalid"
        ),
        "repair_count": repair_count,
        "usage": usage,
        "raw_content": content,
        "repair_raw_content": content_repair,
        "call_observability": list(call_observability or []),
        "planner_max_tokens": int(max_tokens),
    }


def run_planner(
    *,
    llm: LLMClient,
    question: str,
    capability_profile: Dict[str, Any],
    limits: Dict[str, Any],
    max_plan_steps: int = 6,
    temperature: float = 0.0,
    max_tokens: int = PLANNER_MAX_TOKENS,
) -> Dict[str, Any]:
    """Run planner with stage-specific max_tokens (default 4096 in protocol v3)."""
    user = render_planner_user(
        question=question,
        capability_profile=capability_profile,
        limits={
            "max_plan_steps": max_plan_steps,
            "max_total_llm_calls_per_task": limits.get("max_total_llm_calls_per_task", 8),
            "max_total_mcp_invocations": limits.get("max_total_mcp_invocations", 8),
            "no_global_replan": True,
            "no_monitor": True,
            "planner_max_tokens": int(max_tokens),
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
    parsed = parse_and_validate(str(content) if content is not None else "", "plan")
    call_obs: List[Dict[str, Any]] = [
        _obs_from_completion(
            completion,
            stage="planner",
            max_tokens=max_tokens,
            parse_result="ok" if parsed[0] is not None else "failed",
            schema_error=None if parsed[0] is not None else parsed[2],
        )
    ]
    repair_count = 0
    content_repair = None
    if not parsed[0]:
        repair_msg = format_repair_user_message(
            "plan",
            parsed[2] or "invalid plan",
            previous_content=str(content) if content is not None else "",
        )
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
        content_repair = repair_completion.get("content")
        parsed = parse_and_validate(
            str(content) if content is not None else "",
            "plan",
            repair_content=str(content_repair) if content_repair is not None else "",
        )
        call_obs.append(
            _obs_from_completion(
                repair_completion,
                stage="planner_repair",
                max_tokens=max_tokens,
                parse_result="ok" if parsed[0] is not None else "failed",
                schema_error=None if parsed[0] is not None else parsed[2],
            )
        )

    payload, _, error = parsed
    if payload is None:
        return _planner_failure(
            error=error or "planner_failed",
            repair_count=repair_count,
            usage=usage,
            content=content,
            content_repair=content_repair,
            call_observability=call_obs,
            max_tokens=max_tokens,
        )
    try:
        validate_plan_semantics(payload)
    except PlanValidationError as exc:
        return _planner_failure(
            error=str(exc),
            repair_count=repair_count,
            usage=usage,
            content=content,
            content_repair=content_repair,
            call_observability=call_obs,
            max_tokens=max_tokens,
        )
    return {
        "ok": True,
        "plan": payload,
        "error": None,
        "error_type": None,
        "stopping_reason": None,
        "repair_count": repair_count,
        "usage": usage,
        "raw_content": content,
        "repair_raw_content": content_repair,
        "call_observability": call_obs,
        "planner_max_tokens": int(max_tokens),
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
