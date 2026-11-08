"""Plan monitor: bounded decisions + incremental patches."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from hk_equity_mcp.agent.llm_client import LLMClient
from hk_equity_mcp.apv_fin.plan_state import PlanState, PlanStateError
from hk_equity_mcp.apv_fin.planner import _merge_usage, plan_to_public_json
from hk_equity_mcp.apv_fin.prompts import (
    APV_FIN_MONITOR_USER_TEMPLATE,
    APV_FIN_SYSTEM_PROMPT,
)
from hk_equity_mcp.apv_fin.schemas import (
    MONITOR_DECISIONS,
    assert_no_operation_id_in_plan,
    format_repair_user_message,
    parse_and_validate,
)


def run_monitor(
    llm: LLMClient,
    *,
    question: str,
    plan_state: PlanState,
    observations: List[Dict[str, Any]],
    evidence_summary: Dict[str, Any],
    budget: Dict[str, Any],
    temperature: float = 0.0,
    max_tokens: int = 2048,
) -> Dict[str, Any]:
    messages = [
        {"role": "system", "content": APV_FIN_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": APV_FIN_MONITOR_USER_TEMPLATE.format(
                question=question,
                revision=plan_state.revision,
                plan_json=plan_to_public_json(plan_state),
                observations_json=json.dumps(observations[-8:], ensure_ascii=False),
                evidence_summary_json=json.dumps(evidence_summary, ensure_ascii=False),
                budget_json=json.dumps(budget, ensure_ascii=False),
            ),
        },
    ]
    resp = llm.complete(messages, temperature=temperature, max_tokens=max_tokens)
    content = resp["content"]
    payload, repair_count, error = parse_and_validate(content, "apv_fin_monitor_v1")
    if error is not None:
        messages.append({"role": "assistant", "content": content})
        messages.append(
            {
                "role": "user",
                "content": format_repair_user_message("apv_fin_monitor_v1", error),
            }
        )
        repair_resp = llm.complete(
            messages, temperature=temperature, max_tokens=max_tokens
        )
        payload, repair_count2, error2 = parse_and_validate(
            content,
            "apv_fin_monitor_v1",
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
                "fail_closed": True,
                "error": error2,
                "repair_count": repair_count,
                "llm": resp,
                "payload": None,
                "patch_summary": None,
            }

    decision = payload.get("decision")
    if decision not in MONITOR_DECISIONS:
        return {
            "ok": False,
            "fail_closed": True,
            "error": f"invalid monitor decision: {decision!r}",
            "repair_count": repair_count,
            "llm": resp,
            "payload": payload,
            "patch_summary": None,
        }

    patch_summary = None
    try:
        if decision == "replan":
            assert_no_operation_id_in_plan({"initial_plan": {"steps": payload["patch"].get("new_steps") or []}})
            patch_summary = plan_state.apply_monitor_patch(payload.get("patch") or {})
        elif payload.get("patch"):
            # Allow non-replan decisions to mark completed steps without full replan.
            patch = payload["patch"]
            soft = {
                "completed_steps": list(patch.get("completed_steps") or []),
                "updated_steps": [],
                "new_steps": [],
                "removed_steps": [],
            }
            if soft["completed_steps"]:
                patch_summary = plan_state.apply_monitor_patch(soft)
    except (PlanStateError, ValueError) as exc:
        return {
            "ok": False,
            "fail_closed": True,
            "error": f"{type(exc).__name__}: {exc}",
            "repair_count": repair_count,
            "llm": resp,
            "payload": payload,
            "patch_summary": None,
        }

    return {
        "ok": True,
        "fail_closed": False,
        "error": None,
        "repair_count": repair_count,
        "llm": resp,
        "payload": payload,
        "decision": decision,
        "decision_reason": payload.get("decision_reason"),
        "ready_step_ids": list(payload.get("ready_step_ids") or []),
        "patch_summary": patch_summary,
    }
