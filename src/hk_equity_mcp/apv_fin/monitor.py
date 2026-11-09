"""Exceptional-replan Monitor: model-owned plan patches only (no runtime state)."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from hk_equity_mcp.agent.llm_client import LLMClient
from hk_equity_mcp.apv_fin.plan_state import PlanState, PlanStateError
from hk_equity_mcp.apv_fin.planner import _merge_usage
from hk_equity_mcp.apv_fin.prompts import (
    APV_FIN_MONITOR_USER_TEMPLATE,
    APV_FIN_SYSTEM_PROMPT,
)
from hk_equity_mcp.apv_fin.schemas import MONITOR_DECISIONS, assert_no_operation_id_in_plan
from hk_equity_mcp.apv_fin.structured_parse import parse_structured, repair_user_message


def run_monitor(
    llm: LLMClient,
    *,
    question: str,
    plan_state: PlanState,
    observations: List[Dict[str, Any]],
    evidence_summary: Dict[str, Any],
    budget: Dict[str, Any],
    exception_trigger: Optional[str] = None,
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
                plan_json=json.dumps(
                    plan_state.model_plan_dict(), ensure_ascii=False, indent=2
                ),
                observations_json=json.dumps(observations[-8:], ensure_ascii=False),
                evidence_summary_json=json.dumps(evidence_summary, ensure_ascii=False),
                budget_json=json.dumps(budget, ensure_ascii=False),
                exception_trigger=str(
                    exception_trigger
                    or (budget or {}).get("exception_trigger")
                    or "unspecified"
                ),
            ),
        },
    ]
    resp = llm.complete(
        messages, temperature=temperature, max_tokens=max_tokens, stage="monitor"
    )
    parsed = parse_structured(resp.get("content"), "apv_fin_monitor_v1")
    raw_contents = [parsed["raw_content"]]
    if not parsed["ok"]:
        messages.append({"role": "assistant", "content": str(resp.get("content") or "")})
        messages.append(
            {
                "role": "user",
                "content": repair_user_message(
                    "apv_fin_monitor_v1", parsed.get("error") or "parse_failed"
                ),
            }
        )
        repair_resp = llm.complete(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stage="monitor_repair",
        )
        parsed = parse_structured(
            resp.get("content"),
            "apv_fin_monitor_v1",
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
                "fail_closed": True,
                "error": parsed.get("error") or "monitor_parse_fail_closed",
                "repair_count": int(parsed.get("repair_count") or 1),
                "llm": resp,
                "payload": None,
                "patch_summary": None,
                "raw_contents": raw_contents,
            }

    payload = parsed["payload"]
    decision = payload.get("decision")
    if decision not in MONITOR_DECISIONS:
        return {
            "ok": False,
            "fail_closed": True,
            "error": f"invalid monitor decision: {decision!r}",
            "repair_count": int(parsed.get("repair_count") or 0),
            "llm": resp,
            "payload": payload,
            "patch_summary": None,
            "raw_contents": raw_contents,
        }

    patch_summary = None
    try:
        patch = payload.get("patch") or {}
        # Reject runtime-only fields anywhere in the patch before apply.
        _assert_no_runtime_fields_in_patch(patch)
        if decision == "replan":
            assert_no_operation_id_in_plan(
                {"initial_plan": {"steps": patch.get("new_plan_steps") or []}}
            )
            patch_summary = plan_state.apply_monitor_patch(patch)
        elif patch.get("completed_plan_step_ids") or patch.get("completed_steps"):
            # Suggestions only — apply_monitor_patch does not auto-resolve.
            patch_summary = plan_state.apply_monitor_patch(
                {
                    "completed_plan_step_ids": list(
                        patch.get("completed_plan_step_ids")
                        or patch.get("completed_steps")
                        or []
                    ),
                    "updated_plan_steps": [],
                    "new_plan_steps": [],
                    "removed_plan_step_ids": [],
                }
            )
    except (PlanStateError, ValueError) as exc:
        return {
            "ok": False,
            "fail_closed": True,
            "error": f"{type(exc).__name__}: {exc}",
            "repair_count": int(parsed.get("repair_count") or 0),
            "llm": resp,
            "payload": payload,
            "patch_summary": None,
            "raw_contents": raw_contents,
        }

    return {
        "ok": True,
        "fail_closed": False,
        "error": None,
        "repair_count": int(parsed.get("repair_count") or 0),
        "llm": resp,
        "payload": payload,
        "decision": decision,
        "decision_reason": payload.get("decision_reason"),
        "ready_step_ids": list(payload.get("ready_step_ids") or []),
        "patch_summary": patch_summary,
        "raw_contents": raw_contents,
        "exception_trigger": exception_trigger,
    }


def _assert_no_runtime_fields_in_patch(patch: Dict[str, Any]) -> None:
    from hk_equity_mcp.apv_fin.plan_state import RUNTIME_ONLY_STEP_FIELDS

    for key in ("updated_plan_steps", "updated_steps", "new_plan_steps", "new_steps"):
        for entry in patch.get(key) or []:
            if not isinstance(entry, dict):
                continue
            leaked = [k for k in RUNTIME_ONLY_STEP_FIELDS if k in entry]
            if leaked:
                raise PlanStateError(
                    f"monitor patch must not include runtime fields: {leaked}"
                )
