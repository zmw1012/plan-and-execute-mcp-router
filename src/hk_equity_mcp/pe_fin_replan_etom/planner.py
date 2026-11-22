"""One-shot Planner for PE-Fin RePlan-ETOM (+ at most one JSON repair).

Thin wrapper: ETOM prompts + ETOM schemas. No operation_id in the output.
DAG helper ``topological_ready_order`` and ``plan_ops`` are domain-agnostic and
imported from the HK packages unchanged.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from hk_equity_mcp.pe_fin_lite.planner import topological_ready_order  # noqa: F401
from hk_equity_mcp.pe_fin_replan.plan_ops import (  # noqa: F401  (re-exported)
    diff_plans,
    ready_steps,
    remove_step,
    select_one_ready_step,
    validate_updated_remaining_plan,
)
from hk_equity_mcp.pe_fin_replan_etom.prompts import PLANNER_SYSTEM_PROMPT, render_planner_user
from hk_equity_mcp.pe_fin_replan_etom.schemas import (
    PLANNER_MAX_TOKENS,
    format_repair_user_message,
    parse_and_validate,
)


def _merge_usage(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(a or {})
    for key, value in (b or {}).items():
        if isinstance(value, (int, float)) and isinstance(out.get(key), (int, float)):
            out[key] = out[key] + value
        else:
            out[key] = value
    return out


def _failure(*, error: str, repair_count: int, usage: Dict[str, Any], calls: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "ok": False,
        "plan": None,
        "error": error,
        "error_type": "planner_schema_or_semantics",
        "stopping_reason": "planner_invalid_after_repair" if repair_count else "planner_invalid",
        "repair_count": repair_count,
        "usage": usage,
        "call_observability": calls,
    }


def run_planner(
    *,
    llm: Any,
    question: str,
    capability_profile: Dict[str, Any],
    limits: Dict[str, Any],
    max_plan_steps: int = 6,
    temperature: float = 0.0,
    max_tokens: int = PLANNER_MAX_TOKENS,
) -> Dict[str, Any]:
    user = render_planner_user(
        question=question,
        capability_profile=capability_profile,
        limits=limits,
        max_plan_steps=max_plan_steps,
    )
    messages = [
        {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
    completion = llm.complete(messages, temperature=temperature, max_tokens=max_tokens, stage="planner")
    usage = dict(completion.get("usage") or {})
    content = completion.get("content")
    payload, _rc, error = parse_and_validate(str(content) if content is not None else "", "plan")
    type_recs = list((payload or {}).get("step_type_canonicalizations") or [])
    calls: List[Dict[str, Any]] = [
        {
            "stage": "planner",
            "parse_result": "ok" if payload else "failed",
            "schema_error": error,
            "step_type_canonicalizations": type_recs,
        }
    ]
    repair_count = 0
    if payload is None:
        repair = llm.complete(
            messages
            + [
                {"role": "assistant", "content": str(content or "")},
                {
                    "role": "user",
                    "content": format_repair_user_message(
                        "plan", error or "invalid plan", previous_content=str(content or "")
                    ),
                },
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            stage="planner_repair",
        )
        repair_count = 1
        usage = _merge_usage(usage, repair.get("usage") or {})
        payload, _rc, error = parse_and_validate(str(repair.get("content") or ""), "plan")
        type_recs = list((payload or {}).get("step_type_canonicalizations") or [])
        calls.append(
            {
                "stage": "planner_repair",
                "parse_result": "ok" if payload else "failed",
                "schema_error": error,
                "step_type_canonicalizations": type_recs,
            }
        )
    if payload is None:
        return _failure(
            error=error or "planner_failed", repair_count=repair_count, usage=usage, calls=calls
        )
    # Observability stays in call_observability; keep executable plan schema-clean.
    plan = {k: v for k, v in payload.items() if k != "step_type_canonicalizations"}
    return {
        "ok": True,
        "plan": plan,
        "error": None,
        "error_type": None,
        "stopping_reason": None,
        "repair_count": repair_count,
        "usage": usage,
        "call_observability": calls,
        "step_type_canonicalizations": type_recs,
        "planner_max_tokens": int(max_tokens),
    }
