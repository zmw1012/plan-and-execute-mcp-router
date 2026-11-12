"""Replanner LLM wrappers for PE-Fin RePlan-ETOM: Plan or Response.

At most one schema/semantic repair (shared) and one transient transport retry.
ETOM prompts/schemas; transport retry helper is imported from the HK package
(it is provider-agnostic).
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from hk_equity_mcp.pe_fin_lite.planner import topological_ready_order
from hk_equity_mcp.pe_fin_lite.schemas import PlanValidationError
from hk_equity_mcp.pe_fin_replan.plan_ops import diff_plans
from hk_equity_mcp.pe_fin_replan.transport_retry import complete_with_transient_retry
from hk_equity_mcp.pe_fin_replan_etom.prompts import (
    REPLANNER_FINAL_ROUND_SYSTEM_PROMPT,
    REPLANNER_SYSTEM_PROMPT,
    render_replanner_user,
)
from hk_equity_mcp.pe_fin_replan_etom.schemas import (
    format_replanner_repair_message,
    format_semantic_plan_repair_message,
    parse_replanner_output,
)


def _merge_usage(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(a or {})
    for k, v in (b or {}).items():
        if isinstance(v, (int, float)) and isinstance(out.get(k), (int, float)):
            out[k] = out[k] + v
        else:
            out[k] = v
    return out


def _obs(stage: str, parsed: Optional[Dict[str, Any]], err: Optional[str], repair_count: int) -> Dict[str, Any]:
    return {
        "stage": stage,
        "parse_result": "ok" if parsed is not None else "failed",
        "schema_error": None if parsed is not None else err,
        "repair_count": int(repair_count),
    }


def run_replanner(
    *,
    llm: Any,
    compact_state: Dict[str, Any],
    temperature: float = 0.0,
    max_tokens: int = 2048,
    response_only: bool = False,
    execution_round: int = 0,
    max_execution_rounds: int = 7,
) -> Dict[str, Any]:
    system = REPLANNER_FINAL_ROUND_SYSTEM_PROMPT if response_only else REPLANNER_SYSTEM_PROMPT
    user = render_replanner_user(
        compact_state=compact_state,
        response_only=response_only,
        execution_round=execution_round,
        max_execution_rounds=max_execution_rounds,
    )
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    calls: List[Dict[str, Any]] = []
    transport_meta: Dict[str, Any] = {}
    try:
        completion, transport_meta = complete_with_transient_retry(
            llm, messages, temperature=temperature, max_tokens=max_tokens, stage="replanner"
        )
    except Exception as exc:  # noqa: BLE001
        if type(exc).__name__ in ("LlmCallBudgetExhausted", "McpBudgetExhausted"):
            raise
        return {
            "ok": False,
            "payload": None,
            "error": str(exc),
            "error_type": "llm_transport_failure",
            "repair_count": 0,
            "usage": {},
            "stopping_reason": "llm_transport_failure_after_retry",
            "response_only": response_only,
            "observability": {"transport": transport_meta, "calls": calls},
        }
    usage = dict(completion.get("usage") or {})
    content = completion.get("content")
    parsed, _, err = parse_replanner_output(str(content or ""), response_only=response_only)
    calls.append(_obs("replanner", parsed, err, 0))
    repair_count = 0
    repair_content = None
    if parsed is None:
        json_path = err.split(":", 1)[0].strip() if err and err.startswith("$") else "$"
        try:
            repair, _rt = complete_with_transient_retry(
                llm,
                messages
                + [
                    {"role": "assistant", "content": str(content or "")},
                    {
                        "role": "user",
                        "content": format_replanner_repair_message(
                            err or "invalid", response_only=response_only, json_path=json_path
                        ),
                    },
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                stage="replanner_repair",
            )
        except Exception as exc:  # noqa: BLE001
            if type(exc).__name__ in ("LlmCallBudgetExhausted", "McpBudgetExhausted"):
                raise
            return {
                "ok": False,
                "payload": None,
                "error": str(exc),
                "error_type": "llm_transport_failure",
                "repair_count": 1,
                "usage": usage,
                "stopping_reason": "llm_transport_failure_after_retry",
                "response_only": response_only,
                "observability": {"transport": transport_meta, "calls": calls},
            }
        usage = _merge_usage(usage, repair.get("usage") or {})
        repair_count = 1
        repair_content = repair.get("content")
        parsed, _, err = parse_replanner_output(
            str(content or ""), repair_content=str(repair_content or ""), response_only=response_only
        )
        calls.append(_obs("replanner_repair", parsed, err, 1))
    assistant_content = str(repair_content or "") if repair_count else str(content or "")
    if parsed is None:
        return {
            "ok": False,
            "payload": None,
            "error": err or "replanner_invalid_after_repair",
            "error_type": "replanner_schema_or_semantics",
            "repair_count": repair_count,
            "usage": usage,
            "stopping_reason": "invalid_replanner_output_after_repair",
            "response_only": response_only,
            "assistant_content": assistant_content,
            "messages": messages,
            "observability": {"transport": transport_meta, "calls": calls},
        }
    if response_only and parsed.get("action") != "response":
        return {
            "ok": False,
            "payload": None,
            "error": "forced_final_round_plan_not_allowed",
            "error_type": "replanner_schema_or_semantics",
            "repair_count": repair_count,
            "usage": usage,
            "stopping_reason": "forced_final_round_plan_after_repair",
            "response_only": response_only,
            "assistant_content": assistant_content,
            "messages": messages,
            "observability": {"transport": transport_meta, "calls": calls},
        }
    return {
        "ok": True,
        "payload": parsed,
        "error": None,
        "repair_count": repair_count,
        "usage": usage,
        "action": parsed.get("action"),
        "decision_reason": parsed.get("decision_reason"),
        "response_only": response_only,
        "assistant_content": assistant_content,
        "messages": messages,
        "observability": {"transport": transport_meta, "calls": calls},
    }


def run_replanner_semantic_repair(
    *,
    llm: Any,
    messages: List[Dict[str, str]],
    previous_content: str,
    semantic_error: str,
    max_new_steps: int = 2,
    execution_round: int = 0,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    response_only: bool = False,
) -> Dict[str, Any]:
    """One full semantic repair (shares the single repair budget with schema repair)."""
    try:
        repair, _rt = complete_with_transient_retry(
            llm,
            list(messages)
            + [
                {"role": "assistant", "content": str(previous_content or "")},
                {
                    "role": "user",
                    "content": format_semantic_plan_repair_message(
                        semantic_error,
                        max_new_steps=max_new_steps,
                        execution_round=execution_round,
                        response_only=response_only,
                    ),
                },
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            stage="replanner_semantic_repair",
        )
    except Exception as exc:  # noqa: BLE001
        if type(exc).__name__ in ("LlmCallBudgetExhausted", "McpBudgetExhausted"):
            raise
        return {
            "ok": False,
            "payload": None,
            "error": str(exc),
            "error_type": "llm_transport_failure",
            "repair_count": 1,
            "usage": {},
            "stopping_reason": "llm_transport_failure_after_retry",
            "response_only": response_only,
        }
    usage = dict(repair.get("usage") or {})
    content = str(repair.get("content") or "")
    parsed, _, err = parse_replanner_output(content, response_only=response_only)
    obs = {"calls": [_obs("replanner_semantic_repair", parsed, err, 1)]}
    if parsed is None:
        return {
            "ok": False,
            "payload": None,
            "error": err or "semantic_repair_invalid_after_repair",
            "repair_count": 1,
            "usage": usage,
            "stopping_reason": "invalid_plan_update_after_semantic_repair",
            "response_only": response_only,
            "assistant_content": content,
            "observability": obs,
        }
    if response_only and parsed.get("action") != "response":
        return {
            "ok": False,
            "payload": None,
            "error": "forced_final_round_plan_not_allowed",
            "repair_count": 1,
            "usage": usage,
            "stopping_reason": "forced_final_round_plan_after_repair",
            "response_only": response_only,
            "assistant_content": content,
            "observability": obs,
        }
    return {
        "ok": True,
        "payload": parsed,
        "error": None,
        "repair_count": 1,
        "usage": usage,
        "action": parsed.get("action"),
        "decision_reason": parsed.get("decision_reason"),
        "response_only": response_only,
        "assistant_content": content,
        "observability": obs,
    }


def validate_remaining_plan_update(
    *,
    proposed: Sequence[Dict[str, Any]],
    previous_remaining: Sequence[Dict[str, Any]],
    completed_ids: Set[str],
    completed_objectives: Set[str],
    max_new_steps: int = 2,
    executed_ids: Optional[Set[str]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """ETOM-local plan-update validation (generic; no HK scope keys).

    * unique ids, <= ``max_new_steps`` ids not present in the previous remaining plan;
    * a completed id may never reappear;
    * repeating a completed objective is refused unless ``scope_constraints`` carries
      at least one non-null value (i.e. the step is plausibly a different target);
    * the DAG must be orderable. Never truncates or deletes steps silently.
    """
    steps = [deepcopy(s) for s in proposed]
    ids = [str(s.get("id")) for s in steps]
    if len(ids) != len(set(ids)):
        raise PlanValidationError("duplicate step ids in remaining_steps")
    prev_ids = {str(s.get("id")) for s in previous_remaining}
    new_ids = [i for i in ids if i not in prev_ids]
    if len(new_ids) > max_new_steps:
        raise PlanValidationError(f"too_many_new_steps:{len(new_ids)}>max={max_new_steps}")
    known = set(ids) | set(completed_ids) | set(executed_ids or ())
    for step in steps:
        sid = str(step.get("id"))
        if sid in completed_ids:
            raise PlanValidationError(f"completed_step_readded:{sid}")
        for dep in step.get("depends_on") or []:
            if str(dep) not in known:
                raise PlanValidationError(f"step {sid} depends on unknown {dep}")
        obj = str(step.get("objective") or "").strip().lower()
        if obj and obj in completed_objectives and sid not in prev_ids:
            scope = step.get("scope_constraints")
            has_scope = isinstance(scope, dict) and any(
                v not in (None, "") for v in scope.values()
            )
            if not has_scope:
                raise PlanValidationError(
                    f"completed_objective_repeat_without_scope_change:{obj[:80]}"
                )
    if steps:
        try:
            topological_ready_order(steps)
        except PlanValidationError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise PlanValidationError(f"remaining_plan_not_orderable:{exc}") from exc
    return steps, diff_plans(previous_remaining, steps)
