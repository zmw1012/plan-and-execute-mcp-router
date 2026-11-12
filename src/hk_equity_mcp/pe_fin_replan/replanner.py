"""Replanner LLM call: Plan or Response (at most one repair + one transport retry)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from hk_equity_mcp.agent.llm_client import LLMClient
from hk_equity_mcp.pe_fin_lite.temporal_policy import DEFAULT_DATASET_CUTOFF
from hk_equity_mcp.pe_fin_replan.prompts import (
    REPLANNER_FINAL_ROUND_SYSTEM_PROMPT,
    REPLANNER_SYSTEM_PROMPT,
    render_replanner_user,
)
from hk_equity_mcp.pe_fin_replan.schemas import (
    format_replanner_repair_message,
    format_semantic_plan_repair_message,
    parse_replanner_output,
)
from hk_equity_mcp.pe_fin_replan.transport_retry import complete_with_transient_retry


def _merge_usage(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(a or {})
    for k, v in (b or {}).items():
        if isinstance(v, (int, float)) and isinstance(out.get(k), (int, float)):
            out[k] = out[k] + v
        else:
            out[k] = v
    return out


def _obs_from_completion(
    completion: Dict[str, Any],
    *,
    stage: str,
    max_tokens: int,
    parse_result: Optional[str] = None,
    schema_error: Optional[str] = None,
    repair_count: int = 0,
) -> Dict[str, Any]:
    """Secret-free observability; never includes hidden reasoning text or API keys."""
    base = dict(completion.get("response_observability") or {})
    content = str(completion.get("content") or "")
    out = {
        "stage": stage,
        "content_present": bool(content.strip())
        if "content_present" not in base
        else bool(base.get("content_present")),
        "content_length": len(content)
        if "content_length" not in base
        else int(base.get("content_length") or 0),
        "finish_reason": completion.get("finish_reason") or base.get("finish_reason"),
        "response_message_keys": list(
            base.get("response_message_keys")
            or completion.get("response_message_keys")
            or []
        ),
        "max_tokens_requested": int(
            base.get("max_tokens_requested")
            if base.get("max_tokens_requested") is not None
            else max_tokens
        ),
        "parse_result": parse_result,
        "schema_error": schema_error,
        "repair_count": int(repair_count),
    }
    # Drop any accidental secret-ish keys if present upstream.
    for banned in ("api_key", "authorization", "reasoning_content", "content"):
        out.pop(banned, None)
    return out


def run_replanner(
    *,
    llm: LLMClient,
    compact_state: Dict[str, Any],
    temperature: float = 0.0,
    max_tokens: int = 2048,
    benchmark_cutoff: str = DEFAULT_DATASET_CUTOFF,
    response_only: bool = False,
    execution_round: int = 0,
    max_execution_rounds: int = 7,
) -> Dict[str, Any]:
    system = (
        REPLANNER_FINAL_ROUND_SYSTEM_PROMPT
        if response_only
        else REPLANNER_SYSTEM_PROMPT
    )
    user = render_replanner_user(
        compact_state=compact_state,
        benchmark_cutoff=benchmark_cutoff,
        response_only=response_only,
        execution_round=execution_round,
        max_execution_rounds=max_execution_rounds,
    )
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    call_obs: List[Dict[str, Any]] = []
    transport_meta: Dict[str, Any] = {}
    try:
        completion, transport_meta = complete_with_transient_retry(
            llm,
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stage="replanner",
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "payload": None,
            "error": str(exc),
            "error_type": "llm_transport_failure",
            "repair_count": 0,
            "usage": {},
            "stopping_reason": "llm_transport_failure_after_retry",
            "response_only": response_only,
            "observability": {
                "transport": transport_meta,
                "calls": call_obs,
            },
        }

    usage = dict(completion.get("usage") or {})
    content = completion.get("content")
    parsed, _, err = parse_replanner_output(
        str(content or ""), response_only=response_only
    )
    call_obs.append(
        _obs_from_completion(
            completion,
            stage="replanner",
            max_tokens=max_tokens,
            parse_result="ok" if parsed is not None else "failed",
            schema_error=None if parsed is not None else err,
            repair_count=0,
        )
    )
    if transport_meta.get("transport_retry_attempted"):
        call_obs[-1]["transport_retry"] = dict(transport_meta)

    repair_count = 0
    repair_content = None
    if parsed is None:
        # Extract JSON path hint from error if present (e.g. "$.anyOf[0]: ...")
        json_path = "$"
        if err and err.startswith("$"):
            json_path = err.split(":", 1)[0].strip() or "$"
        repair_msg = format_replanner_repair_message(
            err or "invalid",
            response_only=response_only,
            json_path=json_path,
        )
        try:
            repair, repair_transport = complete_with_transient_retry(
                llm,
                messages
                + [
                    {"role": "assistant", "content": str(content or "")},
                    {"role": "user", "content": repair_msg},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                stage="replanner_repair",
            )
        except Exception as exc:  # noqa: BLE001
            return {
                "ok": False,
                "payload": None,
                "error": str(exc),
                "error_type": "llm_transport_failure",
                "repair_count": 0,
                "usage": usage,
                "stopping_reason": "llm_transport_failure_after_retry",
                "response_only": response_only,
                "observability": {
                    "transport": transport_meta,
                    "calls": call_obs,
                    "repair_transport_error": str(exc),
                },
            }
        usage = _merge_usage(usage, repair.get("usage") or {})
        repair_count = 1
        repair_content = repair.get("content")
        parsed, _, err = parse_replanner_output(
            str(content or ""),
            repair_content=str(repair_content or ""),
            response_only=response_only,
        )
        call_obs.append(
            _obs_from_completion(
                repair,
                stage="replanner_repair",
                max_tokens=max_tokens,
                parse_result="ok" if parsed is not None else "failed",
                schema_error=None if parsed is not None else err,
                repair_count=1,
            )
        )
        if repair_transport.get("transport_retry_attempted"):
            call_obs[-1]["transport_retry"] = dict(repair_transport)

    assistant_content = (
        str(repair_content or "") if repair_count else str(content or "")
    )
    if parsed is None:
        return {
            "ok": False,
            "payload": None,
            "error": err or "replanner_invalid_after_repair",
            "repair_count": repair_count,
            "usage": usage,
            "stopping_reason": "invalid_replanner_output_after_repair",
            "response_only": response_only,
            "assistant_content": assistant_content,
            "messages": messages,
            "observability": {"transport": transport_meta, "calls": call_obs},
        }
    if response_only and parsed.get("action") != "response":
        return {
            "ok": False,
            "payload": None,
            "error": "forced_final_round_plan_not_allowed",
            "repair_count": repair_count,
            "usage": usage,
            "stopping_reason": "forced_final_round_plan_after_repair",
            "response_only": response_only,
            "assistant_content": assistant_content,
            "messages": messages,
            "observability": {"transport": transport_meta, "calls": call_obs},
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
        "observability": {"transport": transport_meta, "calls": call_obs},
    }


def run_replanner_semantic_repair(
    *,
    llm: LLMClient,
    messages: List[Dict[str, str]],
    previous_content: str,
    semantic_error: str,
    max_new_steps: int = 2,
    execution_round: int = 0,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    response_only: bool = False,
) -> Dict[str, Any]:
    """One complete semantic repair after schema-valid Plan fails plan semantics.

    Shares the single repair budget with schema repair (caller must ensure
    prior repair_count == 0). Counts toward LLM≤16 via the budgeted client.
    """
    repair_msg = format_semantic_plan_repair_message(
        semantic_error,
        max_new_steps=max_new_steps,
        execution_round=execution_round,
        response_only=response_only,
    )
    call_obs: List[Dict[str, Any]] = []
    try:
        repair, repair_transport = complete_with_transient_retry(
            llm,
            list(messages)
            + [
                {"role": "assistant", "content": str(previous_content or "")},
                {"role": "user", "content": repair_msg},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            stage="replanner_semantic_repair",
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "payload": None,
            "error": str(exc),
            "error_type": "llm_transport_failure",
            "repair_count": 1,
            "usage": {},
            "stopping_reason": "llm_transport_failure_after_retry",
            "response_only": response_only,
            "observability": {"calls": call_obs, "repair_transport_error": str(exc)},
        }

    usage = dict(repair.get("usage") or {})
    repair_content = str(repair.get("content") or "")
    parsed, _, err = parse_replanner_output(
        repair_content, response_only=response_only
    )
    call_obs.append(
        _obs_from_completion(
            repair,
            stage="replanner_semantic_repair",
            max_tokens=max_tokens,
            parse_result="ok" if parsed is not None else "failed",
            schema_error=None if parsed is not None else err,
            repair_count=1,
        )
    )
    if repair_transport.get("transport_retry_attempted"):
        call_obs[-1]["transport_retry"] = dict(repair_transport)

    if parsed is None:
        return {
            "ok": False,
            "payload": None,
            "error": err or "semantic_repair_invalid_after_repair",
            "repair_count": 1,
            "usage": usage,
            "stopping_reason": "invalid_plan_update_after_semantic_repair",
            "response_only": response_only,
            "assistant_content": repair_content,
            "observability": {"calls": call_obs},
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
            "assistant_content": repair_content,
            "observability": {"calls": call_obs},
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
        "assistant_content": repair_content,
        "observability": {"calls": call_obs},
    }
