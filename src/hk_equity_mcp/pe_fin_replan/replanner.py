"""Replanner LLM call: Plan or Response (at most one repair)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from hk_equity_mcp.agent.llm_client import LLMClient
from hk_equity_mcp.pe_fin_lite.temporal_policy import DEFAULT_DATASET_CUTOFF
from hk_equity_mcp.pe_fin_replan.prompts import (
    REPLANNER_SYSTEM_PROMPT,
    render_replanner_user,
)
from hk_equity_mcp.pe_fin_replan.schemas import (
    format_replanner_repair_message,
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


def run_replanner(
    *,
    llm: LLMClient,
    compact_state: Dict[str, Any],
    temperature: float = 0.0,
    max_tokens: int = 2048,
    benchmark_cutoff: str = DEFAULT_DATASET_CUTOFF,
) -> Dict[str, Any]:
    user = render_replanner_user(
        compact_state=compact_state, benchmark_cutoff=benchmark_cutoff
    )
    messages = [
        {"role": "system", "content": REPLANNER_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
    completion = llm.complete(
        messages, temperature=temperature, max_tokens=max_tokens, stage="replanner"
    )
    usage = dict(completion.get("usage") or {})
    content = completion.get("content")
    parsed, _, err = parse_replanner_output(str(content or ""))
    repair_count = 0
    repair_content = None
    if parsed is None:
        repair_msg = format_replanner_repair_message(err or "invalid")
        repair = llm.complete(
            messages
            + [
                {"role": "assistant", "content": str(content or "")},
                {"role": "user", "content": repair_msg},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            stage="replanner_repair",
        )
        usage = _merge_usage(usage, repair.get("usage") or {})
        repair_count = 1
        repair_content = repair.get("content")
        parsed, _, err = parse_replanner_output(
            str(content or ""), repair_content=str(repair_content or "")
        )
    if parsed is None:
        return {
            "ok": False,
            "payload": None,
            "error": err or "replanner_invalid_after_repair",
            "repair_count": repair_count,
            "usage": usage,
            "stopping_reason": "invalid_replanner_output_after_repair",
        }
    return {
        "ok": True,
        "payload": parsed,
        "error": None,
        "repair_count": repair_count,
        "usage": usage,
        "action": parsed.get("action"),
        "decision_reason": parsed.get("decision_reason"),
    }
