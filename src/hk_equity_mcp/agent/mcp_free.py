"""MCP-Free closed-book condition: question only, no tools.

No MCP / Embedding / RAG / web / catalog. Limited JSON parse repair only;
never falls back to tools on failure.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

from hk_equity_mcp.agent.llm_client import LLMClient
from hk_equity_mcp.experiment.normalize import empty_resolver_calls, normalize_final_payload
from hk_equity_mcp.experiment.prompts import MCP_FREE_SYSTEM_PROMPT

# One generation + at most one parse-repair generation.
DEFAULT_MAX_PARSE_REPAIRS = 1

PARSE_REPAIR_USER_MESSAGE = (
    "Your previous response was not valid JSON. "
    "Reply again with the required JSON answer object only."
)


def _parse_json(content: str) -> Dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return json.loads(text)


class MCPFreeAgent:
    """Closed-book agent: no tools, no gold/reference leakage."""

    def __init__(
        self,
        llm: LLMClient,
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        system_prompt: Optional[str] = None,
        max_parse_repairs: int = DEFAULT_MAX_PARSE_REPAIRS,
    ):
        self.llm = llm
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = (
            system_prompt if system_prompt is not None else MCP_FREE_SYSTEM_PROMPT
        )
        self.max_parse_repairs = max(0, int(max_parse_repairs))

    def exposed_tools(self) -> List[str]:
        return []

    def run(self, question: str, history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        started = time.perf_counter()
        history = history or []
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": self.system_prompt},
            *history,
            {"role": "user", "content": question},
        ]
        trace: List[Dict[str, Any]] = []
        error = None
        predicted_status = "unavailable"
        final_text = ""
        final_obj: Optional[Dict[str, Any]] = None
        total_in = 0
        total_out = 0
        llm_calls = 0
        parse_repair_count = 0

        try:
            attempts = 1 + self.max_parse_repairs
            for attempt in range(attempts):
                llm_resp = self.llm.complete(
                    messages, temperature=self.temperature, max_tokens=self.max_tokens
                )
                llm_calls += 1
                usage = llm_resp.get("usage") or {}
                total_in += int(usage.get("input_tokens") or 0)
                total_out += int(usage.get("output_tokens") or 0)
                content = llm_resp["content"]
                messages.append({"role": "assistant", "content": content})
                event: Dict[str, Any] = {
                    "type": "llm_turn",
                    "turn": attempt + 1,
                    "content": content,
                    "usage": usage,
                    "latency_ms": llm_resp.get("latency_ms"),
                    "provider": llm_resp.get("provider"),
                    "model": llm_resp.get("model"),
                    "parse_repair": attempt > 0,
                }
                try:
                    parsed = _parse_json(content)
                    predicted_status, final_text = normalize_final_payload(parsed)
                    final_obj = parsed if isinstance(parsed, dict) else None
                    event["parsed"] = parsed
                    trace.append(event)
                    error = None
                    break
                except Exception as exc:  # noqa: BLE001
                    event["parse_error"] = type(exc).__name__
                    trace.append(event)
                    predicted_status = "unavailable"
                    final_text = content
                    final_obj = None
                    error = f"parse_error:{type(exc).__name__}"
                    if attempt + 1 >= attempts:
                        break
                    parse_repair_count += 1
                    messages.append(
                        {"role": "user", "content": PARSE_REPAIR_USER_MESSAGE}
                    )
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}"
            predicted_status = "unavailable"
            final_text = ""
            final_obj = None

        latency_ms = (time.perf_counter() - started) * 1000
        status = "full" if final_text and error is None else "failed"
        return {
            "question": question,
            "architecture": "mcp_free",
            "condition": "mcp_free",
            "execution_backend": "none",
            "exposed_tools": [],
            "final_answer": final_text,
            "final_answer_obj": final_obj,
            "predicted_status": predicted_status,
            "status": status,
            "llm_calls": llm_calls,
            "parse_repair_count": parse_repair_count,
            "tokens": {
                "input_tokens": total_in,
                "output_tokens": total_out,
                "total_tokens": total_in + total_out,
            },
            "resolver_calls": empty_resolver_calls(),
            "underlying_operation_calls": 0,
            "embedding_calls": 0,
            "embedding_tokens": 0,
            "real_mcp_execution_count": 0,
            "retry_count": parse_repair_count,
            "latency_ms": round(latency_ms, 3),
            "provider_metadata": {
                "llm_provider": getattr(self.llm, "provider", None),
                "llm_model": getattr(self.llm, "model", None),
                "embedding_model": None,
            },
            "trace": trace,
            "error": error,
            # Explicit no-tool markers (not vacuous tool-success).
            "tool_metrics_applicable": False,
            "catalog_exposed": False,
        }
