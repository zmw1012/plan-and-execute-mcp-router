"""MCP-Free closed-book condition: question only, no tools."""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

from hk_equity_mcp.agent.llm_client import LLMClient
from hk_equity_mcp.experiment.normalize import empty_resolver_calls, normalize_final_payload
from hk_equity_mcp.experiment.prompts import MCP_FREE_SYSTEM_PROMPT


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
    ):
        self.llm = llm
        self.temperature = temperature
        self.max_tokens = max_tokens

    def exposed_tools(self) -> List[str]:
        return []

    def run(self, question: str, history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        started = time.perf_counter()
        history = history or []
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": MCP_FREE_SYSTEM_PROMPT},
            *history,
            {"role": "user", "content": question},
        ]
        trace: List[Dict[str, Any]] = []
        error = None
        predicted_status = "unavailable"
        final_text = ""
        total_in = 0
        total_out = 0
        llm_calls = 0

        try:
            llm_resp = self.llm.complete(
                messages, temperature=self.temperature, max_tokens=self.max_tokens
            )
            llm_calls = 1
            usage = llm_resp.get("usage") or {}
            total_in = int(usage.get("input_tokens") or 0)
            total_out = int(usage.get("output_tokens") or 0)
            content = llm_resp["content"]
            event = {
                "type": "llm_turn",
                "turn": 1,
                "content": content,
                "usage": usage,
                "latency_ms": llm_resp.get("latency_ms"),
                "provider": llm_resp.get("provider"),
                "model": llm_resp.get("model"),
            }
            try:
                parsed = _parse_json(content)
                predicted_status, final_text = normalize_final_payload(parsed)
                event["parsed"] = parsed
            except Exception as exc:  # noqa: BLE001
                event["parse_error"] = type(exc).__name__
                predicted_status = "unavailable"
                final_text = content
                error = f"parse_error:{type(exc).__name__}"
            trace.append(event)
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}"
            predicted_status = "unavailable"
            final_text = ""

        latency_ms = (time.perf_counter() - started) * 1000
        return {
            "question": question,
            "architecture": "mcp_free",
            "exposed_tools": [],
            "final_answer": final_text,
            "predicted_status": predicted_status,
            "status": "full" if final_text and error is None else "failed",
            "llm_calls": llm_calls,
            "tokens": {
                "input_tokens": total_in,
                "output_tokens": total_out,
                "total_tokens": total_in + total_out,
            },
            "resolver_calls": empty_resolver_calls(),
            "underlying_operation_calls": 0,
            "latency_ms": round(latency_ms, 3),
            "provider_metadata": {
                "llm_provider": getattr(self.llm, "provider", None),
                "llm_model": getattr(self.llm, "model", None),
                "embedding_model": None,
            },
            "trace": trace,
            "error": error,
        }
