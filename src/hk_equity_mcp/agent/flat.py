"""Flat condition: central model sees underlying Financial tools directly."""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Sequence

from hk_equity_mcp.agent.llm_client import LLMClient
from hk_equity_mcp.experiment.normalize import (
    agent_completion_to_fallback,
    empty_resolver_calls,
    normalize_final_payload,
)
from hk_equity_mcp.experiment.prompts import build_flat_system_prompt
from hk_equity_mcp.mcp_servers.financial import FinancialServer

FLAT_TOOLS = ("get_financial_metric", "get_filing_summary")


def _parse_agent_json(content: str) -> Dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return json.loads(text)


class FlatAgent:
    """ReAct agent with Flat financial tool exposure (no embedding / resolver)."""

    def __init__(
        self,
        llm: LLMClient,
        *,
        financial_server: Optional[FinancialServer] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        max_llm_turns: int = 8,
        max_invocations: int = 6,
        timeout_sec: float = 120.0,
    ):
        self.llm = llm
        self.financial = financial_server or FinancialServer()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_llm_turns = max_llm_turns
        self.max_invocations = max_invocations
        self.timeout_sec = timeout_sec
        self.system_prompt = build_flat_system_prompt()

    def exposed_tools(self) -> List[str]:
        return list(FLAT_TOOLS)

    def run(self, question: str, history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        started = time.perf_counter()
        history = history or []
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": self.system_prompt},
            *history,
            {"role": "user", "content": question},
        ]
        trace: List[Dict[str, Any]] = []
        invoke_calls = 0
        llm_calls = 0
        total_in = 0
        total_out = 0
        final_payload: Any = None
        status = "failed"
        error = None
        deadline = started + self.timeout_sec

        for turn in range(1, self.max_llm_turns + 1):
            if time.perf_counter() > deadline:
                status = "failed"
                final_payload = {
                    "status": "unavailable",
                    "final_answer": "Timed out before completing the request.",
                    "reasoning_brief": "timeout",
                }
                error = "timeout"
                break

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
                "turn": turn,
                "content": content,
                "usage": usage,
                "latency_ms": llm_resp.get("latency_ms"),
                "provider": llm_resp.get("provider"),
                "model": llm_resp.get("model"),
            }
            try:
                parsed = _parse_agent_json(content)
            except Exception as exc:  # noqa: BLE001
                event["parse_error"] = type(exc).__name__
                trace.append(event)
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Your previous response was not valid JSON. "
                            "Reply again with the required JSON schema only."
                        ),
                    }
                )
                continue

            tool_calls = parsed.get("tool_calls") or []
            event["parsed"] = {
                "thought": parsed.get("thought"),
                "tool_calls": tool_calls,
                "final_answer": parsed.get("final_answer"),
            }
            trace.append(event)

            if parsed.get("final_answer") is not None and not tool_calls:
                final_payload = parsed.get("final_answer")
                status = "full"
                break

            if not tool_calls:
                final_payload = parsed.get("final_answer") or {
                    "status": "unavailable",
                    "final_answer": "No tool calls and no answer.",
                    "reasoning_brief": "",
                }
                status = "partial"
                break

            observations = self._execute_tool_calls(tool_calls)
            invoke_calls += observations["invoke_calls"]
            for obs in observations["results"]:
                trace.append(obs)
            messages.append(
                {
                    "role": "user",
                    "content": "Tool results (visible):\n"
                    + json.dumps(observations["results"], ensure_ascii=False),
                }
            )
            if invoke_calls > self.max_invocations:
                final_payload = {
                    "status": "unavailable",
                    "final_answer": "Exceeded max invocations.",
                    "reasoning_brief": "",
                }
                status = "failed"
                error = "max_invocations"
                break
        else:
            if final_payload is None:
                final_payload = {
                    "status": "unavailable",
                    "final_answer": "Reached max LLM turns without a final answer.",
                    "reasoning_brief": "",
                }
                status = "failed"
                error = "max_llm_turns"

        predicted_status, final_text = normalize_final_payload(
            final_payload,
            fallback_status=agent_completion_to_fallback(status),
        )
        latency_ms = (time.perf_counter() - started) * 1000
        return {
            "question": question,
            "architecture": "flat",
            "exposed_tools": self.exposed_tools(),
            "final_answer": final_text,
            "predicted_status": predicted_status,
            "status": status,
            "llm_calls": llm_calls,
            "tokens": {
                "input_tokens": total_in,
                "output_tokens": total_out,
                "total_tokens": total_in + total_out,
            },
            "resolver_calls": empty_resolver_calls(),
            "underlying_operation_calls": invoke_calls,
            "latency_ms": round(latency_ms, 3),
            "provider_metadata": {
                "llm_provider": getattr(self.llm, "provider", None),
                "llm_model": getattr(self.llm, "model", None),
                "embedding_model": None,
            },
            "trace": trace,
            "error": error,
        }

    def _execute_tool_calls(self, tool_calls: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        prepared: List[Dict[str, Any]] = []
        for idx, call in enumerate(tool_calls):
            prepared.append(
                {
                    "index": idx,
                    "name": call.get("name"),
                    "arguments": call.get("arguments") or {},
                }
            )

        # Contiguous independent read-only Flat tools may parallelize.
        results: List[Optional[Dict[str, Any]]] = [None] * len(prepared)
        invoke_calls = 0
        i = 0
        while i < len(prepared):
            item = prepared[i]
            if item["name"] in FLAT_TOOLS and isinstance(item["arguments"], dict):
                batch = [item]
                j = i + 1
                while j < len(prepared) and prepared[j]["name"] in FLAT_TOOLS:
                    batch.append(prepared[j])
                    j += 1
                if len(batch) > 1:
                    with ThreadPoolExecutor(max_workers=min(4, len(batch))) as pool:
                        futs = {
                            pool.submit(self._dispatch, b["name"], b["arguments"]): b
                            for b in batch
                        }
                        for fut in as_completed(futs):
                            b = futs[fut]
                            payload = fut.result()
                            invoke_calls += 1
                            results[b["index"]] = {
                                "type": "tool_result",
                                "name": b["name"],
                                "arguments": b["arguments"],
                                "parallel": True,
                                "result": payload,
                            }
                else:
                    payload = self._dispatch(item["name"], item["arguments"])
                    invoke_calls += 1
                    results[item["index"]] = {
                        "type": "tool_result",
                        "name": item["name"],
                        "arguments": item["arguments"],
                        "parallel": False,
                        "result": payload,
                    }
                i = j
                continue

            payload = self._dispatch(item["name"], item["arguments"])
            invoke_calls += 1 if item["name"] in FLAT_TOOLS else 0
            results[item["index"]] = {
                "type": "tool_result",
                "name": item["name"],
                "arguments": item["arguments"],
                "parallel": False,
                "result": payload,
            }
            i += 1

        return {
            "results": [r for r in results if r is not None],
            "invoke_calls": invoke_calls,
        }

    def _dispatch(self, name: Optional[str], arguments: Dict[str, Any]) -> Dict[str, Any]:
        if name not in FLAT_TOOLS:
            return {
                "success": False,
                "error": (
                    f"Unknown or disallowed Flat tool '{name}'. "
                    f"Allowed: {list(FLAT_TOOLS)}"
                ),
            }
        try:
            return self.financial.call(name, **arguments)
        except TypeError as exc:
            return {"success": False, "error": f"Invalid tool arguments: {exc}"}
        except Exception as exc:  # noqa: BLE001
            return {
                "success": False,
                "error": f"Tool execution failed: {type(exc).__name__}",
            }
