"""Routing loop. The model sees the user query and tool observations only."""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

from hk_equity_mcp.etom import FIDELITY_NAME
from hk_equity_mcp.etom.adapter import EtomRoutingResolver, FIND_ALIASES, META_TOOLS
from hk_equity_mcp.etom.prompts import ETOM_ROUTING_SYSTEM_PROMPT, assert_prompt_has_no_gold


def _parse_agent_json(content: str) -> Dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("Agent JSON must be an object.")
    return payload


def assert_no_gold_leak(messages: List[Dict[str, str]], ground_truth_tools: Any) -> None:
    blob = json.dumps(ground_truth_tools, ensure_ascii=False, sort_keys=True)
    for message in messages:
        text = message.get("content") or ""
        lowered = text.lower()
        if "ground_truth" in lowered or "gold_tool" in lowered:
            raise RuntimeError("Gold graph leaked into the model context.")
        if ground_truth_tools and len(blob) > 80 and blob in text:
            raise RuntimeError("Serialized gold graph leaked into the model context.")


class EtomRoutingAgent:
    def __init__(
        self,
        llm,
        resolver: EtomRoutingResolver,
        *,
        temperature: float,
        max_tokens: int,
        max_llm_turns: int = 8,
        max_search_reformulations: int = 3,
        max_invocations: int = 8,
        timeout_sec: float = 180.0,
        max_parse_repairs: int = 2,
    ):
        assert_prompt_has_no_gold()
        self.llm = llm
        self.resolver = resolver
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_llm_turns = max_llm_turns
        self.max_search_reformulations = max_search_reformulations
        self.max_invocations = max_invocations
        self.timeout_sec = timeout_sec
        self.max_parse_repairs = max_parse_repairs

    def run(self, query: str, ground_truth_tools: Any = None) -> Dict[str, Any]:
        """Run one query. ``ground_truth_tools`` is used only as a leak sentinel."""
        started = time.perf_counter()
        self.resolver.reset_route()
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": ETOM_ROUTING_SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ]
        trace: List[Dict[str, Any]] = []
        parse_repairs = 0
        llm_calls = 0
        input_tokens = 0
        output_tokens = 0
        search_calls = 0
        describe_calls = 0
        invoke_calls = 0
        error = None
        status = "failed"
        final_answer = None
        timed_out = False

        for turn in range(1, self.max_llm_turns + 1):
            if time.perf_counter() - started > self.timeout_sec:
                status = "failed"
                error = "timeout"
                timed_out = True
                break
            assert_no_gold_leak(messages, ground_truth_tools)
            llm_resp = self.llm.complete(
                messages, temperature=self.temperature, max_tokens=self.max_tokens
            )
            llm_calls += 1
            usage = llm_resp.get("usage") or {}
            input_tokens += int(usage.get("input_tokens") or 0)
            output_tokens += int(usage.get("output_tokens") or 0)
            content = llm_resp["content"]
            messages.append({"role": "assistant", "content": content})
            try:
                parsed = _parse_agent_json(content)
            except Exception as exc:  # noqa: BLE001
                parse_repairs += 1
                trace.append(
                    {
                        "type": "parse_error",
                        "turn": turn,
                        "error": type(exc).__name__,
                    }
                )
                if parse_repairs > self.max_parse_repairs:
                    status = "failed"
                    error = "parse_error"
                    break
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Your previous response was not valid JSON. "
                            "Reply again with the required JSON object only."
                        ),
                    }
                )
                continue
            tool_calls = parsed.get("tool_calls") or []
            trace.append(
                {
                    "type": "llm_turn",
                    "turn": turn,
                    "tool_calls": tool_calls,
                    "final_answer": parsed.get("final_answer"),
                    "usage": usage,
                    "latency_ms": llm_resp.get("latency_ms"),
                    "model": llm_resp.get("model"),
                }
            )
            if parsed.get("final_answer") is not None and not tool_calls:
                final_answer = parsed.get("final_answer")
                status = "completed"
                break
            if not tool_calls:
                final_answer = parsed.get("final_answer")
                status = "completed"
                break
            observations = []
            for call in tool_calls:
                if not isinstance(call, dict):
                    observations.append(
                        {
                            "name": None,
                            "arguments": {},
                            "result": {
                                "success": False,
                                "executed": False,
                                "business_result": None,
                                "error": "tool call must be an object",
                            },
                        }
                    )
                    continue
                name = call.get("name")
                arguments = call.get("arguments") if isinstance(call.get("arguments"), dict) else {}
                depends_on = arguments.get("depends_on", call.get("depends_on"))
                try:
                    payload = self._dispatch(
                        name,
                        arguments,
                        depends_on,
                        search_calls=search_calls,
                        invoke_calls=invoke_calls,
                    )
                except (TypeError, ValueError):
                    payload = {
                        "success": False,
                        "executed": False,
                        "business_result": None,
                        "error": "invalid_arguments",
                    }
                if name in FIND_ALIASES and payload.get("success"):
                    search_calls += 1
                elif name == "describe_operation" and payload.get("success"):
                    describe_calls += 1
                elif name == "invoke_api" and payload.get("success"):
                    invoke_calls += 1
                observations.append(
                    {
                        "name": name,
                        "arguments": arguments,
                        "result": payload,
                    }
                )
            trace.append({"type": "tool_results", "results": observations})
            messages.append(
                {
                    "role": "user",
                    "content": "Tool results (visible):\n"
                    + json.dumps(observations, ensure_ascii=False),
                }
            )
        else:
            if status != "completed":
                status = "failed"
                error = error or "max_llm_turns"

        latency_ms = (time.perf_counter() - started) * 1000
        nodes = list(self.resolver.predicted_nodes)
        return {
            "fidelity": FIDELITY_NAME,
            "status": status,
            "error": error,
            "timeout": timed_out,
            "final_answer": final_answer,
            "predicted_nodes": nodes,
            "predicted_tools": [
                {
                    "server_name": node["server_name"],
                    "tool_name": node["tool_name"],
                    "operation_id": node["operation_id"],
                    "order": node["order"],
                }
                for node in nodes
            ],
            "dependency_edges": [
                edge
                for node in nodes
                for edge in node.get("dependency_edges") or []
            ],
            "llm_calls": llm_calls,
            "tokens": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            },
            "resolver_calls": {
                "find_operations": search_calls,
                "describe_operation": describe_calls,
                "invoke_api": invoke_calls,
            },
            "parse_repairs": parse_repairs,
            "real_tool_executions": self.resolver.real_tool_executions,
            "latency_ms": round(latency_ms, 3),
            "exposed_tools": list(META_TOOLS),
            "trace": trace,
        }

    def _dispatch(
        self,
        name,
        arguments,
        depends_on,
        *,
        search_calls: int,
        invoke_calls: int,
    ):
        if name in FIND_ALIASES:
            if search_calls > self.max_search_reformulations:
                return {
                    "success": False,
                    "executed": False,
                    "business_result": None,
                    "error": "max_search_reformulations",
                    "candidates": [],
                }
            return self.resolver.find_operations(
                query=str(arguments.get("query") or ""),
                top_k=arguments.get("top_k", self.resolver.top_k_default),
            )
        if name == "describe_operation":
            return self.resolver.describe_operation(str(arguments.get("operation_id") or ""))
        if name == "invoke_api":
            if invoke_calls >= self.max_invocations:
                return {
                    "success": False,
                    "executed": False,
                    "business_result": None,
                    "error": "max_invocations",
                }
            nested = arguments.get("arguments")
            return self.resolver.invoke_api(
                operation_id=str(arguments.get("operation_id") or ""),
                arguments=nested if isinstance(nested, dict) else {},
                depends_on=depends_on if isinstance(depends_on, list) else None,
            )
        return self.resolver.call_meta(str(name))
