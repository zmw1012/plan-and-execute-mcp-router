"""Operation-Visible Agent — simplified reimplementation adapted from Ghoshal et al. (2026).

Central ReAct loop with visible Operation-Resolver meta-tools only:
find_operations, describe_operation, invoke_api.

Underlying financial tools are NOT directly exposed to the central model.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import yaml

from hk_equity_mcp.agent.llm_client import LLMClient
from hk_equity_mcp.operation_resolver.retrieval import EmbeddingBackend
from hk_equity_mcp.operation_resolver.server import META_TOOLS, OperationResolverServer

SYSTEM_PROMPT = """You are an Operation-Visible enterprise agent for HK equity Q&A.
You may call ONLY these meta-tools:
- find_operations(query, top_k?): search the operation registry
- describe_operation(operation_id): inspect schema/constraints
- invoke_api(operation_id, arguments): validate and execute

Do NOT invent underlying tool names. Do NOT claim hidden planning.
Use find_operations first for a subgoal, then describe_operation before invoke_api
unless the contract is already known from this trajectory.
You may issue multiple independent read-only invoke_api calls in one turn when
arguments are already known and calls have no dependency.
If validation/execution errors occur, recover using the visible structured error.

Respond with a single JSON object of the form:
{
  "thought": "brief visible plan (not private chain-of-thought)",
  "tool_calls": [
     {"name": "find_operations", "arguments": {"query": "...", "top_k": 20}},
     {"name": "describe_operation", "arguments": {"operation_id": "..."}},
     {"name": "invoke_api", "arguments": {"operation_id": "...", "arguments": {...}}}
  ],
  "final_answer": null
}
When ready to answer, set tool_calls to [] and put the answer in final_answer.
If blocked, set final_answer to a clarification/unavailable message.
"""


def load_operation_visible_config(path: Optional[str] = None) -> Dict[str, Any]:
    if path is None:
        path = str(
            Path(__file__).resolve().parents[3] / "configs" / "operation_visible.yaml"
        )
    cfg_path = Path(path)
    if not cfg_path.exists():
        return {}
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _parse_agent_json(content: str) -> Dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        # Strip optional markdown fences without inventing content.
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return json.loads(text)


class OperationVisibleAgent:
    """Centralized ReAct Parallel Tool Calling + Operation-Resolver MCP."""

    def __init__(
        self,
        llm: LLMClient,
        embedding_backend: EmbeddingBackend,
        config: Optional[Dict[str, Any]] = None,
        resolver: Optional[OperationResolverServer] = None,
    ):
        self.llm = llm
        self.config = config or load_operation_visible_config()
        limits = self.config.get("limits", {})
        self.max_llm_turns = int(limits.get("max_llm_turns", 8))
        self.max_search_reformulations = int(limits.get("max_search_reformulations", 3))
        self.max_invocations = int(limits.get("max_invocations", 6))
        self.timeout_sec = float(limits.get("timeout_sec", 120))
        self.temperature = float(self.config.get("llm", {}).get("temperature", 0.0))
        self.max_tokens = int(self.config.get("llm", {}).get("max_tokens", 1024))
        self.resolver = resolver or OperationResolverServer(
            embedding_backend=embedding_backend
        )

    def exposed_tools(self) -> List[str]:
        """Tools visible to the central model (meta-tools only)."""
        return list(META_TOOLS)

    def run(self, question: str, history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        started = time.perf_counter()
        history = history or []
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *history,
            {"role": "user", "content": question},
        ]
        trace: List[Dict[str, Any]] = []
        search_calls = 0
        describe_calls = 0
        invoke_calls = 0
        llm_calls = 0
        total_in = 0
        total_out = 0
        final_answer = None
        status = "failed"
        deadline = started + self.timeout_sec

        for turn in range(1, self.max_llm_turns + 1):
            if time.perf_counter() > deadline:
                status = "failed"
                final_answer = "Timed out before completing the request."
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
            event = {
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
                event["parse_error"] = f"{type(exc).__name__}"
                trace.append(event)
                # Ask model to repair JSON once via visible message.
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

            if parsed.get("final_answer") and not tool_calls:
                final_answer = parsed.get("final_answer")
                status = "full"  # placeholder label; human/judge labeling later
                break

            if not tool_calls:
                final_answer = parsed.get("final_answer") or "No tool calls and no answer."
                status = "partial" if final_answer else "failed"
                break

            # Execute tool calls for this turn.
            observations = self._execute_tool_calls(
                tool_calls,
                counters={
                    "search_calls": search_calls,
                    "describe_calls": describe_calls,
                    "invoke_calls": invoke_calls,
                },
            )
            search_calls = observations["counters"]["search_calls"]
            describe_calls = observations["counters"]["describe_calls"]
            invoke_calls = observations["counters"]["invoke_calls"]
            for obs in observations["results"]:
                trace.append(obs)
            messages.append(
                {
                    "role": "user",
                    "content": "Tool results (visible):\n"
                    + json.dumps(observations["results"], ensure_ascii=False),
                }
            )

            if search_calls > self.max_search_reformulations + 1:
                final_answer = "Exceeded max search reformulations."
                status = "failed"
                break
            if invoke_calls > self.max_invocations:
                final_answer = "Exceeded max invocations."
                status = "failed"
                break
        else:
            if final_answer is None:
                final_answer = "Reached max LLM turns without a final answer."
                status = "failed"

        latency_ms = (time.perf_counter() - started) * 1000
        return {
            "question": question,
            "architecture": "operation_visible",
            "fidelity_note": (
                "simplified reimplementation adapted from Ghoshal et al. (2026)"
            ),
            "exposed_tools": self.exposed_tools(),
            "final_answer": final_answer,
            "status": status,
            "llm_calls": llm_calls,
            "tokens": {
                "input_tokens": total_in,
                "output_tokens": total_out,
                "total_tokens": total_in + total_out,
            },
            "resolver_calls": {
                "find_operations": search_calls,
                "describe_operation": describe_calls,
            },
            "underlying_operation_calls": invoke_calls,
            "latency_ms": round(latency_ms, 3),
            "provider_metadata": {
                "llm_provider": getattr(self.llm, "provider", None),
                "llm_model": getattr(self.llm, "model", None),
                "embedding_model": getattr(
                    self.resolver.retriever.backend, "model_name", None
                ),
                # Never include API keys.
            },
            "trace": trace,
        }

    def _execute_tool_calls(
        self, tool_calls: Sequence[Dict[str, Any]], counters: Dict[str, int]
    ) -> Dict[str, Any]:
        # Partition: independent read-only invoke_api may run in parallel;
        # find/describe and dependent invokes stay ordered.
        prepared: List[Dict[str, Any]] = []
        for idx, call in enumerate(tool_calls):
            name = call.get("name")
            args = call.get("arguments") or {}
            prepared.append({"index": idx, "name": name, "arguments": args})

        independent_invokes: List[Dict[str, Any]] = []
        sequential: List[Dict[str, Any]] = []
        for item in prepared:
            if item["name"] == "invoke_api" and self._is_independent_readonly(item):
                independent_invokes.append(item)
            else:
                sequential.append(item)

        results: List[Optional[Dict[str, Any]]] = [None] * len(prepared)

        # Run independent invokes in parallel when >1.
        if len(independent_invokes) > 1:
            with ThreadPoolExecutor(max_workers=min(4, len(independent_invokes))) as pool:
                futs = {
                    pool.submit(self._dispatch, item["name"], item["arguments"]): item
                    for item in independent_invokes
                }
                for fut in as_completed(futs):
                    item = futs[fut]
                    payload = fut.result()
                    counters["invoke_calls"] += 1
                    results[item["index"]] = {
                        "type": "tool_result",
                        "name": item["name"],
                        "arguments": item["arguments"],
                        "parallel": True,
                        "result": payload,
                    }
        elif len(independent_invokes) == 1:
            item = independent_invokes[0]
            payload = self._dispatch(item["name"], item["arguments"])
            counters["invoke_calls"] += 1
            results[item["index"]] = {
                "type": "tool_result",
                "name": item["name"],
                "arguments": item["arguments"],
                "parallel": False,
                "result": payload,
            }

        for item in sequential:
            name = item["name"]
            payload = self._dispatch(name, item["arguments"])
            if name == "find_operations":
                counters["search_calls"] += 1
            elif name == "describe_operation":
                counters["describe_calls"] += 1
            elif name == "invoke_api":
                counters["invoke_calls"] += 1
            results[item["index"]] = {
                "type": "tool_result",
                "name": name,
                "arguments": item["arguments"],
                "parallel": False,
                "result": payload,
            }

        return {
            "results": [r for r in results if r is not None],
            "counters": counters,
        }

    def _is_independent_readonly(self, item: Dict[str, Any]) -> bool:
        args = item.get("arguments") or {}
        op_id = args.get("operation_id")
        if not op_id:
            return False
        op = self.resolver.registry.get(op_id)
        if op is None:
            return False
        if not op.read_only:
            return False
        if op.depends_on:
            return False
        # Arguments must already be present/known.
        call_args = args.get("arguments")
        return isinstance(call_args, dict)

    def _dispatch(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        if name not in META_TOOLS:
            return {
                "success": False,
                "error": (
                    f"Refused direct underlying tool '{name}'. "
                    f"Operation-Visible exposes only {list(META_TOOLS)}."
                ),
            }
        try:
            return self.resolver.call_meta(name, **arguments)
        except TypeError as exc:
            return {"success": False, "error": f"Invalid meta-tool arguments: {exc}"}
        except Exception as exc:  # noqa: BLE001
            return {
                "success": False,
                "error": f"Meta-tool execution failed: {type(exc).__name__}",
            }
