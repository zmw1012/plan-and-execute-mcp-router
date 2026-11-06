"""MCP-Zero agent for HK-155: Active Tool Request → hierarchical retrieval →
Real MCP invoke → iterative extension → final answer.

Uses the same answer contract and UnifiedRealMcpGateway as Operation-Visible.
Does not read gold / reference answers. Does not fall back to mock domain servers.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional, Set

from hk_equity_mcp.agent.llm_client import LLMClient
from hk_equity_mcp.experiment.normalize import (
    agent_completion_to_fallback,
    normalize_final_payload,
)
from hk_equity_mcp.mcp_zero.active_request import parse_active_tool_request
from hk_equity_mcp.mcp_zero.extension import (
    TERMINATION_MAX_LLM_TURNS,
    TERMINATION_TASK_COMPLETE,
    TERMINATION_TIMEOUT,
    ExtensionController,
)
from hk_equity_mcp.mcp_zero.prompt import HK_MCP_ZERO_SYSTEM_PROMPT
from hk_equity_mcp.mcp_zero.router import HierarchicalRouter
from hk_equity_mcp.mcp_zero.schemas import McpZeroTrace, ToolHit
from hk_equity_mcp.real_mcp.unified_gateway import UnifiedRealMcpGateway


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


class McpZeroAgent:
    """Active hierarchical tool-discovery agent over Real MCP."""

    def __init__(
        self,
        llm: LLMClient,
        router: HierarchicalRouter,
        gateway: UnifiedRealMcpGateway,
        *,
        max_llm_turns: int = 8,
        max_invocations: int = 6,
        max_extension_rounds: int = 3,
        timeout_sec: float = 120.0,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        system_prompt: Optional[str] = None,
    ):
        self.llm = llm
        self.router = router
        self.gateway = gateway
        self.max_llm_turns = int(max_llm_turns)
        self.max_invocations = int(max_invocations)
        self.max_extension_rounds = int(max_extension_rounds)
        self.timeout_sec = float(timeout_sec)
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.system_prompt = system_prompt or HK_MCP_ZERO_SYSTEM_PROMPT

    def run(self, question: str) -> Dict[str, Any]:
        started = time.perf_counter()
        deadline = started + self.timeout_sec
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": question},
        ]
        trace: List[Dict[str, Any]] = []
        mcp_trace = McpZeroTrace()
        extension = ExtensionController(max_extension_rounds=self.max_extension_rounds)
        available: Dict[str, ToolHit] = {}
        selected_operations: List[str] = []
        invoke_calls = 0
        llm_calls = 0
        total_in = 0
        total_out = 0
        generation_latency_ms = 0.0
        retrieval_latency_ms = 0.0
        embedding_calls = 0
        embedding_tokens = 0
        final_payload: Any = None
        status = "failed"
        error: Optional[str] = None
        parse_repair_count = 0

        for turn in range(1, self.max_llm_turns + 1):
            if time.perf_counter() > deadline:
                status = "failed"
                final_payload = {
                    "status": "unavailable",
                    "final_answer": "Timed out before completing the request.",
                    "reasoning_brief": "timeout",
                    "evidence": [],
                }
                error = "timeout"
                extension.finalize_trace(mcp_trace, reason=TERMINATION_TIMEOUT)
                break

            t0 = time.perf_counter()
            llm_resp = self.llm.complete(
                messages, temperature=self.temperature, max_tokens=self.max_tokens
            )
            generation_latency_ms += (time.perf_counter() - t0) * 1000.0
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

            # Prefer Active Tool Request when present (even if JSON also appears).
            request = parse_active_tool_request(
                content,
                round_index=len(mcp_trace.rounds),
                source="extension" if extension.initial_done else "llm",
            )
            if request is not None:
                decision = extension.decide(request)
                event["active_tool_request"] = request.to_dict()
                event["extension_decision"] = {
                    "allow": decision.allow,
                    "reason": decision.reason,
                    "is_extension": decision.is_extension,
                }
                trace.append(event)
                if not decision.allow:
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Active Tool Request rejected by extension policy "
                                f"({decision.reason}). Continue with currently "
                                "available operations, invoke_api, or final_answer JSON."
                            ),
                        }
                    )
                    continue

                round_ = self.router.retrieve(
                    request,
                    exclude_operation_ids=extension.already_seen_ops(),
                    is_extension=decision.is_extension,
                )
                round_ = extension.commit_round(round_)
                mcp_trace.rounds.append(round_)
                retrieval_latency_ms += float(round_.latency_ms)
                embedding_calls += int(round_.embedding_calls)
                embedding_tokens += int(round_.embedding_tokens)

                for hit in round_.tools:
                    available[hit.operation_id] = hit
                    if hit.operation_id not in selected_operations:
                        selected_operations.append(hit.operation_id)

                retrieve_event = {
                    "type": "tool_result",
                    "name": "mcp_zero_retrieve",
                    "arguments": {
                        "server": request.server,
                        "tool": request.tool,
                        "round_index": round_.round_index,
                        "is_extension": round_.is_extension,
                    },
                    "result": {
                        "query": request.tool,
                        "servers": [s.to_dict() for s in round_.servers],
                        "candidates": [
                            {
                                "operation_id": t.operation_id,
                                "server_id": t.server_id,
                                "tool_name": t.tool_name,
                                "score": t.final_score,
                                "server_score": t.server_score,
                                "tool_score": t.tool_score,
                                "rank": t.rank,
                            }
                            for t in round_.tools
                        ],
                        "new_operation_ids": list(round_.new_operation_ids),
                        "stop_reason": round_.stop_reason,
                    },
                    "parallel": False,
                }
                trace.append(retrieve_event)
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Hierarchical retrieval result "
                            f"(round={round_.round_index}, extension={round_.is_extension}, "
                            f"stop_reason={round_.stop_reason}):\n"
                            + self.router.format_tools_for_prompt(round_.tools)
                            + "\n\nAvailable operation_ids now: "
                            + json.dumps(sorted(available.keys()), ensure_ascii=False)
                            + "\nRespond with invoke_api JSON, a refined "
                            "<tool_assistant> request, or final_answer JSON."
                        ),
                    }
                )
                continue

            # Otherwise treat as JSON action / answer.
            try:
                parsed = _parse_agent_json(content)
            except Exception as exc:  # noqa: BLE001
                event["parse_error"] = type(exc).__name__
                trace.append(event)
                parse_repair_count += 1
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Your previous response was neither a valid "
                            "<tool_assistant> block nor valid JSON. Reply again "
                            "with the required format only."
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
                extension.mark_model_proceeded()
                extension.finalize_trace(mcp_trace, reason=TERMINATION_TASK_COMPLETE)
                break

            if not tool_calls:
                final_payload = parsed.get("final_answer") or {
                    "status": "unavailable",
                    "final_answer": "No tool calls and no answer.",
                    "reasoning_brief": "",
                    "evidence": [],
                }
                status = "partial"
                extension.mark_model_proceeded()
                extension.finalize_trace(mcp_trace)
                break

            observations = self._execute_invokes(
                tool_calls, available=available, invoke_calls=invoke_calls
            )
            invoke_calls = observations["invoke_calls"]
            for obs in observations["results"]:
                trace.append(obs)
            extension.mark_model_proceeded()
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
                    "evidence": [],
                }
                status = "failed"
                error = "max_invocations"
                extension.finalize_trace(mcp_trace)
                break
        else:
            if final_payload is None:
                final_payload = {
                    "status": "unavailable",
                    "final_answer": "Reached max LLM turns without a final answer.",
                    "reasoning_brief": "",
                    "evidence": [],
                }
                status = "failed"
                error = "max_llm_turns"
                extension.finalize_trace(mcp_trace, reason=TERMINATION_MAX_LLM_TURNS)

        if mcp_trace.termination_reason is None:
            extension.finalize_trace(mcp_trace)

        predicted_status, final_text = normalize_final_payload(
            final_payload,
            fallback_status=agent_completion_to_fallback(status),
        )
        latency_ms = (time.perf_counter() - started) * 1000.0
        # Query-time embedding counters from the shared CountingEmbedder.
        emb_snap = self.router.embedder.snapshot()
        return {
            "question": question,
            "architecture": "mcp_zero",
            "condition": "mcp_zero",
            "fidelity_note": (
                "simplified reimplementation adapted from Fei et al. (2025) MCP-Zero"
            ),
            "final_answer": final_text,
            "final_answer_obj": final_payload if isinstance(final_payload, dict) else None,
            "predicted_status": predicted_status,
            "status": status,
            "llm_calls": llm_calls,
            "tokens": {
                "input_tokens": total_in,
                "output_tokens": total_out,
                "total_tokens": total_in + total_out,
            },
            "resolver_calls": {
                "find_operations": 0,
                "describe_operation": 0,
                "mcp_zero_retrieve": len(mcp_trace.rounds),
            },
            "underlying_operation_calls": invoke_calls,
            "latency_ms": round(latency_ms, 3),
            "generation_latency_ms": round(generation_latency_ms, 3),
            "retrieval_latency_ms": round(retrieval_latency_ms, 3),
            "embedding_calls": emb_snap["calls"],
            "embedding_tokens": emb_snap["tokens"],
            "parse_repair_count": parse_repair_count,
            "provider_metadata": {
                "llm_provider": getattr(self.llm, "provider", None),
                "llm_model": getattr(self.llm, "model", None),
                "embedding_model": self.router.embedder.model_name,
            },
            "trace": trace,
            "error": error,
            # MCP-Zero-specific visible fields
            "active_tool_request": mcp_trace.active_requests(),
            "retrieved_servers": mcp_trace.retrieved_servers(),
            "retrieved_tools": mcp_trace.retrieved_tools(),
            "expanded_servers": mcp_trace.expanded_servers(),
            "expanded_tools": mcp_trace.expanded_tools(),
            "selected_operations": list(selected_operations),
            "invoked_operations": [
                e.get("arguments", {}).get("operation_id")
                for e in trace
                if e.get("type") == "tool_result" and e.get("name") == "invoke_api"
            ],
            "extension_rounds": [r.to_dict() for r in mcp_trace.extension_rounds],
            "number_of_extension_rounds": mcp_trace.number_of_extension_rounds,
            "termination_reason": mcp_trace.termination_reason,
            "mcp_zero_trace": mcp_trace.to_dict(),
            "candidate_operation_ids": mcp_trace.candidate_operation_ids(),
        }

    def _execute_invokes(
        self,
        tool_calls: List[Dict[str, Any]],
        *,
        available: Dict[str, ToolHit],
        invoke_calls: int,
    ) -> Dict[str, Any]:
        results: List[Dict[str, Any]] = []
        allowed: Set[str] = set(available)
        for call in tool_calls:
            name = call.get("name")
            args = dict(call.get("arguments") or {})
            if name != "invoke_api":
                results.append(
                    {
                        "type": "tool_result",
                        "name": name,
                        "arguments": args,
                        "parallel": False,
                        "result": {
                            "success": False,
                            "error": (
                                f"MCP-Zero refuses tool '{name}'. "
                                "Only invoke_api on retrieved operations is allowed "
                                "(or emit <tool_assistant> for extension)."
                            ),
                            "error_type": "refused_tool",
                        },
                    }
                )
                continue
            oid = args.get("operation_id")
            call_args = args.get("arguments") or {}
            if not oid or oid not in allowed:
                results.append(
                    {
                        "type": "tool_result",
                        "name": "invoke_api",
                        "arguments": args,
                        "parallel": False,
                        "result": {
                            "success": False,
                            "error": (
                                f"operation_id {oid!r} is not in the retrieved "
                                "candidate set for this trajectory."
                            ),
                            "error_type": "not_retrieved",
                        },
                    }
                )
                continue
            # Gateway enforces blocked / unknown fail-closed.
            invoke_calls += 1
            mcp_result = self.gateway.invoke(str(oid), dict(call_args))
            payload = {
                "success": bool(mcp_result.success),
                "operation_id": oid,
                "server_id": getattr(mcp_result, "server_id", None),
                "normalized": getattr(mcp_result, "normalized", None),
                "evidence": getattr(mcp_result, "evidence", None),
                "error": getattr(mcp_result, "error", None),
                "error_type": getattr(mcp_result, "error_type", None),
                "protocol_success": getattr(mcp_result, "protocol_success", None),
                "execution_success": getattr(mcp_result, "execution_success", None),
                "real_mcp": getattr(mcp_result, "real_mcp", None),
                "transport": getattr(mcp_result, "transport", None),
            }
            results.append(
                {
                    "type": "tool_result",
                    "name": "invoke_api",
                    "arguments": {"operation_id": oid, "arguments": call_args},
                    "parallel": False,
                    "result": payload,
                }
            )
        return {"results": results, "invoke_calls": invoke_calls}
