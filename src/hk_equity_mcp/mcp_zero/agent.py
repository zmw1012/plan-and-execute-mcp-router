"""MCP-Zero agent for HK-155: Active Tool Request → hierarchical retrieval →
Real MCP invoke → iterative extension → final answer.

Uses the same answer contract and UnifiedRealMcpGateway as Operation-Visible.
Does not read gold / reference answers. Does not fall back to mock domain servers.
"""

from __future__ import annotations

import json
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from hk_equity_mcp.agent.llm_client import LLMClient
from hk_equity_mcp.experiment.normalize import (
    agent_completion_to_fallback,
    normalize_final_payload,
)
from hk_equity_mcp.experiment.runner import redact_secrets
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

ATR_EXAMPLE = (
    "<tool_assistant>\n"
    "server: [brief server/platform description]\n"
    "tool: [brief operation type + target]\n"
    "</tool_assistant>"
)

INVOKE_EXAMPLE = (
    '{"thought":"...","tool_calls":[{"name":"invoke_api","arguments":'
    '{"operation_id":"<retrieved_id>","arguments":{}}}],"final_answer":null}'
)

FINAL_EXAMPLE = (
    '{"thought":"...","tool_calls":[],"final_answer":{"status":"answered|'
    'clarification_needed|unavailable","final_answer":"...","reasoning_brief":"...",'
    '"evidence":[]}}'
)


def build_protocol_repair_message(*, available_operation_ids: List[str]) -> str:
    """State-aware repair: show the minimal legal formats for the current turn."""
    lines = [
        "Invalid empty or malformed action. Choose exactly ONE legal response:",
        "",
        "A) Active Tool Request (raw text tags; do NOT wrap in JSON):",
        ATR_EXAMPLE,
        "",
    ]
    if available_operation_ids:
        lines.extend(
            [
                "B) JSON invoke on a retrieved operation_id:",
                INVOKE_EXAMPLE,
                "Available operation_ids now: "
                + json.dumps(sorted(available_operation_ids), ensure_ascii=False),
                "",
                "C) JSON final answer:",
                FINAL_EXAMPLE,
                "",
                "If candidates are insufficient after tools, emit A (extension ATR).",
            ]
        )
    else:
        lines.extend(
            [
                "No operations retrieved yet — prefer A (Active Tool Request).",
                "If you must refuse without tools, use C:",
                FINAL_EXAMPLE,
            ]
        )
    lines.append(
        "Empty tool_calls with null final_answer is never a valid end."
    )
    return "\n".join(lines)


def build_parse_repair_message(*, available_operation_ids: List[str]) -> str:
    return (
        "Your previous response was neither a valid <tool_assistant> block nor "
        "valid action/final JSON. Reply with exactly one of the formats below.\n\n"
        + build_protocol_repair_message(
            available_operation_ids=available_operation_ids
        )
    )


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def _parse_agent_json(content: str) -> Dict[str, Any]:
    text = _strip_code_fence(content)
    parsed = json.loads(text)
    # Some providers wrap a single object in a one-element array.
    if isinstance(parsed, list) and len(parsed) == 1 and isinstance(parsed[0], dict):
        parsed = parsed[0]
    if not isinstance(parsed, dict):
        raise ValueError("Agent JSON must be an object")
    return parsed


def _is_valid_final_answer_object(value: Any) -> bool:
    return isinstance(value, dict) and value.get("status") is not None


def _classify_turn_validity(
    *,
    atr: bool,
    parsed: Optional[Dict[str, Any]],
    parse_error: bool,
    protocol_repair: bool,
) -> bool:
    if atr:
        return True
    if parse_error or protocol_repair:
        return False
    if not isinstance(parsed, dict):
        return False
    tool_calls = parsed.get("tool_calls") or []
    if not isinstance(tool_calls, list):
        tool_calls = []
    if tool_calls:
        return True
    return _is_valid_final_answer_object(parsed.get("final_answer"))


def _safe_error_message(exc: BaseException, *, limit: int = 500) -> str:
    text = str(exc) or type(exc).__name__
    redacted = redact_secrets(text)
    if not isinstance(redacted, str):
        redacted = str(redacted)
    for needle in ("api_key", "authorization", "bearer ", "sk-"):
        if needle in redacted.lower():
            return type(exc).__name__
    return redacted[:limit]


def _write_debug_traceback(
    *,
    debug_dir: Optional[Path],
    task_label: str,
    stage: str,
    exc: BaseException,
    trace: List[Dict[str, Any]],
) -> Optional[str]:
    if debug_dir is None:
        return None
    debug_dir = Path(debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)
    safe_name = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in task_label)[
        :80
    ] or "task"
    path = debug_dir / f"{safe_name}_traceback.txt"
    tb = traceback.format_exc()
    payload = redact_secrets(
        {
            "task_label": task_label,
            "failure_stage": stage,
            "error_type": type(exc).__name__,
            "error_message": _safe_error_message(exc),
            "traceback": tb,
            "partial_trace_event_types": [
                e.get("type") or e.get("name") for e in trace[-20:]
            ],
        }
    )
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    # Strip absolute home/workspace secret-looking paths from traceback text.
    text = text.replace(str(Path.home()), "~")
    path.write_text(text, encoding="utf-8")
    return str(path.name)


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
        debug_dir: Optional[Path] = None,
        task_label: str = "task",
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
        self.debug_dir = Path(debug_dir) if debug_dir else None
        self.task_label = task_label

    def run(self, question: str, *, task_label: Optional[str] = None) -> Dict[str, Any]:
        if task_label:
            self.task_label = task_label
        started = time.perf_counter()
        deadline = started + self.timeout_sec
        emb_start = self.router.embedder.snapshot()
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": question},
        ]
        trace: List[Dict[str, Any]] = []
        mcp_trace = McpZeroTrace()
        extension = ExtensionController(max_extension_rounds=self.max_extension_rounds)
        available: Dict[str, ToolHit] = {}
        selected_operations: List[str] = []
        invalid_selections: List[str] = []
        invoked_operations: List[str] = []
        successful_invoked_operations: List[str] = []
        invoke_calls = 0
        llm_calls = 0
        total_in = 0
        total_out = 0
        generation_latency_ms = 0.0
        retrieval_latency_ms = 0.0
        retrieval_embedding_calls = 0
        retrieval_embedding_tokens = 0
        final_payload: Any = None
        status = "failed"
        error: Optional[str] = None
        error_type: Optional[str] = None
        error_message: Optional[str] = None
        failure_stage: Optional[str] = None
        parse_repair_count = 0
        protocol_repair_count = 0
        valid_turn_count = 0
        debug_traceback_file: Optional[str] = None

        try:
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
                    error_type = "timeout"
                    failure_stage = "timeout"
                    extension.finalize_trace(mcp_trace, reason=TERMINATION_TIMEOUT)
                    break

                failure_stage = "llm_complete"
                t0 = time.perf_counter()
                llm_resp = self.llm.complete(
                    messages, temperature=self.temperature, max_tokens=self.max_tokens
                )
                generation_latency_ms += (time.perf_counter() - t0) * 1000.0
                llm_calls += 1
                usage = llm_resp.get("usage") or {}
                total_in += int(usage.get("input_tokens") or 0)
                total_out += int(usage.get("output_tokens") or 0)
                content = llm_resp.get("content")
                # Always persist auditable raw content (never drop to empty silently).
                if content is None:
                    raw_content = ""
                    event_note = "null_content"
                elif isinstance(content, str):
                    raw_content = content
                    event_note = None
                else:
                    raw_content = str(content)
                    event_note = "non_string_content"
                messages.append({"role": "assistant", "content": raw_content})
                event: Dict[str, Any] = {
                    "type": "llm_turn",
                    "turn": turn,
                    "content": raw_content,
                    "raw_content": raw_content,
                    "usage": usage,
                    "latency_ms": llm_resp.get("latency_ms"),
                    "provider": llm_resp.get("provider"),
                    "model": llm_resp.get("model"),
                }
                if event_note:
                    event["content_note"] = event_note

                failure_stage = "active_tool_request_parse"
                # Parser order: ATR first, then JSON action/final.
                request = parse_active_tool_request(
                    raw_content,
                    round_index=len(mcp_trace.rounds),
                    source="extension" if extension.initial_done else "llm",
                )
                if request is not None:
                    valid_turn_count += 1
                    decision = extension.decide(request)
                    event["active_tool_request"] = request.to_dict()
                    event["turn_valid"] = True
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

                    failure_stage = "hierarchical_retrieve"
                    round_ = self.router.retrieve(
                        request,
                        exclude_operation_ids=extension.already_seen_ops(),
                        is_extension=decision.is_extension,
                    )
                    round_ = extension.commit_round(round_)
                    mcp_trace.rounds.append(round_)
                    retrieval_latency_ms += float(round_.latency_ms)
                    retrieval_embedding_calls += int(round_.embedding_calls)
                    retrieval_embedding_tokens += int(round_.embedding_tokens)

                    for hit in round_.tools:
                        available[hit.operation_id] = hit

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
                                + json.dumps(
                                    sorted(available.keys()), ensure_ascii=False
                                )
                                + "\nRespond with invoke_api JSON, a refined "
                                "<tool_assistant> request, or final_answer JSON."
                            ),
                        }
                    )
                    continue

                failure_stage = "agent_json_parse"
                try:
                    parsed = _parse_agent_json(raw_content)
                except Exception as exc:  # noqa: BLE001
                    event["parse_error"] = type(exc).__name__
                    event["turn_valid"] = False
                    trace.append(event)
                    parse_repair_count += 1
                    messages.append(
                        {
                            "role": "user",
                            "content": build_parse_repair_message(
                                available_operation_ids=list(available.keys())
                            ),
                        }
                    )
                    continue

                tool_calls = parsed.get("tool_calls") or []
                if not isinstance(tool_calls, list):
                    tool_calls = []
                event["parsed"] = {
                    "thought": parsed.get("thought"),
                    "tool_calls": tool_calls,
                    "final_answer": parsed.get("final_answer"),
                }

                if _is_valid_final_answer_object(parsed.get("final_answer")) and not tool_calls:
                    event["turn_valid"] = True
                    valid_turn_count += 1
                    trace.append(event)
                    final_payload = parsed.get("final_answer")
                    status = "full"
                    extension.mark_model_proceeded()
                    extension.finalize_trace(mcp_trace, reason=TERMINATION_TASK_COMPLETE)
                    break

                if not tool_calls:
                    # Empty action: thought-only / null final_answer → protocol repair.
                    protocol_repair_count += 1
                    event["protocol_repair"] = True
                    event["turn_valid"] = False
                    trace.append(event)
                    messages.append(
                        {
                            "role": "user",
                            "content": build_protocol_repair_message(
                                available_operation_ids=list(available.keys())
                            ),
                        }
                    )
                    continue

                event["turn_valid"] = True
                valid_turn_count += 1
                trace.append(event)

                failure_stage = "invoke_api"
                observations = self._execute_invokes(
                    tool_calls,
                    available=available,
                    invoke_calls=invoke_calls,
                    selected_operations=selected_operations,
                    invalid_selections=invalid_selections,
                    invoked_operations=invoked_operations,
                    successful_invoked_operations=successful_invoked_operations,
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
                    error_type = "max_invocations"
                    failure_stage = "max_invocations"
                    extension.finalize_trace(mcp_trace)
                    break
            else:
                if final_payload is None:
                    final_payload = {
                        "status": "unavailable",
                        "final_answer": (
                            "Reached max LLM turns without a valid final_answer."
                        ),
                        "reasoning_brief": "",
                        "evidence": [],
                    }
                    status = "failed"
                    error = "max_llm_turns"
                    error_type = "max_llm_turns"
                    failure_stage = "max_llm_turns"
                    extension.finalize_trace(
                        mcp_trace, reason=TERMINATION_MAX_LLM_TURNS
                    )
        except Exception as exc:  # noqa: BLE001
            status = "failed"
            error = type(exc).__name__
            error_type = type(exc).__name__
            error_message = _safe_error_message(exc)
            failure_stage = failure_stage or "agent_run"
            debug_traceback_file = _write_debug_traceback(
                debug_dir=self.debug_dir,
                task_label=self.task_label,
                stage=failure_stage,
                exc=exc,
                trace=trace,
            )
            if final_payload is None:
                final_payload = {
                    "status": "unavailable",
                    "final_answer": "Agent failed before producing a final answer.",
                    "reasoning_brief": error_type,
                    "evidence": [],
                }
            extension.finalize_trace(mcp_trace)

        if mcp_trace.termination_reason is None:
            extension.finalize_trace(mcp_trace)

        predicted_status, final_text = normalize_final_payload(
            final_payload,
            fallback_status=agent_completion_to_fallback(status),
        )
        latency_ms = (time.perf_counter() - started) * 1000.0
        emb_end = self.router.embedder.snapshot()
        embedding_calls = max(0, int(emb_end["calls"]) - int(emb_start["calls"]))
        embedding_tokens = max(0, int(emb_end["tokens"]) - int(emb_start["tokens"]))
        terminal_contract_valid = status == "full" and _is_valid_final_answer_object(
            final_payload
        )
        repair_count = int(protocol_repair_count) + int(parse_repair_count)
        total_turn_count = int(llm_calls)
        task_protocol_success = bool(
            terminal_contract_valid and status == "full" and error is None
        )
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
            "total_turn_count": total_turn_count,
            "valid_turn_count": valid_turn_count,
            "repair_count": repair_count,
            "terminal_contract_valid": terminal_contract_valid,
            "task_protocol_success": task_protocol_success,
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
            "embedding_calls": embedding_calls,
            "embedding_tokens": embedding_tokens,
            "retrieval_embedding_calls": retrieval_embedding_calls,
            "retrieval_embedding_tokens": retrieval_embedding_tokens,
            "parse_repair_count": parse_repair_count,
            "protocol_repair_count": protocol_repair_count,
            "provider_metadata": {
                "llm_provider": getattr(self.llm, "provider", None),
                "llm_model": getattr(self.llm, "model", None),
                "embedding_model": self.router.embedder.model_name,
            },
            "trace": trace,
            "error": error,
            "error_type": error_type,
            "error_message": error_message,
            "failure_stage": failure_stage if status == "failed" else None,
            "debug_traceback_file": debug_traceback_file,
            "active_tool_request": mcp_trace.active_requests(),
            "retrieved_servers": mcp_trace.retrieved_servers(),
            "retrieved_tools": mcp_trace.retrieved_tools(),
            "expanded_servers": mcp_trace.expanded_servers(),
            "expanded_tools": mcp_trace.expanded_tools(),
            "selected_operations": list(selected_operations),
            "invalid_selections": list(invalid_selections),
            "invoked_operations": list(invoked_operations),
            "successful_invoked_operations": list(successful_invoked_operations),
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
        selected_operations: List[str],
        invalid_selections: List[str],
        invoked_operations: List[str],
        successful_invoked_operations: List[str],
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
            if oid:
                oid_s = str(oid)
                if oid_s not in selected_operations:
                    selected_operations.append(oid_s)
            if not oid or str(oid) not in allowed:
                if oid:
                    invalid_selections.append(str(oid))
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
            oid_s = str(oid)
            invoke_calls += 1
            invoked_operations.append(oid_s)
            mcp_result = self.gateway.invoke(oid_s, dict(call_args))
            success = bool(mcp_result.success)
            if success and oid_s not in successful_invoked_operations:
                successful_invoked_operations.append(oid_s)
            payload = {
                "success": success,
                "operation_id": oid_s,
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
                    "arguments": {"operation_id": oid_s, "arguments": call_args},
                    "parallel": False,
                    "result": payload,
                }
            )
        return {"results": results, "invoke_calls": invoke_calls}
