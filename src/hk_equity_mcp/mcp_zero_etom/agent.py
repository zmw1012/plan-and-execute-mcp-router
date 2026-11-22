"""MCP-Zero agent for ETOM routing-only evaluation (no tool execution)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from hk_equity_mcp.mcp_zero.active_request import parse_active_tool_request
from hk_equity_mcp.mcp_zero.agent import (
    _parse_agent_json,
    _safe_error_message,
    _write_debug_traceback,
    build_parse_repair_message,
    build_protocol_repair_message,
)
from hk_equity_mcp.mcp_zero.extension import (
    TERMINATION_MAX_LLM_TURNS,
    TERMINATION_TASK_COMPLETE,
    TERMINATION_TIMEOUT,
    ExtensionController,
)
from hk_equity_mcp.mcp_zero.router import HierarchicalRouter
from hk_equity_mcp.mcp_zero.schemas import McpZeroTrace, ToolHit
from hk_equity_mcp.mcp_zero_etom.prompt import ETOM_MCP_ZERO_SYSTEM_PROMPT
from hk_equity_mcp.mcp_zero_etom.recorder import EtomRouteRecorder


def _is_valid_route_final(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    status = value.get("status")
    if status not in {"route_complete", "infeasible"}:
        return False
    if "infeasible" in value and not isinstance(value.get("infeasible"), bool):
        return False
    if status == "infeasible" and value.get("infeasible") is False:
        return False
    if status == "route_complete" and value.get("infeasible") is True:
        return False
    return True


def _leak_check(text: str, ground_truth_tools: Any) -> None:
    if ground_truth_tools is None:
        return
    blob = json.dumps(ground_truth_tools, ensure_ascii=False)
    if len(blob) > 40 and blob in text:
        raise RuntimeError("Gold ground_truth leaked into model-visible text.")


class EtomMcpZeroAgent:
    """Active hierarchical routing agent; records routes; executes no tools."""

    def __init__(
        self,
        llm: Any,
        router: HierarchicalRouter,
        recorder: EtomRouteRecorder,
        *,
        max_llm_turns: int = 8,
        max_invocations: int = 8,
        max_extension_rounds: int = 3,
        timeout_sec: float = 180.0,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        system_prompt: Optional[str] = None,
        debug_dir: Optional[Path] = None,
        task_label: str = "task",
    ):
        self.llm = llm
        self.router = router
        self.recorder = recorder
        self.max_llm_turns = int(max_llm_turns)
        self.max_invocations = int(max_invocations)
        self.max_extension_rounds = int(max_extension_rounds)
        self.timeout_sec = float(timeout_sec)
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.system_prompt = system_prompt or ETOM_MCP_ZERO_SYSTEM_PROMPT
        self.debug_dir = Path(debug_dir) if debug_dir else None
        self.task_label = task_label

    def run(
        self,
        query: str,
        *,
        task_label: Optional[str] = None,
        ground_truth_tools: Any = None,
    ) -> Dict[str, Any]:
        if task_label:
            self.task_label = task_label
        started = time.perf_counter()
        deadline = started + self.timeout_sec
        emb_start = self.router.embedder.snapshot()
        self.recorder.reset()
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": query},
        ]
        _leak_check(self.system_prompt, ground_truth_tools)
        _leak_check(query, ground_truth_tools)

        trace: List[Dict[str, Any]] = []
        mcp_trace = McpZeroTrace()
        extension = ExtensionController(max_extension_rounds=self.max_extension_rounds)
        available: Dict[str, ToolHit] = {}
        selected_operations: List[str] = []
        invalid_selections: List[str] = []
        invoked_operations: List[str] = []
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
        predicted_rejection = False
        debug_traceback_file: Optional[str] = None

        try:
            for turn in range(1, self.max_llm_turns + 1):
                if time.perf_counter() > deadline:
                    status = "failed"
                    final_payload = {
                        "status": "infeasible",
                        "infeasible": True,
                        "reasoning_brief": "timeout",
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
                if content is None:
                    raw_content = ""
                    event_note = "null_content"
                elif isinstance(content, str):
                    raw_content = content
                    event_note = None
                else:
                    raw_content = str(content)
                    event_note = "non_string_content"
                _leak_check(raw_content, ground_truth_tools)
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
                    self.recorder.set_retrieved(set(available.keys()))
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
                                + "\nRespond with invoke_api JSON (route record only), "
                                "a refined <tool_assistant> request, or final_answer JSON."
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

                if _is_valid_route_final(parsed.get("final_answer")) and not tool_calls:
                    event["turn_valid"] = True
                    valid_turn_count += 1
                    trace.append(event)
                    final_payload = parsed.get("final_answer")
                    predicted_rejection = bool(final_payload.get("infeasible"))
                    if predicted_rejection and self.recorder.predicted_nodes:
                        status = "failed"
                        error = "infeasible_with_nonempty_route"
                        error_type = "protocol_contract"
                        failure_stage = "final_contract"
                    else:
                        status = "completed"
                    extension.mark_model_proceeded()
                    extension.finalize_trace(mcp_trace, reason=TERMINATION_TASK_COMPLETE)
                    break

                if not tool_calls:
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
                self.recorder.set_retrieved(set(available.keys()))

                failure_stage = "invoke_api"
                observations = self._execute_invokes(
                    tool_calls,
                    available=available,
                    invoke_calls=invoke_calls,
                    selected_operations=selected_operations,
                    invalid_selections=invalid_selections,
                    invoked_operations=invoked_operations,
                )
                invoke_calls = observations["invoke_calls"]
                for obs in observations["results"]:
                    trace.append(obs)
                extension.mark_model_proceeded()
                messages.append(
                    {
                        "role": "user",
                        "content": "Route-record results (no tool execution):\n"
                        + json.dumps(observations["results"], ensure_ascii=False),
                    }
                )
                if invoke_calls > self.max_invocations:
                    status = "failed"
                    error = "max_invocations"
                    error_type = "max_invocations"
                    failure_stage = "max_invocations"
                    extension.finalize_trace(mcp_trace)
                    break
            else:
                if final_payload is None:
                    final_payload = {
                        "status": "infeasible",
                        "infeasible": True,
                        "reasoning_brief": "max_llm_turns",
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
                    "status": "infeasible",
                    "infeasible": True,
                    "reasoning_brief": error_type or "agent_failed",
                }
            extension.finalize_trace(mcp_trace)

        if mcp_trace.termination_reason is None:
            extension.finalize_trace(mcp_trace)
        if self.recorder.real_tool_executions != 0:
            raise RuntimeError("Recorder reported non-zero real_tool_executions")

        latency_ms = (time.perf_counter() - started) * 1000.0
        emb_end = self.router.embedder.snapshot()
        embedding_calls = max(0, int(emb_end["calls"]) - int(emb_start["calls"]))
        embedding_tokens = max(0, int(emb_end["tokens"]) - int(emb_start["tokens"]))
        terminal_contract_valid = status == "completed" and _is_valid_route_final(
            final_payload
        )
        return {
            "query": query,
            "architecture": "mcp_zero",
            "condition": "mcp_zero",
            "routing_only": True,
            "real_tool_executions": 0,
            "predicted_tools": self.recorder.predicted_tools,
            "predicted_nodes": list(self.recorder.predicted_nodes),
            "predicted_rejection": predicted_rejection,
            "final_answer_obj": final_payload if isinstance(final_payload, dict) else None,
            "status": status,
            "llm_calls": llm_calls,
            "total_turn_count": llm_calls,
            "valid_turn_count": valid_turn_count,
            "tokens": {
                "input_tokens": total_in,
                "output_tokens": total_out,
                "total_tokens": total_in + total_out,
            },
            "input_tokens": total_in,
            "output_tokens": total_out,
            "total_tokens": total_in + total_out,
            "latency_ms": round(latency_ms, 3),
            "generation_latency_ms": round(generation_latency_ms, 3),
            "retrieval_latency_ms": round(retrieval_latency_ms, 3),
            "embedding_calls": embedding_calls,
            "embedding_tokens": embedding_tokens,
            "query_embedding_calls": embedding_calls,
            "query_embedding_tokens": embedding_tokens,
            "retrieval_embedding_calls": retrieval_embedding_calls,
            "retrieval_embedding_tokens": retrieval_embedding_tokens,
            "parse_repairs": parse_repair_count,
            "parse_repair_count": parse_repair_count,
            "protocol_repair_count": protocol_repair_count,
            "http_retries": 0,
            "timeout": error_type == "timeout",
            "trace": trace,
            "error": error,
            "error_type": error_type,
            "error_message": error_message,
            "failure_stage": failure_stage if status != "completed" else None,
            "debug_traceback_file": debug_traceback_file,
            "active_tool_request": mcp_trace.active_requests(),
            "retrieved_servers": mcp_trace.retrieved_servers(),
            "retrieved_tools": mcp_trace.retrieved_tools(),
            "expanded_servers": mcp_trace.expanded_servers(),
            "expanded_tools": mcp_trace.expanded_tools(),
            "selected_operations": list(selected_operations),
            "invalid_selections": list(invalid_selections),
            "invoked_operations": list(invoked_operations),
            "extension_rounds": [r.to_dict() for r in mcp_trace.extension_rounds],
            "number_of_extension_rounds": mcp_trace.number_of_extension_rounds,
            "termination_reason": mcp_trace.termination_reason,
            "mcp_zero_trace": mcp_trace.to_dict(),
            "candidate_operation_ids": mcp_trace.candidate_operation_ids(),
            "terminal_contract_valid": terminal_contract_valid,
            "task_protocol_success": bool(terminal_contract_valid and error is None),
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
                            "executed": False,
                            "real_tool_executions": 0,
                            "error": (
                                f"MCP-Zero ETOM refuses tool '{name}'. "
                                "Only invoke_api route recording is allowed."
                            ),
                            "error_type": "refused_tool",
                        },
                    }
                )
                continue
            oid = args.get("operation_id")
            call_args = args.get("arguments") or {}
            depends_on = args.get("depends_on") or []
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
                            "executed": False,
                            "real_tool_executions": 0,
                            "error": (
                                f"operation_id {oid!r} is not in the retrieved "
                                "candidate set for this trajectory."
                            ),
                            "error_type": "not_retrieved",
                        },
                    }
                )
                continue
            oid_s = str(oid)
            invoke_calls += 1
            invoked_operations.append(oid_s)
            recorded = self.recorder.invoke(oid_s, dict(call_args), depends_on)
            results.append(
                {
                    "type": "tool_result",
                    "name": "invoke_api",
                    "arguments": {
                        "operation_id": oid_s,
                        "arguments": call_args,
                        "depends_on": depends_on,
                    },
                    "parallel": False,
                    "result": recorded,
                }
            )
        return {"results": results, "invoke_calls": invoke_calls}
