"""Flat All Tools agent: all runtime operations directly exposed (no routing).

Uses the same prompt-JSON ReAct loop as Operation-Visible, but replaces the
meta-tool boundary with direct operation calls via UnifiedRealMcpGateway.
No embedding, find_operations, describe_operation, or Top-K retrieval.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Sequence, Set

from hk_equity_mcp.agent.llm_client import LLMClient
from hk_equity_mcp.experiment.normalize import (
    agent_completion_to_fallback,
    empty_resolver_calls,
    normalize_final_payload,
)
from hk_equity_mcp.experiment.unified_flat_action_mapping import FlatActionMapping
from hk_equity_mcp.experiment.unified_flat_schema_bundle import FlatSchemaBundle
from hk_equity_mcp.operation_resolver.schemas import OperationSpec, ParameterSpec
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


def _validate_arguments(
    op: OperationSpec, arguments: Dict[str, Any]
) -> Optional[str]:
    if not isinstance(arguments, dict):
        return "arguments must be an object/dict"
    known = op.param_map()
    unknown = sorted(set(arguments) - set(known))
    if unknown:
        return f"Unknown arguments: {unknown}"
    missing = [p.name for p in op.required_params() if p.name not in arguments]
    if missing:
        return f"Missing required arguments: {missing}"
    for name, value in arguments.items():
        err = _validate_value(known[name], value)
        if err:
            return err
    return None


def _validate_value(spec: ParameterSpec, value: Any) -> Optional[str]:
    if value is None:
        return f"Argument '{spec.name}' cannot be null"
    expected = spec.type
    if expected == "string" and not isinstance(value, str):
        return f"Argument '{spec.name}' must be string"
    if expected == "number" and not isinstance(value, (int, float)):
        return f"Argument '{spec.name}' must be number"
    if expected == "boolean" and not isinstance(value, bool):
        return f"Argument '{spec.name}' must be boolean"
    if expected == "object" and not isinstance(value, dict):
        return f"Argument '{spec.name}' must be object"
    if expected == "array" and not isinstance(value, list):
        return f"Argument '{spec.name}' must be array"
    if spec.enum is not None and value not in spec.enum:
        return f"Argument '{spec.name}' must be one of {spec.enum}; got {value!r}"
    return None


class FlatAllToolsAgent:
    """Centralized ReAct loop with all runtime operations directly exposed."""

    def __init__(
        self,
        llm: LLMClient,
        *,
        gateway: UnifiedRealMcpGateway,
        schema_bundle: FlatSchemaBundle,
        system_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        max_llm_turns: int = 8,
        max_invocations: int = 6,
        timeout_sec: float = 120.0,
    ):
        self.llm = llm
        self.gateway = gateway
        self.schema_bundle = schema_bundle
        self.mapping: FlatActionMapping = schema_bundle.mapping
        self.registry: Dict[str, OperationSpec] = dict(gateway.registry)
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_llm_turns = max_llm_turns
        self.max_invocations = max_invocations
        self.timeout_sec = timeout_sec
        self._runtime_ids: Set[str] = set(schema_bundle.runtime_operation_ids)
        self._blocked_ids: Set[str] = set(schema_bundle.blocked_operation_ids)
        # Fail closed if gateway registry does not match the schema bundle.
        if set(self.registry) != self._runtime_ids:
            missing = sorted(self._runtime_ids - set(self.registry))
            extra = sorted(set(self.registry) - self._runtime_ids)
            raise RuntimeError(
                "Flat All Tools fail-closed: gateway registry mismatch "
                f"missing={missing[:5]} extra={extra[:5]}"
            )
        # Never retain a FinancialServer fallback path.
        if getattr(gateway, "mock_transport", None) is None:
            # Live path must still go through UnifiedRealMcpGateway only.
            pass

    def exposed_tools(self) -> List[str]:
        """Action names visible to the model (all 39 runtime ops)."""
        return [
            self.mapping.action_name(oid)
            for oid in self.schema_bundle.runtime_operation_ids
        ]

    def exposed_operation_ids(self) -> List[str]:
        return list(self.schema_bundle.runtime_operation_ids)

    def run(self, question: str, history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        started = time.perf_counter()
        history = history or []
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": self.system_prompt},
            *history,
            {"role": "user", "content": question},
        ]
        # Guard: gold/reference must never appear in the model context.
        for msg in messages:
            content = str(msg.get("content") or "")
            for banned in (
                "gold_operations",
                "reference_answer",
                "gold_values",
                "expected_tool_path",
            ):
                if banned in content:
                    raise RuntimeError(
                        f"Flat All Tools refuse: banned substring {banned} in model context"
                    )

        trace: List[Dict[str, Any]] = []
        selected_operations: List[str] = []
        invoked_operations: List[str] = []
        successful_invoked_operations: List[str] = []
        invalid_selections: List[Dict[str, Any]] = []
        invoke_calls = 0
        llm_calls = 0
        parse_repair_count = 0
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
                parse_repair_count += 1
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

            observations = self._execute_tool_calls(
                tool_calls,
                selected_operations=selected_operations,
                invoked_operations=invoked_operations,
                successful_invoked_operations=successful_invoked_operations,
                invalid_selections=invalid_selections,
            )
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
            "architecture": "flat_all_tools",
            "condition": "flat",
            "execution_backend": "real_mcp",
            "fidelity_note": (
                "Flat All Tools routing ablation: all runtime operations exposed; "
                "no embedding/retrieval/find/describe"
            ),
            "exposed_tools": self.exposed_tools(),
            "exposed_operation_ids": self.exposed_operation_ids(),
            "final_answer": final_text,
            "final_answer_obj": final_payload if isinstance(final_payload, dict) else None,
            "predicted_status": predicted_status,
            "status": status,
            "llm_calls": llm_calls,
            "parse_repair_count": parse_repair_count,
            "protocol_repair_count": parse_repair_count,
            "tokens": {
                "input_tokens": total_in,
                "output_tokens": total_out,
                "total_tokens": total_in + total_out,
            },
            "resolver_calls": empty_resolver_calls(),
            "find_operations_calls": 0,
            "describe_operation_calls": 0,
            "retrieval_calls": 0,
            "embedding_calls": 0,
            "embedding_tokens": 0,
            "underlying_operation_calls": invoke_calls,
            "real_mcp_execution_count": invoke_calls,
            "selected_operations": list(dict.fromkeys(selected_operations)),
            "invoked_operations": list(dict.fromkeys(invoked_operations)),
            "successful_invoked_operations": list(
                dict.fromkeys(successful_invoked_operations)
            ),
            "invalid_selections": list(invalid_selections),
            "schema_bundle_overhead": {
                "serialized_schema_char_count": (
                    self.schema_bundle.serialized_schema_char_count
                ),
                "token_estimate": self.schema_bundle.token_estimate,
                "schema_bundle_sha256": self.schema_bundle.schema_bundle_sha256,
            },
            "latency_ms": round(latency_ms, 3),
            "provider_metadata": {
                "llm_provider": getattr(self.llm, "provider", None),
                "llm_model": getattr(self.llm, "model", None),
                "embedding_model": None,
            },
            "trace": trace,
            "error": error,
            "tool_metrics_applicable": True,
            "retrieval_metrics": "N/A",
        }

    def _execute_tool_calls(
        self,
        tool_calls: Sequence[Dict[str, Any]],
        *,
        selected_operations: List[str],
        invoked_operations: List[str],
        successful_invoked_operations: List[str],
        invalid_selections: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        prepared: List[Dict[str, Any]] = []
        for idx, call in enumerate(tool_calls):
            name = call.get("name")
            args = call.get("arguments") or {}
            prepared.append({"index": idx, "name": name, "arguments": args})

        results: List[Optional[Dict[str, Any]]] = [None] * len(prepared)
        invoke_calls = 0
        i = 0
        while i < len(prepared):
            item = prepared[i]
            resolved = self.mapping.resolve(item["name"])
            if (
                resolved
                and resolved in self._runtime_ids
                and self._is_independent_readonly(resolved, item["arguments"])
            ):
                batch = [item]
                j = i + 1
                while j < len(prepared):
                    nxt = prepared[j]
                    nxt_oid = self.mapping.resolve(nxt["name"])
                    if (
                        nxt_oid
                        and nxt_oid in self._runtime_ids
                        and self._is_independent_readonly(nxt_oid, nxt["arguments"])
                    ):
                        batch.append(nxt)
                        j += 1
                    else:
                        break
                invoke_calls += self._run_batch(
                    batch,
                    results,
                    selected_operations=selected_operations,
                    invoked_operations=invoked_operations,
                    successful_invoked_operations=successful_invoked_operations,
                    invalid_selections=invalid_selections,
                )
                i = j
                continue

            payload = self._dispatch_one(
                item["name"],
                item["arguments"],
                selected_operations=selected_operations,
                invoked_operations=invoked_operations,
                successful_invoked_operations=successful_invoked_operations,
                invalid_selections=invalid_selections,
            )
            if payload.get("dispatched"):
                invoke_calls += 1
            results[item["index"]] = payload["event"]
            i += 1

        return {
            "results": [r for r in results if r is not None],
            "invoke_calls": invoke_calls,
        }

    def _run_batch(
        self,
        batch: List[Dict[str, Any]],
        results: List[Optional[Dict[str, Any]]],
        *,
        selected_operations: List[str],
        invoked_operations: List[str],
        successful_invoked_operations: List[str],
        invalid_selections: List[Dict[str, Any]],
    ) -> int:
        invoke_calls = 0
        parallel = len(batch) > 1
        if parallel:
            with ThreadPoolExecutor(max_workers=min(4, len(batch))) as pool:
                futs = {
                    pool.submit(
                        self._dispatch_one,
                        item["name"],
                        item["arguments"],
                        selected_operations=selected_operations,
                        invoked_operations=invoked_operations,
                        successful_invoked_operations=successful_invoked_operations,
                        invalid_selections=invalid_selections,
                    ): item
                    for item in batch
                }
                for fut in as_completed(futs):
                    item = futs[fut]
                    payload = fut.result()
                    if payload.get("dispatched"):
                        invoke_calls += 1
                    event = dict(payload["event"])
                    event["parallel"] = True
                    results[item["index"]] = event
            return invoke_calls

        item = batch[0]
        payload = self._dispatch_one(
            item["name"],
            item["arguments"],
            selected_operations=selected_operations,
            invoked_operations=invoked_operations,
            successful_invoked_operations=successful_invoked_operations,
            invalid_selections=invalid_selections,
        )
        if payload.get("dispatched"):
            invoke_calls += 1
        event = dict(payload["event"])
        event["parallel"] = False
        results[item["index"]] = event
        return invoke_calls

    def _is_independent_readonly(
        self, operation_id: str, arguments: Dict[str, Any]
    ) -> bool:
        op = self.registry.get(operation_id)
        if op is None:
            return False
        if not op.read_only:
            return False
        if op.depends_on:
            return False
        return isinstance(arguments, dict)

    def _dispatch_one(
        self,
        name: Optional[str],
        arguments: Dict[str, Any],
        *,
        selected_operations: List[str],
        invoked_operations: List[str],
        successful_invoked_operations: List[str],
        invalid_selections: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        arguments = dict(arguments or {}) if isinstance(arguments, dict) else {}
        # Refuse meta-tools explicitly.
        if name in {"find_operations", "describe_operation", "invoke_api"}:
            invalid_selections.append(
                {
                    "name": name,
                    "error_type": "meta_tool_not_allowed",
                    "error": (
                        f"Flat All Tools refuses meta-tool '{name}'. "
                        "Call underlying operations by action_name directly."
                    ),
                }
            )
            return {
                "dispatched": False,
                "event": {
                    "type": "tool_result",
                    "name": name,
                    "operation_id": None,
                    "invalid_selection": True,
                    "arguments": arguments,
                    "parallel": False,
                    "result": {
                        "success": False,
                        "error_type": "meta_tool_not_allowed",
                        "error": f"Meta-tool '{name}' is not available in Flat All Tools",
                        "result": None,
                    },
                },
            }

        oid = self.mapping.resolve(name)
        if oid is None:
            invalid_selections.append(
                {
                    "name": name,
                    "error_type": "unknown_operation",
                    "error": f"Unknown Flat action/operation: {name}",
                }
            )
            return {
                "dispatched": False,
                "event": {
                    "type": "tool_result",
                    "name": name,
                    "operation_id": None,
                    "invalid_selection": True,
                    "arguments": arguments,
                    "parallel": False,
                    "result": {
                        "success": False,
                        "error_type": "unknown_operation",
                        "error": (
                            f"Unknown or disallowed Flat tool '{name}'. "
                            "Fail closed — no silent fallback."
                        ),
                        "result": None,
                    },
                },
            }

        selected_operations.append(oid)

        if oid in self._blocked_ids or oid not in self._runtime_ids:
            invalid_selections.append(
                {
                    "name": name,
                    "operation_id": oid,
                    "error_type": "blocked_operation",
                    "error": f"Blocked/non-runtime operation not callable: {oid}",
                }
            )
            return {
                "dispatched": False,
                "event": {
                    "type": "tool_result",
                    "name": name,
                    "operation_id": oid,
                    "invalid_selection": True,
                    "arguments": arguments,
                    "parallel": False,
                    "result": {
                        "success": False,
                        "operation_id": oid,
                        "error_type": "blocked_operation",
                        "error": f"Blocked/rejected operation not executable: {oid}",
                        "result": None,
                    },
                },
            }

        op = self.registry.get(oid)
        if op is None:
            invalid_selections.append(
                {
                    "name": name,
                    "operation_id": oid,
                    "error_type": "unknown_operation",
                    "error": f"Operation not in runtime registry: {oid}",
                }
            )
            return {
                "dispatched": False,
                "event": {
                    "type": "tool_result",
                    "name": name,
                    "operation_id": oid,
                    "invalid_selection": True,
                    "arguments": arguments,
                    "parallel": False,
                    "result": {
                        "success": False,
                        "operation_id": oid,
                        "error_type": "unknown_operation",
                        "error": f"Unknown Real-MCP operation_id (fail-closed): {oid}",
                        "result": None,
                    },
                },
            }

        validation_error = _validate_arguments(op, arguments)
        if validation_error is not None:
            invalid_selections.append(
                {
                    "name": name,
                    "operation_id": oid,
                    "error_type": "validation_error",
                    "error": validation_error,
                }
            )
            return {
                "dispatched": False,
                "event": {
                    "type": "tool_result",
                    "name": name,
                    "operation_id": oid,
                    "invalid_selection": True,
                    "arguments": arguments,
                    "parallel": False,
                    # Emit invoke_api-shaped args so evaluator can still see the attempt.
                    "result": {
                        "success": False,
                        "operation_id": oid,
                        "error_type": "validation_error",
                        "error": validation_error,
                        "result": None,
                    },
                },
            }

        # Apply defaults for optional params.
        call_args = dict(arguments)
        for p in op.optional_params():
            if p.name not in call_args and p.default is not None:
                call_args[p.name] = p.default

        invoked_operations.append(oid)
        try:
            mcp_result = self.gateway.invoke(oid, call_args)
        except Exception as exc:  # noqa: BLE001
            return {
                "dispatched": True,
                "event": self._tool_event(
                    name=name,
                    operation_id=oid,
                    call_args=call_args,
                    success=False,
                    error=f"Real MCP invoke failed: {type(exc).__name__}: {exc}",
                    error_type="execution_error",
                    nested=None,
                ),
            }

        payload = (
            mcp_result.to_dict()
            if hasattr(mcp_result, "to_dict")
            else dict(mcp_result)
        )
        success = bool(payload.get("success"))
        if success:
            successful_invoked_operations.append(oid)
        return {
            "dispatched": True,
            "event": self._tool_event(
                name=name,
                operation_id=oid,
                call_args=call_args,
                success=success,
                error=payload.get("error"),
                error_type=(
                    None
                    if success
                    else (payload.get("error_type") or "underlying_tool_error")
                ),
                nested=payload,
            ),
        }

    def _tool_event(
        self,
        *,
        name: Optional[str],
        operation_id: str,
        call_args: Dict[str, Any],
        success: bool,
        error: Any,
        error_type: Any,
        nested: Any,
    ) -> Dict[str, Any]:
        """Emit a dual-shape tool_result: Flat fields + invoke_api-compatible args."""
        return {
            "type": "tool_result",
            "name": name,
            "flat_action_name": name,
            "operation_id": operation_id,
            "arguments": call_args,
            # Evaluator v1.2 compatibility: also expose invoke_api-shaped fields.
            "invoke_api_arguments": {
                "operation_id": operation_id,
                "arguments": call_args,
            },
            "parallel": False,
            "result": {
                "success": success,
                "operation_id": operation_id,
                "arguments": call_args,
                "result": nested,
                "error_type": error_type,
                "error": error,
            },
        }
