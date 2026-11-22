"""Offline scripted LLM for MCP-Zero × ETOM mock runs."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence


def atr_text(server: str, tool: str) -> str:
    return (
        "<tool_assistant>\n"
        f"server: {server}\n"
        f"tool: {tool}\n"
        "</tool_assistant>"
    )


def invoke_json(operation_id: str, depends_on: Optional[Sequence[int]] = None) -> str:
    payload = {
        "thought": "record route node",
        "tool_calls": [
            {
                "name": "invoke_api",
                "arguments": {
                    "operation_id": operation_id,
                    "arguments": {},
                    "depends_on": list(depends_on or []),
                },
            }
        ],
        "final_answer": None,
    }
    return json.dumps(payload, ensure_ascii=False)


def final_json(*, infeasible: bool = False) -> str:
    payload = {
        "thought": "done",
        "tool_calls": [],
        "final_answer": {
            "status": "infeasible" if infeasible else "route_complete",
            "infeasible": bool(infeasible),
            "reasoning_brief": "mock final",
        },
    }
    return json.dumps(payload, ensure_ascii=False)


def build_route_script(
    *,
    server_desc: str,
    tool_desc: str,
    operation_ids: Sequence[str],
    depends_on_map: Optional[Dict[int, Sequence[int]]] = None,
    infeasible: bool = False,
) -> List[str]:
    if infeasible:
        return [final_json(infeasible=True)]
    turns = [atr_text(server_desc, tool_desc)]
    dep_map = depends_on_map or {}
    for index, oid in enumerate(operation_ids):
        turns.append(invoke_json(oid, dep_map.get(index, [])))
    turns.append(final_json(infeasible=False))
    return turns


class EtomMockLLM:
    """Standalone scripted LLM (no network)."""

    def __init__(self, scripts: Optional[Dict[str, List[str]]] = None):
        self.provider = "mock"
        self.model = "mock-etom-mcp-zero"
        self.scripts = dict(scripts or {})
        self._active: List[str] = []
        self._cursor = 0
        self.current_task_id: Optional[str] = None

    def set_task(self, task_id: str, responses: Optional[Sequence[str]] = None) -> None:
        self.current_task_id = task_id
        if responses is not None:
            self._active = list(responses)
        else:
            self._active = list(self.scripts.get(task_id) or [final_json(infeasible=True)])
        self._cursor = 0

    def complete(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        if self._cursor >= len(self._active):
            content = final_json(infeasible=False)
        else:
            content = self._active[self._cursor]
            self._cursor += 1
        return {
            "content": content,
            "usage": {
                "input_tokens": max(1, len(json.dumps(messages)) // 4),
                "output_tokens": max(1, len(content) // 4),
            },
            "latency_ms": 0.0,
            "provider": self.provider,
            "model": self.model,
        }
