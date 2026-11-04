"""Fail-closed chat client for the ETOM pilot. Never logs API keys."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from hk_equity_mcp.etom import EXPECTED_LLM_MODEL, EXPECTED_LLM_PROVIDER
from hk_equity_mcp.etom.embeddings import ModelMismatchError, scrub_secret, urlopen_with_deadline


class FailClosedChatClient:
    def __init__(
        self,
        *,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout_sec: float = 120.0,
        max_retries: int = 1,
        json_mode: bool = True,
    ):
        self.provider = provider or os.environ.get("LLM_PROVIDER") or ""
        self.model = model or os.environ.get("LLM_MODEL") or ""
        self.base_url = (base_url or os.environ.get("LLM_BASE_URL") or "").rstrip("/")
        self.api_key = api_key or os.environ.get("LLM_API_KEY") or ""
        self.timeout_sec = timeout_sec
        self.max_retries = max_retries
        self.json_mode = json_mode
        self.retries = 0
        if self.provider != EXPECTED_LLM_PROVIDER or self.model != EXPECTED_LLM_MODEL:
            raise ModelMismatchError(
                "Generation provider/model mismatch before the request: "
                f"expected {EXPECTED_LLM_PROVIDER}/{EXPECTED_LLM_MODEL}."
            )
        if not self.base_url or not self.api_key:
            raise RuntimeError(
                "Missing generation endpoint configuration. "
                "Load the existing generation env file. The key is not printed."
            )

    def complete(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
    ) -> Dict[str, Any]:
        url = f"{self.base_url}/chat/completions"
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if self.json_mode:
            payload["response_format"] = {"type": "json_object"}
        data = json.dumps(payload).encode("utf-8")
        last_name = "RuntimeError"
        for attempt in range(self.max_retries + 1):
            request = urllib.request.Request(
                url,
                data=data,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
                method="POST",
            )
            started = time.perf_counter()
            try:
                raw = urlopen_with_deadline(request, self.timeout_sec)
                body = json.loads(raw.decode("utf-8"))
                latency_ms = (time.perf_counter() - started) * 1000
                return self._parse(body, latency_ms)
            except ModelMismatchError:
                raise
            except TimeoutError:
                last_name = "TimeoutError"
                if attempt >= self.max_retries:
                    raise RuntimeError(
                        f"LLM request exceeded {self.timeout_sec:.0f}s for model {self.model}."
                    ) from None
                self.retries += 1
            except urllib.error.HTTPError as exc:
                last_name = f"HTTP{exc.code}"
                if exc.code not in {429, 500, 502, 503, 504} or attempt >= self.max_retries:
                    raise RuntimeError(
                        scrub_secret(
                            f"LLM request failed with HTTP {exc.code} for model {self.model}."
                        )
                    ) from None
                self.retries += 1
            except Exception as exc:  # noqa: BLE001
                last_name = type(exc).__name__
                if attempt >= self.max_retries:
                    raise RuntimeError(
                        scrub_secret(
                            f"LLM request failed for model {self.model}: {last_name}."
                        )
                    ) from None
                self.retries += 1
        raise RuntimeError(f"LLM request failed: {last_name}.")

    def _parse(self, body: dict, latency_ms: float) -> Dict[str, Any]:
        echoed = body.get("model")
        if echoed and echoed != self.model:
            raise ModelMismatchError(
                f"Generation provider/model mismatch: expected {self.model}, echoed {echoed}."
            )
        choices = body.get("choices") or []
        if not choices:
            raise RuntimeError("LLM response missing choices.")
        content = choices[0].get("message", {}).get("content") or ""
        usage = body.get("usage") or {}
        input_tokens = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
        output_tokens = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
        return {
            "content": content,
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": int(usage.get("total_tokens") or (input_tokens + output_tokens)),
            },
            "latency_ms": round(latency_ms, 3),
            "provider": self.provider,
            "model": echoed or self.model,
            "model_echo_absent": not bool(echoed),
            "http_retries": self.retries,
        }


class ScriptedChatClient:
    """Deterministic offline client. It does not read gold graphs."""

    provider = "mock"
    model = "mock-llm"

    def __init__(self, policy: str = "route_first_candidate"):
        self.policy = policy
        self.retries = 0
        self.calls = 0

    def complete(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
    ) -> Dict[str, Any]:
        self.calls += 1
        full = "\n".join(message.get("content") or "" for message in messages)
        if "ground_truth" in full or "GOLD_SENTINEL_TOOL_ZZZ" in full:
            raise RuntimeError("Mock client observed gold in the model context.")
        tool_blob = "\n".join(
            message.get("content") or ""
            for message in messages
            if (message.get("content") or "").startswith("Tool results (visible):")
        )
        content = self._next(tool_blob)
        return {
            "content": content,
            "usage": {
            "input_tokens": max(1, len(full) // 4),
            "output_tokens": max(1, len(content) // 4),
            "total_tokens": max(1, (len(full) + len(content)) // 4),
            },
            "latency_ms": 0.1,
            "provider": self.provider,
            "model": self.model,
            "model_echo_absent": False,
            "http_retries": 0,
        }

    def _next(self, blob: str) -> str:
        if "Tool results (visible):" not in blob:
            return json.dumps(
                {
                    "thought": "Search the catalog.",
                    "tool_calls": [
                        {
                            "name": "find_operations",
                            "arguments": {"query": "task tools", "top_k": 5},
                        }
                    ],
                    "final_answer": None,
                }
            )
        if '"name": "describe_operation"' not in blob and "input_schema" not in blob:
            operation_id = _first_operation_id(blob)
            if operation_id is None:
                return _final(infeasible=True)
            return json.dumps(
                {
                    "thought": "Inspect the first retrieved operation.",
                    "tool_calls": [
                        {
                            "name": "describe_operation",
                            "arguments": {"operation_id": operation_id},
                        }
                    ],
                    "final_answer": None,
                }
            )
        if '"execution_mode": "etom_routing_only"' not in blob:
            operation_id = _described_operation_id(blob) or _first_operation_id(blob)
            if operation_id is None:
                return _final(infeasible=True)
            return json.dumps(
                {
                    "thought": "Record one route node. Do not expect business data.",
                    "tool_calls": [
                        {
                            "name": "invoke_api",
                            "arguments": {
                                "operation_id": operation_id,
                                "arguments": {},
                                "depends_on": [],
                            },
                        }
                    ],
                    "final_answer": None,
                }
            )
        return _final(infeasible=False)


def _first_operation_id(blob: str) -> Optional[str]:
    marker = '"operation_id": "'
    start = blob.find(marker)
    if start < 0:
        return None
    start += len(marker)
    end = blob.find('"', start)
    if end < 0:
        return None
    return blob[start:end]


def _described_operation_id(blob: str) -> Optional[str]:
    marker = '"input_schema"'
    pos = blob.rfind(marker)
    if pos < 0:
        return _first_operation_id(blob)
    window = blob[max(0, pos - 500) : pos]
    return _first_operation_id(window) or _first_operation_id(blob)


def _final(infeasible: bool) -> str:
    return json.dumps(
        {
            "thought": "Finish the route.",
            "tool_calls": [],
            "final_answer": {
                "status": "infeasible" if infeasible else "route_complete",
                "infeasible": infeasible,
                "reasoning_brief": "mock route",
            },
        }
    )
