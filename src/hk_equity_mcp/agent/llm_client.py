"""LLM client interfaces for Operation-Visible / Flat agents.

Never log or print API key values. Pin LLM_MODEL explicitly (no model=auto).
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class LLMClient(ABC):
    provider: str
    model: str

    @abstractmethod
    def complete(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> Dict[str, Any]:
        """Return {content, usage:{input_tokens,output_tokens,total_tokens}, latency_ms}."""
        raise NotImplementedError


class OpenAICompatibleChatClient(LLMClient):
    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        timeout_sec: float = 120.0,
        json_mode: bool = True,
    ):
        self.base_url = (base_url or os.environ.get("LLM_BASE_URL") or "").rstrip("/")
        self.api_key = api_key or os.environ.get("LLM_API_KEY") or ""
        self.model = model or os.environ.get("LLM_MODEL") or ""
        self.provider = provider or os.environ.get("LLM_PROVIDER") or "openai_compatible"
        self.timeout_sec = timeout_sec
        self.json_mode = bool(json_mode)
        if not self.base_url:
            raise RuntimeError("Missing LLM_BASE_URL. Load ENV_FILE=.env.aliyun.")
        if not self.api_key:
            raise RuntimeError("Missing LLM_API_KEY. Do not fall back silently.")
        if not self.model:
            raise RuntimeError("Missing LLM_MODEL. Pin an explicit model (no auto).")
        if self.model.strip().lower() == "auto":
            raise RuntimeError("LLM_MODEL=auto is not allowed.")

    def complete(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
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
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise RuntimeError(
                f"LLM request failed with HTTP {exc.code} for model {self.model}."
            ) from exc
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"LLM request failed for model {self.model}: {type(exc).__name__}."
            ) from exc
        latency_ms = (time.perf_counter() - started) * 1000
        choices = body.get("choices") or []
        if not choices:
            raise RuntimeError("LLM response missing choices.")
        # message may be explicitly null; dict.get("message", {}) still returns None.
        message = choices[0].get("message")
        if not isinstance(message, dict):
            message = {}
        content = message.get("content") or ""
        if not isinstance(content, str):
            content = str(content)
        usage = body.get("usage") or {}
        in_tok = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
        out_tok = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
        return {
            "content": content,
            "usage": {
                "input_tokens": in_tok,
                "output_tokens": out_tok,
                "total_tokens": int(usage.get("total_tokens") or (in_tok + out_tok)),
            },
            "latency_ms": round(latency_ms, 3),
            "provider": self.provider,
            "model": self.model,
        }


class ScriptedLLMClient(LLMClient):
    """Deterministic mock LLM for unit/integration tests (no network)."""

    def __init__(self, scripted_contents: List[str], model: str = "mock-llm"):
        self.provider = "mock"
        self.model = model
        self._queue = list(scripted_contents)

    def complete(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> Dict[str, Any]:
        if not self._queue:
            raise RuntimeError("ScriptedLLMClient exhausted.")
        content = self._queue.pop(0)
        return {
            "content": content,
            "usage": {
                "input_tokens": sum(len(m.get("content", "")) for m in messages) // 4,
                "output_tokens": len(content) // 4,
                "total_tokens": (
                    sum(len(m.get("content", "")) for m in messages) + len(content)
                )
                // 4,
            },
            "latency_ms": 0.1,
            "provider": self.provider,
            "model": self.model,
        }
