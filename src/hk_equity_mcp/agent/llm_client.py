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
        stage: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Return {content, usage:{input_tokens,output_tokens,total_tokens}, latency_ms}.

        ``stage`` is optional observability for budgeted wrappers; base clients ignore it.
        """
        raise NotImplementedError


def _host_from_base_url(base_url: str) -> Optional[str]:
    raw = (base_url or "").strip()
    if not raw:
        return None
    if "://" in raw:
        raw = raw.split("://", 1)[1]
    raw = raw.split("/", 1)[0]
    return raw.lower() or None


def extract_final_text_content(message: Any) -> tuple[str, Any, Dict[str, Any]]:
    """Null-safe extraction of final assistant text.

    Never treats ``reasoning_content`` / hidden CoT as final JSON content.
    Supports string content and list-of-parts content shapes.
    """
    meta: Dict[str, Any] = {
        "message_was_null": message is None,
        "content_was_null": True,
        "message_keys": [],
        "reasoning_content_present": False,
        "reasoning_content_length": 0,
        "content_shape": None,
    }
    if message is None:
        return "", None, meta
    if not isinstance(message, dict):
        meta["content_shape"] = type(message).__name__
        return "", None, meta
    meta["message_keys"] = sorted(str(k) for k in message.keys())
    # Observability only — never copy hidden reasoning text into content.
    for rk in ("reasoning_content", "reasoning", "thinking"):
        if rk in message and message.get(rk) is not None:
            meta["reasoning_content_present"] = True
            val = message.get(rk)
            if isinstance(val, str):
                meta["reasoning_content_length"] = len(val)
            else:
                meta["reasoning_content_length"] = len(json.dumps(val, ensure_ascii=False))
            break

    raw_content = message.get("content")
    meta["content_was_null"] = raw_content is None
    if raw_content is None:
        # Some providers nest final text under content parts only; still ignore reasoning.
        return "", None, meta
    if isinstance(raw_content, str):
        meta["content_shape"] = "string"
        return raw_content, raw_content, meta
    if isinstance(raw_content, (dict, list)):
        # OpenAI-compatible multimodal / parts list.
        if isinstance(raw_content, list):
            parts: List[str] = []
            for item in raw_content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    for key in ("text", "content", "output_text"):
                        if isinstance(item.get(key), str):
                            parts.append(item[key])
                            break
            joined = "".join(parts)
            meta["content_shape"] = "list_parts"
            return joined, raw_content, meta
        meta["content_shape"] = "object"
        return json.dumps(raw_content, ensure_ascii=False), raw_content, meta
    meta["content_shape"] = type(raw_content).__name__
    return str(raw_content), raw_content, meta


def build_response_observability(
    *,
    stage: Optional[str],
    content: str,
    usage: Dict[str, Any],
    latency_ms: float,
    max_tokens_requested: int,
    provider: str,
    model: str,
    host: Optional[str],
    finish_reason: Optional[str],
    message_meta: Dict[str, Any],
    parse_result: Optional[str] = None,
    schema_error: Optional[str] = None,
) -> Dict[str, Any]:
    """Public, secret-free response metadata for planner traces."""
    return {
        "stage": stage,
        "content_present": bool((content or "").strip()),
        "content_length": len(content or ""),
        "finish_reason": finish_reason,
        "response_message_keys": list(message_meta.get("message_keys") or []),
        "reasoning_content_present": bool(message_meta.get("reasoning_content_present")),
        "reasoning_content_length": int(message_meta.get("reasoning_content_length") or 0),
        "content_shape": message_meta.get("content_shape"),
        "message_was_null": bool(message_meta.get("message_was_null")),
        "content_was_null": bool(message_meta.get("content_was_null")),
        "prompt_tokens": usage.get("input_tokens") or usage.get("prompt_tokens"),
        "input_tokens": usage.get("input_tokens") or usage.get("prompt_tokens"),
        "output_tokens": usage.get("output_tokens") or usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "max_tokens_requested": int(max_tokens_requested),
        "latency_ms": latency_ms,
        "provider": provider,
        "model": model,
        "host": host,
        "parse_result": parse_result,
        "schema_error": schema_error,
        # Explicitly never include api_key / secrets / reasoning text.
    }


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
        self.host = _host_from_base_url(self.base_url)
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
        stage: Optional[str] = None,
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
        choice0 = choices[0] if isinstance(choices[0], dict) else {}
        message = choice0.get("message")
        finish_reason = choice0.get("finish_reason") or choice0.get("stop_reason")
        content, raw_content, message_meta = extract_final_text_content(message)
        usage_raw = body.get("usage") or {}
        in_tok = int(usage_raw.get("prompt_tokens") or usage_raw.get("input_tokens") or 0)
        out_tok = int(
            usage_raw.get("completion_tokens") or usage_raw.get("output_tokens") or 0
        )
        usage = {
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "total_tokens": int(usage_raw.get("total_tokens") or (in_tok + out_tok)),
        }
        observability = build_response_observability(
            stage=stage,
            content=content,
            usage=usage,
            latency_ms=round(latency_ms, 3),
            max_tokens_requested=max_tokens,
            provider=self.provider,
            model=self.model,
            host=self.host,
            finish_reason=finish_reason,
            message_meta=message_meta,
        )
        return {
            "content": content,
            "raw_content": raw_content,
            "message_was_null": message is None,
            "content_was_null": raw_content is None,
            "finish_reason": finish_reason,
            "usage": usage,
            "latency_ms": round(latency_ms, 3),
            "provider": self.provider,
            "model": self.model,
            "host": self.host,
            "response_observability": observability,
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
        stage: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not self._queue:
            raise RuntimeError("ScriptedLLMClient exhausted.")
        content = self._queue.pop(0)
        usage = {
            "input_tokens": sum(len(m.get("content", "")) for m in messages) // 4,
            "output_tokens": len(content) // 4,
            "total_tokens": (
                sum(len(m.get("content", "")) for m in messages) + len(content)
            )
            // 4,
        }
        message_meta = {
            "message_was_null": False,
            "content_was_null": content is None,
            "message_keys": ["role", "content"],
            "reasoning_content_present": False,
            "reasoning_content_length": 0,
            "content_shape": "string",
        }
        obs = build_response_observability(
            stage=stage,
            content=str(content or ""),
            usage=usage,
            latency_ms=0.1,
            max_tokens_requested=max_tokens,
            provider=self.provider,
            model=self.model,
            host="local_mock",
            finish_reason="stop",
            message_meta=message_meta,
        )
        return {
            "content": content,
            "usage": usage,
            "latency_ms": 0.1,
            "provider": self.provider,
            "model": self.model,
            "host": "local_mock",
            "finish_reason": "stop",
            "response_observability": obs,
        }
