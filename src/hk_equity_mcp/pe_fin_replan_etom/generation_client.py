"""Fail-closed OpenAI-compatible chat client for pe_fin_replan_etom live runs.

Gates against the proposed relay provider profile (openai_compatible /
api.zhizengzeng.com / deepseek-v4.1-flash). Does NOT use
``hk_equity_mcp.etom.EXPECTED_LLM_PROVIDER`` (historical aliyun baseline).

Never logs or returns API keys.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from hk_equity_mcp.etom.embeddings import scrub_secret, urlopen_with_deadline
from hk_equity_mcp.pe_fin_replan_etom.generation_provider import (
    EXPECTED_HOST,
    EXPECTED_MODEL,
    EXPECTED_PROVIDER,
    GenerationProviderIdentityError,
    assert_live_generation_identity,
    host_from_base_url,
)


class PeFinReplanEtomChatClient:
    """Urllib OpenAI-compatible chat completions client with proposed-method gate."""

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
        self.provider = (provider or os.environ.get("LLM_PROVIDER") or "").strip()
        self.model = (model or os.environ.get("LLM_MODEL") or "").strip()
        self.base_url = (base_url or os.environ.get("LLM_BASE_URL") or "").rstrip("/")
        self.api_key = api_key or os.environ.get("LLM_API_KEY") or ""
        self.timeout_sec = timeout_sec
        self.max_retries = max_retries
        self.json_mode = json_mode
        self.retries = 0
        host = host_from_base_url(self.base_url)
        assert_live_generation_identity(
            provider=self.provider, model=self.model, host=host
        )
        if not self.base_url or not self.api_key:
            raise RuntimeError(
                "Missing generation endpoint configuration. "
                "Load .env.relay-v41. The key is not printed."
            )
        if host != EXPECTED_HOST:
            raise GenerationProviderIdentityError(
                f"base_url host {host!r} != {EXPECTED_HOST!r}"
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
            except GenerationProviderIdentityError:
                raise
            except TimeoutError:
                if attempt >= self.max_retries:
                    raise RuntimeError(
                        f"LLM request exceeded {self.timeout_sec:.0f}s for model {self.model}."
                    ) from None
                self.retries += 1
            except urllib.error.HTTPError as exc:
                if exc.code not in {429, 500, 502, 503, 504} or attempt >= self.max_retries:
                    raise RuntimeError(
                        scrub_secret(
                            f"LLM request failed with HTTP {exc.code} for model {self.model}."
                        )
                    ) from None
                self.retries += 1
            except Exception as exc:  # noqa: BLE001
                if attempt >= self.max_retries:
                    raise RuntimeError(
                        scrub_secret(
                            f"LLM request failed ({type(exc).__name__}) for model {self.model}."
                        )
                    ) from None
                self.retries += 1
        raise RuntimeError(f"LLM request failed for model {self.model}.")

    def _parse(self, body: Dict[str, Any], latency_ms: float) -> Dict[str, Any]:
        choices = body.get("choices") or []
        if not choices:
            raise RuntimeError("LLM response missing choices.")
        message = (choices[0] or {}).get("message") or {}
        content = message.get("content")
        if content is None:
            raise RuntimeError("LLM response missing message content.")
        usage = body.get("usage") or {}
        return {
            "content": content if isinstance(content, str) else json.dumps(content),
            "provider": self.provider,
            "model": self.model,
            "host": EXPECTED_HOST,
            "latency_ms": latency_ms,
            "usage": {
                "prompt_tokens": int(usage.get("prompt_tokens") or 0),
                "completion_tokens": int(usage.get("completion_tokens") or 0),
                "total_tokens": int(usage.get("total_tokens") or 0),
            },
            # Never echo keys / Authorization headers.
            "expected_provider": EXPECTED_PROVIDER,
            "expected_model": EXPECTED_MODEL,
        }
