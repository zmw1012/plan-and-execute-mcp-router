"""Transient LLM transport retry for RePlan generation (at most once)."""

from __future__ import annotations

import http.client
import socket
import ssl
import urllib.error
from typing import Any, Dict, List, Optional, Tuple

from hk_equity_mcp.agent.llm_client import LLMClient

_RETRYABLE_HTTP = frozenset({429, 500, 502, 503, 504})
_RETRYABLE_DISCONNECT = (
    http.client.RemoteDisconnected,
    http.client.IncompleteRead,
    ConnectionResetError,
    ConnectionAbortedError,
    BrokenPipeError,
)
_SSL_INTERRUPT_MARKERS = (
    "eof",
    "unexpected_eof",
    "connection reset",
    "broken pipe",
    "connection aborted",
    "syscall",
)
_RETRYABLE_TYPE_NAMES = (
    "TimeoutError",
    "URLError",
    "RemoteDisconnected",
    "IncompleteRead",
    "ConnectionResetError",
    "ConnectionAbortedError",
    "BrokenPipeError",
    "SSLError",
)


def _is_ssl_interrupt(exc: BaseException) -> bool:
    if not isinstance(exc, ssl.SSLError):
        return False
    text = str(exc).lower()
    return any(m in text for m in _SSL_INTERRUPT_MARKERS)


def is_transient_llm_transport_error(exc: BaseException) -> bool:
    """Allow at most one retry for transient transport / overload failures."""
    if isinstance(exc, (TimeoutError, socket.timeout)):
        return True
    if isinstance(exc, _RETRYABLE_DISCONNECT):
        return True
    if _is_ssl_interrupt(exc):
        return True
    if isinstance(exc, urllib.error.HTTPError):
        return int(exc.code) in _RETRYABLE_HTTP
    if isinstance(exc, urllib.error.URLError):
        return True
    # Wrapped RuntimeError from OpenAICompatibleChatClient
    if isinstance(exc, RuntimeError):
        text = str(exc)
        if any(f"HTTP {code}" in text for code in _RETRYABLE_HTTP):
            return True
        # Non-429 HTTP 4xx must not retry.
        if "HTTP 4" in text and "HTTP 429" not in text:
            return False
        if any(name in text for name in _RETRYABLE_TYPE_NAMES):
            # SSLError without interrupt markers: still check cause
            if "SSLError" in text and "URLError" not in text:
                cause = exc.__cause__
                if isinstance(cause, BaseException) and cause is not exc:
                    return is_transient_llm_transport_error(cause)
                return _is_ssl_interrupt(exc) or any(
                    m in text.lower() for m in _SSL_INTERRUPT_MARKERS
                )
            return True
        cause = exc.__cause__
        if isinstance(cause, BaseException) and cause is not exc:
            return is_transient_llm_transport_error(cause)
        return False
    cause = getattr(exc, "__cause__", None)
    if isinstance(cause, BaseException) and cause is not exc:
        return is_transient_llm_transport_error(cause)
    return False


def complete_with_transient_retry(
    llm: LLMClient,
    messages: List[Dict[str, str]],
    *,
    temperature: float,
    max_tokens: int,
    stage: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Call LLM once; on transient transport error, retry once under a new stage.

    Both attempts go through the budgeted client when ``llm`` is budgeted, so
    retries count toward the LLM call budget.
    """
    meta: Dict[str, Any] = {
        "stage": stage,
        "transport_retry_attempted": False,
        "transport_retry_succeeded": False,
        "transport_error_type": None,
    }
    try:
        return llm.complete(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stage=stage,
        ), meta
    except Exception as exc:  # noqa: BLE001
        if not is_transient_llm_transport_error(exc):
            raise
        meta["transport_retry_attempted"] = True
        meta["transport_error_type"] = type(exc).__name__
        retry_stage = f"{stage}_transport_retry"
        meta["retry_stage"] = retry_stage
        try:
            out = llm.complete(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stage=retry_stage,
            )
            meta["transport_retry_succeeded"] = True
            return out, meta
        except Exception as exc2:  # noqa: BLE001
            meta["transport_retry_succeeded"] = False
            meta["transport_retry_error_type"] = type(exc2).__name__
            # Controlled failure for caller — second failure
            raise RuntimeError(
                f"LLM transport retry failed for stage={stage}: "
                f"{type(exc2).__name__}."
            ) from exc2
