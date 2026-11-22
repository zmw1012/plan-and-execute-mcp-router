"""Secret / sensitive-path redaction for Real MCP traces."""

from __future__ import annotations

import re
from typing import Any

_SENSITIVE_KEY = re.compile(
    r"(api[_-]?key|token|password|secret|authorization|bearer|xq_a_token|ftshare)",
    re.I,
)
_HOME_PATH = re.compile(r"(/Users/[^/\s]+|/home/[^/\s]+)")


def redact_text(text: str) -> str:
    if not text:
        return text
    out = _HOME_PATH.sub("<HOME>", text)
    out = re.sub(
        r"(?i)(api[_-]?key|token|password|secret|authorization)\s*[:=]\s*\S+",
        r"\1=<REDACTED>",
        out,
    )
    return out


def redact_obj(value: Any) -> Any:
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            if _SENSITIVE_KEY.search(str(k)):
                out[k] = "<REDACTED>"
            else:
                out[k] = redact_obj(v)
        return out
    if isinstance(value, list):
        return [redact_obj(v) for v in value]
    if isinstance(value, str):
        return redact_text(value)
    return value
