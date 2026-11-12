"""Sina HK equity symbol normalizer for aktools.stock_hk_daily only.

AKShare stock_hk_daily expects five-digit numeric codes (e.g. 09988), not 09988.HK.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Dict, Optional

SINA_HK_SYMBOL_NORMALIZER_VERSION = "normalize_sina_hk_symbol_v1"

_DOT_HK = re.compile(r"\.hk$", re.IGNORECASE)
_HK_PREFIX = re.compile(r"^hk", re.IGNORECASE)


class SinaHkSymbolError(ValueError):
    """Fail-closed invalid Sina HK symbol format."""

    failure_class = "invalid_symbol_format"


def sina_symbol_normalizer_hash() -> str:
    blob = json.dumps(
        {
            "version": SINA_HK_SYMBOL_NORMALIZER_VERSION,
            "rules": [
                "strip_case_insensitive_dot_hk",
                "strip_hk_prefix",
                "digits_only",
                "zfill_5",
                "reject_empty_non_digit_gt5",
                "no_company_task_gold_lookup",
                "no_llm",
                "applies_only_to_aktools_stock_hk_daily",
            ],
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def normalize_sina_hk_symbol_v1(raw: Any) -> Dict[str, Any]:
    """Normalize binder/catalog aliases to Sina five-digit numeric invoke format.

    Allowed examples: 9988.HK, 09988.HK, hk09988, 9988, 09988 → 09988.
    """
    original = None if raw is None else str(raw).strip()
    if original is None or original == "":
        raise SinaHkSymbolError("invalid_symbol_format:empty")

    text = original
    text = _DOT_HK.sub("", text)
    text = text.strip()
    text = _HK_PREFIX.sub("", text)
    text = text.strip()
    # Reject remaining dots/letters/spaces
    if not text.isdigit():
        raise SinaHkSymbolError(
            f"invalid_symbol_format:non_numeric:{original!r}"
        )
    if len(text) > 5:
        raise SinaHkSymbolError(
            f"invalid_symbol_format:too_many_digits:{original!r}"
        )
    normalized = text.zfill(5)
    return {
        "original_symbol": original,
        "normalized_symbol": normalized,
        "symbol_format_original": original,
        "symbol_format_normalized": normalized,
        "symbol_normalizer_version": SINA_HK_SYMBOL_NORMALIZER_VERSION,
        "symbol_normalizer_hash": sina_symbol_normalizer_hash(),
        "argument_transform_applied": True,
    }
