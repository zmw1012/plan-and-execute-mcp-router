"""Canonical HK issuer_code acceptance (company-generic; no task-id branches).

Rejects statement labels / Chinese table names (e.g. 利润表) that must never
be written into issuer_code.
"""

from __future__ import annotations

import re
from typing import Any, Optional

# Digits with optional .HK suffix.
_CODE_FULL = re.compile(r"^(\d{1,5})(?:\.HK)?$", re.IGNORECASE)
_HAS_CJK = re.compile(r"[\u4e00-\u9fff]")
_STATEMENT_LABEL = re.compile(
    r"(利润表|资产负债表|现金流量表|合并|income\s*statement|balance\s*sheet|"
    r"cash\s*flow|statement|indicator|表$)",
    re.IGNORECASE,
)


def canonicalize_hk_issuer_code(raw: Any) -> Optional[str]:
    """Return normalized 5-digit HK code or None if not a valid issuer code."""
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    if _HAS_CJK.search(text) or _STATEMENT_LABEL.search(text):
        return None
    m = _CODE_FULL.fullmatch(text)
    if m:
        return m.group(1).zfill(5)
    # Allow bare punctuation-stripped digit codes only when the whole token is digits(+optional .HK).
    stripped = text.upper().replace(".HK", "")
    if stripped.isdigit() and 1 <= len(stripped) <= 5:
        return stripped.zfill(5)
    return None


def issuer_codes_equivalent(a: Any, b: Any) -> bool:
    ca = canonicalize_hk_issuer_code(a)
    cb = canonicalize_hk_issuer_code(b)
    if ca is not None and cb is not None:
        return ca == cb
    # If either side is non-canonical, fall back to strict string equality of stripped forms.
    if a in (None, "") or b in (None, ""):
        return a in (None, "") and b in (None, "")
    return str(a).strip() == str(b).strip()
