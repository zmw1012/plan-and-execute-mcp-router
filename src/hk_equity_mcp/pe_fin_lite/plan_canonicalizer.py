"""Versioned structural plan canonicalizer (unambiguous rules only).

Must run before schema validation. Never invents answers, operations,
required_outputs, entities, periods, or non-root depends_on.
"""

from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

CANONICALIZER_VERSION = "pe_fin_lite_plan_structural_canonicalizer_v1"

# Domain aliases that are structurally unambiguous given step type.
REASON_DOMAIN_ALIASES = {
    "analysis": "general",
    "reasoning": "general",
}
COMPUTE_DOMAIN_ALIASES = {
    "calculation": "compute",
}


def plan_canonicalizer_hash() -> str:
    payload = {
        "version": CANONICALIZER_VERSION,
        "reason_domain_aliases": REASON_DOMAIN_ALIASES,
        "compute_domain_aliases": COMPUTE_DOMAIN_ALIASES,
        "rules": [
            "first_step_missing_depends_on_to_empty_array",
            "reason_analysis_or_reasoning_domain_to_general",
            "compute_calculation_domain_to_compute",
            "json_text_whitespace_fence_unicode_quote_trailing_comma",
            "runtime_owned_fields_stripped_elsewhere",
        ],
    }
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def normalize_json_text(text: Optional[str]) -> Tuple[str, List[str]]:
    """Generic text normalization before JSON parse. No fragment splicing."""
    applied: List[str] = []
    if text is None:
        return "", applied
    raw = str(text)
    if not raw.strip():
        return raw, applied

    stripped = raw.strip()
    if stripped != raw:
        applied.append("strip_whitespace")
        raw = stripped

    fence = re.search(r"```(?:json)?\s*(\{[\s\S]*\})\s*```", raw, flags=re.IGNORECASE)
    if fence:
        raw = fence.group(1).strip()
        applied.append("strip_markdown_fence")

    # Unicode / smart quotes → ASCII
    translated = (
        raw.replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\ufeff", "")
    )
    if translated != raw:
        applied.append("unicode_quotes_to_ascii")
        raw = translated

    # Trailing commas before } or ] (structural only; does not invent tokens).
    cleaned = re.sub(r",\s*([}\]])", r"\1", raw)
    if cleaned != raw:
        applied.append("remove_trailing_commas")
        raw = cleaned

    return raw, applied


def canonicalize_plan_payload(
    payload: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[str]]:
    """Apply unambiguous structural rules. Does not invent semantic content."""
    if not isinstance(payload, dict):
        return payload, []
    out = deepcopy(payload)
    applied: List[str] = []
    steps = out.get("steps")
    if not isinstance(steps, list) or not steps:
        return out, applied

    first = steps[0]
    if isinstance(first, dict) and "depends_on" not in first:
        first["depends_on"] = []
        applied.append("first_step_missing_depends_on_to_empty_array")

    for step in steps:
        if not isinstance(step, dict):
            continue
        stype = str(step.get("type") or "")
        domain = step.get("domain")
        if not isinstance(domain, str):
            continue
        dlow = domain.strip().lower()
        if stype == "reason" and dlow in REASON_DOMAIN_ALIASES:
            step["domain"] = REASON_DOMAIN_ALIASES[dlow]
            applied.append(f"reason_domain_{dlow}_to_general")
        elif stype == "compute" and dlow in COMPUTE_DOMAIN_ALIASES:
            step["domain"] = COMPUTE_DOMAIN_ALIASES[dlow]
            applied.append(f"compute_domain_{dlow}_to_compute")

    return out, applied


def describe_canonicalizer() -> Dict[str, Any]:
    return {
        "version": CANONICALIZER_VERSION,
        "hash": plan_canonicalizer_hash(),
        "allowed_rules": [
            "first_step_missing_depends_on → []",
            "reason domain analysis|reasoning → general",
            "compute domain calculation → compute",
            "JSON text: whitespace / fence / unicode quotes / trailing commas",
            "runtime-owned schema/version fields stripped/re-injected by runtime",
        ],
        "forbidden": [
            "non-first step missing depends_on auto-fill",
            "missing domain without unique type mapping",
            "missing required_outputs invention",
            "missing compute_spec invention",
            "overlength semantic truncation",
            "truncated JSON fragment splicing",
            "gold/task-id driven inference",
        ],
    }
