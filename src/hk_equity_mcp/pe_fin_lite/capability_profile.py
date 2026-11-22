"""Static capability profile loader (not an LLM module)."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

DEFAULT_PROFILE_REL = "configs/capabilities/hk155_capability_profile_v1.yaml"

# Fail closed if a profile smuggles evaluation leakage.
_BANNED_PROFILE_PATTERNS = (
    r"\bgold_operations?\b",
    r"\breference_answer\b",
    r'"task_id"\s*:',
    r"\btask_id\s*:",
    r"\bfin_cf_\d+\b",
    r"\bxd_(dev|ht)_\d+\b",
    r"\bmkt_(dev|ht)_\d+\b",
    r"\bnews_(dev|ht)_\d+\b",
    r"\boperation_id\s*:",
)


class CapabilityProfileError(ValueError):
    pass


def profile_path(root: Path, rel: Optional[str] = None) -> Path:
    return Path(root) / (rel or DEFAULT_PROFILE_REL)


def load_capability_profile(
    root: Path,
    *,
    path: Optional[Path] = None,
) -> Dict[str, Any]:
    cfg_path = path or profile_path(root)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise CapabilityProfileError("capability profile must be a mapping")
    validate_capability_profile(raw)
    return raw


def validate_capability_profile(profile: Dict[str, Any]) -> None:
    required = (
        "profile_id",
        "domain",
        "capability_summary",
        "source_types",
        "known_limitations",
        "composite_workflows",
    )
    for key in required:
        if key not in profile:
            raise CapabilityProfileError(f"missing required profile field: {key}")
    if profile.get("company_generic") is False:
        raise CapabilityProfileError("profile must be company-generic")
    blob = json.dumps(profile, ensure_ascii=False).lower()
    for pat in _BANNED_PROFILE_PATTERNS:
        if re.search(pat, blob, flags=re.IGNORECASE):
            raise CapabilityProfileError(
                f"capability profile contains banned pattern: {pat}"
            )
    # Explicit company whitelist keys forbidden.
    for banned_key in ("issuer_whitelist", "company_whitelist", "task_whitelist"):
        if banned_key in profile:
            raise CapabilityProfileError(f"banned profile key: {banned_key}")


def capability_profile_hash(profile: Dict[str, Any]) -> str:
    blob = json.dumps(profile, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def profile_for_planner(profile: Dict[str, Any]) -> Dict[str, Any]:
    """Compact view injected into the planner prompt (still static)."""
    from hk_equity_mcp.pe_fin_lite.output_vocabulary import vocabulary_for_profile

    source_types = profile.get("source_types") or {}
    compact_sources = {
        name: {
            "summary": (meta or {}).get("summary"),
            "examples": list((meta or {}).get("examples") or []),
        }
        for name, meta in source_types.items()
        if isinstance(meta, dict)
    }
    workflows = []
    for wf in profile.get("composite_workflows") or []:
        if not isinstance(wf, dict):
            continue
        workflows.append(
            {
                "workflow_id": wf.get("workflow_id"),
                "domain": wf.get("domain"),
                "description": wf.get("description"),
            }
        )
    vocab = profile.get("canonical_output_vocabulary") or vocabulary_for_profile()
    scope = profile.get("scope_constraints_contract") or {
        "version": "pe_fin_lite_scope_constraints_v1",
        "keys": [
            "entity",
            "period",
            "trade_date",
            "from_date",
            "to_date",
            "metric_basis",
            "adjustment",
            "title_keyword",
        ],
    }
    return {
        "profile_id": profile.get("profile_id"),
        "domain": profile.get("domain"),
        "capability_summary": profile.get("capability_summary"),
        "source_types": compact_sources,
        "known_limitations": list(profile.get("known_limitations") or []),
        "composite_workflows": workflows,
        "canonical_output_vocabulary": vocab,
        "scope_constraints_contract": scope,
    }


def disclosure_workflow_id(profile: Dict[str, Any]) -> Optional[str]:
    for wf in profile.get("composite_workflows") or []:
        if isinstance(wf, dict) and wf.get("workflow_id"):
            if "disclosure" in str(wf.get("domain") or "").lower() or "disclosure" in str(
                wf.get("workflow_id") or ""
            ):
                return str(wf["workflow_id"])
    return None
