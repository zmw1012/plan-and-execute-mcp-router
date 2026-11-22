"""Capability workflow expansion driven by capability contracts (v1.2/v1.4)."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Optional, Sequence

from hk_equity_mcp.apv_fin.capability_ontology import DISCLOSURE_NARRATIVE_WORKFLOW

DEPENDENCY_PROGRESSION_VERSION = "apv_fin_dependency_progression_v1_4_1"


def dependency_progression_hash() -> str:
    blob = json.dumps(
        {
            "version": DEPENDENCY_PROGRESSION_VERSION,
            "workflow": list(DISCLOSURE_NARRATIVE_WORKFLOW),
            "policy": [
                "prefix_list_when_local_key_missing",
                "apply_after_capability_compiler",
                "requires_provides_graph",
                "ledger_availability_gate",
                "no_fabricated_local_key",
            ],
        },
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def is_narrative_capability(capability: Optional[str]) -> bool:
    return str(capability or "") in {
        "financial.official_narrative_evidence",
        "news.official_announcement",
    }


def extract_provided_values(
    projection: Dict[str, Any],
    *,
    provides: Sequence[str],
) -> Dict[str, Any]:
    """Pull concrete dependency values from a projected/adapted observation."""
    out: Dict[str, Any] = {}
    provides_set = {str(p) for p in provides}
    proj = projection.get("projected") or projection.get("adapted") or projection
    rows = proj.get("rows") or proj.get("documents") or proj.get("snippets") or []
    if "local_key" in provides_set:
        keys = []
        if proj.get("local_key"):
            keys.append(proj["local_key"])
        keys += list(proj.get("local_keys") or [])
        for row in rows:
            if isinstance(row, dict) and row.get("local_key"):
                keys.append(row["local_key"])
        if keys:
            out["local_key"] = keys[0]
            out["local_keys"] = list(dict.fromkeys(keys))
    if "news_id" in provides_set:
        if proj.get("news_id"):
            out["news_id"] = proj["news_id"]
        else:
            for row in rows:
                if isinstance(row, dict) and row.get("news_id"):
                    out["news_id"] = row["news_id"]
                    break
    if "release_date" in provides_set:
        for row in rows:
            if isinstance(row, dict) and row.get("release_date"):
                out["release_date"] = row["release_date"]
                break
    if "page_number" in provides_set:
        if proj.get("page_number") is not None:
            out["page_number"] = proj.get("page_number")
        else:
            for row in rows:
                if isinstance(row, dict) and row.get("page_number") is not None:
                    out["page_number"] = row.get("page_number")
                    break
                if isinstance(row, dict) and row.get("page") is not None:
                    out["page_number"] = row.get("page")
                    break
    return out


def next_disclosure_capability(
    *,
    current_capability: Optional[str],
    target_capability: Optional[str],
) -> Optional[str]:
    """Next capability in the generic disclosure narrative workflow."""
    if not is_narrative_capability(target_capability) and target_capability not in {
        "disclosure.keyword_search",
        "disclosure.page_read",
    }:
        # Still allow expansion when current is list toward narrative target.
        if not is_narrative_capability(target_capability):
            target_capability = "financial.official_narrative_evidence"
    workflow = list(DISCLOSURE_NARRATIVE_WORKFLOW)
    cur = str(current_capability or "disclosure.list")
    if cur not in workflow:
        # List-like / narrative-target retrieve that already listed docs → treat as list.
        if (
            "list" in cur
            or cur == "news.official_announcement"
            or is_narrative_capability(cur)
        ):
            cur = "disclosure.list"
        else:
            return workflow[0]
    idx = workflow.index(cur)
    if idx + 1 >= len(workflow):
        return None
    return workflow[idx + 1]


def next_capability_operations(
    *,
    contracts: Dict[str, Dict[str, Any]],
    next_capability: str,
    available: Dict[str, Any],
    already_invoked: Sequence[str],
) -> List[Dict[str, Any]]:
    """Suggest runtime ops whose contract capabilities include next_capability
    and whose requires_fields ⊆ available keys.
    """
    invoked = set(already_invoked)
    available_keys = set(available.keys())
    out: List[Dict[str, Any]] = []
    for oid, row in contracts.items():
        if oid in invoked:
            continue
        if str(row.get("runtime_availability") or "") != "runtime":
            continue
        caps = set(row.get("capabilities") or [])
        if next_capability not in caps:
            continue
        requires = [str(r) for r in (row.get("requires_fields") or [])]
        soft = {"issuer_identity", "interface_name"}
        hard = [r for r in requires if r not in soft]
        if hard and not set(hard).issubset(available_keys):
            continue
        out.append(
            {
                "operation_id": oid,
                "capability": next_capability,
                "requires": hard,
                "provides": list(row.get("provides_fields") or []),
                "contract": row,
            }
        )
    # Prefer download → parse → keyword → pages by capability order.
    priority = {c: i for i, c in enumerate(DISCLOSURE_NARRATIVE_WORKFLOW)}
    out.sort(key=lambda c: (priority.get(c["capability"], 99), c["operation_id"]))
    return out


def build_followup_retrieve_step(
    *,
    parent_step_id: str,
    next_capability: str,
    operation_hint: Optional[Dict[str, Any]] = None,
    available: Optional[Dict[str, Any]] = None,
    index: int = 1,
    required_outputs: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Model-owned follow-up retrieve step for disclosure capability chain."""
    available = available or {}
    outputs = list(required_outputs or [])
    if next_capability in {
        "disclosure.keyword_search",
        "disclosure.page_read",
        "disclosure.parse",
    }:
        if "snippet" not in outputs:
            outputs.append("snippet")
        if "local_key" not in outputs:
            outputs.append("local_key")
    elif next_capability == "disclosure.download":
        if "local_key" not in outputs:
            outputs.append("local_key")
    slots = {}
    if available.get("local_key"):
        slots["local_key"] = available["local_key"]
    if available.get("page_number") is not None:
        slots["page_number"] = available["page_number"]
    if available.get("news_id"):
        slots["news_id"] = available["news_id"]
    if next_capability == "disclosure.keyword_search" and "keywords" not in slots:
        slots["keywords"] = "non-IFRS OR adjusted profit OR attributable"
    hint_op = (operation_hint or {}).get("operation_id") or next_capability
    return {
        "step_id": f"{parent_step_id}_cap_{index}",
        "step_type": "retrieve",
        "objective": f"Advance disclosure workflow via {next_capability}."[:180],
        "depends_on": [parent_step_id],
        "mandatory": True,
        "success_criteria": f"Obtain {', '.join(outputs) or next_capability}"[:160],
        "retrieval_query_hint": f"{hint_op} local_key={available.get('local_key')}"[:160],
        "required_capability": next_capability,
        "required_outputs": outputs,
        "argument_slots": slots,
        "source_preference": ["official_disclosure"],
        "fallback_capabilities": [],
        "compute_spec": None,
        "notes": f"auto_dependency_expansion:{next_capability}",
    }


def list_only_should_not_resolve_narrative(
    *,
    operation_id: str,
    objective: str = "",
    query_hint: Optional[str] = None,
    required_capability: Optional[str] = None,
) -> bool:
    """List identifies documents; narrative evidence needs download/parse/search."""
    if is_narrative_capability(required_capability):
        oid = (operation_id or "").lower()
        return "list" in oid
    if str(required_capability or "") == "disclosure.list":
        return True
    blob = f"{objective or ''} {query_hint or ''} {required_capability or ''}".lower()
    if "narrative" in blob or "non-ifrs" in blob or "official" in blob:
        oid = (operation_id or "").lower()
        return "list" in oid and ("hkex" in oid or "disclosure" in oid or "announcement" in oid)
    return False


def capability_requires_local_key(
    *,
    contracts: Dict[str, Dict[str, Any]],
    capability: Optional[str],
) -> bool:
    """True when runtime providers for capability require local_key."""
    cap = str(capability or "")
    if not cap:
        return False
    for row in contracts.values():
        if str(row.get("runtime_availability") or "") != "runtime":
            continue
        caps = set(row.get("capabilities") or [])
        if cap not in caps:
            continue
        requires = {str(r) for r in (row.get("requires_fields") or [])}
        req_args = {str(r) for r in (row.get("required_arguments") or [])}
        if "local_key" in requires or "local_key" in req_args:
            return True
    return False


def available_deps_from_ledger(ledger: Any) -> Dict[str, Any]:
    """Extract disclosure identity / parse provenance already present in the ledger."""
    out: Dict[str, Any] = {}
    items = []
    if ledger is None:
        return out
    if hasattr(ledger, "items"):
        items = list(ledger.items or [])
    elif isinstance(ledger, list):
        items = ledger
    for item in items:
        payload = item.to_dict() if hasattr(item, "to_dict") else item
        if not isinstance(payload, dict):
            continue
        blobs: List[Any] = [
            payload.get("raw_value"),
            payload.get("normalized_value"),
            payload.get("extras"),
            payload,
        ]
        for blob in blobs:
            if not isinstance(blob, dict):
                continue
            for key in (
                "local_key",
                "news_id",
                "release_date",
                "page_number",
                "stock_code",
                "title",
            ):
                if blob.get(key) is not None and key not in out:
                    out[key] = blob.get(key)
            if blob.get("documents") or blob.get("document_identified"):
                out.setdefault("document_identified", True)
            if (
                blob.get("parsed")
                or blob.get("pages")
                or blob.get("snippets")
                or blob.get("textual_content")
                or str(payload.get("claim") or "").startswith("parsed")
            ):
                out["parsed_document"] = True
            adapted = blob.get("adapted") if isinstance(blob.get("adapted"), dict) else None
            if adapted:
                for key in ("local_key", "news_id", "release_date", "page_number"):
                    if adapted.get(key) is not None and key not in out:
                        out[key] = adapted.get(key)
                if adapted.get("documents") or adapted.get("document_identified"):
                    out.setdefault("document_identified", True)
                if adapted.get("snippets") or adapted.get("pages"):
                    out["parsed_document"] = True
    return out


def disclosure_prefix_needed(
    *,
    required_capability: Optional[str],
    argument_slots: Optional[Dict[str, Any]] = None,
    available_deps: Optional[Dict[str, Any]] = None,
    contracts: Optional[Dict[str, Dict[str, Any]]] = None,
) -> bool:
    """Narrative/keyword/page path missing ledger identity → list first (no Binder LLM)."""
    cap = str(required_capability or "")
    slots = dict(argument_slots or {})
    available = dict(available_deps or {})
    if slots.get("local_key") or available.get("local_key"):
        return False
    if cap == "disclosure.list":
        return False
    if is_narrative_capability(cap):
        # Prefix when ledger lacks document identity and parse provenance.
        if (
            available.get("local_key")
            or available.get("news_id")
            or available.get("parsed_document")
        ):
            return False
        return True
    if cap in {
        "disclosure.keyword_search",
        "disclosure.page_read",
        "disclosure.parse",
        "disclosure.download",
    }:
        if available.get("local_key"):
            return False
        if contracts is None:
            return True
        return capability_requires_local_key(contracts=contracts, capability=cap)
    return False


def rewrite_step_to_disclosure_list(
    step: Dict[str, Any],
    *,
    target_capability: Optional[str] = None,
) -> Dict[str, Any]:
    """Deterministically retarget a retrieve step to disclosure.list."""
    out = dict(step)
    target = target_capability or step.get("required_capability")
    out["required_capability"] = "disclosure.list"
    out["required_outputs"] = ["local_key", "title", "release_date"]
    out["source_preference"] = ["official_disclosure"]
    out["fallback_capabilities"] = [
        c
        for c in (
            "disclosure.download",
            "disclosure.parse",
            "disclosure.keyword_search",
            "disclosure.page_read",
            "financial.official_narrative_evidence",
        )
        if c != "disclosure.list"
    ]
    notes = str(out.get("notes") or "")
    tag = f"disclosure_prefix_for:{target}"
    out["notes"] = f"{notes};{tag}".strip(";")
    out["semantic_output_requirements"] = list(
        out.get("semantic_output_requirements") or out.get("required_outputs") or []
    )
    return out
