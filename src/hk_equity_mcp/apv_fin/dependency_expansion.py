"""Capability workflow expansion driven by capability contracts (v1.2)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from hk_equity_mcp.apv_fin.capability_ontology import DISCLOSURE_NARRATIVE_WORKFLOW


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
        # List-like ops map to disclosure.list
        if "list" in cur or cur == "news.official_announcement":
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
    blob = f"{objective or ''} {query_hint or ''} {required_capability or ''}".lower()
    if "narrative" in blob or "non-ifrs" in blob or "official" in blob:
        oid = (operation_id or "").lower()
        return "list" in oid and ("hkex" in oid or "disclosure" in oid or "announcement" in oid)
    return False
