"""Capability dependency expansion driven by metadata requires/provides."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Set


NARRATIVE_EVIDENCE_HINTS = (
    "disclosure",
    "announcement",
    "filing",
    "narrative",
    "annual report",
    "interim",
    "pdf",
    "hkex",
)


def is_narrative_retrieve_objective(objective: str, query_hint: Optional[str] = None) -> bool:
    blob = f"{objective or ''} {query_hint or ''}".lower()
    return any(h in blob for h in NARRATIVE_EVIDENCE_HINTS)


def extract_provided_values(
    projection: Dict[str, Any],
    *,
    provides: Sequence[str],
) -> Dict[str, Any]:
    """Pull concrete dependency values (e.g. local_key) from a projected observation."""
    out: Dict[str, Any] = {}
    provides_set = {str(p) for p in provides}
    rows = (projection.get("projected") or projection).get("rows") or []
    if "local_key" in provides_set:
        keys = []
        for row in rows:
            if isinstance(row, dict) and row.get("local_key"):
                keys.append(row["local_key"])
        keys += list((projection.get("projected") or {}).get("local_keys") or [])
        if keys:
            out["local_key"] = keys[0]
            out["local_keys"] = list(dict.fromkeys(keys))
    if "news_id" in provides_set:
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
        for row in rows:
            if isinstance(row, dict) and row.get("page_number") is not None:
                out["page_number"] = row.get("page_number")
                break
            if isinstance(row, dict) and row.get("page") is not None:
                out["page_number"] = row.get("page")
                break
    return out


def next_capability_operations(
    *,
    metadata_by_id: Dict[str, Dict[str, Any]],
    available: Dict[str, Any],
    already_invoked: Sequence[str],
    domain_pref: str = "financial",
) -> List[Dict[str, Any]]:
    """Suggest next ops whose requires ⊆ available keys, preferring download→parse→search."""
    available_keys = set(available.keys())
    invoked = set(already_invoked)
    priority = [
        "announcement_pdf",
        "download",
        "parse",
        "keyword",
        "pages",
        "get_pages",
    ]
    candidates: List[Dict[str, Any]] = []
    for oid, meta in metadata_by_id.items():
        if oid in invoked:
            continue
        requires = [str(r) for r in (meta.get("requires") or [])]
        soft = {"issuer_identity", "interface_name"}
        hard_requires = [r for r in requires if r not in soft]
        if not hard_requires:
            continue
        if not set(hard_requires).issubset(available_keys):
            continue
        provides = [str(p) for p in (meta.get("provides") or [])]
        # Prefer ops that advance the chain beyond list-only.
        if not provides and "list" in oid:
            continue
        score = 0
        for i, tip in enumerate(priority):
            if tip in oid.lower() or any(tip in p for p in provides):
                score += 10 - i
        if domain_pref and meta.get("domain") == domain_pref:
            score += 1
        candidates.append(
            {
                "operation_id": oid,
                "requires": hard_requires,
                "provides": provides,
                "score": score,
                "meta": meta,
            }
        )
    candidates.sort(key=lambda c: (-c["score"], c["operation_id"]))
    return candidates


def build_followup_retrieve_step(
    *,
    parent_step_id: str,
    operation_hint: Dict[str, Any],
    available: Dict[str, Any],
    index: int = 1,
) -> Dict[str, Any]:
    """Model-owned follow-up retrieve step unlocked after list provides local_key."""
    oid = operation_hint["operation_id"]
    provides = operation_hint.get("provides") or []
    objective = f"Advance disclosure chain via {oid} using provided dependencies."
    hint = f"{oid} local_key={available.get('local_key')}"
    return {
        "step_id": f"{parent_step_id}_cap_{index}",
        "step_type": "retrieve",
        "objective": objective[:180],
        "depends_on": [parent_step_id],
        "mandatory": True,
        "success_criteria": f"Obtain {', '.join(provides) or 'next disclosure artifact'}"[:160],
        "retrieval_query_hint": hint[:160],
        "compute_spec": None,
        "notes": f"auto_dependency_expansion:{oid}",
    }


def list_only_should_not_resolve_narrative(
    *,
    operation_id: str,
    objective: str,
    query_hint: Optional[str],
) -> bool:
    """List announcements identifies documents; narrative evidence needs download/parse."""
    if not is_narrative_retrieve_objective(objective, query_hint):
        return False
    oid = operation_id.lower()
    return "list" in oid and "hkex" in oid
