"""Per-retrieve-step query construction, hard filter, embedding Top-K, describe."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Set

DOMAIN_COMPAT: Dict[str, Set[str]] = {
    "financial": {"financial", "cross_domain", "general"},
    "market": {"market", "cross_domain", "general"},
    "disclosure": {"news", "disclosure", "cross_domain", "general"},
    "news": {"news", "disclosure", "cross_domain", "general"},
    "compute": {"general"},
    "general": {"financial", "market", "news", "disclosure", "cross_domain", "general"},
}


def build_step_search_query(step: Dict[str, Any]) -> str:
    parts: List[str] = []
    for key in ("domain", "objective", "entity", "period", "date"):
        val = step.get(key)
        if val:
            parts.append(str(val))
    req = step.get("required_outputs") or []
    if req:
        parts.append("required_outputs: " + ", ".join(str(x) for x in req))
    wf = step.get("workflow")
    if wf:
        parts.append(f"workflow: {wf}")
    return " | ".join(parts).strip()


def _meta_blocked(meta: Dict[str, Any]) -> bool:
    availability = str(meta.get("availability") or "").lower()
    role = str(meta.get("runtime_role") or "").lower()
    if availability in {"blocked", "rejected", "unavailable"}:
        return True
    if role in {"blocked", "rejected"}:
        return True
    if meta.get("blocked") is True or meta.get("rejected") is True:
        return True
    return False


def _meta_runtime_exposed(
    meta: Dict[str, Any],
    *,
    runtime_ids: Set[str],
    operation_id: str,
) -> bool:
    if operation_id not in runtime_ids:
        return False
    availability = str(meta.get("availability") or "runtime").lower()
    if availability not in {"runtime", "available", "exposed", ""}:
        return False
    return True


def domain_compatible(step_domain: str, op_domain: str) -> bool:
    allowed = DOMAIN_COMPAT.get(str(step_domain or "general").lower(), DOMAIN_COMPAT["general"])
    return str(op_domain or "general").lower() in allowed or str(op_domain or "").lower() in {
        "general",
        "",
    }


def hard_filter_operations(
    *,
    registry_ids: Sequence[str],
    metadata_by_id: Dict[str, Dict[str, Any]],
    runtime_operation_ids: Sequence[str],
    step_domain: str,
) -> Dict[str, Any]:
    runtime = set(runtime_operation_ids)
    kept: List[str] = []
    rejected: List[Dict[str, str]] = []
    for oid in registry_ids:
        meta = metadata_by_id.get(oid) or {}
        if not _meta_runtime_exposed(meta, runtime_ids=runtime, operation_id=oid):
            rejected.append({"operation_id": oid, "reason": "not_runtime_exposed"})
            continue
        if _meta_blocked(meta):
            rejected.append({"operation_id": oid, "reason": "blocked_or_rejected"})
            continue
        op_domain = str(meta.get("domain") or "general")
        if not domain_compatible(step_domain, op_domain):
            rejected.append({"operation_id": oid, "reason": "domain_incompatible"})
            continue
        # Cutoff / availability notes are advisory; blocked already handled.
        kept.append(oid)
    return {
        "kept": kept,
        "rejected": rejected,
        "kept_count": len(kept),
        "rejected_count": len(rejected),
    }


def select_embedding_topk(
    candidates: Sequence[Dict[str, Any]],
    *,
    allowed_ids: Set[str],
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    rows = [
        c
        for c in candidates
        if str(c.get("operation_id") or "") in allowed_ids
    ]
    # Candidates already ranked by resolver; keep order, cap top_k.
    out = []
    for row in rows[: max(1, int(top_k))]:
        item = dict(row)
        item["score"] = float(
            item.get("score")
            or item.get("embedding_score")
            or item.get("similarity")
            or 0.0
        )
        out.append(item)
    return out


def select_describe_candidates(
    topk: Sequence[Dict[str, Any]],
    *,
    top_n: int = 3,
) -> List[Dict[str, Any]]:
    n = max(2, min(3, int(top_n))) if top_n else 3
    # Spec: describe top 2–3
    n = min(max(n, 2), 3)
    return [dict(r) for r in list(topk)[:n]]


def enrich_operation_document(
    *,
    operation_id: str,
    contract: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    index_document: Optional[str] = None,
) -> str:
    """Build a retrieval document richer than function name alone."""
    contract = contract or {}
    metadata = metadata or {}
    parts = [
        f"operation_id: {operation_id}",
        f"name: {contract.get('name') or operation_id}",
        f"description: {contract.get('description') or contract.get('summary') or ''}",
        f"domain: {metadata.get('domain') or ''}",
        f"capabilities: {', '.join(metadata.get('capabilities') or [])}",
        f"provides: {', '.join(str(x) for x in (metadata.get('provides') or []))}",
        f"requires: {', '.join(str(x) for x in (metadata.get('requires') or []))}",
        f"constraints: {'; '.join(contract.get('constraints') or [])}",
        f"source_type: {metadata.get('source_type') or ''}",
    ]
    req = contract.get("required_parameters") or []
    opt = contract.get("optional_parameters") or []
    if req:
        parts.append(
            "input_summary: "
            + ", ".join(
                f"{p.get('name')}:{p.get('type')}" for p in req if isinstance(p, dict)
            )
        )
    if opt:
        parts.append(
            "optional_inputs: "
            + ", ".join(
                f"{p.get('name')}:{p.get('type')}" for p in opt if isinstance(p, dict)
            )
        )
    provides = metadata.get("provides") or []
    if provides:
        parts.append("output_summary: " + ", ".join(str(x) for x in provides))
    if index_document:
        parts.append("index_document: " + index_document)
    return "\n".join(parts)
