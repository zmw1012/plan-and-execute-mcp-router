"""Compatibility-tier reranking over Operation-Visible base retrieval scores.

Tier A: metadata clearly compatible
Tier B: unknown / partial compatibility without hard conflict
Tier C: explicit domain/market/period/source conflict (fallback only)

Hard filters remove blocked/unavailable/non-HK/schema-impossible/cutoff violations.
No learned weights in v1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set


COMPAT_TIERS = ("A", "B", "C")


@dataclass
class RetrievalContext:
    """Query-side hints used for compatibility checks (company-generic)."""

    domain_hints: Sequence[str] = field(default_factory=list)
    market_scope: str = "hk"
    entity_scope_hints: Sequence[str] = field(default_factory=list)
    identifier_types: Sequence[str] = field(default_factory=list)
    temporal_needs: Sequence[str] = field(default_factory=list)
    source_type_prefs: Sequence[str] = field(default_factory=list)
    metric_basis_hints: Sequence[str] = field(default_factory=list)
    required_provides: Sequence[str] = field(default_factory=list)
    cutoff_date: Optional[str] = "2026-08-12"
    disallow_post_cutoff_spot: bool = True
    available_dependencies: Sequence[str] = field(default_factory=list)


def _as_set(values: Sequence[str]) -> Set[str]:
    return {str(v) for v in values if v}


def hard_filter_reason(
    meta: Dict[str, Any],
    ctx: RetrievalContext,
    *,
    in_runtime_registry: bool,
) -> Optional[str]:
    availability = str(meta.get("availability") or "")
    if availability in {"blocked", "rejected", "unavailable", "deferred"}:
        return f"availability={availability}"
    if not in_runtime_registry:
        return "not_in_runtime_registry"
    if meta.get("runtime_role") in {"blocked", "rejected"}:
        return f"runtime_role={meta.get('runtime_role')}"
    market = str(meta.get("market_scope") or "unknown")
    if market in {"cn_mainland"} and ctx.market_scope == "hk":
        # Explicit non-HK market scope conflict → hard filter when HK-only.
        return "explicit_non_hk_market_scope"
    requires = list(meta.get("requires") or [])
    available = _as_set(ctx.available_dependencies)
    missing = [r for r in requires if r not in available and r not in {
        # Soft dependencies that the agent may still supply as invoke args.
        "issuer_identity",
        "interface_name",
        "announcement_id",
        "announcement_pdf",
        "page_text",
    }]
    # Only hard-fail on requires marked ungeneratable via ctx flag list.
    ungen = _as_set(getattr(ctx, "ungeneratable_requires", []) or [])
    hard_missing = [r for r in requires if r in ungen and r not in available]
    if hard_missing:
        return f"missing_ungeneratable_requires:{hard_missing}"
    if ctx.disallow_post_cutoff_spot and ctx.cutoff_date:
        temporal = str(meta.get("temporal_granularity") or "")
        # Spot/latest feeds are cutoff-unsafe for fixed historical gold.
        if temporal == "spot_latest" and "point_in_time" in _as_set(ctx.temporal_needs):
            return "cutoff_violation_spot_latest_for_point_in_time"
    return None


def classify_compatibility(
    meta: Dict[str, Any],
    ctx: RetrievalContext,
) -> Dict[str, Any]:
    reasons: List[str] = []
    conflicts: List[str] = []
    supports: List[str] = []

    domain = str(meta.get("domain") or "")
    domain_hints = _as_set(ctx.domain_hints)
    if domain_hints:
        if domain in domain_hints or domain in {"meta", "utility"}:
            supports.append(f"domain:{domain}")
        elif domain:
            conflicts.append(f"domain_conflict:{domain} vs {sorted(domain_hints)}")

    market = str(meta.get("market_scope") or "unknown")
    if ctx.market_scope == "hk":
        if market in {"hk", "hk_and_global"}:
            supports.append(f"market:{market}")
        elif market in {"global", "unknown"}:
            reasons.append(f"market_partial:{market}")
        else:
            conflicts.append(f"market_conflict:{market}")

    entity = str(meta.get("entity_scope") or "unknown")
    entity_hints = _as_set(ctx.entity_scope_hints)
    if entity_hints:
        if entity in entity_hints or entity == "unknown":
            if entity in entity_hints:
                supports.append(f"entity:{entity}")
            else:
                reasons.append("entity_unknown")
        else:
            conflicts.append(f"entity_conflict:{entity}")

    id_types = _as_set(meta.get("accepted_identifier_types") or [])
    needed_ids = _as_set(ctx.identifier_types)
    if needed_ids:
        if id_types & needed_ids or "unknown" in id_types or "none" in id_types:
            if id_types & needed_ids:
                supports.append("identifier_overlap")
            else:
                reasons.append("identifier_unspecified")
        else:
            conflicts.append("identifier_conflict")

    temporal = str(meta.get("temporal_granularity") or "unknown")
    temporal_needs = _as_set(ctx.temporal_needs)
    # Soft compatibility groups: historical/fiscal queries can use date_range tools.
    temporal_compat = {
        "fiscal_period": {"fiscal_period", "date_range", "point_in_time", "as_of"},
        "date_range": {"date_range", "point_in_time", "fiscal_period", "as_of"},
        "point_in_time": {"point_in_time", "date_range", "as_of", "fiscal_period"},
        "as_of": {"as_of", "point_in_time", "date_range", "fiscal_period"},
        "calendar": {"calendar"},
        "spot_latest": {"spot_latest"},
        "none": {"none", "unknown"},
        "unknown": set(),
    }
    if temporal_needs:
        compatible = set()
        for need in temporal_needs:
            compatible |= temporal_compat.get(need, {need})
        if temporal in temporal_needs:
            supports.append(f"temporal:{temporal}")
        elif temporal in compatible or temporal in {"unknown", "none", "as_of"}:
            if temporal in compatible:
                supports.append(f"temporal_compat:{temporal}")
            else:
                reasons.append(f"temporal_partial:{temporal}")
        elif temporal == "spot_latest" and (
            "point_in_time" in temporal_needs
            or "date_range" in temporal_needs
            or "fiscal_period" in temporal_needs
        ):
            conflicts.append("temporal_spot_vs_historical")
        else:
            conflicts.append(f"temporal_conflict:{temporal}")

    source = str(meta.get("source_type") or "unknown")
    source_prefs = _as_set(ctx.source_type_prefs)
    if source_prefs:
        if source in source_prefs:
            supports.append(f"source:{source}")
        elif source in {"unknown", "structured_vendor"}:
            reasons.append(f"source_partial:{source}")
        else:
            conflicts.append(f"source_conflict:{source}")

    basis = str(meta.get("metric_basis") or "unknown")
    basis_hints = _as_set(ctx.metric_basis_hints)
    accounting_bases = {
        "reported_ifrs",
        "adjusted_non_ifrs",
        "mixed_or_unspecified",
    }
    price_bases = {"price_unadjusted", "price_adjusted", "price"}
    if basis_hints:
        if basis in basis_hints or basis in {
            "unknown",
            "mixed_or_unspecified",
            "not_applicable",
        }:
            if basis in basis_hints:
                supports.append(f"basis:{basis}")
            else:
                reasons.append(f"basis_partial:{basis}")
        elif (basis in price_bases and basis_hints & accounting_bases) or (
            basis in accounting_bases and basis_hints & price_bases
        ):
            # Cross-family basis mismatch is soft when domain/provides already speak.
            reasons.append(f"basis_partial_cross_family:{basis}")
        else:
            conflicts.append(f"basis_conflict:{basis}")

    provides = _as_set(meta.get("provides") or [])
    needed_provides = _as_set(ctx.required_provides)
    if needed_provides:
        if provides & needed_provides:
            supports.append("provides_overlap")
        else:
            reasons.append("provides_unknown")

    if conflicts:
        tier = "C"
    elif supports and not reasons:
        tier = "A"
    elif supports or reasons or not (domain_hints or entity_hints or temporal_needs):
        # Unknown / partial without hard conflict → B
        tier = "A" if supports and not reasons else "B"
        if supports and reasons:
            tier = "B"
        if not supports and not reasons and not conflicts:
            tier = "B"
    else:
        tier = "B"

    # Strengthen: clear supports and zero conflicts/partials → A
    if supports and not conflicts and not reasons:
        tier = "A"
    elif conflicts:
        tier = "C"
    else:
        tier = "B" if reasons or not supports else "A"

    return {
        "compatibility_tier": tier,
        "compatibility_reasons": {
            "supports": supports,
            "partial": reasons,
            "conflicts": conflicts,
        },
    }


def rerank_candidates(
    candidates: Sequence[Dict[str, Any]],
    metadata_by_id: Dict[str, Dict[str, Any]],
    ctx: RetrievalContext,
    *,
    runtime_operation_ids: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Apply hard filters then Tier A→B→C ordering by original base score."""
    runtime_set = set(runtime_operation_ids or metadata_by_id.keys())
    kept: List[Dict[str, Any]] = []
    filtered: List[Dict[str, Any]] = []

    for original_rank, cand in enumerate(candidates, start=1):
        oid = str(
            cand.get("operation_id")
            or cand.get("id")
            or cand.get("name")
            or ""
        )
        base_score = cand.get("score")
        if base_score is None:
            base_score = cand.get("combined_score")
        if base_score is None:
            base_score = 0.0
        meta = metadata_by_id.get(oid) or {
            "operation_id": oid,
            "availability": "unavailable" if oid not in runtime_set else "runtime",
            "domain": "unknown",
            "market_scope": "unknown",
            "entity_scope": "unknown",
            "accepted_identifier_types": ["unknown"],
            "temporal_granularity": "unknown",
            "source_type": "unknown",
            "metric_basis": "unknown",
            "requires": [],
            "provides": [],
            "runtime_role": "relevant",
        }
        reason = hard_filter_reason(
            meta, ctx, in_runtime_registry=oid in runtime_set
        )
        row = {
            "operation_id": oid,
            "original_rank": original_rank,
            "base_score": float(base_score),
            "candidate": cand,
        }
        if reason is not None:
            row["hard_filtered"] = True
            row["hard_filter_reason"] = reason
            filtered.append(row)
            continue
        compat = classify_compatibility(meta, ctx)
        row.update(compat)
        row["hard_filtered"] = False
        kept.append(row)

    tier_rank = {"A": 0, "B": 1, "C": 2}
    kept.sort(
        key=lambda r: (
            tier_rank.get(r["compatibility_tier"], 9),
            -float(r["base_score"]),
            int(r["original_rank"]),
        )
    )
    for final_rank, row in enumerate(kept, start=1):
        row["final_rank"] = final_rank

    return {
        "reranked": kept,
        "hard_filtered": filtered,
        "tier_counts": {
            t: sum(1 for r in kept if r["compatibility_tier"] == t) for t in COMPAT_TIERS
        },
    }
