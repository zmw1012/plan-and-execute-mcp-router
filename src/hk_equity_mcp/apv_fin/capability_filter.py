"""Deterministic capability filtering + catalog injection (not gold injection)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Set

from hk_equity_mcp.apv_fin.capability_contract import (
    covers_required_outputs,
    operations_for_capability,
)
from hk_equity_mcp.apv_fin.capability_ontology import compatible_capabilities


def filter_by_capability_contract(
    *,
    candidates: Sequence[Dict[str, Any]],
    contracts: Dict[str, Dict[str, Any]],
    required_capability: str,
    required_outputs: Sequence[str] = (),
    runtime_operation_ids: Optional[Sequence[str]] = None,
    source_preference: Sequence[str] = (),
) -> Dict[str, Any]:
    """Pipeline stages 1–5 before describe/bind.

    Returns filtered candidate rows with injection flags.
    """
    runtime = set(runtime_operation_ids or [])
    wanted_caps = compatible_capabilities(required_capability)

    # 1) runtime availability
    avail: List[Dict[str, Any]] = []
    hard_filtered: List[Dict[str, Any]] = []
    for row in candidates:
        oid = str(row.get("operation_id") or "")
        contract = contracts.get(oid) or {}
        if runtime and oid not in runtime:
            hard_filtered.append({**row, "hard_filter_reason": "not_runtime"})
            continue
        if str(contract.get("runtime_availability") or "runtime") != "runtime":
            hard_filtered.append({**row, "hard_filter_reason": "unavailable"})
            continue
        avail.append(row)

    # 2–3) capability + required_outputs
    compatible: List[Dict[str, Any]] = []
    for row in avail:
        oid = str(row.get("operation_id") or "")
        contract = contracts.get(oid) or {}
        caps = set(contract.get("capabilities") or [])
        if not (caps & wanted_caps):
            hard_filtered.append({**row, "hard_filter_reason": "capability_mismatch"})
            continue
        if required_outputs and not covers_required_outputs(contract, required_outputs):
            hard_filtered.append({**row, "hard_filter_reason": "required_outputs_gap"})
            continue
        compatible.append(
            {
                **row,
                "capability_compatible": True,
                "matched_capabilities": sorted(caps & wanted_caps),
            }
        )

    # 4) source preference soft boost (does not drop)
    pref = [str(p) for p in source_preference or []]
    if pref:
        def pref_score(row: Dict[str, Any]) -> int:
            src = str((contracts.get(row["operation_id"]) or {}).get("source_type") or "")
            for i, p in enumerate(pref):
                if p == src or p in src:
                    return len(pref) - i
            return 0

        compatible = sorted(
            compatible,
            key=lambda r: (
                -pref_score(r),
                int(r.get("final_rank") or r.get("original_rank") or 10**6),
                -float(r.get("base_score") or r.get("score") or 0.0),
            ),
        )

    # 5) If embedding set missed capability-compatible ops, inject from catalog index.
    injected: List[Dict[str, Any]] = []
    present_ids = {str(r.get("operation_id")) for r in compatible}
    catalog_ids = operations_for_capability(
        contracts, required_capability, include_compatible=True
    )
    if required_outputs:
        catalog_ids = [
            oid
            for oid in catalog_ids
            if covers_required_outputs(contracts[oid], required_outputs)
        ]
    for oid in catalog_ids:
        if oid in present_ids:
            continue
        if runtime and oid not in runtime:
            continue
        injected.append(
            {
                "operation_id": oid,
                "original_rank": None,
                "final_rank": None,
                "base_score": 0.0,
                "score": 0.0,
                "compatibility_tier": "A",
                "compatibility_reasons": {
                    "supports": ["capability_contract_injection"],
                    "partial": [],
                    "conflicts": [],
                },
                "capability_compatible": True,
                "capability_injected_candidate": True,
                "matched_capabilities": sorted(
                    set(contracts[oid].get("capabilities") or []) & wanted_caps
                ),
                "candidate": {"operation_id": oid, "score": 0.0},
            }
        )
        present_ids.add(oid)

    merged = list(compatible) + injected

    # Assign final ranks within compatible set by base_score then injection last.
    def sort_key(row: Dict[str, Any]) -> tuple:
        injected_flag = 1 if row.get("capability_injected_candidate") else 0
        return (
            injected_flag,
            -float(row.get("base_score") or row.get("score") or 0.0),
            str(row.get("operation_id") or ""),
        )

    merged = sorted(merged, key=sort_key)
    for i, row in enumerate(merged, start=1):
        row["final_rank"] = i
        # Promote capability-compatible to tier A/B for binder gate.
        if row.get("capability_injected_candidate"):
            row["compatibility_tier"] = "A"
        elif row.get("compatibility_tier") == "C":
            row["compatibility_tier"] = "B"

    # Binder gate: if any A/B compatible exists, drop Tier C from binder pool.
    has_ab = any(r.get("compatibility_tier") in {"A", "B"} for r in merged)
    binder_pool = [
        r
        for r in merged
        if (not has_ab) or r.get("compatibility_tier") in {"A", "B"}
    ]

    return {
        "compatible_candidates": merged,
        "binder_candidates": binder_pool,
        "hard_filtered": hard_filtered,
        "capability_filter_candidate_count": len(merged),
        "capability_injected_candidate_count": len(injected),
        "required_capability": required_capability,
        "required_outputs": list(required_outputs or []),
    }
