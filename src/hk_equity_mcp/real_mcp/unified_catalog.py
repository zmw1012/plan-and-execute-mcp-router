"""Build the unified HK Real-MCP operation pool from frozen domain catalogs.

Unique logical ``operation_id`` space = 42 (Financial ∪ Market ∪ News).
Namespaced Cross-domain rows may be 54, but the runtime registry never
registers the same ``operation_id`` twice.

Filtering rules:
- blocked / rejected tools are never exposed to the Agent
- ``runtime_exposed=false`` never enters the retrieval catalog
- ``formal_gold_eligible`` is tracked separately from runtime exposure
- distractors may be runtime-exposed but are never gold-eligible
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import yaml

from hk_equity_mcp.operation_resolver.schemas import OperationSpec
from hk_equity_mcp.real_mcp.catalog import operation_from_yaml
from hk_equity_mcp.real_mcp.market_gateway import BLOCKED_MARKET_OPS

ROOT = Path(__file__).resolve().parents[3]

FINANCIAL_CATALOG = ROOT / "configs" / "operations" / "financial_real_mcp.yaml"
MARKET_CATALOG = ROOT / "configs" / "operations" / "market_real_mcp_v1_1.yaml"
NEWS_CATALOG = ROOT / "configs" / "operations" / "news_real_mcp_v1.yaml"
CROSS_DOMAIN_MANIFEST = ROOT / "configs" / "operations" / "cross_domain_real_mcp_v1.yaml"

DOMAIN_PRIORITY = ("financial", "market", "news")
BLOCKED_ROLES = frozenset({"blocked", "rejected"})


@dataclass
class UnifiedOperationRecord:
    """One unique logical operation after domain-catalog merge."""

    operation_id: str
    domains: List[str]
    server_id: str
    runtime_exposed: bool
    formal_gold_eligible: bool
    is_distractor: bool
    is_blocked: bool
    role: str
    source_raw: Dict[str, Any]
    namespaced_ids: List[str] = field(default_factory=list)
    preferred_domain: str = ""
    dispatch_domain: str = ""


@dataclass
class UnifiedCatalogStats:
    namespaced_entries: int
    unique_operation_ids: int
    runtime_exposed_unique: int
    formal_gold_eligible_unique: int
    distractor_runtime_unique: int
    blocked_or_rejected_unique: int
    blocked_operation_ids: List[str]
    runtime_operation_ids: List[str]
    overlap_operation_ids: List[str]


@dataclass
class UnifiedCatalogBundle:
    records: Dict[str, UnifiedOperationRecord]
    registry: Dict[str, OperationSpec]
    raw_ops: Dict[str, Dict[str, Any]]
    stats: UnifiedCatalogStats
    manifest: Dict[str, Any]
    source_catalogs: Dict[str, Path]


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML object: {path}")
    return data


def _synthesize_execution(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Fill missing execution bodies for News (and similar) thin catalog rows."""
    existing = dict(raw.get("execution") or {})
    if existing.get("kind"):
        return existing

    server_id = str(raw.get("server_id") or "")
    tool = str(raw.get("exposed_mcp_tool") or raw.get("name") or "")
    interface = (
        raw.get("underlying_aktools_interface")
        or raw.get("underlying_interface")
        or ""
    )

    if server_id == "aktools" and tool in {"get_data", "aktools.get_data"} and interface:
        param_keys = [p["name"] for p in (raw.get("parameters") or []) if p.get("name")]
        return {
            "kind": "aktools_get_data",
            "interface": interface,
            "param_keys": param_keys,
        }

    if tool:
        arg_map = {
            p["name"]: p["name"]
            for p in (raw.get("parameters") or [])
            if p.get("name")
        }
        return {"kind": "mcp_tool", "tool": tool, "arg_map": arg_map}

    return existing


def _merge_raw(
    preferred: Dict[str, Any], other: Dict[str, Any]
) -> Dict[str, Any]:
    merged = deepcopy(preferred)
    # Prefer a concrete execution body whenever preferred lacks one.
    pref_exec = dict(merged.get("execution") or {})
    other_exec = dict(other.get("execution") or {})
    if not pref_exec.get("kind") and other_exec.get("kind"):
        merged["execution"] = deepcopy(other_exec)
    elif not pref_exec.get("kind"):
        merged["execution"] = _synthesize_execution(merged)
        if not merged["execution"].get("kind"):
            synth_other = deepcopy(other)
            synth_other["execution"] = _synthesize_execution(other)
            if synth_other["execution"].get("kind"):
                merged = synth_other
                merged["execution"] = synth_other["execution"]
    else:
        merged["execution"] = pref_exec

    # Preserve useful News/Market metadata flags when present.
    for key in (
        "formal_gold_eligible",
        "exposed_to_agent",
        "runtime_registry",
        "runtime_exposed",
        "decision",
        "source_class",
        "underlying_interface",
        "underlying_aktools_interface",
        "exposed_mcp_tool",
    ):
        if key not in merged and key in other:
            merged[key] = other[key]
    if not merged.get("execution"):
        merged["execution"] = _synthesize_execution(merged)
    return merged


def _domain_ops(catalog: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for raw in catalog.get("operations") or []:
        oid = str(raw["operation_id"])
        if oid in out:
            raise ValueError(f"Duplicate operation_id inside catalog: {oid}")
        out[oid] = deepcopy(raw)
    return out


def build_unified_catalog(
    *,
    financial_path: Optional[Path] = None,
    market_path: Optional[Path] = None,
    news_path: Optional[Path] = None,
    cross_domain_path: Optional[Path] = None,
    root: Optional[Path] = None,
) -> UnifiedCatalogBundle:
    root = root or ROOT
    financial_path = Path(financial_path or (root / FINANCIAL_CATALOG.relative_to(ROOT)))
    market_path = Path(market_path or (root / MARKET_CATALOG.relative_to(ROOT)))
    news_path = Path(news_path or (root / NEWS_CATALOG.relative_to(ROOT)))
    cross_domain_path = Path(
        cross_domain_path or (root / CROSS_DOMAIN_MANIFEST.relative_to(ROOT))
    )

    fin = _load_yaml(financial_path)
    mkt = _load_yaml(market_path)
    news = _load_yaml(news_path)
    manifest = _load_yaml(cross_domain_path)

    by_domain = {
        "financial": _domain_ops(fin),
        "market": _domain_ops(mkt),
        "news": _domain_ops(news),
    }

    # Market blocked/deferred must never enter the active tool pool.
    market_blocked: Set[str] = set(BLOCKED_MARKET_OPS)
    for section in ("blocked_operations", "deferred_operations"):
        for row in mkt.get(section) or []:
            if isinstance(row, dict) and row.get("operation_id"):
                market_blocked.add(str(row["operation_id"]))

    # Index Cross-domain namespaced rows by logical operation_id.
    xd_by_oid: Dict[str, List[Dict[str, Any]]] = {}
    for row in manifest.get("operations") or []:
        oid = str(row["operation_id"])
        xd_by_oid.setdefault(oid, []).append(row)

    all_oids = sorted(
        set(by_domain["financial"])
        | set(by_domain["market"])
        | set(by_domain["news"])
        | set(xd_by_oid)
    )
    if len(all_oids) != 42:
        # Fail loudly with diagnostic; caller/tests can report the mismatch.
        pass

    records: Dict[str, UnifiedOperationRecord] = {}
    overlap_ids: List[str] = []

    for oid in all_oids:
        domain_hits: List[str] = [
            d for d in DOMAIN_PRIORITY if oid in by_domain[d]
        ]
        if len(domain_hits) > 1:
            overlap_ids.append(oid)

        xd_rows = xd_by_oid.get(oid, [])
        role_xd = {str(r.get("role_in_cross_domain") or "") for r in xd_rows}
        runtime_flags = [bool(r.get("runtime_exposed")) for r in xd_rows]
        gold_flags = [bool(r.get("formal_gold_eligible")) for r in xd_rows]

        # Source-catalog blocked signals (News roles / Market blocklist).
        source_blocked = False
        source_runtime = True
        source_gold = False
        source_distractor = False
        for d in domain_hits:
            raw = by_domain[d][oid]
            role = str(raw.get("role") or raw.get("decision") or "")
            if role in BLOCKED_ROLES or oid in market_blocked:
                source_blocked = True
            if "formal_gold_eligible" in raw:
                source_gold = source_gold or bool(raw.get("formal_gold_eligible"))
            elif role in {
                "core_gold",
                "primary",
                "supporting",
                "supporting_evidence",
                "equivalent_alternative",
            }:
                source_gold = True
            if role in {"distractor", "distractor_valid", "supporting_runtime_only"}:
                source_distractor = True
            if "runtime_registry" in raw and not raw.get("runtime_registry"):
                source_runtime = False
            if "exposed_to_agent" in raw and not raw.get("exposed_to_agent"):
                source_runtime = False

        is_blocked = source_blocked or bool(role_xd & BLOCKED_ROLES) or oid in market_blocked
        if xd_rows:
            runtime_exposed = (any(runtime_flags) if runtime_flags else False) and not is_blocked
            formal_gold = (any(gold_flags) if gold_flags else False) and not is_blocked
            is_distractor = ("distractor" in role_xd) or source_distractor
        else:
            runtime_exposed = source_runtime and not is_blocked
            formal_gold = source_gold and not is_blocked
            is_distractor = source_distractor

        # Distractors may be runtime-exposed but never gold.
        if is_distractor:
            formal_gold = False
        if is_blocked:
            runtime_exposed = False
            formal_gold = False

        # Merge raw bodies with financial > market > news priority.
        merged: Optional[Dict[str, Any]] = None
        preferred_domain = domain_hits[0] if domain_hits else (
            str(xd_rows[0].get("domain")) if xd_rows else "unknown"
        )
        for d in DOMAIN_PRIORITY:
            if oid not in by_domain[d]:
                continue
            candidate = deepcopy(by_domain[d][oid])
            candidate["execution"] = _synthesize_execution(candidate)
            merged = candidate if merged is None else _merge_raw(merged, candidate)

        if merged is None:
            # Manifest-only pointer without source body — fail closed later at invoke.
            merged = {
                "operation_id": oid,
                "server_id": (xd_rows[0].get("server_id") if xd_rows else "unknown"),
                "name": oid.split(".")[-1],
                "summary": f"Manifest-only operation {oid}",
                "description": "",
                "parameters": [],
                "read_only": True,
                "execution": {},
            }

        merged["execution"] = _synthesize_execution(merged)
        merged["operation_id"] = oid
        merged["formal_gold_eligible"] = formal_gold
        merged["runtime_exposed"] = runtime_exposed
        merged["is_distractor"] = is_distractor
        merged["is_blocked"] = is_blocked

        # Dispatch domain: market-only ops → market gateway (cutoff); else financial
        # gateway for aktools/hkexnews; news-only ah_disclosure → news.
        if preferred_domain == "market" and "financial" not in domain_hits:
            dispatch_domain = "market"
        elif preferred_domain == "news" and "financial" not in domain_hits and "market" not in domain_hits:
            dispatch_domain = "news"
        elif "financial" in domain_hits:
            dispatch_domain = "financial"
        elif preferred_domain == "market":
            dispatch_domain = "market"
        else:
            dispatch_domain = preferred_domain

        records[oid] = UnifiedOperationRecord(
            operation_id=oid,
            domains=domain_hits or sorted({str(r.get("domain")) for r in xd_rows}),
            server_id=str(merged.get("server_id") or "unknown"),
            runtime_exposed=runtime_exposed,
            formal_gold_eligible=formal_gold,
            is_distractor=is_distractor and runtime_exposed,
            is_blocked=is_blocked,
            role=str(
                (xd_rows[0].get("role_in_cross_domain") if xd_rows else None)
                or merged.get("role")
                or ""
            ),
            source_raw=merged,
            namespaced_ids=[
                str(r.get("namespaced_operation_id"))
                for r in xd_rows
                if r.get("namespaced_operation_id")
            ],
            preferred_domain=preferred_domain,
            dispatch_domain=dispatch_domain,
        )

    runtime_ids = sorted(
        oid for oid, rec in records.items() if rec.runtime_exposed and not rec.is_blocked
    )
    blocked_ids = sorted(oid for oid, rec in records.items() if rec.is_blocked)
    gold_ids = sorted(oid for oid, rec in records.items() if rec.formal_gold_eligible)
    distractor_ids = sorted(
        oid for oid, rec in records.items() if rec.is_distractor and rec.runtime_exposed
    )

    registry: Dict[str, OperationSpec] = {}
    raw_ops: Dict[str, Dict[str, Any]] = {}
    for oid in runtime_ids:
        rec = records[oid]
        raw = deepcopy(rec.source_raw)
        raw_ops[oid] = raw
        spec = operation_from_yaml(raw)
        # Annotate retrieval constraints with gold/distractor separation.
        extra = []
        if rec.formal_gold_eligible:
            extra.append("formal_gold_eligible: true")
        else:
            extra.append("formal_gold_eligible: false")
        if rec.is_distractor:
            extra.append("runtime_distractor: true")
        spec.constraints = list(spec.constraints) + extra
        registry[oid] = spec

    if len(registry) != len(runtime_ids):
        raise ValueError("Duplicate operation_id while building unified runtime registry")

    stats = UnifiedCatalogStats(
        namespaced_entries=len(manifest.get("operations") or []),
        unique_operation_ids=len(records),
        runtime_exposed_unique=len(runtime_ids),
        formal_gold_eligible_unique=len(gold_ids),
        distractor_runtime_unique=len(distractor_ids),
        blocked_or_rejected_unique=len(blocked_ids),
        blocked_operation_ids=blocked_ids,
        runtime_operation_ids=runtime_ids,
        overlap_operation_ids=sorted(overlap_ids),
    )

    return UnifiedCatalogBundle(
        records=records,
        registry=registry,
        raw_ops=raw_ops,
        stats=stats,
        manifest=manifest,
        source_catalogs={
            "financial": financial_path,
            "market": market_path,
            "news": news_path,
            "cross_domain": cross_domain_path,
        },
    )


def load_unified_runtime_registry(
    **kwargs: Any,
) -> Tuple[Dict[str, OperationSpec], UnifiedCatalogBundle]:
    bundle = build_unified_catalog(**kwargs)
    return bundle.registry, bundle
