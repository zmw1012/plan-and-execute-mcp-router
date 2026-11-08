"""Structured operation metadata sidecar for APV-Fin (company-generic).

Does not modify frozen catalog operation_id semantics. Validates coverage of
all runtime-exposed operations and keeps blocked tools out of runtime.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

import yaml

from hk_equity_mcp.real_mcp.unified_catalog import (
    UnifiedCatalogBundle,
    build_unified_catalog,
)

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_METADATA_PATH = (
    ROOT / "configs" / "operations" / "hk_unified_operation_metadata_v1.yaml"
)

METADATA_VERSION = "hk_unified_operation_metadata_v1"

DOMAIN_ENUM = ("financial", "market", "news", "cross_domain", "meta", "utility")
CAPABILITY_ENUM = (
    "identity_resolve",
    "issuer_profile",
    "financial_statement",
    "financial_ratio",
    "disclosure_search",
    "disclosure_download",
    "disclosure_parse",
    "price_history",
    "index_history",
    "spot_quote",
    "calendar",
    "news_media",
    "meta_discovery",
    "rate_yield",
    "dividend",
    "valuation",
    "comparison",
    "other",
)
MARKET_SCOPE_ENUM = ("hk", "hk_and_global", "global", "cn_mainland", "unknown")
ENTITY_SCOPE_ENUM = (
    "hk_listed_issuer",
    "hk_equity",
    "index",
    "bond_rate",
    "macro_news",
    "meta",
    "unknown",
)
IDENTIFIER_ENUM = (
    "hk_code_5digit",
    "hk_code_4digit",
    "hk_ticker_dot",
    "index_symbol",
    "company_name",
    "isin",
    "interface_name",
    "none",
    "unknown",
)
TEMPORAL_ENUM = (
    "point_in_time",
    "date_range",
    "fiscal_period",
    "as_of",
    "spot_latest",
    "calendar",
    "none",
    "unknown",
)
SOURCE_TYPE_ENUM = (
    "official_disclosure",
    "exchange_feed",
    "structured_vendor",
    "media",
    "meta",
    "synthetic_blocked",
    "unknown",
)
METRIC_BASIS_ENUM = (
    "reported_ifrs",
    "adjusted_non_ifrs",
    "mixed_or_unspecified",
    "price_unadjusted",
    "price_qfq",
    "price_hfq",
    "index_points",
    "not_applicable",
    "unknown",
)
AVAILABILITY_ENUM = (
    "runtime",
    "blocked",
    "rejected",
    "deferred",
    "unavailable",
)
ROLE_ENUM = (
    "relevant",
    "distractor",
    "blocked",
    "rejected",
    "supporting",
    "meta",
)

REQUIRED_FIELDS = (
    "operation_id",
    "domain",
    "capabilities",
    "market_scope",
    "entity_scope",
    "accepted_identifier_types",
    "temporal_granularity",
    "source_type",
    "metric_basis",
    "availability",
    "requires",
    "provides",
    "formal_gold_eligible",
    "runtime_role",
    "provenance",
    "confidence",
)

# Company-specific issuer names must not appear in metadata text fields.
BANNED_ISSUER_SUBSTRINGS = (
    "tencent",
    "hsbc",
    "alibaba",
    "pop mart",
    "xiaomi",
    "0700",
    "00005",
    "09988",
    "09992",
    "01810",
    "81810",
)


class MetadataValidationError(ValueError):
    pass


def load_metadata_yaml(path: Optional[Path] = None) -> Dict[str, Any]:
    path = Path(path or DEFAULT_METADATA_PATH)
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise MetadataValidationError(f"metadata root must be object: {path}")
    return data


def metadata_index(data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    ops = data.get("operations") or []
    out: Dict[str, Dict[str, Any]] = {}
    for row in ops:
        oid = str(row.get("operation_id") or "")
        if not oid:
            raise MetadataValidationError("operation missing operation_id")
        if oid in out:
            raise MetadataValidationError(f"duplicate metadata operation_id: {oid}")
        out[oid] = row
    return out


def _require_enum(value: Any, allowed: Sequence[str], field: str, oid: str) -> None:
    if value not in allowed:
        raise MetadataValidationError(
            f"{oid}.{field}={value!r} not in {list(allowed)}"
        )


def _require_enum_list(
    values: Any, allowed: Sequence[str], field: str, oid: str
) -> None:
    if not isinstance(values, list) or not values:
        raise MetadataValidationError(f"{oid}.{field} must be non-empty list")
    for item in values:
        _require_enum(item, allowed, field, oid)


def _assert_company_generic(row: Dict[str, Any]) -> None:
    blob = yaml.safe_dump(row, sort_keys=True).lower()
    for banned in BANNED_ISSUER_SUBSTRINGS:
        if banned in blob:
            raise MetadataValidationError(
                f"{row.get('operation_id')}: metadata must be company-generic; "
                f"found banned token {banned!r}"
            )


def validate_metadata_bundle(
    data: Dict[str, Any],
    *,
    catalog: Optional[UnifiedCatalogBundle] = None,
    root: Optional[Path] = None,
) -> Dict[str, Any]:
    if data.get("metadata_version") != METADATA_VERSION:
        raise MetadataValidationError(
            f"expected metadata_version={METADATA_VERSION}, "
            f"got {data.get('metadata_version')!r}"
        )
    catalog = catalog or build_unified_catalog(root=root)
    index = metadata_index(data)
    runtime_ids = set(catalog.stats.runtime_operation_ids)
    blocked_ids = set(catalog.stats.blocked_operation_ids)

    missing = sorted(runtime_ids - set(index))
    if missing:
        raise MetadataValidationError(
            f"metadata missing runtime operations: {missing}"
        )

    for oid, row in index.items():
        for field in REQUIRED_FIELDS:
            if field not in row:
                raise MetadataValidationError(f"{oid} missing field {field}")
        _require_enum(row["domain"], DOMAIN_ENUM, "domain", oid)
        _require_enum_list(row["capabilities"], CAPABILITY_ENUM, "capabilities", oid)
        _require_enum(row["market_scope"], MARKET_SCOPE_ENUM, "market_scope", oid)
        _require_enum(row["entity_scope"], ENTITY_SCOPE_ENUM, "entity_scope", oid)
        _require_enum_list(
            row["accepted_identifier_types"],
            IDENTIFIER_ENUM,
            "accepted_identifier_types",
            oid,
        )
        _require_enum(
            row["temporal_granularity"], TEMPORAL_ENUM, "temporal_granularity", oid
        )
        _require_enum(row["source_type"], SOURCE_TYPE_ENUM, "source_type", oid)
        _require_enum(row["metric_basis"], METRIC_BASIS_ENUM, "metric_basis", oid)
        _require_enum(row["availability"], AVAILABILITY_ENUM, "availability", oid)
        _require_enum(row["runtime_role"], ROLE_ENUM, "runtime_role", oid)
        if not isinstance(row["requires"], list):
            raise MetadataValidationError(f"{oid}.requires must be list")
        if not isinstance(row["provides"], list):
            raise MetadataValidationError(f"{oid}.provides must be list")
        if not isinstance(row["formal_gold_eligible"], bool):
            raise MetadataValidationError(f"{oid}.formal_gold_eligible must be bool")
        if not isinstance(row["provenance"], str) or not row["provenance"].strip():
            raise MetadataValidationError(f"{oid}.provenance must be non-empty string")
        conf = row["confidence"]
        if not isinstance(conf, (int, float)) or isinstance(conf, bool):
            raise MetadataValidationError(f"{oid}.confidence must be number")
        if not (0.0 <= float(conf) <= 1.0):
            raise MetadataValidationError(f"{oid}.confidence must be in [0,1]")
        _assert_company_generic(row)

        if oid in runtime_ids:
            if row["availability"] != "runtime":
                raise MetadataValidationError(
                    f"runtime operation {oid} must have availability=runtime"
                )
            if row["runtime_role"] in {"blocked", "rejected"}:
                raise MetadataValidationError(
                    f"runtime operation {oid} cannot have blocked/rejected role"
                )
        if oid in blocked_ids or row["availability"] in {
            "blocked",
            "rejected",
            "unavailable",
        }:
            if oid in catalog.registry:
                raise MetadataValidationError(
                    f"blocked/unavailable {oid} must not be in runtime registry"
                )

        # Cross-check catalog formal_gold / distractor when present in both.
        if oid in catalog.records:
            rec = catalog.records[oid]
            if bool(row["formal_gold_eligible"]) != bool(rec.formal_gold_eligible):
                raise MetadataValidationError(
                    f"{oid} formal_gold_eligible mismatch vs frozen catalog"
                )

    return {
        "metadata_version": METADATA_VERSION,
        "runtime_coverage": len(runtime_ids),
        "metadata_rows": len(index),
        "blocked_ids": sorted(blocked_ids),
        "ok": True,
    }


def load_and_validate_metadata(
    path: Optional[Path] = None,
    *,
    catalog: Optional[UnifiedCatalogBundle] = None,
    root: Optional[Path] = None,
) -> Dict[str, Dict[str, Any]]:
    data = load_metadata_yaml(path)
    validate_metadata_bundle(data, catalog=catalog, root=root)
    return metadata_index(data)
