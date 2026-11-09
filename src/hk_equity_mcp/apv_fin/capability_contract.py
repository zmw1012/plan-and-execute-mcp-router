"""Load + validate per-operation capability contracts (APV-Fin v1.2)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

import yaml

from hk_equity_mcp.apv_fin.capability_ontology import (
    CAPABILITY_IDS,
    CAPABILITY_ONTOLOGY_VERSION,
    compatible_capabilities,
)
from hk_equity_mcp.real_mcp.unified_catalog import (
    UnifiedCatalogBundle,
    build_unified_catalog,
)

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONTRACT_PATH = ROOT / "configs" / "operations" / "hk_capability_contract_v1.yaml"
CONTRACT_VERSION = "hk_capability_contract_v1"


class CapabilityContractError(ValueError):
    pass


def load_capability_contract_yaml(path: Optional[Path] = None) -> Dict[str, Any]:
    path = Path(path or DEFAULT_CONTRACT_PATH)
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise CapabilityContractError(f"contract root must be object: {path}")
    return data


def contract_index(data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for row in data.get("operations") or []:
        oid = str(row.get("operation_id") or "")
        if not oid:
            raise CapabilityContractError("operation missing operation_id")
        if oid in out:
            raise CapabilityContractError(f"duplicate contract operation_id: {oid}")
        out[oid] = row
    return out


def capability_index(index: Dict[str, Dict[str, Any]]) -> Dict[str, List[str]]:
    """capability_id → operation_ids that provide it."""
    by_cap: Dict[str, List[str]] = {c: [] for c in CAPABILITY_IDS}
    for oid, row in index.items():
        if str(row.get("runtime_availability") or "") != "runtime":
            continue
        for cap in row.get("capabilities") or []:
            by_cap.setdefault(str(cap), []).append(oid)
    return by_cap


def validate_capability_contracts(
    data: Optional[Dict[str, Any]] = None,
    *,
    catalog: Optional[UnifiedCatalogBundle] = None,
    root: Optional[Path] = None,
) -> Dict[str, Any]:
    data = data or load_capability_contract_yaml()
    if data.get("contract_version") != CONTRACT_VERSION:
        raise CapabilityContractError(
            f"expected contract_version={CONTRACT_VERSION}, "
            f"got {data.get('contract_version')!r}"
        )
    if data.get("ontology_version") != CAPABILITY_ONTOLOGY_VERSION:
        raise CapabilityContractError(
            f"expected ontology_version={CAPABILITY_ONTOLOGY_VERSION}, "
            f"got {data.get('ontology_version')!r}"
        )
    catalog = catalog or build_unified_catalog(root=root)
    index = contract_index(data)
    runtime_ids = set(catalog.stats.runtime_operation_ids)
    missing = sorted(runtime_ids - set(index))
    if missing:
        raise CapabilityContractError(
            f"capability contract missing runtime operations: {missing}"
        )

    required_fields = (
        "operation_id",
        "capabilities",
        "required_arguments",
        "provides_fields",
        "requires_fields",
        "supported_market",
        "temporal_granularity",
        "source_type",
        "accounting_basis_support",
        "runtime_availability",
    )
    ontology_set = set(CAPABILITY_IDS)
    provided_caps: Set[str] = set()
    for oid, row in index.items():
        for field in required_fields:
            if field not in row:
                raise CapabilityContractError(f"{oid} missing field {field}")
        caps = row["capabilities"]
        if not isinstance(caps, list) or not caps:
            raise CapabilityContractError(f"{oid}.capabilities must be non-empty list")
        for cap in caps:
            if cap not in ontology_set:
                raise CapabilityContractError(
                    f"{oid}.capabilities contains unknown capability {cap!r}"
                )
            provided_caps.add(str(cap))
        for list_field in (
            "required_arguments",
            "provides_fields",
            "requires_fields",
            "accounting_basis_support",
        ):
            if not isinstance(row[list_field], list):
                raise CapabilityContractError(f"{oid}.{list_field} must be list")
        if oid in runtime_ids and row["runtime_availability"] != "runtime":
            raise CapabilityContractError(
                f"runtime op {oid} must have runtime_availability=runtime"
            )

    uncovered = sorted(set(CAPABILITY_IDS) - provided_caps)
    if uncovered:
        raise CapabilityContractError(
            f"ontology capabilities with no runtime provider: {uncovered}"
        )

    return {
        "ok": True,
        "contract_version": CONTRACT_VERSION,
        "ontology_version": CAPABILITY_ONTOLOGY_VERSION,
        "runtime_coverage": len(runtime_ids),
        "contract_rows": len(index),
        "capability_providers": {
            c: len(v) for c, v in capability_index(index).items()
        },
    }


def load_and_validate_capability_contracts(
    path: Optional[Path] = None,
    *,
    catalog: Optional[UnifiedCatalogBundle] = None,
    root: Optional[Path] = None,
) -> Dict[str, Dict[str, Any]]:
    data = load_capability_contract_yaml(path)
    validate_capability_contracts(data, catalog=catalog, root=root)
    return contract_index(data)


def capability_contract_hash(path: Optional[Path] = None) -> str:
    data = load_capability_contract_yaml(path)
    blob = json.dumps(data, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def operations_for_capability(
    contracts: Dict[str, Dict[str, Any]],
    required_capability: str,
    *,
    include_compatible: bool = True,
) -> List[str]:
    wanted = (
        compatible_capabilities(required_capability)
        if include_compatible
        else {required_capability}
    )
    out: List[str] = []
    for oid, row in contracts.items():
        if str(row.get("runtime_availability") or "") != "runtime":
            continue
        caps = set(row.get("capabilities") or [])
        if caps & wanted:
            out.append(oid)
    return sorted(out)


def covers_required_outputs(
    contract: Dict[str, Any], required_outputs: Sequence[str]
) -> bool:
    if not required_outputs:
        return True
    provides = {str(p) for p in (contract.get("provides_fields") or [])}
    aliases = contract.get("field_aliases") or {}
    # aliases map raw→canonical; also treat alias values as provided.
    provides |= {str(v) for v in aliases.values()}
    return all(str(o) in provides for o in required_outputs)
