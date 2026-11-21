"""Direct-operation schema bundle for Flat All Tools (all 39 runtime ops).

Fail closed when a runtime operation lacks a usable schema. Never invent schemas.
Never include blocked/rejected operations.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

from hk_equity_mcp.experiment.unified_flat_action_mapping import (
    FlatActionMapping,
    build_flat_action_mapping,
)
from hk_equity_mcp.operation_resolver.schemas import OperationSpec
from hk_equity_mcp.real_mcp.unified_catalog import UnifiedCatalogBundle


def _sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def estimate_tokens(char_count: int) -> int:
    """Deterministic, reproducible token estimate (chars/4, ceil).

    Not a provider tokenizer; used only for offline schema-overhead accounting.
    """
    if char_count <= 0:
        return 0
    return int(math.ceil(char_count / 4.0))


def _assert_usable_schema(op: OperationSpec) -> Dict[str, Any]:
    if not op.operation_id:
        raise RuntimeError("Runtime operation missing operation_id (fail-closed)")
    if not (op.name or op.summary):
        raise RuntimeError(
            f"Runtime operation {op.operation_id} missing name/summary (fail-closed)"
        )
    if not (op.description or op.summary):
        raise RuntimeError(
            f"Runtime operation {op.operation_id} missing description (fail-closed)"
        )
    try:
        schema = op.json_schema()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Runtime operation {op.operation_id} schema build failed "
            f"(fail-closed): {type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(schema, dict) or schema.get("type") != "object":
        raise RuntimeError(
            f"Runtime operation {op.operation_id} has unusable JSON schema "
            "(fail-closed; refusing to invent schema)"
        )
    if "properties" not in schema or not isinstance(schema["properties"], dict):
        raise RuntimeError(
            f"Runtime operation {op.operation_id} schema missing properties "
            "(fail-closed; refusing to invent schema)"
        )
    return schema


def build_operation_schema_entry(
    op: OperationSpec, *, action_name: str
) -> Dict[str, Any]:
    schema = _assert_usable_schema(op)
    return {
        "action_name": action_name,
        "operation_id": op.operation_id,
        "name": op.name,
        "summary": op.summary,
        "description": op.description or op.summary,
        "input_json_schema": schema,
        "constraints": list(op.constraints or []),
        "read_only": bool(op.read_only),
        "depends_on": list(op.depends_on or []),
        "invocation_guidance": op.invocation_guidance or "",
        "service_id": op.service_id,
    }


@dataclass(frozen=True)
class FlatSchemaBundle:
    entries: List[Dict[str, Any]]
    mapping: FlatActionMapping
    canonical_json: str
    schema_bundle_sha256: str
    serialized_schema_char_count: int
    token_estimate: int
    runtime_operation_ids: List[str]
    blocked_operation_ids: List[str]

    def prompt_section(self) -> str:
        """Human/model-readable section listing every runtime operation schema."""
        lines: List[str] = [
            "Directly exposed runtime operations (Flat All Tools — no retrieval).",
            "Call tools by action_name. Always treat operation_id as the true identity.",
            f"Exposed count: {len(self.entries)} (blocked/rejected are not listed).",
            "",
        ]
        for entry in self.entries:
            lines.append("---")
            lines.append(f"action_name: {entry['action_name']}")
            lines.append(f"operation_id: {entry['operation_id']}")
            lines.append(f"name: {entry['name']}")
            lines.append(f"summary: {entry.get('summary') or ''}")
            lines.append(f"description: {entry['description']}")
            lines.append(
                "input_json_schema: "
                + json.dumps(entry["input_json_schema"], ensure_ascii=False, sort_keys=True)
            )
            constraints = entry.get("constraints") or []
            lines.append(
                "constraints: "
                + (
                    "; ".join(str(c) for c in constraints)
                    if constraints
                    else "(none)"
                )
            )
            if entry.get("depends_on"):
                lines.append(
                    "depends_on: " + ", ".join(str(x) for x in entry["depends_on"])
                )
            if entry.get("invocation_guidance"):
                lines.append(f"invocation_guidance: {entry['invocation_guidance']}")
            lines.append(f"read_only: {entry.get('read_only')}")
            lines.append(f"service_id: {entry.get('service_id')}")
        lines.append("---")
        return "\n".join(lines)

    def to_metadata(self) -> Dict[str, Any]:
        return {
            "schema_bundle_sha256": self.schema_bundle_sha256,
            "serialized_schema_char_count": self.serialized_schema_char_count,
            "token_estimate": self.token_estimate,
            "token_estimate_method": "ceil(char_count/4)",
            "runtime_exposed_count": len(self.runtime_operation_ids),
            "runtime_operation_ids": list(self.runtime_operation_ids),
            "blocked_operation_ids": list(self.blocked_operation_ids),
            "action_mapping_fingerprint": self.mapping.fingerprint,
            "action_mapping": self.mapping.to_dict(),
        }


def build_flat_schema_bundle(
    catalog_bundle: UnifiedCatalogBundle,
    *,
    mapping: Optional[FlatActionMapping] = None,
) -> FlatSchemaBundle:
    """Build the Flat direct-operation schema bundle from the frozen catalog."""
    stats = catalog_bundle.stats
    runtime_ids = list(stats.runtime_operation_ids)
    # Deduplicate by operation_id (catalog already unique; enforce again).
    runtime_ids = sorted(set(runtime_ids))
    blocked_ids = list(stats.blocked_operation_ids)

    if len(runtime_ids) != int(stats.runtime_exposed_unique):
        raise RuntimeError(
            "Catalog drift: runtime_operation_ids length != runtime_exposed_unique"
        )
    for oid in blocked_ids:
        if oid in runtime_ids or oid in catalog_bundle.registry:
            raise RuntimeError(
                f"Blocked/rejected operation leaked into Flat runtime exposure: {oid}"
            )
    for oid in runtime_ids:
        rec = catalog_bundle.records.get(oid)
        if rec is not None and rec.is_blocked:
            raise RuntimeError(f"Blocked record present in runtime ids: {oid}")
        if oid not in catalog_bundle.registry:
            raise RuntimeError(
                f"Runtime operation_id missing from registry (fail-closed): {oid}"
            )

    mapping = mapping or build_flat_action_mapping(runtime_ids)
    if set(mapping.operation_to_action) != set(runtime_ids):
        raise RuntimeError(
            "Flat action mapping does not exactly cover runtime operation_ids"
        )

    entries: List[Dict[str, Any]] = []
    for oid in runtime_ids:
        op = catalog_bundle.registry[oid]
        action = mapping.action_name(oid)
        entries.append(build_operation_schema_entry(op, action_name=action))

    if len(entries) != len(runtime_ids):
        raise RuntimeError("Flat schema bundle entry count mismatch")

    canonical = {
        "entries": entries,
        "runtime_operation_ids": runtime_ids,
        "blocked_operation_ids": blocked_ids,
        "action_mapping_fingerprint": mapping.fingerprint,
    }
    canonical_json = json.dumps(
        canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    digest = _sha256_hex(canonical_json)
    char_count = len(canonical_json)
    return FlatSchemaBundle(
        entries=entries,
        mapping=mapping,
        canonical_json=canonical_json,
        schema_bundle_sha256=digest,
        serialized_schema_char_count=char_count,
        token_estimate=estimate_tokens(char_count),
        runtime_operation_ids=runtime_ids,
        blocked_operation_ids=blocked_ids,
    )


def assert_no_blocked_in_bundle(
    bundle: FlatSchemaBundle, blocked_ids: Sequence[str]
) -> None:
    exposed = set(bundle.runtime_operation_ids)
    leaked = sorted(set(blocked_ids) & exposed)
    if leaked:
        raise RuntimeError(f"Blocked ops in Flat bundle: {leaked}")
