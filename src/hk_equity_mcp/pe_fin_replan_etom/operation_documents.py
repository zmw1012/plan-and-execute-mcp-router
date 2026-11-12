"""Operation documents built from the official ETOM registry (no gold, no tasks).

Documents are the ``OperationSpec.index_document()`` text produced by
``hk_equity_mcp.etom.convert.convert_registry`` (summary / service / operation /
action terms / required & optional parameters / server categories). This keeps
the retrieval catalog identical to the Operation-Visible ETOM baseline while the
embedding CACHE for this condition lives in its own namespace.

Binder contracts come from the official tool view (input_schema as published).
"""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

from hk_equity_mcp.operation_resolver.schemas import OperationSpec

DOCUMENTS_VERSION = "pe_fin_replan_etom_operation_documents_v1"
_SUMMARY_MAX = 320


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def documents_for_registry(registry: Dict[str, OperationSpec]) -> List[Tuple[str, str, str]]:
    """Ordered (operation_id, document_text, document_sha256)."""
    rows: List[Tuple[str, str, str]] = []
    for oid in sorted(registry):
        doc = registry[oid].index_document()
        rows.append((oid, doc, _sha256_text(doc)))
    return rows


def documents_hash(registry: Dict[str, OperationSpec]) -> str:
    parts = [f"{oid}\t{dhash}" for oid, _doc, dhash in documents_for_registry(registry)]
    return _sha256_text("\n".join(parts) + "\n")


def build_operation_documents(
    registry: Dict[str, OperationSpec],
    official: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Build the doc table, a document hash, and per-operation metadata.

    ``official`` (optional) is only used to cross-check that every registry
    operation has an official description (fail closed on mismatch).
    """
    if official is not None and set(official) != set(registry):
        missing = sorted(set(registry) ^ set(official))[:5]
        raise ValueError(f"registry/official operation_id mismatch, e.g. {missing}")
    rows = documents_for_registry(registry)
    return {
        "version": DOCUMENTS_VERSION,
        "operation_count": len(rows),
        "documents_sha256": documents_hash(registry),
        "documents": {oid: {"text": doc, "sha256": dsha} for oid, doc, dsha in rows},
    }


def contract_from_official(operation_id: str, view: Dict[str, Any]) -> Dict[str, Any]:
    """Binder-facing contract for one described operation (no gold)."""
    schema = copy.deepcopy(view.get("input_schema") or {})
    if not isinstance(schema, dict):
        schema = {}
    description = str(view.get("description") or "")
    return {
        "operation_id": operation_id,
        "name": view.get("tool_name"),
        "server_name": view.get("server_name"),
        "summary": description[:_SUMMARY_MAX],
        "description": description,
        "json_schema": schema,
        "required": list(schema.get("required") or []),
        "constraints": [
            f"server: {view.get('server_name')}",
            f"categories: {', '.join(view.get('categories') or [])}",
        ],
    }


def registry_from_payload(
    payload: Dict[str, Any],
) -> Tuple[Dict[str, OperationSpec], Dict[str, Dict[str, Any]]]:
    """Official ETOM ``{"servers": [...]}`` payload -> (registry, official views).

    Thin wrapper over ``hk_equity_mcp.etom.convert.convert_registry`` so tests,
    mock runs and the live path build the catalog the same way.
    """
    from hk_equity_mcp.etom.convert import convert_registry

    return convert_registry(payload)


def registry_catalog_fingerprint(official: Dict[str, Dict[str, Any]]) -> str:
    """Stable fingerprint over (operation_id, description, input_schema)."""
    payload = json.dumps(
        [
            {
                "operation_id": oid,
                "description": official[oid].get("description"),
                "input_schema": official[oid].get("input_schema"),
            }
            for oid in sorted(official)
        ],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return _sha256_text(payload)


def operation_ids_sorted(registry: Dict[str, OperationSpec]) -> Sequence[str]:
    return [oid for oid, _d, _h in documents_for_registry(registry)]
