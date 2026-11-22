"""Frozen operation-document construction + integrity checks (eval/index only)."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

from hk_equity_mcp.experiment.pe_fin_lite_retrieval_audit import OPERATION_DOCUMENT_VERSION
from hk_equity_mcp.pe_fin_lite.tool_retrieval import enrich_operation_document
from hk_equity_mcp.real_mcp.unified_catalog import UnifiedCatalogBundle


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_operation_document(
    *,
    operation_id: str,
    raw: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    metadata = metadata or {}
    params = raw.get("parameters") or []
    inputs = []
    for p in params:
        if isinstance(p, dict) and p.get("name"):
            inputs.append(
                {
                    "name": p.get("name"),
                    "type": p.get("type") or p.get("schema", {}).get("type"),
                    "required": bool(p.get("required")),
                }
            )
    outputs = list(raw.get("provides") or metadata.get("provides") or [])
    constraints = []
    for key in ("constraints", "notes", "cutoff_note", "availability_note"):
        val = raw.get(key) or metadata.get(key)
        if isinstance(val, list):
            constraints.extend(str(x) for x in val)
        elif val:
            constraints.append(str(val))
    contract = {
        "name": raw.get("name") or operation_id,
        "description": raw.get("description") or raw.get("summary") or "",
        "required_parameters": [p for p in params if isinstance(p, dict) and p.get("required")],
        "optional_parameters": [
            p for p in params if isinstance(p, dict) and not p.get("required")
        ],
        "constraints": constraints,
    }
    text = enrich_operation_document(
        operation_id=operation_id,
        contract=contract,
        metadata={
            "domain": metadata.get("domain") or raw.get("domain") or "",
            "capabilities": metadata.get("capabilities") or raw.get("capabilities") or [],
            "provides": outputs,
            "requires": metadata.get("requires") or raw.get("requires") or [],
            "source_type": metadata.get("source_type") or raw.get("source_class") or "",
        },
    )
    doc = {
        "operation_id": operation_id,
        "name": contract["name"],
        "description": contract["description"],
        "domain": metadata.get("domain") or raw.get("domain") or "",
        "capability_summary": list(
            metadata.get("capabilities") or raw.get("capabilities") or []
        ),
        "inputs": inputs,
        "outputs": outputs,
        "constraints": constraints,
        "document_text": text,
        "document_sha256": _sha(text),
        "operation_document_version": OPERATION_DOCUMENT_VERSION,
    }
    return doc


def build_all_operation_documents(bundle: UnifiedCatalogBundle) -> Dict[str, Any]:
    docs: Dict[str, Dict[str, Any]] = {}
    for oid, rec in sorted(bundle.records.items()):
        meta = {
            "domain": rec.preferred_domain or (rec.domains[0] if rec.domains else ""),
            "capabilities": list((rec.source_raw or {}).get("capabilities") or []),
            "provides": list((rec.source_raw or {}).get("provides") or []),
            "source_type": (rec.source_raw or {}).get("source_class"),
        }
        docs[oid] = build_operation_document(
            operation_id=oid, raw=rec.source_raw or {}, metadata=meta
        )
    integrity = audit_operation_documents(docs, bundle=bundle)
    ordered_hashes = [docs[oid]["document_sha256"] for oid in sorted(docs)]
    documents_sha256 = _sha("\n".join(f"{oid}:{h}" for oid, h in zip(sorted(docs), ordered_hashes)))
    return {
        "documents": docs,
        "operation_documents_sha256": documents_sha256,
        "operation_document_version": OPERATION_DOCUMENT_VERSION,
        "integrity": integrity,
        "n_documents": len(docs),
    }


def audit_operation_documents(
    docs: Dict[str, Dict[str, Any]],
    *,
    bundle: Optional[UnifiedCatalogBundle] = None,
) -> Dict[str, Any]:
    empty = []
    missing_domain = []
    missing_io = []
    duplicate_texts: Dict[str, List[str]] = {}
    for oid, doc in docs.items():
        text = str(doc.get("document_text") or "").strip()
        if not text or text == f"operation_id: {oid}":
            empty.append(oid)
        if not str(doc.get("domain") or "").strip():
            missing_domain.append(oid)
        if not doc.get("inputs") and not doc.get("outputs"):
            missing_io.append(oid)
        duplicate_texts.setdefault(doc.get("document_sha256") or "", []).append(oid)
    dup_groups = [ids for ids in duplicate_texts.values() if len(ids) > 1]
    gold_without_doc = []
    if bundle is not None:
        for oid, rec in bundle.records.items():
            if rec.formal_gold_eligible and oid not in docs:
                gold_without_doc.append(oid)
            if rec.formal_gold_eligible and oid in docs:
                if not str(docs[oid].get("document_text") or "").strip():
                    gold_without_doc.append(oid)
    return {
        "empty_documents": sorted(empty),
        "duplicate_document_groups": dup_groups,
        "duplicate_operation_ids": [],  # keys are unique by construction
        "missing_domain": sorted(missing_domain),
        "missing_inputs_outputs": sorted(missing_io),
        "gold_eligible_without_valid_document": sorted(set(gold_without_doc)),
        "ok": not empty and not gold_without_doc,
    }
