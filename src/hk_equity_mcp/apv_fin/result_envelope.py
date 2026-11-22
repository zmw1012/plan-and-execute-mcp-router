"""UnifiedRealMcpGateway result envelope normalizer (APV-Fin v1.4).

Only accepts the known Gateway/MCP nesting — never recursively scavenges for
arbitrary ``rows`` keys elsewhere in the tree.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

ENVELOPE_NORMALIZER_VERSION = "apv_fin_result_envelope_v1"


class EnvelopeNormalizationError(ValueError):
    def __init__(self, message: str, *, error_type: str = "envelope_normalize_failed"):
        super().__init__(message)
        self.error_type = error_type
        self.message = message


def result_envelope_hash() -> str:
    blob = json.dumps(
        {
            "version": ENVELOPE_NORMALIZER_VERSION,
            "accepted_paths": [
                "outer.success+operation_id+result",
                "result.raw.content[*].json",
                "result.normalized",
                "result.raw.structured_content.result",
            ],
        },
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _is_aktools_table_payload(obj: Any) -> bool:
    if not isinstance(obj, dict):
        return False
    # Explicit AKTools/Gateway table shape.
    if "rows" in obj and isinstance(obj.get("rows"), list):
        return True
    if "columns" in obj and isinstance(obj.get("columns"), list) and "rows" in obj:
        return True
    return False


def _extract_json_payloads_from_content(
    content: Sequence[Any],
) -> Tuple[List[Dict[str, Any]], List[str], List[str]]:
    payloads: List[Dict[str, Any]] = []
    texts: List[str] = []
    paths: List[str] = []
    for i, entry in enumerate(content or []):
        if not isinstance(entry, dict):
            continue
        if "json" in entry and isinstance(entry["json"], dict):
            payloads.append(entry["json"])
            paths.append(f"raw.content[{i}].json")
        if "text" in entry and entry["text"] is not None:
            texts.append(str(entry["text"]))
            paths.append(f"raw.content[{i}].text")
        if "resource" in entry and entry["resource"] is not None:
            # Preserve provenance only; do not treat resource as table rows.
            paths.append(f"raw.content[{i}].resource")
    return payloads, texts, paths


def _merge_table_fields(payloads: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    columns: List[str] = []
    total_rows = None
    returned_rows = None
    offset = None
    truncated = None
    name = None
    for payload in payloads:
        if not _is_aktools_table_payload(payload):
            continue
        name = name or payload.get("name")
        if total_rows is None and payload.get("total_rows") is not None:
            total_rows = payload.get("total_rows")
        if returned_rows is None and payload.get("returned_rows") is not None:
            returned_rows = payload.get("returned_rows")
        if offset is None and payload.get("offset") is not None:
            offset = payload.get("offset")
        if truncated is None and payload.get("truncated") is not None:
            truncated = bool(payload.get("truncated"))
        cols = payload.get("columns")
        if isinstance(cols, list) and not columns:
            columns = [str(c) for c in cols]
        for row in payload.get("rows") or []:
            if isinstance(row, dict):
                rows.append(row)
    if returned_rows is None:
        returned_rows = len(rows)
    return {
        "name": name,
        "rows": rows,
        "columns": columns,
        "total_rows": total_rows,
        "returned_rows": returned_rows,
        "offset": int(offset or 0),
        "truncated": bool(truncated) if truncated is not None else False,
    }


def normalize_gateway_envelope(raw: Any) -> Dict[str, Any]:
    """Normalize a Gateway invoke payload into a stable envelope view.

    Accepts either:
    - full outer invoke object ``{success, operation_id, arguments, result}``
    - mid-layer ``result`` object ``{success, server_id, raw, normalized, ...}``
    """
    errors: List[str] = []
    provenance: List[str] = []
    protocol_success: Optional[bool] = None
    execution_success: Optional[bool] = None
    operation_id: Optional[str] = None
    outer = raw if isinstance(raw, dict) else None
    if outer is None:
        raise EnvelopeNormalizationError(
            "envelope must be an object",
            error_type="envelope_type_error",
        )

    # Outer invoke wrapper.
    mid = outer
    if "result" in outer and isinstance(outer.get("result"), dict):
        if "success" in outer and ("operation_id" in outer or "arguments" in outer):
            protocol_success = bool(outer.get("success"))
            operation_id = (
                str(outer.get("operation_id")) if outer.get("operation_id") else None
            )
            mid = outer["result"]
            provenance.append("outer.result")
            if outer.get("protocol_success") is not None:
                protocol_success = bool(outer.get("protocol_success"))
            if outer.get("execution_success") is not None:
                execution_success = bool(outer.get("execution_success"))

    if not isinstance(mid, dict):
        raise EnvelopeNormalizationError(
            "result layer must be an object",
            error_type="envelope_result_type_error",
        )

    if protocol_success is None and "protocol_success" in mid:
        protocol_success = bool(mid.get("protocol_success"))
    if execution_success is None and "execution_success" in mid:
        execution_success = bool(mid.get("execution_success"))
    if execution_success is None and "success" in mid:
        execution_success = bool(mid.get("success"))
    if protocol_success is None and "success" in mid:
        protocol_success = bool(mid.get("success"))
    if operation_id is None and mid.get("operation_id"):
        operation_id = str(mid.get("operation_id"))

    payloads: List[Dict[str, Any]] = []
    textual: List[str] = []

    # Preferred single table source (avoid double-counting identical pages).
    normalized = mid.get("normalized")
    raw_layer = mid.get("raw") if isinstance(mid.get("raw"), dict) else None
    if isinstance(normalized, dict) and _is_aktools_table_payload(normalized):
        payloads.append(normalized)
        provenance.append("result.normalized")
    elif isinstance(raw_layer, dict):
        if raw_layer.get("is_error") is True:
            errors.append("raw.is_error=true")
        content = raw_layer.get("content")
        if isinstance(content, list):
            c_payloads, c_texts, c_paths = _extract_json_payloads_from_content(content)
            payloads.extend(c_payloads)
            textual.extend(c_texts)
            provenance.extend([f"result.{p}" for p in c_paths])
        if not payloads:
            structured = raw_layer.get("structured_content")
            if isinstance(structured, dict):
                sc_result = structured.get("result")
                if isinstance(sc_result, dict) and _is_aktools_table_payload(sc_result):
                    payloads.append(sc_result)
                    provenance.append("result.raw.structured_content.result")
    elif _is_aktools_table_payload(mid):
        # Mid-layer itself is already an AKTools table (offline fixtures).
        payloads.append(mid)
        provenance.append("result(table)")

    # Always collect textual content even when table came from normalized.
    if isinstance(raw_layer, dict):
        content = raw_layer.get("content")
        if isinstance(content, list):
            _, c_texts, c_paths = _extract_json_payloads_from_content(content)
            for t in c_texts:
                if t not in textual:
                    textual.append(t)
            for p in c_paths:
                if p.endswith(".text"):
                    provenance.append(f"result.{p}")

    table = _merge_table_fields(payloads)
    if not payloads and not textual:
        errors.append("no_supported_payload_path")

    ok = bool(execution_success) and not (
        errors and "no_supported_payload_path" in errors and not table["rows"] and not textual
    )
    return {
        "protocol_success": protocol_success,
        "execution_success": execution_success,
        "operation_id": operation_id,
        "normalized_payloads": payloads,
        "rows": table["rows"],
        "columns": table["columns"],
        "total_rows": table["total_rows"],
        "returned_rows": table["returned_rows"],
        "offset": table["offset"],
        "truncated": table["truncated"],
        "name": table.get("name"),
        "textual_content": textual,
        "raw_provenance_paths": provenance,
        "normalization_errors": errors,
        "ok": bool(execution_success) if execution_success is not None else bool(payloads or textual),
        "envelope_version": ENVELOPE_NORMALIZER_VERSION,
        "adapter_ready": bool(table["rows"] or textual),
    }


def envelope_supports_pagination(envelope: Dict[str, Any]) -> bool:
    if not envelope.get("truncated"):
        return False
    total = envelope.get("total_rows")
    returned = int(envelope.get("returned_rows") or 0)
    offset = int(envelope.get("offset") or 0)
    if total is None:
        return returned > 0
    try:
        return offset + returned < int(total)
    except (TypeError, ValueError):
        return returned > 0
