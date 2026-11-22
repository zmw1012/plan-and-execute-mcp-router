"""Evidence Store with provenance classes for PE-Fin RePlan-ETOM.

Every verified value carries exactly one provenance class:

* ``observed``       – came out of a tool result envelope (business data);
* ``grounded_input`` – an argument value that appears verbatim in the question
                       (scope/identifier input, not tool evidence);
* ``derived``        – produced by the safe calculator from verified operands.

Wraps ``pe_fin_lite.evidence_store.EvidenceStore`` generically (no HK semantics).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from hk_equity_mcp.pe_fin_lite.evidence_store import EvidenceStore

PROVENANCE_OBSERVED = "observed"
PROVENANCE_GROUNDED_INPUT = "grounded_input"
PROVENANCE_DERIVED = "derived"
PROVENANCE_CLASSES = (PROVENANCE_OBSERVED, PROVENANCE_GROUNDED_INPUT, PROVENANCE_DERIVED)


class EtomEvidenceStore(EvidenceStore):
    def __init__(self) -> None:
        super().__init__()
        self.provenance_class_by_key: Dict[str, str] = {}

    # -- writers -----------------------------------------------------------
    def _set(self, key: str, value: Any, klass: str, provenance: Dict[str, Any]) -> None:
        if klass not in PROVENANCE_CLASSES:
            raise ValueError(f"unknown provenance class {klass!r}")
        self.values[key] = value
        self.provenance_by_key[key] = {**dict(provenance), "provenance_class": klass}
        self.provenance_class_by_key[key] = klass

    def add_observed(
        self,
        *,
        step_id: str,
        values: Dict[str, Any],
        provenance: Optional[Dict[str, Any]] = None,
    ) -> None:
        prov = dict(provenance or {})
        prov.setdefault("step_id", step_id)
        for name, value in (values or {}).items():
            self._set(str(name), value, PROVENANCE_OBSERVED, prov)
            self._set(f"{step_id}.{name}", value, PROVENANCE_OBSERVED, prov)
        self.items.append(
            {"kind": "observed", "step_id": step_id, "keys": sorted(str(k) for k in values or {})}
        )

    def add_grounded_input(
        self, *, step_id: str, name: str, value: Any, source: str = "question"
    ) -> None:
        key = f"{step_id}.input.{name}"
        self._set(
            key,
            value,
            PROVENANCE_GROUNDED_INPUT,
            {"step_id": step_id, "source": source, "argument": name},
        )
        self.items.append({"kind": "grounded_input", "step_id": step_id, "key": key})

    def add_retrieve_result(self, *, step_id: str, result: Dict[str, Any]) -> None:
        before = set(self.values)
        super().add_retrieve_result(step_id=step_id, result=result)
        for key in set(self.values) - before:
            self.provenance_class_by_key.setdefault(key, PROVENANCE_OBSERVED)
            self.provenance_by_key[key] = {
                **dict(self.provenance_by_key.get(key) or {}),
                "provenance_class": PROVENANCE_OBSERVED,
            }

    def add_compute_record(self, record: Dict[str, Any]) -> None:
        super().add_compute_record(record)
        if record.get("ok") and record.get("output_name") is not None:
            name = str(record["output_name"])
            self.provenance_class_by_key[name] = PROVENANCE_DERIVED
            self.provenance_by_key[name] = {
                **dict(self.provenance_by_key.get(name) or {}),
                "provenance_class": PROVENANCE_DERIVED,
            }
            step_key = f"{record.get('step_id')}.{name}"
            self._set(
                step_key,
                record.get("result"),
                PROVENANCE_DERIVED,
                {"source": "compute", "step_id": record.get("step_id"), "formula": record.get("formula")},
            )

    # -- readers -----------------------------------------------------------
    def class_of(self, key: str) -> Optional[str]:
        return self.provenance_class_by_key.get(key)

    def values_by_class(self, *classes: str) -> Dict[str, Any]:
        want = set(classes)
        return {
            k: v for k, v in self.values.items() if self.provenance_class_by_key.get(k) in want
        }

    def evidence_summary(self) -> Dict[str, Any]:
        counts = {c: 0 for c in PROVENANCE_CLASSES}
        for klass in self.provenance_class_by_key.values():
            counts[klass] = counts.get(klass, 0) + 1
        return {"n_values": len(self.values), "by_class": counts}

    def to_dict(self) -> Dict[str, Any]:
        out = super().to_dict()
        out["provenance_class_by_key"] = dict(self.provenance_class_by_key)
        out["summary"] = self.evidence_summary()
        return out


def make_evidence_store() -> EtomEvidenceStore:
    return EtomEvidenceStore()


def record_tool_evidence(
    store: EtomEvidenceStore,
    *,
    projection: Dict[str, Any],
    scope: Optional[Dict[str, Any]] = None,
    missing_outputs: Optional[List[str]] = None,
    step_id: Optional[str] = None,
) -> str:
    provenance = str(projection.get("provenance") or PROVENANCE_OBSERVED)
    if provenance not in PROVENANCE_CLASSES:
        provenance = PROVENANCE_OBSERVED
    values = dict(projection.get("observed_values") or {})
    for key in ("operation_id", "server_name", "tool_name"):
        if projection.get(key) is not None:
            values.setdefault(key, projection.get(key))
    if projection.get("identifiers"):
        values["identifiers"] = list(projection.get("identifiers") or [])
    if projection.get("normalized_result") is not None:
        values["normalized_result"] = projection.get("normalized_result")
    if projection.get("projected_text") is not None:
        values["projected_text"] = projection.get("projected_text")
    sid = step_id or f"s{len(store.items) + 1}"
    result = {
        "status": "ok" if projection.get("success") else "error",
        "execution_ok": bool(projection.get("success")),
        "values": values,
        "provenance": {
            "class": provenance,
            "operation_id": projection.get("operation_id"),
            "server_name": projection.get("server_name"),
            "tool_name": projection.get("tool_name"),
            "arguments": projection.get("arguments") or {},
            "raw_result_ref": projection.get("raw_result_ref"),
            "scope": scope or {},
        },
        "missing_outputs": list(missing_outputs or []),
        "scope_match": True,
        "failure_reason": None if projection.get("success") else "invoke_failed",
    }
    store.add_retrieve_result(step_id=sid, result=result)
    return sid


def record_derived_evidence(
    store: EtomEvidenceStore,
    *,
    step_id: str,
    output_name: str,
    result: Any,
    formula: Optional[str] = None,
    ok: bool = True,
) -> None:
    store.add_compute_record(
        {
            "step_id": step_id,
            "output_name": output_name,
            "result": result,
            "formula": formula,
            "ok": ok,
            "provenance": PROVENANCE_DERIVED,
        }
    )


def compress_evidence(store: EvidenceStore, *, limit: int = 24) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    class_map = getattr(store, "provenance_class_by_key", {}) or {}
    for item in store.items[-limit:]:
        prov = item.get("provenance") or {}
        class_name = None
        if isinstance(prov, dict):
            class_name = prov.get("class") or prov.get("provenance_class")
        if item.get("kind") == "compute":
            class_name = PROVENANCE_DERIVED
        if item.get("kind") == "observed":
            class_name = PROVENANCE_OBSERVED
        if item.get("kind") == "grounded_input":
            class_name = PROVENANCE_GROUNDED_INPUT
        values = item.get("values") if isinstance(item.get("values"), dict) else {}
        sid = item.get("step_id") or item.get("kind")
        if not class_name and sid in class_map:
            class_name = class_map.get(sid)
        rows.append(
            {
                "evidence_id": sid,
                "kind": item.get("kind") or "retrieve",
                "provenance": class_name or PROVENANCE_OBSERVED,
                "normalized_result": values or item.get("result"),
                "observed_values": values or {},
                "identifiers": list(values.get("identifiers") or []) if values else [],
                "success": bool(
                    item.get("execution_ok")
                    if "execution_ok" in item
                    else item.get("ok", True)
                ),
                "source_server": (prov or {}).get("server_name")
                if isinstance(prov, dict)
                else None,
                "source_tool": (prov or {}).get("tool_name")
                if isinstance(prov, dict)
                else None,
                "arguments": (prov or {}).get("arguments") if isinstance(prov, dict) else {},
                "raw_result_ref": (prov or {}).get("raw_result_ref")
                if isinstance(prov, dict)
                else None,
                "scope": (prov or {}).get("scope") if isinstance(prov, dict) else {},
                "missing_outputs": list(item.get("missing_outputs") or []),
            }
        )
    return rows
