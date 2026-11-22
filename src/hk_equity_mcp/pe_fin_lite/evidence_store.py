"""Verified Evidence Store — only completed retrieve/compute values enter."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from hk_equity_mcp.pe_fin_lite.step_result import retrieve_step_succeeded


class EvidenceStore:
    def __init__(self) -> None:
        self.items: List[Dict[str, Any]] = []
        self.values: Dict[str, Any] = {}
        self.provenance_by_key: Dict[str, Dict[str, Any]] = {}
        self.missing: List[Dict[str, Any]] = []
        self.compute_records: List[Dict[str, Any]] = []
        self.reason_artifacts: List[Dict[str, Any]] = []

    def add_retrieve_result(self, *, step_id: str, result: Dict[str, Any]) -> None:
        record = {
            "step_id": step_id,
            "kind": "retrieve",
            "status": result.get("status"),
            "execution_ok": result.get("execution_ok"),
            "values": dict(result.get("values") or {}),
            "provenance": dict(result.get("provenance") or {}),
            "missing_outputs": list(result.get("missing_outputs") or []),
            "scope_match": result.get("scope_match"),
            "failure_reason": result.get("failure_reason"),
        }
        self.items.append(record)
        if retrieve_step_succeeded(result):
            for key, value in (result.get("values") or {}).items():
                self.values[key] = value
                self.values[f"{step_id}.{key}"] = value
                self.provenance_by_key[key] = dict(result.get("provenance") or {})
                self.provenance_by_key[f"{step_id}.{key}"] = dict(
                    result.get("provenance") or {}
                )
        else:
            self.missing.append(
                {
                    "step_id": step_id,
                    "missing_outputs": list(result.get("missing_outputs") or []),
                    "failure_reason": result.get("failure_reason"),
                    "status": result.get("status"),
                }
            )

    def add_compute_record(self, record: Dict[str, Any]) -> None:
        self.compute_records.append(dict(record))
        self.items.append({"kind": "compute", **dict(record)})
        if record.get("ok") and record.get("output_name") is not None:
            name = str(record["output_name"])
            self.values[name] = record.get("result")
            self.provenance_by_key[name] = {
                "source": "compute",
                "step_id": record.get("step_id"),
                "formula": record.get("formula"),
            }

    def add_reason_artifact(self, artifact: Dict[str, Any]) -> None:
        self.reason_artifacts.append(dict(artifact))
        self.items.append({"kind": "reason", **dict(artifact)})

    def resolve_operand(self, ref: str) -> Optional[Any]:
        if ref in self.values:
            return self.values[ref]
        return None

    def compact_for_prompt(self, *, max_chars: int = 12000) -> Dict[str, Any]:
        payload = {
            "verified_values": dict(self.values),
            "provenance_by_key": dict(self.provenance_by_key),
            "missing": list(self.missing),
            "compute_records": list(self.compute_records),
            "reason_artifacts": list(self.reason_artifacts),
        }
        # Soft trim: drop oldest retrieve item details if huge.
        text = str(payload)
        if len(text) <= max_chars:
            return payload
        return {
            "verified_values": dict(self.values),
            "provenance_by_key": {
                k: {"operation_id": (v or {}).get("operation_id"), "source": (v or {}).get("source")}
                for k, v in self.provenance_by_key.items()
            },
            "missing": list(self.missing),
            "compute_records": [
                {
                    "step_id": r.get("step_id"),
                    "output_name": r.get("output_name"),
                    "result": r.get("result"),
                    "ok": r.get("ok"),
                }
                for r in self.compute_records
            ],
            "reason_artifacts": [
                {"step_id": a.get("step_id"), "summary": a.get("summary")}
                for a in self.reason_artifacts
            ],
            "truncated": True,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "items": list(self.items),
            "verified_values": dict(self.values),
            "provenance_by_key": dict(self.provenance_by_key),
            "missing": list(self.missing),
            "compute_records": list(self.compute_records),
            "reason_artifacts": list(self.reason_artifacts),
        }
