"""Deterministic compute operand binding for PE-Fin Lite v1.2-EEA."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Dict, List, Mapping, Optional, Tuple

from hk_equity_mcp.pe_fin_lite.calculator import CalculatorError, run_compute_spec

COMPUTE_BINDING_EEA_VERSION = "pe_fin_lite_compute_binding_eea_v1"

COMPUTE_FAILURE_CLASSES = (
    "upstream_evidence_missing",
    "operand_binding_failed",
    "unit_incompatible",
    "period_incompatible",
    "entity_incompatible",
    "expression_invalid",
    "calculation_error",
    "non_numeric_operand",
    "division_by_zero",
)


def compute_binding_eea_hash() -> str:
    blob = json.dumps(
        {
            "version": COMPUTE_BINDING_EEA_VERSION,
            "failure_classes": list(COMPUTE_FAILURE_CLASSES),
            "no_eval": True,
            "safe_calculator_only": True,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _meta_for_ref(store: Any, ref: str) -> Dict[str, Any]:
    prov = {}
    if hasattr(store, "provenance_by_key"):
        prov = dict((store.provenance_by_key or {}).get(ref) or {})
    field_prov = {}
    if isinstance(prov.get("field_provenance"), dict):
        # Prefer exact field if ref ends with .field
        field = ref.split(".")[-1]
        field_prov = prov["field_provenance"].get(field) or {}
    return {
        "provenance": prov,
        "field": field_prov,
        "entity": field_prov.get("entity") or prov.get("entity"),
        "period": field_prov.get("period") or prov.get("period"),
        "basis": field_prov.get("basis") or prov.get("basis"),
        "unit": field_prov.get("unit") or prov.get("unit"),
    }


def bind_compute_operands(
    *,
    spec: Dict[str, Any],
    store: Any,
) -> Dict[str, Any]:
    """Resolve operand_refs → Evidence Store with typed failure classes."""
    refs = dict(spec.get("operand_refs") or {})
    expected_unit = spec.get("expected_unit") or spec.get("output_unit")
    bindings: List[Dict[str, Any]] = []
    resolved: Dict[str, float] = {}
    failure_class: Optional[str] = None
    errors: List[str] = []

    # Upstream retrieve blocked?
    missing_entries = list(getattr(store, "missing", None) or [])
    upstream_blocked = bool(missing_entries) and not refs

    for name, ref in refs.items():
        ref_s = str(ref)
        raw = store.resolve_operand(ref_s) if hasattr(store, "resolve_operand") else None
        meta = _meta_for_ref(store, ref_s)
        binding = {
            "operand": name,
            "operand_ref": ref_s,
            "value": raw,
            "entity": meta.get("entity"),
            "period": meta.get("period"),
            "basis": meta.get("basis"),
            "unit": meta.get("unit"),
            "ok": False,
        }
        if raw is None:
            # Distinguish upstream missing vs bad ref
            if any(ref_s in str(m.get("missing_outputs") or []) or ref_s.split(".")[0] == str(m.get("step_id")) for m in missing_entries):
                failure_class = failure_class or "upstream_evidence_missing"
                errors.append(f"upstream_evidence_missing:{ref_s}")
            else:
                failure_class = failure_class or "operand_binding_failed"
                errors.append(f"operand_binding_failed:{ref_s}")
            bindings.append(binding)
            continue
        try:
            num = float(raw) if not isinstance(raw, (int, float)) else float(raw)
        except (TypeError, ValueError):
            failure_class = failure_class or "non_numeric_operand"
            errors.append(f"non_numeric_operand:{ref_s}:{raw!r}")
            bindings.append(binding)
            continue
        # Unit check vs expected when both present
        unit = meta.get("unit")
        if expected_unit and unit and str(unit).lower() != str(expected_unit).lower():
            from hk_equity_mcp.pe_fin_lite.evidence_acceptance_eea import (
                units_compatible_or_convertible,
            )

            ok_u, scale, _ = units_compatible_or_convertible(expected_unit, unit)
            if not ok_u:
                failure_class = failure_class or "unit_incompatible"
                errors.append(f"unit_incompatible:{ref_s}")
                bindings.append(binding)
                continue
            if scale and scale != 1.0:
                num = num * float(scale)
                binding["transformation"] = f"unit_scale:{scale}"
        binding["value"] = num
        binding["ok"] = True
        resolved[name] = num
        bindings.append(binding)

    # Cross-operand entity/period compatibility
    entities = {b.get("entity") for b in bindings if b.get("ok") and b.get("entity")}
    periods = {str(b.get("period")).upper() for b in bindings if b.get("ok") and b.get("period")}
    if len({str(e).lower() for e in entities if e}) > 1:
        # soft digit match
        digits = set()
        for e in entities:
            d = re.sub(r"\D", "", str(e))
            if d:
                digits.add(d.zfill(5))
        if len(digits) > 1:
            failure_class = failure_class or "entity_incompatible"
            errors.append("entity_incompatible")
    if len(periods) > 1:
        failure_class = failure_class or "period_incompatible"
        errors.append("period_incompatible")

    if failure_class or len(resolved) != len(refs):
        if upstream_blocked and failure_class is None:
            failure_class = "upstream_evidence_missing"
        return {
            "ok": False,
            "resolved": resolved,
            "bindings": bindings,
            "failure_class": failure_class or "operand_binding_failed",
            "errors": errors,
            "expected_unit": expected_unit,
        }

    try:
        evaluated = run_compute_spec(spec, resolved_operands=resolved)
        return {
            "ok": True,
            "resolved": resolved,
            "bindings": bindings,
            "evaluated": evaluated,
            "failure_class": None,
            "errors": [],
            "expected_unit": expected_unit,
        }
    except CalculatorError as exc:
        msg = str(exc).lower()
        if "division by zero" in msg:
            fc = "division_by_zero"
        elif "invalid expression" in msg or "syntax" in msg or "disallowed" in msg:
            fc = "expression_invalid"
        else:
            fc = "calculation_error"
        return {
            "ok": False,
            "resolved": resolved,
            "bindings": bindings,
            "failure_class": fc,
            "errors": [str(exc)],
            "expected_unit": expected_unit,
        }
