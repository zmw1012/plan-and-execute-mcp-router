"""Temporal cutoff contamination scanner for RePlan HK-155 (read-only)."""

from __future__ import annotations

import re
from datetime import date
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from hk_equity_mcp.experiment.pe_fin_replan_r7_hk155_final_audit import (
    BENCHMARK_CUTOFF,
)

# Allow YYYY-MM-DD and ISO datetime prefixes (YYYY-MM-DDTHH:MM:SS...).
DATE_RE = re.compile(r"(?<!\d)(20\d{2})-(\d{2})-(\d{2})(?!\d)")
DATE_FIELD_HINTS = (
    "report_date",
    "trade_date",
    "release_date",
    "to_date",
    "from_date",
    "end_date",
    "start_date",
    "as_of",
    "effective_date",
    "announcement_date",
    "filing_date",
    "date",
)


def parse_cutoff(s: str = BENCHMARK_CUTOFF) -> date:
    y, m, d = s.split("-")
    return date(int(y), int(m), int(d))


def parse_iso_date(text: Any) -> Optional[date]:
    if text is None:
        return None
    if isinstance(text, date) and not isinstance(text, type(None)):
        try:
            return date(text.year, text.month, text.day)
        except Exception:  # noqa: BLE001
            return None
    s = str(text).strip()
    m = DATE_RE.search(s)
    if not m:
        return None
    try:
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except ValueError:
        return None


def is_post_cutoff(d: Optional[date], cutoff: date) -> bool:
    return bool(d and d > cutoff)


def _walk_dates(
    obj: Any,
    *,
    path: str = "$",
    out: Optional[List[Tuple[str, str, date]]] = None,
) -> List[Tuple[str, str, date]]:
    """Collect (json_path, field_or_context, date) from nested structures."""
    if out is None:
        out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = str(k)
            child = f"{path}.{key}"
            if isinstance(v, (str, int, float)) or v is None:
                d = parse_iso_date(v)
                if d is not None:
                    out.append((child, key, d))
            else:
                _walk_dates(v, path=child, out=out)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _walk_dates(v, path=f"{path}[{i}]", out=out)
    elif isinstance(obj, str):
        for m in DATE_RE.finditer(obj):
            try:
                d = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            except ValueError:
                continue
            out.append((path, "text", d))
    return out


def _canonical_key(
    *,
    operation_id: str,
    field: str,
    d: date,
    entity: str = "",
) -> Tuple[str, str, str, str]:
    return (operation_id or "", field or "", d.isoformat(), entity or "")


def analyze_task_temporal(
    row: Dict[str, Any],
    *,
    cutoff: str = BENCHMARK_CUTOFF,
) -> Dict[str, Any]:
    """Return per-task temporal contamination layers A–I (descriptive)."""
    cut = parse_cutoff(cutoff)
    tid = str(row.get("task_id") or row.get("id") or "")
    question = str(row.get("question") or "")
    gold_values = row.get("gold_values")
    final_answer = str(row.get("final_answer") or "")
    answer_correct = row.get("answer_correct")

    # Layer A: question/gold only (NOT automatic leakage)
    layer_a_dates = []
    for path, field, d in _walk_dates({"question": question, "gold_values": gold_values}):
        if is_post_cutoff(d, cut):
            layer_a_dates.append(
                {"path": path, "field": field, "date": d.isoformat(), "layer": "A"}
            )

    # Layer B: planner/replanner scopes (plans / decisions)
    layer_b_hits = []
    for key in (
        "initial_plan",
        "remaining_plan_history",
        "replan_decisions",
        "plan_diff_history",
    ):
        for path, field, d in _walk_dates(row.get(key)):
            if not is_post_cutoff(d, cut):
                continue
            fl = field.lower()
            if fl in DATE_FIELD_HINTS or any(h in fl for h in ("date", "period")):
                layer_b_hits.append(
                    {"path": f"{key}:{path}", "field": field, "date": d.isoformat()}
                )

    # Layers C–G from executed steps + evidence store
    evidence_events: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}
    layer_c: List[Dict[str, Any]] = []
    layer_d_e: List[Dict[str, Any]] = []
    layer_f_g: List[Dict[str, Any]] = []

    for step in list(row.get("executed_steps") or row.get("past_steps") or []):
        funnel = step.get("funnel") or {}
        values = step.get("values") or {}
        binding = funnel.get("binding") or {}
        args = (
            binding.get("arguments")
            or binding.get("arguments_redacted")
            or funnel.get("invoke_arguments")
            or {}
        )
        op_ids = list(funnel.get("invoked_operations") or [])
        op = op_ids[0] if op_ids else str(
            (funnel.get("primary") or {}).get("operation_id")
            or binding.get("operation_id")
            or ""
        )
        # C: request args
        for path, field, d in _walk_dates(args):
            if is_post_cutoff(d, cut):
                layer_c.append(
                    {
                        "operation_id": op,
                        "field": field,
                        "date": d.isoformat(),
                        "path": path,
                        "arguments": args,
                    }
                )
        # D/E: returned/projected *date fields* on step (ignore prose blobs).
        for path, field, d in _walk_dates(values):
            if not is_post_cutoff(d, cut):
                continue
            base_field = field.split(".")[-1]
            if base_field.lower() not in DATE_FIELD_HINTS and field.lower() not in DATE_FIELD_HINTS:
                continue
            ent = str(values.get("issuer_code") or values.get("symbol") or "")
            ck = _canonical_key(operation_id=op, field=base_field, d=d, entity=ent)
            layer_d_e.append(
                {
                    "operation_id": op,
                    "field": base_field,
                    "date": d.isoformat(),
                    "path": path,
                    "values": {
                        k: values.get(k)
                        for k in list(values)[:12]
                    },
                    "canonical_key": list(ck),
                }
            )
            evidence_events.setdefault(
                ck,
                {
                    "operation_id": op,
                    "field": base_field,
                    "date": d.isoformat(),
                    "entity": ent,
                    "sources": [],
                },
            )["sources"].append("executed_step_values")

    store = row.get("evidence_store") or row.get("evidence_ledger") or {}
    verified = store.get("verified_values") or store.get("values") or {}
    provenance = store.get("provenance_by_key") or {}

    def _accept_store_date(
        *,
        field: str,
        d,
        path: str,
        values_obj: Dict[str, Any],
        prov_obj: Dict[str, Any],
        source_tag: str,
    ) -> None:
        base_field = field.split(".")[-1]
        if base_field.lower() not in DATE_FIELD_HINTS and field.lower() not in DATE_FIELD_HINTS:
            return
        if not is_post_cutoff(d, cut):
            return
        op = str(prov_obj.get("operation_id") or "")
        args = prov_obj.get("arguments_redacted") or prov_obj.get("arguments") or {}
        ent = str(
            values_obj.get("issuer_code")
            or values_obj.get("symbol")
            or args.get("symbol")
            or ""
        )
        ck = _canonical_key(operation_id=op, field=base_field, d=d, entity=ent)
        layer_f_g.append(
            {
                "operation_id": op,
                "field": base_field,
                "date": d.isoformat(),
                "path": path,
                "provenance": {
                    "operation_id": op,
                    "arguments_redacted": args,
                    "source": prov_obj.get("source"),
                    "raw_result_ref": prov_obj.get("raw_result_ref"),
                },
                "canonical_key": list(ck),
                "observed_or_verified": True,
            }
        )
        evidence_events.setdefault(
            ck,
            {
                "operation_id": op,
                "field": base_field,
                "date": d.isoformat(),
                "entity": ent,
                "sources": [],
            },
        )["sources"].append(source_tag)
        for apath, afield, ad in _walk_dates(args):
            if is_post_cutoff(ad, cut):
                layer_c.append(
                    {
                        "operation_id": op,
                        "field": afield,
                        "date": ad.isoformat(),
                        "path": f"provenance:{apath}",
                        "arguments": args,
                    }
                )

    for path, field, d in _walk_dates(verified):
        _accept_store_date(
            field=field,
            d=d,
            path=path,
            values_obj=verified if isinstance(verified, dict) else {},
            prov_obj=provenance.get(field)
            or provenance.get(field.split(".")[-1])
            or {},
            source_tag="evidence_store_verified",
        )

    # Also scan evidence items (may hold dates even when verified_values empty).
    for i, item in enumerate(list(store.get("items") or [])):
        if not isinstance(item, dict):
            continue
        vals = item.get("values") or {}
        item_prov = item.get("provenance") or {}
        field_prov = (item_prov.get("field_provenance") or {})
        op = str(item_prov.get("operation_id") or item.get("operation_id") or "")
        for path, field, d in _walk_dates(vals):
            fp = field_prov.get(field) or field_prov.get(field.split(".")[-1]) or {}
            prov_obj = {
                "operation_id": op or fp.get("operation_id"),
                "arguments_redacted": fp.get("arguments_redacted")
                or item_prov.get("arguments_redacted")
                or {},
                "source": item_prov.get("source") or fp.get("source"),
                "raw_result_ref": item_prov.get("raw_result_ref") or fp.get("raw_result_ref"),
            }
            _accept_store_date(
                field=field,
                d=d,
                path=f"$.items[{i}].values.{field}",
                values_obj=vals if isinstance(vals, dict) else {},
                prov_obj=prov_obj,
                source_tag="evidence_store_items",
            )

    # Deduplicate layer lists by canonical (op,field,date,entity)
    def _dedup(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen: Set[Tuple[Any, ...]] = set()
        out: List[Dict[str, Any]] = []
        for it in items:
            key = (
                it.get("operation_id"),
                it.get("field"),
                it.get("date"),
                str((it.get("arguments") or {}).get("symbol") or ""),
            )
            if key in seen:
                continue
            seen.add(key)
            out.append(it)
        return out

    layer_c = _dedup(layer_c)
    layer_d_e = _dedup(layer_d_e)
    layer_f_g = _dedup(layer_f_g)
    unique_evidence = list(evidence_events.values())
    for ev in unique_evidence:
        ev["sources"] = sorted(set(ev.get("sources") or []))

    # Layer H: final answer uses post-cutoff date that also appears in accepted evidence
    evidence_dates = {e["date"] for e in unique_evidence}
    final_post = [
        d.isoformat()
        for _, _, d in _walk_dates(final_answer)
        if is_post_cutoff(d, cut)
    ]
    layer_h_hits = []
    for d in sorted(set(final_post)):
        if d in evidence_dates:
            layer_h_hits.append({"date": d, "in_final_answer": True, "in_evidence": True})

    # Layer I: contamination affects evaluator correctness
    answer_bearing_fields = {
        "report_date",
        "trade_date",
        "release_date",
        "close",
        "open",
        "high",
        "low",
        "eps",
        "basic_eps",
        "revenue",
        "net_profit",
    }
    answer_bearing = [
        e
        for e in unique_evidence
        if e.get("field") in answer_bearing_fields or e.get("date") in set(final_post)
    ]
    layer_i = bool(answer_correct is True and answer_bearing)

    layers = {
        "A_question_or_gold_post_cutoff": bool(layer_a_dates),
        "B_plan_scope_post_cutoff": bool(layer_b_hits),
        "C_mcp_request_post_cutoff": bool(layer_c),
        "D_E_returned_or_projected_post_cutoff": bool(layer_d_e),
        "F_G_evidence_store_accepted_post_cutoff": bool(layer_f_g),
        "H_final_answer_uses_post_cutoff_evidence_date": bool(layer_h_hits),
        "I_affects_evaluator_correctness": layer_i,
    }

    # Provider post-cutoff date values returned/projected/accepted.
    temporal_contamination = bool(unique_evidence)
    answer_bearing_contamination = bool(answer_bearing)

    return {
        "task_id": tid,
        "question": question,
        "final_answer": final_answer,
        "answer_correct": answer_correct,
        "status": row.get("status"),
        "layers": layers,
        "layer_a_dates": layer_a_dates,
        "layer_b": layer_b_hits[:20],
        "layer_c": layer_c,
        "layer_d_e": layer_d_e,
        "layer_f_g": layer_f_g,
        "layer_h": layer_h_hits,
        "unique_evidence_events": unique_evidence,
        "unique_evidence_n": len(unique_evidence),
        "temporal_contamination": temporal_contamination,
        "answer_bearing_contamination": answer_bearing_contamination,
        "operations": sorted(
            {e.get("operation_id") for e in unique_evidence if e.get("operation_id")}
        ),
    }
