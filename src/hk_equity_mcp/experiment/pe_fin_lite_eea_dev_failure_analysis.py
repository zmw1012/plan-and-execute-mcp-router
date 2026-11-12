"""Development-origin-only failure analysis for PE-Fin Lite v1.2-EEA.

Read-only. Never mutates formal HK-155 raw artifacts. Never uses reserved85.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

FORMAL_RUN = Path("outputs/pe_fin_lite_r5_hk155_live_v1")
AUDIT_DIR = FORMAL_RUN / "final_audit_r5_v1"
OUT_DIR = Path("outputs/pe_fin_lite_eea_dev_failure_analysis_v1")

FORMAL_RAW_NAMES = [
    "predictions.jsonl",
    "run_metadata.json",
    "cost_summary.json",
    "error_summary.json",
    "grading_summary.json",
    "per_task_metrics.csv",
]

C07_CLASSES = (
    "missing_required_argument",
    "invalid_argument_format",
    "argument_semantic_mismatch",
    "mcp_transport_or_protocol_failure",
    "empty_result",
    "no_matching_result",
    "gateway_envelope_failure",
    "adapter_failure",
    "scope_mismatch",
    "unsupported_operation",
    "disclosure_chain_stage_failure",
    "other",
)

C09_CLASSES = (
    "canonical_field_alias_missing",
    "required_output_truly_missing",
    "answer_field_present_but_not_projected",
    "context_field_missing",
    "period_mismatch",
    "entity_mismatch",
    "basis_mismatch",
    "unit_mismatch",
    "primary_fallback_merge_failure",
    "provenance_rejected",
    "upstream_c07_failure",
    "other",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def collect_hashes(run_dir: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for name in FORMAL_RAW_NAMES:
        p = run_dir / name
        if not p.exists():
            out[name] = {"exists": False}
            continue
        rows = None
        if name.endswith(".jsonl"):
            rows = sum(1 for line in p.read_text(encoding="utf-8").splitlines() if line.strip())
        elif name.endswith(".csv"):
            rows = max(0, sum(1 for _ in p.read_text(encoding="utf-8").splitlines()) - 1)
        out[name] = {
            "exists": True,
            "sha256": sha256_file(p),
            "bytes": p.stat().st_size,
            "rows": rows,
        }
    return out


def _load_preds(path: Path) -> List[Dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _tid(row: Dict[str, Any]) -> str:
    return str(row.get("task_id") or row.get("id") or "")


def _classify_c07_step(pred: Dict[str, Any], step_status: Dict[str, Any], funnel: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    fr = str(step_status.get("failure_reason") or pred.get("error_type") or "").lower()
    gw = (funnel or {}).get("invoke_gateway_success") or {}
    inv = list((funnel or {}).get("invoked_operations") or [])
    primary = ((funnel or {}).get("primary") or {}).get("operation_id")
    fallback = ((funnel or {}).get("fallback") or {}).get("operation_id")
    missing = list(step_status.get("missing_outputs") or [])
    dc = (funnel or {}).get("disclosure_composite") or {}

    primary_class = "other"
    if "mcp_transport" in fr or "protocol" in fr:
        primary_class = "mcp_transport_or_protocol_failure"
    elif "scope_mismatch" in fr:
        primary_class = "scope_mismatch"
    elif dc and dc.get("execution_ok") is False and inv:
        primary_class = "disclosure_chain_stage_failure"
    elif inv and all(gw.get(x) is False for x in inv):
        primary_class = "mcp_transport_or_protocol_failure"
    elif "empty" in fr or step_status.get("status") == "empty":
        primary_class = "empty_result"
    elif "no_matching" in fr:
        primary_class = "no_matching_result"
    elif "envelope" in fr or "normalization" in fr:
        primary_class = "gateway_envelope_failure"
    elif "adapter" in fr:
        primary_class = "adapter_failure"
    elif "binder_failed" in fr or "argument" in fr:
        if "missing" in fr:
            primary_class = "missing_required_argument"
        elif "invalid" in fr:
            primary_class = "invalid_argument_format"
        else:
            primary_class = "argument_semantic_mismatch"
    elif "unsupported" in fr or "unknown_operation" in fr:
        primary_class = "unsupported_operation"
    elif missing and any(gw.get(x) for x in inv):
        # gateway ok but still failed at C07 layer — often empty/adapter
        primary_class = "empty_result" if not (step_status.get("values") or {}) else "adapter_failure"

    return {
        "task_id": _tid(pred),
        "step_id": step_status.get("id") or (funnel or {}).get("step_id"),
        "operation_id": primary or (inv[0] if inv else None),
        "primary_or_fallback": "primary",
        "fallback_operation_id": fallback,
        "grounded_arguments": ((funnel or {}).get("primary") or {}).get("arguments"),
        "gateway_success_map": gw,
        "raw_error_type": pred.get("error_type"),
        "failure_reason": step_status.get("failure_reason"),
        "adapter_success": None,
        "required_outputs": ((pred.get("initial_plan") or {}).get("steps") or []),
        "missing_outputs": missing,
        "scope_match": step_status.get("scope_match"),
        "earliest_failing_stage": "C07_mcp",
        "primary_class": primary_class,
        "invoked_operations": inv,
    }


def _classify_c09(pred: Dict[str, Any], step_status: Dict[str, Any]) -> Dict[str, Any]:
    missing = list(step_status.get("missing_outputs") or [])
    values = dict(step_status.get("values") or {})
    fr = str(step_status.get("failure_reason") or "").lower()
    pool_keys = []
    for f in pred.get("retrieval_funnels") or []:
        if str(f.get("step_id")) == str(step_status.get("id")):
            obs = ((f.get("binding") or {}) if False else None)
            # raw observability on step result may not be in step_statuses; check funnel enrich
            rro = f.get("raw_result_observability") or {}
            pool_keys = list(rro.get("pool_keys") or [])
            sat = f.get("satisfaction") or {}
            if sat:
                pass
            # Also look at step_result-like fields in funnel
            if f.get("pool_keys"):
                pool_keys = list(f.get("pool_keys") or [])
    # Heuristics without gold
    primary = "other"
    if any(x in fr for x in ("merge", "conflict")):
        primary = "primary_fallback_merge_failure"
    elif "scope" in fr and "period" in fr:
        primary = "period_mismatch"
    elif "entity" in fr or "scope_mismatch" in fr:
        primary = "entity_mismatch"
    elif "basis" in fr:
        primary = "basis_mismatch"
    elif "unit" in fr:
        primary = "unit_mismatch"
    elif "provenance" in fr or "grounded_input" in fr:
        primary = "provenance_rejected"
    elif not missing and not values:
        primary = "upstream_c07_failure"
    elif missing and pool_keys:
        # something in pool but not projected
        alias_hit = any(
            m.replace("basic_", "") in " ".join(pool_keys).lower()
            or m in " ".join(pool_keys).lower()
            for m in missing
        )
        if alias_hit:
            primary = "canonical_field_alias_missing"
        else:
            # distinguish context vs answer
            contextish = {"issuer_code", "company_name", "fiscal_period", "currency", "adjustment", "report_date"}
            optionalish = {"source_url", "local_key", "page_number", "snippet"}
            if missing and set(missing) <= (contextish | optionalish) and any(
                k in values for k in ("revenue", "attributable_profit", "basic_eps", "close", "volume", "title")
            ):
                primary = "context_field_missing"
            elif any(k for k in pool_keys if any(m in str(k).lower() for m in missing)):
                primary = "answer_field_present_but_not_projected"
            else:
                primary = "required_output_truly_missing"
    elif missing and not pool_keys:
        # gateway may have succeeded with empty projection
        if pred.get("failed_invocation_count"):
            primary = "upstream_c07_failure"
        else:
            primary = "required_output_truly_missing"
    else:
        primary = "other"

    return {
        "task_id": _tid(pred),
        "step_id": step_status.get("id"),
        "missing_outputs": missing,
        "provided_keys": list(values.keys()),
        "failure_reason": step_status.get("failure_reason"),
        "primary_class": primary,
        "pool_keys_sample": pool_keys[:30],
    }


def _classify_compute(pred: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = []
    plan_steps = ((pred.get("initial_plan") or {}).get("steps") or [])
    compute_steps = [s for s in plan_steps if s.get("type") == "compute"]
    if not compute_steps:
        return out
    status_by_id = {str(s.get("id")): s for s in (pred.get("step_statuses") or [])}
    for cs in compute_steps:
        sid = str(cs.get("id"))
        st = status_by_id.get(sid) or {}
        fr = str(st.get("failure_reason") or st.get("status") or "")
        primary = "other"
        if st.get("status") == "skipped_due_to_dependency" or "depends_on_failed" in fr or "missing_operands" in fr:
            # Check whether upstream retrieve failed
            deps = [str(d) for d in (cs.get("depends_on") or [])]
            up_failed = any(
                (status_by_id.get(d) or {}).get("status") in {"failed", "unavailable", "empty", "skipped_due_to_dependency"}
                for d in deps
            )
            primary = "upstream_retrieve_blocked" if up_failed or "skipped_due_to_dependency" in str(st.get("status")) else "operand_ref_missing"
        elif "non_numeric" in fr.lower():
            primary = "non_numeric_operand"
        elif "unit" in fr.lower():
            primary = "unit_mismatch"
        elif "period" in fr.lower():
            primary = "period_mismatch"
        elif "entity" in fr.lower():
            primary = "entity_mismatch"
        elif "division" in fr.lower():
            primary = "division_by_zero"
        elif "expression" in fr.lower() or "invalid" in fr.lower():
            primary = "expression_invalid"
        elif st.get("status") == "completed":
            primary = "ok"
        elif "operand" in fr.lower() or "missing" in fr.lower():
            primary = "operand_value_missing"
        out.append(
            {
                "task_id": _tid(pred),
                "step_id": sid,
                "status": st.get("status"),
                "failure_reason": st.get("failure_reason"),
                "primary_class": primary,
                "compute_spec": cs.get("compute_spec"),
            }
        )
    for tr in pred.get("compute_traces") or []:
        if tr.get("ok"):
            continue
        # already covered via step statuses mostly
        pass
    return out


def _classify_answerability(pred: Dict[str, Any]) -> Dict[str, Any]:
    aclass = pred.get("answerability_class")
    pred_status = pred.get("predicted_status")
    store = pred.get("evidence_store") or {}
    values = dict(store.get("verified_values") or store.get("values") or {})
    missing = list(store.get("missing") or [])
    has_values = bool(values)
    label = "upstream_execution_failure"
    if aclass == "intentionally_infeasible" and pred_status in {"unavailable", "clarification_needed"}:
        label = "truly_infeasible"
    elif aclass == "should_refuse_or_clarify":
        label = "truly_infeasible" if pred_status in {"unavailable", "clarification_needed"} else "contradictory_evidence"
    elif has_values and pred_status == "unavailable":
        # evidence present but refused
        optionalish = {"source_url", "local_key", "page_number", "snippet", "company_name"}
        miss_fields = []
        for m in missing:
            miss_fields.extend(m.get("missing_outputs") or [])
        if miss_fields and set(miss_fields) <= optionalish:
            label = "partially_supported_but_answerable"
        elif has_values:
            label = "partially_supported_but_answerable"
        else:
            label = "upstream_execution_failure"
    elif pred_status == "answered" and pred.get("answer_correct") is True:
        label = "sufficiently_supported"
    elif pred_status == "answered":
        label = "partially_supported_but_answerable"
    elif not has_values:
        label = "upstream_execution_failure"
    return {
        "task_id": _tid(pred),
        "answerability_class": aclass,
        "predicted_status": pred_status,
        "label": label,
        "n_verified_values": len(values),
        "n_missing_records": len(missing),
    }


def _disclosure_failures(pred: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = []
    for f in pred.get("retrieval_funnels") or []:
        dc = f.get("disclosure_composite")
        if not dc:
            continue
        inv = list(f.get("invoked_operations") or dc.get("invoked_operations") or [])
        out.append(
            {
                "task_id": _tid(pred),
                "step_id": f.get("step_id"),
                "invoked": inv,
                "execution_ok": dc.get("execution_ok"),
                "stopped_after_list": dc.get("stopped_after_list"),
                "missing_outputs": dc.get("missing_outputs"),
                "error": dc.get("error"),
                "incomplete": bool(inv) and dc.get("execution_ok") is not True,
            }
        )
    return out


def build_diagnostic20(
    *,
    dev_rows: Sequence[Dict[str, Any]],
    audit_by_id: Dict[str, Dict[str, Any]],
    order_ids: Sequence[str],
) -> Dict[str, Any]:
    """Deterministic selection; no gold correctness ranking; no hand-picked IDs."""
    by_id = {_tid(r): r for r in dev_rows}
    # Category membership
    cats = {
        "C07": [],
        "C09": [],
        "COMPUTE": [],
        "DISCLOSURE": [],
        "CONTROL": [],
    }
    for tid in order_ids:
        if tid not in by_id:
            continue
        row = by_id[tid]
        aud = audit_by_id.get(tid) or {}
        pb = aud.get("primary_blocking_checkpoint") or ""
        plan_steps = ((row.get("initial_plan") or {}).get("steps") or [])
        compute_planned = any(s.get("type") == "compute" for s in plan_steps)
        c11 = aud.get("C11_compute_status") or ""
        disc = any((f.get("disclosure_composite") for f in (row.get("retrieval_funnels") or [])))
        disc_incomplete = False
        if disc:
            for f in row.get("retrieval_funnels") or []:
                dc = f.get("disclosure_composite") or {}
                if dc and dc.get("execution_ok") is not True:
                    disc_incomplete = True
        status = row.get("status")
        if pb == "C07_mcp":
            cats["C07"].append(tid)
        elif pb == "C09_evidence_acceptance":
            cats["C09"].append(tid)
        if compute_planned and c11 in {"fail", "N/A_upstream_blocked"} or (
            compute_planned and status != "full"
        ):
            if tid not in cats["COMPUTE"]:
                cats["COMPUTE"].append(tid)
        if disc_incomplete and tid not in cats["DISCLOSURE"]:
            cats["DISCLOSURE"].append(tid)
        if status == "full" and aud.get("common_strict_answer_success") in {True, "True", "true", "1"}:
            cats["CONTROL"].append(tid)
        elif status == "full" and row.get("answer_correct") is True:
            cats["CONTROL"].append(tid)

    # Also allow control = full without requiring strict if needed later
    if len(cats["CONTROL"]) < 2:
        for tid in order_ids:
            if tid in by_id and by_id[tid].get("status") == "full" and tid not in cats["CONTROL"]:
                cats["CONTROL"].append(tid)
            if len(cats["CONTROL"]) >= 2:
                break

    quotas = {"C07": 6, "C09": 6, "COMPUTE": 3, "DISCLOSURE": 3, "CONTROL": 2}
    selected: List[str] = []
    selected_meta: List[Dict[str, Any]] = []
    used = set()

    def domain_diff_key(tid: str) -> Tuple[str, str]:
        r = by_id[tid]
        return str(r.get("domain") or ""), str(r.get("project_difficulty") or "")

    for cat, quota in quotas.items():
        pool = [t for t in cats[cat] if t not in used]
        # diversify by domain then difficulty then order
        domain_counts: Counter = Counter()
        picked = []
        # round-robin domains present in pool preserving order_ids order
        ordered_pool = [t for t in order_ids if t in pool]
        # Prefer underrepresented domains
        while len(picked) < quota and ordered_pool:
            # choose next with minimal domain count among remaining
            best = None
            best_score = None
            for t in ordered_pool:
                dom, diff = domain_diff_key(t)
                score = (domain_counts[dom], order_ids.index(t) if t in order_ids else 10**9)
                if best_score is None or score < best_score:
                    best = t
                    best_score = score
            if best is None:
                break
            picked.append(best)
            domain_counts[domain_diff_key(best)[0]] += 1
            ordered_pool = [t for t in ordered_pool if t != best]
        for t in picked:
            used.add(t)
            selected.append(t)
            selected_meta.append(
                {
                    "task_id": t,
                    "selection_bucket": cat,
                    "domain": by_id[t].get("domain"),
                    "project_difficulty": by_id[t].get("project_difficulty"),
                    "answerability_class": by_id[t].get("answerability_class"),
                    "primary_blocking_checkpoint": (audit_by_id.get(t) or {}).get(
                        "primary_blocking_checkpoint"
                    ),
                }
            )

    # If short, fill from remaining development order (still no gold sort)
    if len(selected) < 20:
        for tid in order_ids:
            if tid in by_id and tid not in used:
                selected.append(tid)
                used.add(tid)
                selected_meta.append(
                    {
                        "task_id": tid,
                        "selection_bucket": "FILL",
                        "domain": by_id[tid].get("domain"),
                        "project_difficulty": by_id[tid].get("project_difficulty"),
                        "answerability_class": by_id[tid].get("answerability_class"),
                        "primary_blocking_checkpoint": (audit_by_id.get(tid) or {}).get(
                            "primary_blocking_checkpoint"
                        ),
                    }
                )
            if len(selected) >= 20:
                break

    selected = selected[:20]
    selected_meta = selected_meta[:20]
    blob = json.dumps(selected, separators=(",", ":"))
    ordered_hash = hashlib.sha256(blob.encode("utf-8")).hexdigest()
    manifest = {
        "manifest_id": "hk_pe_fin_lite_eea_diagnostic20_v1",
        "version": "1.0",
        "role": "posthoc_engineering_diagnostic_only",
        "source_split": "development_origin_only",
        "explicitly_not_formal_accuracy": True,
        "cannot_replace_hk155": True,
        "cannot_replace_reserved_evaluation": True,
        "parent_dataset": "hk_unified_exploratory_all_v1",
        "selection_rule": {
            "quotas": quotas,
            "priority": [
                "checkpoint_category",
                "domain_diversity",
                "difficulty",
                "deterministic_exploratory_order",
            ],
            "no_gold_correctness_sort": True,
            "no_hand_picked_task_ids": True,
            "development_origin_only": True,
        },
        "ordered_task_ids": selected,
        "ordered_task_ids_hash": ordered_hash,
        "tasks": selected_meta,
        "n": len(selected),
    }
    man_blob = json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    manifest["manifest_sha256"] = hashlib.sha256(man_blob.encode("utf-8")).hexdigest()
    return manifest


def run_analysis(*, root: Path) -> Dict[str, Any]:
    run_dir = root / FORMAL_RUN
    out_dir = root / OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    before = collect_hashes(run_dir)

    preds = _load_preds(run_dir / "predictions.jsonl")
    audit_rows = list(
        csv.DictReader((run_dir / "final_audit_r5_v1" / "per_task_audit.csv").open(encoding="utf-8"))
    )
    audit_by_id = {r["task_id"]: r for r in audit_rows}
    from hk_equity_mcp.experiment.unified_exploratory import load_exploratory_run_order

    order = load_exploratory_run_order(root)
    order_ids = list(order.get("task_ids") or [])

    dev = [p for p in preds if p.get("source_split") == "development"]
    assert len(dev) == 70, f"expected 70 development tasks, got {len(dev)}"

    matrix_rows = []
    c07_rows = []
    c09_rows = []
    compute_rows = []
    ans_rows = []
    disc_rows = []

    for pred in dev:
        tid = _tid(pred)
        aud = audit_by_id.get(tid) or {}
        funnels = {str(f.get("step_id")): f for f in (pred.get("retrieval_funnels") or [])}
        for st in pred.get("step_statuses") or []:
            if st.get("type") == "retrieve" and (
                aud.get("primary_blocking_checkpoint") == "C07_mcp"
                or st.get("status") in {"failed", "empty"}
            ):
                if aud.get("C07_mcp_status") == "fail" or aud.get("primary_blocking_checkpoint") == "C07_mcp":
                    c07_rows.append(_classify_c07_step(pred, st, funnels.get(str(st.get("id")))))
            if st.get("type") == "retrieve" and (
                aud.get("primary_blocking_checkpoint") == "C09_evidence_acceptance"
                or (st.get("status") == "unavailable" and st.get("missing_outputs"))
            ):
                if aud.get("C09_evidence_acceptance_status") == "fail" or aud.get(
                    "primary_blocking_checkpoint"
                ) == "C09_evidence_acceptance":
                    c09_rows.append(_classify_c09(pred, st))
        compute_rows.extend(_classify_compute(pred))
        ans_rows.append(_classify_answerability(pred))
        disc_rows.extend(_disclosure_failures(pred))
        matrix_rows.append(
            {
                "task_id": tid,
                "domain": pred.get("domain"),
                "project_difficulty": pred.get("project_difficulty"),
                "answerability_class": pred.get("answerability_class"),
                "status": pred.get("status"),
                "predicted_status": pred.get("predicted_status"),
                "answer_correct": pred.get("answer_correct"),
                "primary_blocking_checkpoint": aud.get("primary_blocking_checkpoint"),
                "C07_status": aud.get("C07_mcp_status"),
                "C09_status": aud.get("C09_evidence_acceptance_status"),
                "C11_status": aud.get("C11_compute_status"),
                "exact_tool_set": aud.get("exact_tool_set"),
                "common_strict_answer_success": aud.get("common_strict_answer_success"),
            }
        )

    # Deduplicate c07/c09 by task_id keeping first
    def dedupe(rows: List[Dict[str, Any]], key: str = "task_id") -> List[Dict[str, Any]]:
        seen = set()
        out = []
        for r in rows:
            k = r.get(key)
            if k in seen:
                continue
            seen.add(k)
            out.append(r)
        return out

    c07_rows = dedupe(c07_rows)
    c09_rows = dedupe(c09_rows)

    summary = {
        "n_development": 70,
        "source_split": "development",
        "primary_blocker_counts": dict(
            Counter((audit_by_id.get(_tid(p)) or {}).get("primary_blocking_checkpoint") or "none" for p in dev)
        ),
        "c07_class_counts": dict(Counter(r["primary_class"] for r in c07_rows)),
        "c09_class_counts": dict(Counter(r["primary_class"] for r in c09_rows)),
        "compute_class_counts": dict(Counter(r["primary_class"] for r in compute_rows)),
        "answerability_label_counts": dict(Counter(r["label"] for r in ans_rows)),
        "disclosure_incomplete": sum(1 for r in disc_rows if r.get("incomplete")),
        "status_counts": dict(Counter(p.get("status") for p in dev)),
        "note": "reserved85 unused; no gold-driven patches",
    }

    # Write artifacts
    def write_json(path: Path, obj: Any) -> None:
        path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    write_json(out_dir / "c07_failure_taxonomy.json", {"n": len(c07_rows), "rows": c07_rows, "counts": summary["c07_class_counts"]})
    write_json(out_dir / "c09_failure_taxonomy.json", {"n": len(c09_rows), "rows": c09_rows, "counts": summary["c09_class_counts"]})
    write_json(out_dir / "compute_binding_failures.json", {"n": len(compute_rows), "rows": compute_rows, "counts": summary["compute_class_counts"]})
    write_json(out_dir / "answerability_failures.json", {"n": len(ans_rows), "rows": ans_rows, "counts": summary["answerability_label_counts"]})
    write_json(out_dir / "disclosure_failures.json", {"n": len(disc_rows), "rows": disc_rows, "incomplete": summary["disclosure_incomplete"]})
    write_json(out_dir / "development_only_summary.json", summary)

    with (out_dir / "dev_failure_matrix.csv").open("w", encoding="utf-8", newline="") as fh:
        fields = list(matrix_rows[0].keys()) if matrix_rows else []
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in matrix_rows:
            w.writerow(r)

    # Diagnostic-20
    man = build_diagnostic20(dev_rows=dev, audit_by_id=audit_by_id, order_ids=order_ids)
    man_path = root / "benchmarks" / "manifests" / "hk_pe_fin_lite_eea_diagnostic20_v1.json"
    man_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(man_path, man)
    write_json(out_dir / "diagnostic20_manifest_copy.json", man)

    after = collect_hashes(run_dir)
    if before != after:
        raise RuntimeError("Formal HK-155 raw artifacts changed during analysis")
    write_json(out_dir / "raw_hashes.json", {"before": before, "after": after, "unchanged": True})

    return {
        "out_dir": str(out_dir),
        "manifest_path": str(man_path),
        "summary": summary,
        "diagnostic20": man,
        "formal_hashes_unchanged": True,
    }


def main() -> int:
    root = Path(__file__).resolve().parents[3]
    # pe_fin_lite path: src/hk_equity_mcp/experiment/ → parents[3]=repo? 
    # experiment -> hk_equity_mcp -> src -> repo = parents[3]
    result = run_analysis(root=root)
    print("EEA DEV FAILURE ANALYSIS OK")
    print(json.dumps({k: result[k] for k in ("out_dir", "manifest_path", "formal_hashes_unchanged")}, indent=2))
    print("diagnostic20_ids", result["diagnostic20"]["ordered_task_ids"])
    print("summary_blockers", result["summary"]["primary_blocker_counts"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
