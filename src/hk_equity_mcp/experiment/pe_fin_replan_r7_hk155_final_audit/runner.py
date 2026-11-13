"""Read-only RePlan-7 HK-155 final audit. Never mutates raw run artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from hk_equity_mcp.experiment.pe_fin_replan_r7_hk155_final_audit import (
    AUDIT_VERSION,
    BENCHMARK_CUTOFF,
    DEFAULT_AUDIT_SUBDIR,
    EXPECTED_ARCHITECTURE,
    EXPECTED_CONDITION,
    EXPECTED_DOMAIN_COUNTS,
    EXPECTED_FORMULA_HASH,
    EXPECTED_FORMULA_NAME,
    EXPECTED_INDEX_IDENTITY,
    EXPECTED_METHOD,
    EXPECTED_META_SHA,
    EXPECTED_N,
    EXPECTED_PRED_SHA,
    EXPECTED_PROTOCOL,
    EXPECTED_ROUND_CAP,
    EXPECTED_RUN_COMMIT,
    RAW_NAMES,
    RESULT_ROLE,
)
from hk_equity_mcp.experiment.pe_fin_replan_r7_hk155_final_audit.temporal import (
    analyze_task_temporal,
)
from hk_equity_mcp.experiment.unified_exploratory import load_exploratory_run_order
from hk_equity_mcp.pe_fin_lite.retrieval_formula import (
    RETRIEVAL_FORMULA_NAME,
    retrieval_formula_hash,
)

FROZEN_BASELINES = {
    "MCP-Free": {
        "final_answer_accuracy": 0.1290,
        "common_strict_answer_success": None,
        "exact_tool_set": None,
        "tokens_total": 79133,
        "llm_calls": None,
        "mcp_calls": None,
        "note": "frozen baseline figures from protocol; not re-derived",
    },
    "Flat All Tools": {
        "final_answer_accuracy": 0.5677,
        "common_strict_answer_success": 0.5484,
        "exact_tool_set": 0.2323,
        "tokens_total": 9464124,
        "llm_calls": None,
        "mcp_calls": None,
    },
    "Operation-Visible": {
        "final_answer_accuracy": 0.6129,
        "common_strict_answer_success": None,
        "exact_tool_set": 0.4129,
        "condition_specific_strict": 0.2903,
        "tokens_total": 9959757,
        "llm_calls": 834,
        "mcp_calls": 396,
        "note": "from OV cost_summary + final_audit_v2 grading; condition-specific E2E not cross-comparable",
    },
    "MCP-Zero": {
        "final_answer_accuracy": 0.5806,
        "common_strict_answer_success": 0.5806,
        "exact_tool_set": 0.439,
        "condition_specific_strict": 0.3677,
        "tokens_total": 4050513,
        "note": "condition-specific strict not comparable across methods",
    },
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def collect_raw_hashes(run_dir: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for name in RAW_NAMES:
        path = run_dir / name
        if not path.exists():
            out[name] = {"exists": False}
            continue
        data = path.read_bytes()
        rows = None
        if path.suffix == ".jsonl":
            rows = sum(1 for line in data.splitlines() if line.strip())
        out[name] = {
            "exists": True,
            "path": str(path),
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "rows": rows,
        }
    return out


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(fieldnames), extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def _load_preds(path: Path) -> List[Dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _mean(xs: Sequence[Any]) -> Optional[float]:
    vals = [float(x) for x in xs if x is not None]
    return statistics.mean(vals) if vals else None


def _median(xs: Sequence[Any]) -> Optional[float]:
    vals = [float(x) for x in xs if x is not None]
    return statistics.median(vals) if vals else None


def _p95(xs: Sequence[Any]) -> Optional[float]:
    vals = sorted(float(x) for x in xs if x is not None)
    if not vals:
        return None
    idx = int(round(0.95 * (len(vals) - 1)))
    return vals[idx]


def _mcp(row: Dict[str, Any]) -> int:
    return int(row.get("underlying_operation_calls") or row.get("mcp_calls") or 0)


def _tokens(row: Dict[str, Any]) -> int:
    tok = row.get("tokens") or {}
    if isinstance(tok, dict) and tok.get("total_tokens") is not None:
        return int(tok["total_tokens"])
    return int(row.get("total_tokens") or 0)


def _domain_key(row: Dict[str, Any]) -> str:
    d = str(row.get("domain") or "").strip().lower()
    if d in {"financial", "finance"}:
        return "financial"
    if d in {"market"}:
        return "market"
    if d in {"news"}:
        return "news"
    if d in {"cross_domain", "cross-domain", "xd", "cross"}:
        return "cross_domain"
    # fallback via domain_combination / id prefix
    dc = str(row.get("domain_combination") or "").lower()
    if "cross" in dc:
        return "cross_domain"
    tid = str(row.get("task_id") or "")
    if tid.startswith("fin_"):
        return "financial"
    if tid.startswith("mkt_"):
        return "market"
    if tid.startswith("news_"):
        return "news"
    if tid.startswith("xd_"):
        return "cross_domain"
    return d or "unknown"


def classify_primary_failure(row: Dict[str, Any], temporal: Dict[str, Any]) -> Tuple[str, List[str]]:
    secondary: List[str] = []
    if temporal.get("answer_bearing_contamination"):
        secondary.append("temporal_contamination")
    elif temporal.get("temporal_contamination"):
        secondary.append("temporal_contamination")

    stop = str(row.get("stopping_reason") or "")
    err = str(row.get("error_type") or row.get("error") or "")
    loop = str(row.get("loop_prevention_trigger") or "")
    completion = str(row.get("completion") or "")
    status = str(row.get("status") or "")

    if row.get("answer_correct") is True and status == "full":
        return "ok", secondary
    if "max_total_mcp" in stop or "mcp_budget" in stop:
        return "MCP budget exhaustion", secondary
    if "max_total_llm" in stop or "llm_call_budget" in stop:
        return "LLM call budget", secondary
    if "ungrounded" in stop or "grounding" in err:
        return "grounding failure", secondary
    if "schema" in err or "invalid_replanner" in stop or "round7_plan" in stop or "forced_final_round_plan" in stop:
        return "Replanner schema failure", secondary
    if "semantic" in stop or "invalid_plan_update" in stop or "too_many_new_steps" in stop:
        if "too_many_new_steps" in stop:
            return "too_many_new_steps", secondary
        return "Replanner semantic-plan failure", secondary
    if "completed" in stop and "re" in stop:
        return "completed-step re-added", secondary
    if loop == "no_progress" or "no_progress" in stop:
        return "no-progress", secondary
    if "planner" in err or "planner" in stop:
        return "planner failure", secondary
    if "binder" in err or "binding" in err:
        return "binder failure", secondary
    if "transport" in err or "llm_transport" in stop:
        return "LLM transport failure", secondary
    if "provider" in err or "mcp" in err:
        return "MCP transport/provider failure", secondary
    if completion == "insufficient_evidence" or row.get("predicted_status") == "unavailable":
        return "insufficient evidence", secondary
    if status == "failed":
        return "other_failed", secondary
    return "other", secondary


def run_final_audit(
    *,
    root: Path,
    run_dir: Path,
    audit_subdir: str = DEFAULT_AUDIT_SUBDIR,
) -> Dict[str, Any]:
    root = root.resolve()
    run_dir = run_dir.resolve()
    out_dir = run_dir / audit_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    hashes_before = collect_raw_hashes(run_dir)
    meta = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    rows = _load_preds(run_dir / "predictions.jsonl")
    order = load_exploratory_run_order(root)
    expected_ids = list(order.get("task_ids") or [])

    # -------- identity / completeness --------
    got_ids = [str(r.get("task_id") or r.get("id")) for r in rows]
    missing = [t for t in expected_ids if t not in set(got_ids)]
    extras = [t for t in got_ids if t not in set(expected_ids)]
    dupes = [t for t, c in Counter(got_ids).items() if c > 1]
    domain_counts = Counter(_domain_key(r) for r in rows)
    formula_ok = (
        meta.get("retrieval_formula_name") == EXPECTED_FORMULA_NAME
        and meta.get("retrieval_formula_hash") == EXPECTED_FORMULA_HASH
        and RETRIEVAL_FORMULA_NAME == EXPECTED_FORMULA_NAME
        and retrieval_formula_hash() == EXPECTED_FORMULA_HASH
    )
    identity_checks = {
        "n_rows": len(rows) == EXPECTED_N,
        "unique_ids": len(set(got_ids)) == EXPECTED_N,
        "order_match": got_ids == expected_ids,
        "domain_counts_match": dict(domain_counts) == EXPECTED_DOMAIN_COUNTS,
        "run_status_completed": meta.get("run_status") == "completed",
        "processed_155": int(meta.get("processed_task_count") or 0) == EXPECTED_N,
        "missing_ids_empty": not missing,
        "duplicate_ids_empty": not dupes,
        "condition": meta.get("condition") == EXPECTED_CONDITION,
        "method_version": meta.get("method_version") == EXPECTED_METHOD,
        "protocol_version": meta.get("protocol_version") == EXPECTED_PROTOCOL,
        "architecture": meta.get("architecture") == EXPECTED_ARCHITECTURE,
        "run_commit": meta.get("git_commit") == EXPECTED_RUN_COMMIT,
        "git_dirty_false": meta.get("git_dirty") is False,
        "replan_round_cap_7": int(meta.get("replan_round_cap") or 0) == EXPECTED_ROUND_CAP,
        "not_r4": meta.get("condition") != "pe_fin_replan_r4"
        and meta.get("method_version") != "pe_fin_replan_v1_4_1_round4_ablation",
        "max_execution_rounds_7": int(meta.get("max_execution_rounds") or 0) == 7,
        "max_replan_rounds_7": int(meta.get("max_replan_rounds") or 0) == 7,
        "llm_ceiling_16": int((meta.get("limits") or {}).get("max_total_llm_calls_per_task") or 0)
        == 16,
        "mcp_ceiling_8": int((meta.get("limits") or {}).get("max_total_mcp_invocations") or 0)
        == 8,
        "r5_formula": formula_ok,
        "topk_5": int(meta.get("retrieval_top_k") or (meta.get("limits") or {}).get("retrieval_top_k") or 0)
        == 5,
        "describe_n_3": int(meta.get("describe_top_n") or (meta.get("limits") or {}).get("describe_top_n") or 0)
        == 3,
        "embedding_index_identity": meta.get("embedding_index_identity_sha256")
        == EXPECTED_INDEX_IDENTITY,
        "no_mock": meta.get("mock") is False,
        "no_dry_run": meta.get("dry_run") is False,
        "posthoc_exploratory": meta.get("posthoc_exploratory") is True,
        "formal_primary_false": meta.get("formal_primary_result") is False,
        "pred_sha_expected": hashes_before.get("predictions.jsonl", {}).get("sha256")
        == EXPECTED_PRED_SHA,
        "meta_sha_expected": hashes_before.get("run_metadata.json", {}).get("sha256")
        == EXPECTED_META_SHA,
    }
    # per-task budget / rounds
    llm_viol = [r["task_id"] for r in rows if int(r.get("llm_calls") or 0) > 16]
    mcp_viol = [r["task_id"] for r in rows if _mcp(r) > 8]
    round_viol = [
        r["task_id"]
        for r in rows
        if int(r.get("execution_round_count") or 0) > 7
        or int(r.get("replan_round_count") or 0) > 7
        or int(r.get("max_execution_rounds") or 7) != 7
    ]
    eighth_round = [
        r["task_id"]
        for r in rows
        if any(
            int((t or {}).get("round") or 0) >= 8
            for t in (r.get("trace") or r.get("replan_decisions") or [])
        )
    ]
    identity_checks["llm_leq_16_all_tasks"] = not llm_viol
    identity_checks["mcp_leq_8_all_tasks"] = not mcp_viol
    identity_checks["rounds_leq_7_all_tasks"] = not round_viol
    identity_checks["no_eighth_round"] = not eighth_round
    identity_all_pass = all(bool(v) for v in identity_checks.values())

    completeness = {
        "n": len(rows),
        "unique": len(set(got_ids)),
        "missing_ids": missing,
        "duplicate_ids": dupes,
        "extra_ids": extras,
        "domain_counts": dict(domain_counts),
        "expected_domain_counts": EXPECTED_DOMAIN_COUNTS,
        "order_match": got_ids == expected_ids,
        "resume_duplicate_ids": dupes,
    }

    # -------- temporal --------
    temporal_rows = [analyze_task_temporal(r) for r in rows]
    contaminated = [t for t in temporal_rows if t.get("temporal_contamination")]
    answer_bearing = [t for t in temporal_rows if t.get("answer_bearing_contamination")]
    affected_correct = [
        t for t in answer_bearing if t.get("answer_correct") is True
    ]
    request_leak = [t for t in temporal_rows if t["layers"]["C_mcp_request_post_cutoff"]]
    return_leak = [
        t for t in temporal_rows if t["layers"]["D_E_returned_or_projected_post_cutoff"]
    ]
    evidence_leak = [
        t for t in temporal_rows if t["layers"]["F_G_evidence_store_accepted_post_cutoff"]
    ]
    final_leak = [
        t
        for t in temporal_rows
        if t["layers"]["H_final_answer_uses_post_cutoff_evidence_date"]
    ]

    # Accuracy sets
    raw_true = sum(1 for r in rows if r.get("answer_correct") is True)
    raw_false = sum(1 for r in rows if r.get("answer_correct") is False)
    raw_null = sum(1 for r in rows if r.get("answer_correct") is None)
    null_tasks = [r for r in rows if r.get("answer_correct") is None]

    contaminated_ids = {t["task_id"] for t in contaminated}
    answer_bearing_ids = {t["task_id"] for t in answer_bearing}
    # Conservative: answer-bearing contamination => incorrect
    cons_true = 0
    for r in rows:
        tid = r["task_id"]
        if tid in answer_bearing_ids:
            continue
        if r.get("answer_correct") is True:
            cons_true += 1
    # Clean-subset: exclude contaminated tasks
    clean_rows = [r for r in rows if r["task_id"] not in contaminated_ids]
    clean_true = sum(1 for r in clean_rows if r.get("answer_correct") is True)

    if affected_correct:
        temporal_marker = "invalidated_for_accuracy_due_to_non_disclosure_cutoff_leakage"
    elif contaminated:
        temporal_marker = "engineering_complete_with_temporal_evidence_contamination"
    else:
        temporal_marker = "valid_posthoc_exploratory_hk155"

    temporal_verdict = {
        "benchmark_cutoff": BENCHMARK_CUTOFF,
        "engineering_complete": identity_all_pass and meta.get("run_status") == "completed",
        "cutoff_request_compliant": len(request_leak) == 0,
        "cutoff_return_filter_compliant": len(return_leak) == 0,
        "cutoff_evidence_acceptance_compliant": len(evidence_leak) == 0,
        "cutoff_final_answer_compliant": len(final_leak) == 0,
        "usable_for_formal_accuracy": False,  # RePlan is explicitly posthoc exploratory
        "usable_for_posthoc_descriptive_analysis": True,
        "marker": temporal_marker,
        "contaminated_task_n": len(contaminated),
        "answer_bearing_contaminated_task_n": len(answer_bearing),
        "contaminated_and_raw_correct_n": len(affected_correct),
        "contaminated_task_ids": sorted(contaminated_ids),
        "answer_bearing_contaminated_task_ids": sorted(answer_bearing_ids),
        "note": (
            "Question/gold future dates alone are NOT leakage. "
            "Contamination requires provider post-cutoff evidence acceptance."
        ),
    }

    # -------- accuracy --------
    status_counts = Counter(str(r.get("status")) for r in rows)
    answered = sum(1 for r in rows if r.get("predicted_status") == "answered" or r.get("completion") == "answered")
    unavailable = EXPECTED_N - answered
    strict = sum(
        1 for r in rows if r.get("status") == "full" and r.get("answer_correct") is True
    )
    full_correct = sum(
        1 for r in rows if r.get("status") == "full" and r.get("answer_correct") is True
    )
    partial_correct = sum(
        1
        for r in rows
        if r.get("status") == "partial" and r.get("answer_correct") is True
    )
    failed_correct = sum(
        1 for r in rows if r.get("status") == "failed" and r.get("answer_correct") is True
    )
    exact_tool = sum(1 for r in rows if r.get("exact_tool_set") is True)
    op_p = _mean([r.get("operation_precision") for r in rows])
    op_r = _mean([r.get("operation_recall") for r in rows])
    op_f = _mean([r.get("operation_f1") for r in rows])
    all_mcp_ok = sum(1 for r in rows if r.get("all_tool_invocations_successful") is True)

    accuracy_metrics = {
        "n": EXPECTED_N,
        "answer_correct_true": raw_true,
        "answer_correct_false": raw_false,
        "answer_correct_null": raw_null,
        "lower_bound_accuracy": raw_true / EXPECTED_N,
        "auto_gradable_accuracy": (raw_true / (raw_true + raw_false))
        if (raw_true + raw_false)
        else None,
        "upper_bound_if_null_correct": (raw_true + raw_null) / EXPECTED_N,
        "common_strict_success": {
            "definition": "status==full AND answer_correct==true; denom=155",
            "count": strict,
            "rate": strict / EXPECTED_N,
        },
        "descriptive_correct_by_status": {
            "full_correct": full_correct,
            "partial_correct": partial_correct,
            "failed_correct": failed_correct,
        },
        "answered": answered,
        "unavailable": unavailable,
        "status_counts": dict(status_counts),
        "exact_tool_set": {"count": exact_tool, "rate": exact_tool / EXPECTED_N},
        "operation_precision_mean": op_p,
        "operation_recall_mean": op_r,
        "operation_f1_mean": op_f,
        "all_mcp_invocations_successful_count": all_mcp_ok,
        "all_mcp_invocations_successful_rate": all_mcp_ok / EXPECTED_N,
        "sensitivity": {
            "A_frozen_raw_correct": f"{raw_true}/{EXPECTED_N}",
            "A_frozen_raw_rate": raw_true / EXPECTED_N,
            "B_conservative_correct": f"{cons_true}/{EXPECTED_N}",
            "B_conservative_rate": cons_true / EXPECTED_N,
            "B_rule": "answer-bearing temporal contamination tasks forced incorrect; denom=155",
            "C_clean_subset_correct": f"{clean_true}/{len(clean_rows)}",
            "C_clean_subset_rate": (clean_true / len(clean_rows)) if clean_rows else None,
            "C_note": "cannot formally cross-compare to baselines",
        },
        "explicitly_not_formal_accuracy": True,
    }

    # -------- manual review --------
    manual_queue = []
    for r in null_tasks:
        manual_queue.append(
            {
                "task_id": r.get("task_id"),
                "domain": _domain_key(r),
                "status": r.get("status"),
                "predicted_status": r.get("predicted_status"),
                "completion": r.get("completion"),
                "final_answer": r.get("final_answer"),
                "reference_answer": r.get("reference_answer") or r.get("gold_answer"),
                "gold_status": r.get("gold_status"),
                "auto_grade_status": r.get("auto_grade_status"),
                "auto_grade_reason": r.get("auto_grade_reason"),
                "answer_correct": r.get("answer_correct"),
                "human_grade_pending": True,
            }
        )
    human_grades = {
        "schema_version": "pe_fin_replan_r7_hk155_human_grades_v1",
        "pending_task_ids": [r.get("task_id") for r in null_tasks],
        "grades": {
            str(r.get("task_id")): {
                "human_answer_correct": None,
                "human_evidence_correct": None,
                "notes": "",
            }
            for r in null_tasks
        },
        "note": "Do not prefill. Await human grading.",
    }

    # -------- rounds --------
    rounds = [int(r.get("execution_round_count") or 0) for r in rows]
    correct_rounds = [
        int(r.get("execution_round_count") or 0)
        for r in rows
        if r.get("answer_correct") is True
    ]
    incorrect_rounds = [
        int(r.get("execution_round_count") or 0)
        for r in rows
        if r.get("answer_correct") is False
    ]
    by_round_correct = defaultdict(lambda: {"n": 0, "correct": 0})
    for r in rows:
        rr = int(r.get("execution_round_count") or 0)
        by_round_correct[rr]["n"] += 1
        if r.get("answer_correct") is True:
            by_round_correct[rr]["correct"] += 1
    round_dist = Counter(rounds)
    round7_response_only_ok = sum(
        1
        for r in rows
        if int(r.get("execution_round_count") or 0) == 7
        and (
            r.get("forced_final_round") is True
            or r.get("round7_response_only") is True
            or r.get("final_round_response_only") is True
        )
    )
    round_analysis = {
        "mean": _mean(rounds),
        "median": _median(rounds),
        "p95": _p95(rounds),
        "max": max(rounds) if rounds else None,
        "sum": sum(rounds),
        "distribution": {str(k): round_dist[k] for k in range(0, 8)},
        "correct_mean": _mean(correct_rounds),
        "correct_median": _median(correct_rounds),
        "incorrect_mean": _mean(incorrect_rounds),
        "incorrect_median": _median(incorrect_rounds),
        "by_status_mean": {
            st: _mean(
                [
                    int(r.get("execution_round_count") or 0)
                    for r in rows
                    if r.get("status") == st
                ]
            )
            for st in ("full", "partial", "failed")
        },
        "per_round_correct_rate": {
            str(k): {
                "n": by_round_correct[k]["n"],
                "correct": by_round_correct[k]["correct"],
                "rate": (
                    by_round_correct[k]["correct"] / by_round_correct[k]["n"]
                    if by_round_correct[k]["n"]
                    else None
                ),
            }
            for k in sorted(by_round_correct)
        },
        "round7_forced_final_tasks": round7_response_only_ok,
        "no_eighth_round": not eighth_round,
        "eighth_round_task_ids": eighth_round,
        "premature_insufficient_total": sum(
            int(r.get("premature_insufficient_blocked_count") or 0) for r in rows
        ),
        "no_progress_trigger_tasks": sum(
            1 for r in rows if r.get("loop_prevention_trigger") == "no_progress"
        ),
        "failed_signature_blocks_total": sum(
            int(r.get("failed_signature_blocks") or 0) for r in rows
        ),
        "note": "Round means for correct vs incorrect are descriptive correlation, not causal.",
    }

    # -------- failure taxonomy --------
    primary_counts: Counter = Counter()
    per_task_audit = []
    for r, t in zip(rows, temporal_rows):
        primary, secondary = classify_primary_failure(r, t)
        primary_counts[primary] += 1
        per_task_audit.append(
            {
                "task_id": r.get("task_id"),
                "domain": _domain_key(r),
                "answerability_class": r.get("answerability_class"),
                "status": r.get("status"),
                "answer_correct": r.get("answer_correct"),
                "predicted_status": r.get("predicted_status"),
                "completion": r.get("completion"),
                "execution_round_count": r.get("execution_round_count"),
                "replan_round_count": r.get("replan_round_count"),
                "llm_calls": r.get("llm_calls"),
                "mcp_calls": _mcp(r),
                "total_tokens": _tokens(r),
                "latency_ms": r.get("latency_ms"),
                "exact_tool_set": r.get("exact_tool_set"),
                "operation_f1": r.get("operation_f1"),
                "primary_failure": primary,
                "secondary_failures": "|".join(secondary),
                "temporal_contamination": t.get("temporal_contamination"),
                "answer_bearing_contamination": t.get("answer_bearing_contamination"),
                "stopping_reason": r.get("stopping_reason"),
                "error_type": r.get("error_type"),
                "source_split": r.get("source_split"),
                "project_difficulty": r.get("project_difficulty") or r.get("difficulty"),
            }
        )
    failure_taxonomy = {
        "primary_counts": dict(primary_counts),
        "n": EXPECTED_N,
    }

    # -------- resources --------
    llms = [int(r.get("llm_calls") or 0) for r in rows]
    mcps = [_mcp(r) for r in rows]
    toks = [_tokens(r) for r in rows]
    lats = [float(r.get("latency_ms") or 0.0) for r in rows]
    embeds = [int(r.get("embedding_calls") or 0) for r in rows]
    stage_tot: Counter = Counter()
    for r in rows:
        for k, v in (r.get("llm_calls_by_stage") or {}).items():
            stage_tot[str(k)] += int(v or 0)
    cost = {}
    cost_path = run_dir / "cost_summary.json"
    if cost_path.exists():
        cost = json.loads(cost_path.read_text(encoding="utf-8"))
    resource_summary = {
        "llm_calls_total": sum(llms),
        "llm_calls_mean": _mean(llms),
        "llm_calls_median": _median(llms),
        "llm_calls_max": max(llms) if llms else None,
        "calls_by_stage_totals": dict(stage_tot),
        "mcp_calls_total": sum(mcps),
        "mcp_calls_mean": _mean(mcps),
        "mcp_calls_median": _median(mcps),
        "mcp_calls_max": max(mcps) if mcps else None,
        "embedding_calls_total": sum(embeds),
        "fallback_total": sum(int(r.get("fallback_used_count") or 0) for r in rows),
        "execution_rounds_total": sum(rounds),
        "tokens_total_sum_per_task": sum(toks),
        "tokens_total_from_cost_summary": cost.get("total_tokens"),
        "input_tokens_reliable": False,
        "output_tokens_reliable": False,
        "token_note": (
            "Per-task tokens.total_tokens is reliable; prompt/completion often 0; "
            "do not invent input/output split."
        ),
        "latency_ms_total": sum(lats),
        "latency_ms_mean": _mean(lats),
        "latency_ms_median": _median(lats),
        "latency_ms_p95": _p95(lats),
        "estimated_usd": None,
        "budget_violations": {
            "llm_gt_16": llm_viol,
            "mcp_gt_8": mcp_viol,
            "rounds_gt_7": round_viol,
        },
        "resource_limit_tasks": [
            r["task_id"] for r in rows if r.get("resource_limit_failure") is True
        ],
    }

    # -------- breakdowns --------
    def _group(key_fn):
        buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for r in rows:
            buckets[str(key_fn(r) or "unknown")].append(r)
        out = {}
        for k, rs in sorted(buckets.items()):
            tids = {x["task_id"] for x in rs}
            out[k] = {
                "n": len(rs),
                "raw_correct": sum(1 for x in rs if x.get("answer_correct") is True),
                "strict_correct": sum(
                    1
                    for x in rs
                    if x.get("status") == "full" and x.get("answer_correct") is True
                ),
                "status_counts": dict(Counter(x.get("status") for x in rs)),
                "answered": sum(
                    1
                    for x in rs
                    if x.get("predicted_status") == "answered"
                    or x.get("completion") == "answered"
                ),
                "unavailable": sum(
                    1
                    for x in rs
                    if x.get("predicted_status") == "unavailable"
                    or x.get("completion") == "insufficient_evidence"
                ),
                "llm_mean": _mean([x.get("llm_calls") for x in rs]),
                "mcp_mean": _mean([_mcp(x) for x in rs]),
                "rounds_mean": _mean([x.get("execution_round_count") for x in rs]),
                "tokens_mean": _mean([_tokens(x) for x in rs]),
                "primary_failure_mode": Counter(
                    next(
                        p["primary_failure"]
                        for p in per_task_audit
                        if p["task_id"] == x["task_id"]
                    )
                    for x in rs
                ).most_common(1)[0][0]
                if rs
                else None,
                "temporal_contaminated_n": sum(1 for t in tids if t in contaminated_ids),
            }
        return out

    domain_breakdown = _group(_domain_key)
    answerability_breakdown = _group(lambda r: r.get("answerability_class"))
    split_breakdown = _group(
        lambda r: (
            "development70"
            if str(r.get("source_split") or "") == "development"
            else (
                "previously_reserved85"
                if str(r.get("source_split") or "")
                in {"previously_reserved", "heldout", "reserved"}
                else str(r.get("source_split") or "unknown")
            )
        )
    )
    difficulty_breakdown = _group(
        lambda r: r.get("project_difficulty") or r.get("difficulty") or "unknown"
    )

    # -------- paired Lite --------
    lite_dir = root / "outputs" / "pe_fin_lite_r5_hk155_live_v1"
    lite_rows = _load_preds(lite_dir / "predictions.jsonl") if lite_dir.exists() else []
    lite_by = {r["task_id"]: r for r in lite_rows}
    paired = {
        "lite_incorrect_replan_correct": 0,
        "lite_correct_replan_incorrect": 0,
        "both_correct": 0,
        "both_incorrect": 0,
        "null_either": 0,
        "status_improved": 0,
        "status_regressed": 0,
        "status_unchanged": 0,
        "by_domain": {},
        "per_task": [],
    }
    status_rank = {"failed": 0, "partial": 1, "full": 2}

    def _corr(v):
        if v is True:
            return True
        if v is False:
            return False
        return None

    domain_pair = defaultdict(
        lambda: {
            "improved": 0,
            "regressed": 0,
            "both_correct": 0,
            "both_incorrect": 0,
        }
    )
    for r in rows:
        tid = r["task_id"]
        lr = lite_by.get(tid) or {}
        lc, rc = _corr(lr.get("answer_correct")), _corr(r.get("answer_correct"))
        if lc is None or rc is None:
            paired["null_either"] += 1
            outcome = "null_either"
        elif lc and rc:
            paired["both_correct"] += 1
            outcome = "both_correct"
            domain_pair[_domain_key(r)]["both_correct"] += 1
        elif (not lc) and rc:
            paired["lite_incorrect_replan_correct"] += 1
            outcome = "improved"
            domain_pair[_domain_key(r)]["improved"] += 1
        elif lc and (not rc):
            paired["lite_correct_replan_incorrect"] += 1
            outcome = "regressed"
            domain_pair[_domain_key(r)]["regressed"] += 1
        else:
            paired["both_incorrect"] += 1
            outcome = "both_incorrect"
            domain_pair[_domain_key(r)]["both_incorrect"] += 1
        ls, rs = str(lr.get("status") or ""), str(r.get("status") or "")
        if status_rank.get(rs, -1) > status_rank.get(ls, -1):
            paired["status_improved"] += 1
            status_delta = "improved"
        elif status_rank.get(rs, -1) < status_rank.get(ls, -1):
            paired["status_regressed"] += 1
            status_delta = "regressed"
        else:
            paired["status_unchanged"] += 1
            status_delta = "unchanged"
        paired["per_task"].append(
            {
                "task_id": tid,
                "domain": _domain_key(r),
                "outcome": outcome,
                "status_delta": status_delta,
                "lite_correct": lc,
                "replan_correct": rc,
                "lite_status": ls,
                "replan_status": rs,
                "delta_llm": int(r.get("llm_calls") or 0) - int(lr.get("llm_calls") or 0),
                "delta_mcp": _mcp(r)
                - int(lr.get("underlying_operation_calls") or lr.get("mcp_calls") or 0),
                "delta_tokens": _tokens(r) - int((lr.get("tokens") or {}).get("total_tokens") or lr.get("total_tokens") or 0),
            }
        )
    paired["by_domain"] = dict(domain_pair)
    paired["lite_raw_correct"] = sum(1 for r in lite_rows if r.get("answer_correct") is True)
    paired["replan_raw_correct"] = raw_true
    paired["note"] = "Descriptive only; no method change from paired deltas."

    # -------- baselines --------
    lite_audit = lite_dir / "final_audit_r5_v1" / "audit_summary.json"
    lite_res = lite_dir / "final_audit_r5_v1" / "resource_summary.json"
    lite_summary = json.loads(lite_audit.read_text()) if lite_audit.exists() else {}
    lite_resources = json.loads(lite_res.read_text()) if lite_res.exists() else {}
    baseline_comparison = {
        "RePlan-7_v1_4_1": {
            "final_answer_accuracy_lower_bound": raw_true / EXPECTED_N,
            "final_answer_accuracy_upper_bound": (raw_true + raw_null) / EXPECTED_N,
            "common_strict_answer_success": strict / EXPECTED_N,
            "exact_tool_set": exact_tool / EXPECTED_N,
            "operation_f1_mean": op_f,
            "status_counts": dict(status_counts),
            "llm_calls": sum(llms),
            "mcp_calls": sum(mcps),
            "tokens_total": sum(toks),
            "latency_ms_mean": _mean(lats),
            "provider_model": {
                "generation": meta.get("generation_model"),
                "embedding": meta.get("embedding_model"),
                "market_provider": meta.get("market_provider_profile_id"),
            },
            "posthoc_exploratory": True,
            "formal_primary_result": False,
        },
        "PE-Fin Lite v1.2 HK-155": {
            "final_answer_accuracy_lower_bound": (
                lite_summary.get("answer_metrics") or {}
            ).get("final_answer_accuracy_lower_bound"),
            "common_strict_answer_success": (
                (lite_summary.get("answer_metrics") or {})
                .get("common_strict_answer_success")
                or {}
            ).get("rate"),
            "exact_tool_set": (
                (lite_summary.get("answer_metrics") or {}).get("exact_tool_set") or {}
            ).get("rate"),
            "llm_calls": lite_resources.get("llm_calls_total"),
            "mcp_calls": lite_resources.get("mcp_invocations_total"),
            "tokens_total": lite_resources.get("tokens_total_from_cost_summary"),
            "status_counts": (lite_summary.get("status_counts")),
            "role": "frozen primary proposed-method result",
        },
        **{k: v for k, v in FROZEN_BASELINES.items()},
        "caveats": [
            "Condition-specific E2E metrics are not directly cross-comparable.",
            "No significance tests.",
            "No provider-effect causal claims.",
            "RePlan is post-hoc exploratory.",
            "previously_reserved is no longer unseen held-out.",
            "PE-Fin Lite v1.2 remains frozen primary proposed-method result.",
        ],
    }

    # -------- write artifacts --------
    _write_json(out_dir / "raw_artifact_sha256_before.json", hashes_before)
    _write_json(out_dir / "identity_checks.json", {
        "checks": identity_checks,
        "all_pass": identity_all_pass,
        "llm_violations": llm_viol,
        "mcp_violations": mcp_viol,
        "round_violations": round_viol,
        "eighth_round_task_ids": eighth_round,
    })
    _write_json(out_dir / "completeness.json", completeness)
    _write_json(
        out_dir / "temporal_cutoff_audit.json",
        {
            "verdict": temporal_verdict,
            "request_leak_n": len(request_leak),
            "return_leak_n": len(return_leak),
            "evidence_leak_n": len(evidence_leak),
            "final_leak_n": len(final_leak),
            "unique_evidence_event_total": sum(
                int(t.get("unique_evidence_n") or 0) for t in temporal_rows
            ),
        },
    )
    with (out_dir / "temporal_affected_tasks.jsonl").open("w", encoding="utf-8") as fh:
        for t in contaminated:
            fh.write(json.dumps(t, ensure_ascii=False) + "\n")
    _write_json(out_dir / "accuracy_metrics.json", accuracy_metrics)
    _write_csv(
        out_dir / "manual_review_queue.csv",
        manual_queue,
        fieldnames=list(manual_queue[0].keys())
        if manual_queue
        else ["task_id", "human_grade_pending"],
    )
    _write_json(out_dir / "human_grades.json", human_grades)
    _write_csv(
        out_dir / "per_task_audit.csv",
        per_task_audit,
        fieldnames=list(per_task_audit[0].keys()) if per_task_audit else ["task_id"],
    )
    _write_json(out_dir / "domain_breakdown.json", domain_breakdown)
    _write_json(out_dir / "answerability_breakdown.json", answerability_breakdown)
    _write_json(
        out_dir / "split_difficulty_breakdown.json",
        {"source_split": split_breakdown, "difficulty": difficulty_breakdown},
    )
    _write_json(out_dir / "round_analysis.json", round_analysis)
    _write_json(out_dir / "failure_taxonomy.json", failure_taxonomy)
    _write_json(out_dir / "resource_summary.json", resource_summary)
    _write_json(out_dir / "baseline_comparison.json", baseline_comparison)
    _write_json(out_dir / "paired_lite_replan.json", paired)
    _write_json(
        out_dir / "VALIDITY_MARKER.json",
        {
            "marker": temporal_marker,
            "explicitly_not_formal_accuracy": True,
            "usable_for_formal_accuracy": False,
            "engineering_valid": identity_all_pass,
            "posthoc_exploratory": True,
        },
    )

    hashes_after = collect_raw_hashes(run_dir)
    raw_unchanged = hashes_before == hashes_after
    _write_json(out_dir / "raw_artifact_sha256.json", {
        "before": hashes_before,
        "after": hashes_after,
        "raw_hashes_unchanged": raw_unchanged,
    })

    report_lines = [
        "# PE-Fin RePlan-7 HK-155 Final Offline Audit",
        "",
        f"- audit_version: `{AUDIT_VERSION}`",
        f"- result_role: `{RESULT_ROLE}`",
        f"- marker: `{temporal_marker}`",
        f"- explicitly_not_formal_accuracy: `true`",
        f"- engineering_valid: `{identity_all_pass}`",
        f"- raw_hashes_unchanged: `{raw_unchanged}`",
        f"- run_commit: `{EXPECTED_RUN_COMMIT}`",
        f"- cutoff: `{BENCHMARK_CUTOFF}`",
        "",
        "## Accuracy (descriptive)",
        f"- frozen raw correct: **{raw_true}/{EXPECTED_N}**",
        f"- common strict: **{strict}/{EXPECTED_N}**",
        f"- conservative (answer-bearing contamination forced incorrect): **{cons_true}/{EXPECTED_N}**",
        f"- clean-subset: **{clean_true}/{len(clean_rows)}** (not formally comparable)",
        f"- manual pending: `{[r.get('task_id') for r in null_tasks]}`",
        f"- status full/partial/failed: `{status_counts.get('full',0)}/{status_counts.get('partial',0)}/{status_counts.get('failed',0)}`",
        "",
        "## Temporal",
        f"- contaminated tasks: `{len(contaminated_ids)}`",
        f"- answer-bearing contaminated: `{sorted(answer_bearing_ids)}`",
        f"- raw-correct among answer-bearing contaminated: `{[t['task_id'] for t in affected_correct]}`",
        "",
        "## Resources",
        f"- LLM={sum(llms)} MCP={sum(mcps)} tokens≈{sum(toks)} rounds={sum(rounds)}",
        f"- rounds mean correct/incorrect: `{_mean(correct_rounds)}` / `{_mean(incorrect_rounds)}`",
        "",
        "## Caveats",
        "- RePlan is post-hoc exploratory; Lite v1.2 remains frozen primary proposed-method.",
        "- No significance / provider-effect / unseen-heldout claims.",
        "- No method change, no rerun, no raw mutation, no ETOM, no push.",
        "",
    ]
    (out_dir / "final_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    summary = {
        "audit_version": AUDIT_VERSION,
        "result_role": RESULT_ROLE,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "audit_dir": str(out_dir),
        "git_run_commit": meta.get("git_commit"),
        "engineering_valid": identity_all_pass,
        "identity_all_pass": identity_all_pass,
        "raw_hashes_unchanged": raw_unchanged,
        "marker": temporal_marker,
        "explicitly_not_formal_accuracy": True,
        "usable_for_formal_accuracy": False,
        "temporal_verdict": temporal_verdict,
        "accuracy_metrics": accuracy_metrics,
        "status_counts": dict(status_counts),
        "round_analysis_headline": {
            "mean": round_analysis["mean"],
            "correct_mean": round_analysis["correct_mean"],
            "incorrect_mean": round_analysis["incorrect_mean"],
        },
        "resources": {
            "llm": sum(llms),
            "mcp": sum(mcps),
            "tokens": sum(toks),
            "rounds": sum(rounds),
        },
        "manual_pending": [r.get("task_id") for r in null_tasks],
        "zero_network": True,
        "no_rerun": True,
        "no_raw_mutation": True,
        "no_method_change": True,
        "no_push": True,
        "etom_not_run": True,
    }
    _write_json(out_dir / "audit_summary.json", summary)
    return summary
