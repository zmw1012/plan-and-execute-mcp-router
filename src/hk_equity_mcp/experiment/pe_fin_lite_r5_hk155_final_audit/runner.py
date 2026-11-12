"""Read-only R5 HK-155 final audit. Never mutates raw run artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from hk_equity_mcp.experiment.pe_fin_lite_checkpoint_audit import CHECKPOINT_IDS
from hk_equity_mcp.experiment.pe_fin_lite_checkpoint_audit.evaluate import (
    evaluate_task_checkpoints,
)
from hk_equity_mcp.experiment.pe_fin_lite_r5_hk155_final_audit import (
    AUDIT_VERSION,
    DEFAULT_AUDIT_SUBDIR,
    DOCS_MIRROR_SUBDIR,
    EXPECTED_CANDIDATE_LOCK,
    EXPECTED_DOMAIN_COUNTS,
    EXPECTED_FORMULA_HASH,
    EXPECTED_INDEX_IDENTITY,
    EXPECTED_N,
    RESULT_ROLE,
)
from hk_equity_mcp.experiment.unified_exploratory import load_exploratory_run_order
from hk_equity_mcp.experiment.unified_pe_fin_lite_metrics import (
    aggregate_pe_fin_lite_metrics,
)
from hk_equity_mcp.pe_fin_lite.retrieval_formula import (
    RETRIEVAL_FORMULA_NAME,
    retrieval_formula_hash,
)

RAW_NAMES = [
    "predictions.jsonl",
    "run_metadata.json",
    "cost_summary.json",
    "error_summary.json",
    "grading_summary.json",
    "grouped_aggregates.json",
    "manual_grading.csv",
    "pe_fin_lite_metrics.json",
    "per_task_metrics.csv",
    "retrieval_analysis.json",
    "step_funnel_summary.json",
]

BASELINES = {
    "MCP-Free": {
        "final_answer_accuracy": 0.1290,
        "common_strict_answer_success": None,
        "exact_tool_set": None,
        "tokens_total": 79133,
        "note": "frozen baseline figures from protocol; not re-derived here",
    },
    "Flat All Tools": {
        "final_answer_accuracy": 0.5677,
        "common_strict_answer_success": 0.5484,
        "exact_tool_set": 0.2323,
        "tokens_total": 9464124,
    },
    "Operation-Visible": {
        "final_answer_accuracy": 0.6129,
        "common_strict_answer_success": None,
        "exact_tool_set": 0.4129,
        "condition_specific_strict": 0.2903,
        "tokens_total": None,
        "note": "condition-specific strict not comparable across methods",
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
            "bytes": len(data),
            "sha256": sha256_file(path),
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


def _tid(row: Dict[str, Any]) -> str:
    return str(row.get("task_id") or row.get("id"))


def _mean(xs: Sequence[float]) -> Optional[float]:
    vals = [float(x) for x in xs if x is not None]
    return statistics.mean(vals) if vals else None


def _median(xs: Sequence[float]) -> Optional[float]:
    vals = [float(x) for x in xs if x is not None]
    return statistics.median(vals) if vals else None


def _p95(xs: Sequence[float]) -> Optional[float]:
    vals = sorted(float(x) for x in xs if x is not None)
    if not vals:
        return None
    idx = min(len(vals) - 1, max(0, int(math.ceil(0.95 * len(vals)) - 1)))
    return vals[idx]


def common_strict_answer_success(row: Dict[str, Any]) -> bool:
    """status=full AND answer_correct=true; partial/failed forced incorrect."""
    return row.get("status") == "full" and row.get("answer_correct") is True


def _score_matches_r5(row: Dict[str, Any]) -> bool:
    cos, lex, sc = row.get("cosine_similarity"), row.get("lexical_score"), row.get("score")
    if cos is None or lex is None or sc is None:
        return False
    expect = float(cos) + 0.12 * float(lex)
    return abs(float(sc) - round(expect, 6)) <= 1e-9 or abs(float(sc) - expect) <= 1e-6


def audit_r5_funnels(preds: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    n_emb_steps = 0
    n_disc_steps = 0
    score_ok = 0
    score_checked = 0
    domain_rejects = 0
    bad_filter = 0
    pool39 = 0
    topk_gt5 = 0
    desc_gt3 = 0
    primary_n = 0
    fallback_sel = 0
    fallback_used = 0
    rank45 = 0
    zero_cand = 0
    binder_outside = 0
    formula_name_mismatch = 0
    topk_ops = 0
    described_n = 0
    zero_invocation_steps = 0
    safety_only_steps = 0

    for pred in preds:
        for funnel in pred.get("retrieval_funnels") or []:
            if not isinstance(funnel, dict):
                continue
            if funnel.get("disclosure_composite"):
                n_disc_steps += 1
                continue
            topk = funnel.get("embedding_topk") or []
            if not topk and not funnel.get("retrieval_formula") and not funnel.get("hard_filtered"):
                # empty non-disclosure funnel (rare)
                continue
            n_emb_steps += 1
            hf = funnel.get("hard_filtered") or {}
            if funnel.get("domain_hard_filter") is not False:
                bad_filter += 1
            if hf.get("filter_kind") not in {"safety_only_r5", None}:
                if topk:
                    bad_filter += 1
            if hf.get("filter_kind") == "safety_only_r5":
                safety_only_steps += 1
                if hf.get("kept_count") == 39:
                    pool39 += 1
            rejected = hf.get("rejected") or []
            domain_rejects += sum(
                1
                for r in rejected
                if isinstance(r, dict) and r.get("reason") == "domain_incompatible"
            )
            if len(topk) > 5:
                topk_gt5 += 1
            desc = funnel.get("described_candidates") or []
            if isinstance(desc, list) and len(desc) > 3:
                desc_gt3 += 1
            described_n += len(desc) if isinstance(desc, list) else 0
            topk_ops += len(topk)
            if not topk:
                zero_cand += 1
            inv = funnel.get("invoked_operations") or []
            if not inv:
                zero_invocation_steps += 1
            rf = funnel.get("retrieval_formula") or {}
            if topk and rf.get("name") not in (None, RETRIEVAL_FORMULA_NAME):
                formula_name_mismatch += 1
            for r in topk:
                score_checked += 1
                if _score_matches_r5(r):
                    score_ok += 1
            if (funnel.get("primary") or {}).get("operation_id") if isinstance(funnel.get("primary"), dict) else funnel.get("primary"):
                primary_n += 1
            fb = funnel.get("fallback")
            fb_id = fb.get("operation_id") if isinstance(fb, dict) else fb
            if fb_id:
                fallback_sel += 1
            lf = funnel.get("local_fallback") or {}
            if isinstance(lf, dict) and lf.get("attempted"):
                fallback_used += 1
            rank45 += len(funnel.get("rank4_5_not_described") or [])
            bsv = funnel.get("binder_selection_validation") or {}
            if bsv.get("selected_outside_described") is True:
                binder_outside += 1

    return {
        "embedding_retrieve_steps": n_emb_steps,
        "disclosure_retrieve_steps": n_disc_steps,
        "score_rows_checked": score_checked,
        "score_rows_ok": score_ok,
        "score_formula_ok": score_checked == 0 or score_ok == score_checked,
        "domain_incompatible_rejects": domain_rejects,
        "filter_anomalies": bad_filter,
        "safety_only_r5_steps": safety_only_steps,
        "pool39_steps": pool39,
        "topk_gt5_steps": topk_gt5,
        "describe_gt3_steps": desc_gt3,
        "topk_candidate_rows": topk_ops,
        "described_candidate_rows": described_n,
        "primary_selections": primary_n,
        "fallback_selections": fallback_sel,
        "fallback_attempts": fallback_used,
        "rank4_5_not_described_entries": rank45,
        "zero_candidate_steps": zero_cand,
        "zero_invocation_steps": zero_invocation_steps,
        "binder_outside_described": binder_outside,
        "formula_name_mismatches": formula_name_mismatch,
        "note_zero_invocation": "zero_invocation_steps != tool success; counted separately",
        "r5_runtime_effective": (
            bad_filter == 0
            and domain_rejects == 0
            and topk_gt5 == 0
            and desc_gt3 == 0
            and formula_name_mismatch == 0
            and binder_outside == 0
            and (score_checked == 0 or score_ok == score_checked)
        ),
    }


def evaluation_only_retrieval_metrics(preds: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Gold-grounded Hit/Recall@K / MRR / complete-route — evaluation only."""
    hit = {1: [], 3: [], 5: []}
    recall = {1: [], 3: [], 5: []}
    mrrs: List[float] = []
    complete = {3: [], 5: []}

    for pred in preds:
        gold = {str(x) for x in (pred.get("gold_operations") or []) if x}
        if not gold:
            continue
        # union of topk across retrieve steps, preserving best rank
        best_rank: Dict[str, int] = {}
        step_lists: List[List[str]] = []
        for funnel in pred.get("retrieval_funnels") or []:
            if not isinstance(funnel, dict) or funnel.get("disclosure_composite"):
                continue
            ids = [
                str(r.get("operation_id"))
                for r in (funnel.get("embedding_topk") or [])
                if r.get("operation_id")
            ]
            if ids:
                step_lists.append(ids)
            for i, oid in enumerate(ids, start=1):
                best_rank[oid] = min(best_rank.get(oid, 10**9), i)
        if not best_rank and not step_lists:
            for k in (1, 3, 5):
                hit[k].append(0.0)
                recall[k].append(0.0)
            mrrs.append(0.0)
            complete[3].append(0.0)
            complete[5].append(0.0)
            continue
        ranks = [best_rank[g] for g in gold if g in best_rank]
        first = min(ranks) if ranks else None
        for k in (1, 3, 5):
            hit[k].append(1.0 if first is not None and first <= k else 0.0)
            covered = {g for g in gold if best_rank.get(g, 10**9) <= k}
            recall[k].append(len(covered) / len(gold))
        mrrs.append(0.0 if first is None else 1.0 / float(first))
        for k in (3, 5):
            # complete-route: every gold appears in some step's top-k (multi-step union)
            ok = all(best_rank.get(g, 10**9) <= k for g in gold)
            complete[k].append(1.0 if ok else 0.0)

    def avg(xs: Sequence[float]) -> Optional[float]:
        return statistics.mean(xs) if xs else None

    return {
        "n_tasks_with_gold_ops": len(mrrs),
        "hit@1": avg(hit[1]),
        "hit@3": avg(hit[3]),
        "hit@5": avg(hit[5]),
        "recall@1": avg(recall[1]),
        "recall@3": avg(recall[3]),
        "recall@5": avg(recall[5]),
        "mrr": avg(mrrs),
        "complete_route@3": avg(complete[3]),
        "complete_route@5": avg(complete[5]),
        "note": "evaluation-only; gold ops never enter runtime ranking",
    }


def _slice_rows(preds: Sequence[Dict[str, Any]], key: str, value: Any) -> List[Dict[str, Any]]:
    return [p for p in preds if p.get(key) == value]


def slice_metrics(preds: Sequence[Dict[str, Any]], blockers: Dict[str, str]) -> Dict[str, Any]:
    if not preds:
        return {"n": 0}
    n = len(preds)
    ans = [p.get("answer_correct") for p in preds]
    ans_true = sum(1 for a in ans if a is True)
    graded = [a for a in ans if a is not None]
    return {
        "n": n,
        "answer_correct_true": ans_true,
        "answer_correct_false": sum(1 for a in ans if a is False),
        "answer_correct_null": sum(1 for a in ans if a is None),
        "auto_gradable_accuracy": (sum(1 for a in graded if a) / len(graded)) if graded else None,
        "common_strict_answer_success": sum(1 for p in preds if common_strict_answer_success(p)) / n,
        "exact_tool_set": sum(1 for p in preds if p.get("exact_tool_set")) / n,
        "status_counts": dict(Counter(p.get("status") for p in preds)),
        "primary_blocker_counts": dict(
            Counter(blockers.get(_tid(p), "none") for p in preds)
        ),
        "mean_llm_calls": _mean(
            [
                float((p.get("budget") or {}).get("llm_calls_used")
                or (p.get("budget") or {}).get("llm_calls")
                or p.get("llm_calls")
                or 0)
                for p in preds
            ]
        ),
        "mean_mcp_calls": _mean(
            [float((p.get("budget") or {}).get("mcp_invocations") or 0) for p in preds]
        ),
        "mean_tokens": _mean(
            [
                float(
                    ((p.get("tokens") or {}).get("total_tokens"))
                    or (p.get("input_tokens") or 0) + (p.get("output_tokens") or 0)
                    or 0
                )
                for p in preds
            ]
        ),
        "mean_latency_ms": _mean([float(p.get("latency_ms") or 0) for p in preds]),
    }


def analyze_disclosure(preds: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    counts = Counter()
    details = []
    for pred in preds:
        for funnel in pred.get("retrieval_funnels") or []:
            if not isinstance(funnel, dict):
                continue
            dc = funnel.get("disclosure_composite")
            if not dc:
                continue
            counts["disclosure_steps"] += 1
            inv = list(funnel.get("invoked_operations") or dc.get("invoked_operations") or [])
            inv_s = " ".join(str(x) for x in inv)
            if "list_hkexnews" in inv_s:
                counts["list"] += 1
            if "download_hkexnews" in inv_s:
                counts["download"] += 1
            if "parse_hkexnews" in inv_s:
                counts["parse"] += 1
            if "keyword_search" in inv_s:
                counts["keyword"] += 1
            if "get_pages" in inv_s or "pages" in inv_s:
                counts["pages"] += 1
            for key in (
                "list_ok",
                "download_ok",
                "parse_ok",
                "keyword_ok",
                "pages_ok",
                "shortest_required_prefix_stopped",
                "no_matching_result",
                "envelope_adapter_failure",
                "stopped_after_list",
                "execution_ok",
                "download_budget_exhausted",
                "cutoff_mismatch",
                "title_mismatch",
                "evidence_found_not_accepted",
                "chain_started_incomplete",
                "chain_complete",
            ):
                if dc.get(key) is True:
                    counts[key] += 1
            # Derived chain status if not explicit
            if dc.get("execution_ok") is True and not dc.get("stopped_after_list"):
                counts["chain_complete_proxy"] += 1
            if inv and dc.get("execution_ok") is not True:
                counts["chain_started_but_incomplete_proxy"] += 1
            if dc.get("stopped_after_list") is True:
                counts["list_success_not_body_evidence"] += 1
            missing = dc.get("missing_outputs") or []
            if missing and dc.get("execution_ok") is not True:
                counts["evidence_found_but_not_accepted_proxy"] += 1
            details.append(
                {
                    "task_id": _tid(pred),
                    "step_id": funnel.get("step_id"),
                    "invoked": inv,
                    "execution_ok": dc.get("execution_ok"),
                    "missing_outputs": missing,
                    "stopped_after_list": dc.get("stopped_after_list"),
                }
            )
    return {
        "counts": dict(counts),
        "n_details": len(details),
        "note": "list success ≠ body evidence complete; proxies labeled _proxy when flag absent",
        "sample": details[:20],
    }


def analyze_compute_reason_final(preds: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    compute_planned = compute_exec = compute_ok = 0
    reason_planned = reason_ok = reason_repair = reason_fail = 0
    final = Counter()
    for pred in preds:
        plan_steps = ((pred.get("initial_plan") or {}).get("steps") or [])
        types = [s.get("type") for s in plan_steps]
        if "compute" in types:
            compute_planned += 1
        if "reason" in types:
            reason_planned += 1
        for st in pred.get("step_statuses") or []:
            if st.get("type") == "compute":
                compute_exec += 1
                if st.get("status") == "completed":
                    compute_ok += 1
            if st.get("type") == "reason":
                if st.get("status") == "completed":
                    reason_ok += 1
                if st.get("reason_parse_failed"):
                    reason_fail += 1
        budget = pred.get("budget") or {}
        cbs = budget.get("calls_by_stage") or {}
        if int(cbs.get("reason_repair") or 0) > 0:
            reason_repair += 1
        ps = str(pred.get("predicted_status") or "")
        final[ps] += 1
        if pred.get("status") == "full" and pred.get("answer_correct") is True:
            final["semantically_correct_full"] += 1
    return {
        "compute": {
            "planned": compute_planned,
            "executed_status_rows": compute_exec,
            "completed": compute_ok,
        },
        "reason": {
            "planned": reason_planned,
            "completed_status_rows": reason_ok,
            "repair_tasks": reason_repair,
            "parse_failed_flags": reason_fail,
        },
        "final": dict(final),
    }


def build_identity_audit(meta: Dict[str, Any], *, root: Path, preds: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    order = load_exploratory_run_order(root)
    ids = [_tid(p) for p in preds]
    domain_counts = dict(Counter(p.get("domain") for p in preds))
    formula_hash = retrieval_formula_hash()
    checks = {
        "n_rows_155": len(preds) == EXPECTED_N,
        "unique_ids_155": len(set(ids)) == EXPECTED_N,
        "order_match_exploratory_all": ids == list(order.get("task_ids") or []),
        "domain_counts": domain_counts == EXPECTED_DOMAIN_COUNTS,
        "run_completed": meta.get("run_status") == "completed",
        "processed_155": int(meta.get("processed_task_count") or 0) == EXPECTED_N,
        "method_pe_fin_lite": meta.get("condition") == "pe_fin_lite",
        "method_version": meta.get("method_version") == "pe_fin_lite_v1.2",
        "architecture": meta.get("architecture") == "plan_first_step_level_retrieval_v1",
        "formula_name": meta.get("retrieval_formula_name") == RETRIEVAL_FORMULA_NAME,
        "formula_hash": meta.get("retrieval_formula_hash") == EXPECTED_FORMULA_HASH == formula_hash,
        "candidate_lock": meta.get("candidate_lock_sha256") == EXPECTED_CANDIDATE_LOCK,
        "domain_hard_filter_false": meta.get("domain_hard_filter") is False,
        "topk_5": int(meta.get("retrieval_top_k") or 0) == 5,
        "describe_3": int(meta.get("describe_top_n") or 0) == 3,
        "monitor_off": meta.get("monitor_enabled") is False,
        "replan_off": meta.get("global_replan_enabled") is False,
        "generation_ok": (
            meta.get("generation_provider") == "openai_compatible"
            and meta.get("generation_base_url_host") == "api.zhizengzeng.com"
            and meta.get("generation_model") == "deepseek-v4.1-flash"
        ),
        "embedding_ok": (
            meta.get("embedding_base_url_host") == "api.zhizengzeng.com"
            and meta.get("embedding_model") == "text-embedding-3-small"
            and int(meta.get("embedding_dimension") or 0) == 1536
            and meta.get("embedding_index_identity_sha256") == EXPECTED_INDEX_IDENTITY
        ),
        "git_dirty_false": meta.get("git_dirty") is False,
        "no_mock": meta.get("mock") is False and meta.get("dry_run") is False,
        "real_mcp": meta.get("execution_backend") == "real_mcp",
        "subset_full": meta.get("subset_kind") == "full",
        "not_smoke_resume": meta.get("smoke5") is False,
    }
    return {
        "checks": checks,
        "all_required_pass": all(bool(v) for v in checks.values()),
        "domain_counts": domain_counts,
        "missing_task_ids": [
            tid for tid in (order.get("task_ids") or []) if tid not in set(ids)
        ],
        "duplicate_count": len(ids) - len(set(ids)),
        "run_order_hash": order.get("run_order_hash"),
        "meta_git_commit": meta.get("git_commit"),
        "embedding_provider_null": meta.get("embedding_provider") is None,
        "formula_hash_recomputed": formula_hash,
    }


def run_final_audit(
    *,
    run_dir: Path,
    root: Path,
    output_dir: Optional[Path] = None,
    docs_mirror: Optional[Path] = None,
) -> Dict[str, Any]:
    run_dir = Path(run_dir).resolve()
    root = Path(root).resolve()
    out_dir = Path(output_dir) if output_dir else (run_dir / DEFAULT_AUDIT_SUBDIR)
    if out_dir.resolve() == run_dir.resolve():
        raise RuntimeError("Refuse to write audit into run_dir root")
    out_dir.mkdir(parents=True, exist_ok=True)

    before = collect_raw_hashes(run_dir)
    meta = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    preds = _load_preds(run_dir / "predictions.jsonl")
    identity = build_identity_audit(meta, root=root, preds=preds)

    # Preliminary fact check (descriptive)
    status_counts = Counter(p.get("status") for p in preds)
    ans_counts = Counter(
        ("null" if p.get("answer_correct") is None else bool(p.get("answer_correct")))
        for p in preds
    )
    preliminary = {
        "observed": {
            "rows": len(preds),
            "unique_ids": len({_tid(p) for p in preds}),
            "full": status_counts.get("full", 0),
            "partial": status_counts.get("partial", 0),
            "failed": status_counts.get("failed", 0),
            "answer_correct_true": ans_counts.get(True, 0),
            "answer_correct_false": ans_counts.get(False, 0),
            "answer_correct_null": ans_counts.get("null", 0),
        },
        "expected_snapshot": {
            "rows": 155,
            "unique_ids": 155,
            "full": 14,
            "partial": 47,
            "failed": 94,
            "answer_correct_true": 35,
            "answer_correct_false": 119,
            "answer_correct_null": 1,
        },
    }
    preliminary["matches_expected_snapshot"] = (
        preliminary["observed"] == preliminary["expected_snapshot"]
    )

    # Checkpoints + earliest blockers
    per_task: List[Dict[str, Any]] = []
    checkpoint_status_counts: Dict[str, Counter] = {
        cid: Counter() for cid in CHECKPOINT_IDS
    }
    primary_blockers: Counter = Counter()
    secondary_total = 0

    for pred in preds:
        cp = evaluate_task_checkpoints(pred)
        primary = cp.get("primary_blocking_checkpoint")
        secondary = cp.get("secondary_failures") or []
        secondary_total += len(secondary)
        primary_blockers[primary or "none_or_all_non_fail"] += 1
        for cid in CHECKPOINT_IDS:
            st = ((cp.get("checkpoints") or {}).get(cid) or {}).get("status")
            checkpoint_status_counts[cid][st or "unknown"] += 1
        budget = pred.get("budget") or {}
        llm = int(budget.get("llm_calls_used") or budget.get("llm_calls") or pred.get("llm_calls") or 0)
        mcp = int(budget.get("mcp_invocations") or 0)
        emb = int(budget.get("embedding_query_calls") or 0)
        fb = int(budget.get("local_fallback_count") or 0)
        per_task.append(
            {
                "task_id": _tid(pred),
                "domain": pred.get("domain"),
                "project_difficulty": pred.get("project_difficulty"),
                "source_split": pred.get("source_split"),
                "answerability_class": pred.get("answerability_class"),
                "status": pred.get("status"),
                "predicted_status": pred.get("predicted_status"),
                "answer_correct": pred.get("answer_correct"),
                "auto_grade_status": pred.get("auto_grade_status"),
                "auto_grade_reason": pred.get("auto_grade_reason"),
                "exact_tool_set": bool(pred.get("exact_tool_set")),
                "operation_f1": pred.get("operation_f1"),
                "operation_precision": pred.get("operation_precision"),
                "operation_recall": pred.get("operation_recall"),
                "common_strict_answer_success": common_strict_answer_success(pred),
                "llm_calls": llm,
                "mcp_invocations": mcp,
                "embedding_query_calls": emb,
                "local_fallback_count": fb,
                "monitor_calls": int(budget.get("monitor_calls") or 0),
                "global_replan_rounds": int(budget.get("global_replan_rounds") or 0),
                "latency_ms": pred.get("latency_ms"),
                "tokens_total": (
                    ((pred.get("tokens") or {}).get("total_tokens"))
                    or (pred.get("input_tokens") or 0) + (pred.get("output_tokens") or 0)
                ),
                "primary_blocking_checkpoint": primary,
                "secondary_failures": secondary,
                "checkpoints": cp.get("checkpoints"),
                "budget_ok": llm <= 8 and mcp <= 8 and fb <= 1,
            }
        )

    blockers_by_id = {
        r["task_id"]: r.get("primary_blocking_checkpoint") or "none_or_all_non_fail"
        for r in per_task
    }

    # Budget violations (engineering)
    budget_violations = [
        r["task_id"]
        for r in per_task
        if r["llm_calls"] > 8 or r["mcp_invocations"] > 8 or r["local_fallback_count"] > 1
        or r["monitor_calls"] != 0
        or r["global_replan_rounds"] != 0
    ]

    r5 = audit_r5_funnels(preds)
    retrieval_eval = evaluation_only_retrieval_metrics(preds)
    pe_metrics = aggregate_pe_fin_lite_metrics(list(preds))

    # Answer metrics (separated)
    n = len(preds)
    auto = [p for p in preds if p.get("auto_grade_status") != "manual_required"]
    manual = [p for p in preds if p.get("auto_grade_status") == "manual_required" or p.get("answer_correct") is None]
    auto_correct = sum(1 for p in auto if p.get("answer_correct") is True)
    answer_metrics = {
        "n": n,
        "auto_gradable_n": len(auto),
        "manual_required_n": len(manual),
        "auto_gradable_accuracy": (auto_correct / len(auto)) if auto else None,
        "final_answer_accuracy_lower_bound": auto_correct / n,  # manual counted incorrect
        "final_answer_accuracy_upper_bound": (auto_correct + len(manual)) / n,
        "final_answer_accuracy_pending_manual": True,
        "common_strict_answer_success": {
            "definition": "status==full AND answer_correct==true; partial/failed forced incorrect; denominator=155",
            "count": sum(1 for p in preds if common_strict_answer_success(p)),
            "rate": sum(1 for p in preds if common_strict_answer_success(p)) / n,
        },
        "exact_tool_set": {
            "count": sum(1 for p in preds if p.get("exact_tool_set")),
            "rate": sum(1 for p in preds if p.get("exact_tool_set")) / n,
        },
        "operation_precision_mean": pe_metrics.get("operation_precision_mean"),
        "operation_recall_mean": pe_metrics.get("operation_recall_mean"),
        "operation_f1_mean": pe_metrics.get("operation_f1_mean"),
        "all_mcp_invocations_successful_rate": pe_metrics.get("invocation_success_rate"),
        "evidence_acceptance_proxy_required_output_satisfaction_mean": pe_metrics.get(
            "required_output_satisfaction_mean"
        ),
        "condition_specific_end_to_end_success": {
            "formula": (
                "PE-Fin Lite strict_answer_success_rate from aggregate_pe_fin_lite_metrics: "
                "answer_correct AND predicted_status=='answered' AND status in {full,partial}; "
                "NOT comparable to other conditions' condition-specific strict."
            ),
            "rate": pe_metrics.get("strict_answer_success_rate"),
        },
        "refusal_clarification": {
            "n_should_refuse_or_clarify": sum(
                1 for p in preds if p.get("answerability_class") == "should_refuse_or_clarify"
            ),
            "predicted_unavailable_or_clarify": sum(
                1
                for p in preds
                if str(p.get("predicted_status") or "")
                in {"unavailable", "clarification_needed"}
            ),
        },
        "status_distribution": dict(status_counts),
    }

    # Funnel aggregate — explicit status buckets; N/A ≠ pass; not_evaluated ≠ pass
    STATUS_BUCKETS = (
        "pass",
        "fail",
        "N/A",
        "N/A_not_planned",
        "N/A_not_required",
        "N/A_not_reached",
        "N/A_upstream_blocked",
        "not_evaluated",
        "unknown",
    )
    checkpoint_funnel: Dict[str, Any] = {}
    for cid in CHECKPOINT_IDS:
        cnt = checkpoint_status_counts[cid]
        buckets = {b: int(cnt.get(b, 0)) for b in STATUS_BUCKETS}
        for k, v in cnt.items():
            if k not in buckets:
                buckets[str(k)] = int(v)
        contract_c: Counter = Counter()
        stage_c: Counter = Counter()
        effect_c: Counter = Counter()
        for r in per_task:
            cp = (r.get("checkpoints") or {}).get(cid) or {}
            contract_c[cp.get("contract_status") or "unknown"] += 1
            stage_c[cp.get("stage_outcome_status") or cp.get("status") or "unknown"] += 1
            effect_c[cp.get("downstream_task_effect") or "none"] += 1
        checkpoint_funnel[cid] = {
            "counts": buckets,
            "pass": buckets["pass"],
            "fail": buckets["fail"],
            "N/A_not_planned": buckets["N/A_not_planned"],
            "N/A_not_reached": buckets["N/A_not_reached"],
            "N/A_upstream_blocked": buckets["N/A_upstream_blocked"],
            "not_evaluated": buckets["not_evaluated"],
            "unknown": buckets["unknown"],
            "contract_status_counts": dict(contract_c),
            "stage_outcome_status_counts": dict(stage_c),
            "downstream_task_effect_counts": dict(effect_c),
            "note": "N/A and not_evaluated are never counted as pass; trace presence ≠ stage success",
        }

    # Grouped
    grouped: Dict[str, Any] = {
        "by_domain": {},
        "by_difficulty": {},
        "by_source_split": {},
        "by_answerability": {},
        "by_status": {},
    }
    for dom in ("financial", "market", "news", "cross_domain"):
        grouped["by_domain"][dom] = slice_metrics(_slice_rows(preds, "domain", dom), blockers_by_id)
    for diff in ("D1", "D2", "D3", "D4", "D5"):
        grouped["by_difficulty"][diff] = slice_metrics(
            _slice_rows(preds, "project_difficulty", diff), blockers_by_id
        )
    for split in ("development", "previously_reserved"):
        grouped["by_source_split"][split] = slice_metrics(
            _slice_rows(preds, "source_split", split), blockers_by_id
        )
    for ans in sorted({p.get("answerability_class") for p in preds}):
        grouped["by_answerability"][str(ans)] = slice_metrics(
            _slice_rows(preds, "answerability_class", ans), blockers_by_id
        )
    for st in ("full", "partial", "failed"):
        grouped["by_status"][st] = slice_metrics(_slice_rows(preds, "status", st), blockers_by_id)

    # Primary blockers by domain/difficulty/status
    primary_blockers_payload = {
        "primary_blocking_checkpoint_counts": dict(primary_blockers),
        "secondary_failure_total": secondary_total,
        "by_domain": {
            dom: dict(
                Counter(
                    blockers_by_id[_tid(p)]
                    for p in preds
                    if p.get("domain") == dom
                )
            )
            for dom in ("financial", "market", "news", "cross_domain")
        },
        "by_difficulty": {
            d: dict(
                Counter(
                    blockers_by_id[_tid(p)]
                    for p in preds
                    if p.get("project_difficulty") == d
                )
            )
            for d in ("D1", "D2", "D3", "D4", "D5")
        },
        "by_status": {
            st: dict(
                Counter(
                    blockers_by_id[_tid(p)] for p in preds if p.get("status") == st
                )
            )
            for st in ("full", "partial", "failed")
        },
    }

    # Error taxonomy (lightweight from primary blockers + error_type)
    error_taxonomy = {
        "primary_blocking_checkpoint_counts": dict(primary_blockers),
        "error_type_counts": dict(Counter(p.get("error_type") for p in preds)),
        "auto_grade_reason_counts": dict(
            Counter(p.get("auto_grade_reason") for p in preds if p.get("auto_grade_reason"))
        ),
    }

    disclosure = analyze_disclosure(preds)
    crf = analyze_compute_reason_final(preds)

    # Resources
    latencies = [float(p.get("latency_ms") or 0) for p in preds]
    tokens = [
        float(
            ((p.get("tokens") or {}).get("total_tokens"))
            or (p.get("input_tokens") or 0) + (p.get("output_tokens") or 0)
            or 0
        )
        for p in preds
    ]
    resource = {
        "llm_calls_total": sum(r["llm_calls"] for r in per_task),
        "mcp_invocations_total": sum(r["mcp_invocations"] for r in per_task),
        "embedding_query_calls_total": sum(r["embedding_query_calls"] for r in per_task),
        "tokens_total_sum_per_task": sum(tokens),
        "tokens_total_from_cost_summary": (json.loads((run_dir / "cost_summary.json").read_text()) if (run_dir / "cost_summary.json").exists() else {}).get("tokens_total"),
        "latency_ms_mean": _mean(latencies),
        "latency_ms_median": _median(latencies),
        "latency_ms_p95": _p95(latencies),
        "estimated_usd": None,
        "calls_by_stage_totals": dict(
            Counter(
                k
                for p in preds
                for k, v in ((p.get("budget") or {}).get("calls_by_stage") or {}).items()
                for _ in range(int(v or 0))
            )
        ),
        "fallback_total": sum(r["local_fallback_count"] for r in per_task),
        "budget_violations": budget_violations,
        "catalog_index_build_calls": 0,
        "note": "index reused complete cache; no build during this run expected",
    }

    # Manual review
    manual_rows = []
    for p in preds:
        if p.get("answer_correct") is None or p.get("auto_grade_status") == "manual_required":
            manual_rows.append(
                {
                    "task_id": _tid(p),
                    "domain": p.get("domain"),
                    "question": (p.get("question") or "")[:300],
                    "final_answer": (p.get("final_answer") or "")[:300],
                    "predicted_status": p.get("predicted_status"),
                    "status": p.get("status"),
                    "auto_grade_status": p.get("auto_grade_status"),
                    "auto_grade_reason": p.get("auto_grade_reason"),
                    "answerability_class": p.get("answerability_class"),
                    "gold_status": p.get("gold_status"),
                    "human_answer_correct": "",
                    "human_notes": "",
                }
            )
    human_grades = {
        "schema_version": "pe_fin_lite_r5_hk155_human_grades_v1",
        "instruction": (
            "Fill human_answer_correct with true/false/null per task_id. "
            "Do not copy grades from other conditions. No task-id hardcoding in tools."
        ),
        "pending_task_ids": [r["task_id"] for r in manual_rows],
        "grades": {
            r["task_id"]: {
                "human_answer_correct": None,
                "human_notes": None,
            }
            for r in manual_rows
        },
    }

    # Engineering validity
    engineering_blockers = []
    if not identity["all_required_pass"]:
        failed = [k for k, v in identity["checks"].items() if not v]
        engineering_blockers.append({"type": "identity_check_failed", "failed": failed})
    if not r5["r5_runtime_effective"]:
        engineering_blockers.append({"type": "r5_not_effective", "detail": r5})
    if budget_violations:
        engineering_blockers.append(
            {"type": "budget_counter_or_enforcement_violation", "tasks": budget_violations}
        )
    if identity["duplicate_count"] or identity["missing_task_ids"]:
        engineering_blockers.append({"type": "completeness_failure"})

    # High failure rates alone are NOT engineering invalidation
    engineering_valid = len(engineering_blockers) == 0
    engineering_verdict = {
        "engineering_valid": engineering_valid,
        "engineering_blockers": engineering_blockers,
        "method_valid_failure_note": (
            "High full/partial/failed counts and low answer accuracy are treated as "
            "method-valid outcomes under the frozen R5 PE-Fin Lite contract unless an "
            "identity/R5-wiring/budget-implementation inconsistency is proven."
        ),
        "evidence_r5_wired": r5["r5_runtime_effective"] and identity["checks"]["formula_hash"],
        "evidence_domain_filter_off": r5["domain_incompatible_rejects"] == 0
        and identity["checks"]["domain_hard_filter_false"],
    }

    # Baseline comparison (descriptive)
    pe_final_lower = answer_metrics["final_answer_accuracy_lower_bound"]
    pe_strict = answer_metrics["common_strict_answer_success"]["rate"]
    pe_exact = answer_metrics["exact_tool_set"]["rate"]
    pe_tokens = resource["tokens_total_from_cost_summary"]
    baseline_comparison = {
        "pe_fin_lite_r5": {
            "final_answer_accuracy_lower_bound": pe_final_lower,
            "final_answer_accuracy_upper_bound": answer_metrics[
                "final_answer_accuracy_upper_bound"
            ],
            "common_strict_answer_success": pe_strict,
            "exact_tool_set": pe_exact,
            "tokens_total": pe_tokens,
            "condition_specific_strict": pe_metrics.get("strict_answer_success_rate"),
            "condition_specific_note": "not comparable across methods",
        },
        "baselines_frozen": BASELINES,
        "descriptive_deltas_vs_baselines_final_answer_lower": {
            name: (None if b.get("final_answer_accuracy") is None else pe_final_lower - b["final_answer_accuracy"])
            for name, b in BASELINES.items()
        },
        "descriptive_deltas_vs_baselines_common_strict": {
            name: (
                None
                if b.get("common_strict_answer_success") is None
                else pe_strict - b["common_strict_answer_success"]
            )
            for name, b in BASELINES.items()
        },
        "no_significance_claims": True,
    }

    git_info = {
        "branch": meta.get("git_branch"),
        "commit": meta.get("git_commit"),
        "dirty_at_run": meta.get("git_dirty"),
        "audit_repo_commit_note": "audit commit recorded separately after this artifact write",
    }
    summary = {
        "audit_version": AUDIT_VERSION,
        "result_role": RESULT_ROLE,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "git": git_info,
        "engineering_valid": engineering_valid,
        "identity_all_pass": identity["all_required_pass"],
        "preliminary_facts_match": preliminary["matches_expected_snapshot"],
        "status_counts": dict(status_counts),
        "answer_metrics": answer_metrics,
        "r5_runtime_effective": r5["r5_runtime_effective"],
        "manual_review_required": bool(manual_rows),
        "manual_pending_n": len(manual_rows),
        "zero_network": True,
        "no_rerun": True,
        "no_raw_mutation": True,
        "no_method_change": True,
        "no_push": True,
    }

    # Write artifacts
    _write_json(out_dir / "raw_hashes.json", {"before": before, "after": None, "unchanged": None})
    _write_json(out_dir / "identity_audit.json", identity)
    _write_json(out_dir / "audit_summary.json", summary)
    _write_json(out_dir / "checkpoint_funnel.json", checkpoint_funnel)
    _write_json(out_dir / "primary_blockers.json", primary_blockers_payload)
    _write_json(out_dir / "error_taxonomy.json", error_taxonomy)
    _write_json(out_dir / "grouped_results.json", grouped)
    _write_json(
        out_dir / "retrieval_analysis.json",
        {"r5_runtime": r5, "evaluation_only_metrics": retrieval_eval, "pe_metrics_retrieval_recall_at_5": pe_metrics.get("retrieval_recall_at_5_mean")},
    )
    _write_json(out_dir / "disclosure_analysis.json", disclosure)
    _write_json(out_dir / "compute_reason_final_analysis.json", crf)
    _write_json(out_dir / "resource_summary.json", resource)
    _write_json(out_dir / "baseline_comparison.json", baseline_comparison)
    _write_json(out_dir / "preliminary_facts.json", preliminary)
    _write_json(out_dir / "engineering_verdict.json", engineering_verdict)
    _write_json(out_dir / "human_grades.json", human_grades)
    _write_csv(
        out_dir / "manual_review_queue.csv",
        manual_rows,
        [
            "task_id",
            "domain",
            "question",
            "final_answer",
            "predicted_status",
            "status",
            "auto_grade_status",
            "auto_grade_reason",
            "answerability_class",
            "gold_status",
            "human_answer_correct",
            "human_notes",
        ],
    )

    # per_task CSV (flat)
    flat_rows = []
    for r in per_task:
        flat = {k: r[k] for k in r if k != "checkpoints"}
        flat["secondary_failures"] = "|".join(r.get("secondary_failures") or [])
        for cid in CHECKPOINT_IDS:
            flat[f"{cid}_status"] = ((r.get("checkpoints") or {}).get(cid) or {}).get("status")
        flat_rows.append(flat)
    fields = [
        "task_id",
        "domain",
        "project_difficulty",
        "source_split",
        "answerability_class",
        "status",
        "predicted_status",
        "answer_correct",
        "auto_grade_status",
        "common_strict_answer_success",
        "exact_tool_set",
        "operation_f1",
        "llm_calls",
        "mcp_invocations",
        "embedding_query_calls",
        "local_fallback_count",
        "primary_blocking_checkpoint",
        "secondary_failures",
        "budget_ok",
    ] + [f"{cid}_status" for cid in CHECKPOINT_IDS]
    _write_csv(out_dir / "per_task_audit.csv", flat_rows, fields)
    _write_json(out_dir / "per_task_audit.json", per_task)

    # report.md
    report = _render_report(
        summary=summary,
        identity=identity,
        preliminary=preliminary,
        answer_metrics=answer_metrics,
        checkpoint_funnel=checkpoint_funnel,
        primary_blockers=primary_blockers_payload,
        r5=r5,
        retrieval_eval=retrieval_eval,
        grouped=grouped,
        disclosure=disclosure,
        crf=crf,
        resource=resource,
        baseline_comparison=baseline_comparison,
        engineering_verdict=engineering_verdict,
        manual_rows=manual_rows,
    )
    (out_dir / "report.md").write_text(report, encoding="utf-8")

    # finalize hashes
    after = collect_raw_hashes(run_dir)
    unchanged = before == after
    if not unchanged:
        raise RuntimeError("Raw artifacts changed during audit (fail-closed)")
    _write_json(
        out_dir / "raw_hashes.json",
        {"before": before, "after": after, "unchanged": True},
    )
    summary["raw_hashes_unchanged"] = True
    _write_json(out_dir / "audit_summary.json", summary)

    # docs mirror (repo-internal; allowed)
    docs_dir = Path(docs_mirror) if docs_mirror else (root / DOCS_MIRROR_SUBDIR)
    docs_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "audit_summary.json",
        "raw_hashes.json",
        "identity_audit.json",
        "checkpoint_funnel.json",
        "primary_blockers.json",
        "error_taxonomy.json",
        "grouped_results.json",
        "retrieval_analysis.json",
        "disclosure_analysis.json",
        "compute_reason_final_analysis.json",
        "resource_summary.json",
        "baseline_comparison.json",
        "engineering_verdict.json",
        "preliminary_facts.json",
        "human_grades.json",
        "manual_review_queue.csv",
        "per_task_audit.csv",
        "report.md",
    ):
        src = out_dir / name
        if src.exists():
            (docs_dir / name).write_bytes(src.read_bytes())

    # final raw check again after docs mirror (docs outside run_dir)
    after2 = collect_raw_hashes(run_dir)
    if after2 != before:
        raise RuntimeError("Raw artifacts changed after docs mirror")

    return {
        "out_dir": str(out_dir),
        "docs_dir": str(docs_dir),
        "engineering_valid": engineering_valid,
        "summary": summary,
        "manual_pending_n": len(manual_rows),
    }


def _render_report(**kwargs: Any) -> str:
    s = kwargs["summary"]
    identity = kwargs["identity"]
    am = kwargs["answer_metrics"]
    pb = kwargs["primary_blockers"]
    r5 = kwargs["r5"]
    eng = kwargs["engineering_verdict"]
    cf = kwargs["checkpoint_funnel"]
    grouped = kwargs["grouped"]
    disclosure = kwargs["disclosure"]
    crf = kwargs["crf"]
    resource = kwargs["resource"]
    baseline = kwargs["baseline_comparison"]
    retrieval_eval = kwargs["retrieval_eval"]
    preliminary = kwargs["preliminary"]
    git = s.get("git") or {}

    funnel_lines = []
    for cid, payload in cf.items():
        funnel_lines.append(
            f"- `{cid}`: pass={payload.get('pass')} fail={payload.get('fail')} "
            f"N/A_not_planned={payload.get('N/A_not_planned')} "
            f"N/A_not_reached={payload.get('N/A_not_reached')} "
            f"N/A_upstream_blocked={payload.get('N/A_upstream_blocked')} "
            f"not_evaluated={payload.get('not_evaluated')} unknown={payload.get('unknown')}"
        )

    lines = [
        f"# PE-Fin Lite R5 HK-155 Final Audit ({AUDIT_VERSION})",
        "",
        "## 1. Branch / commit / status",
        f"- run git_branch: `{git.get('branch')}`",
        f"- run git_commit: `{git.get('commit')}`",
        f"- run git_dirty: `{git.get('dirty_at_run')}`",
        f"- audit timestamp_utc: `{s['timestamp_utc']}`",
        f"- identity_all_pass: `{identity['all_required_pass']}`",
        f"- preliminary_facts_match: `{preliminary['matches_expected_snapshot']}`",
        "",
        "## 2. Raw SHA256 before/after",
        "- See `raw_hashes.json` (`unchanged=true` required).",
        f"- audit raw_hashes_unchanged flag (post-write): sealed in artifact.",
        "",
        "## 3. Identity verdict",
        f"- all_required_pass: `{identity['all_required_pass']}`",
        f"- failed_checks: `{[k for k,v in identity['checks'].items() if not v]}`",
        f"- domain_counts: `{identity['domain_counts']}`",
        f"- duplicate_count: `{identity['duplicate_count']}`",
        f"- missing_task_ids: `{identity['missing_task_ids']}`",
        f"- formula_hash_recomputed: `{identity.get('formula_hash_recomputed')}`",
        "",
        "## 4. 155 completeness",
        f"- rows/unique: `{preliminary['observed']['rows']}` / `{preliminary['observed']['unique_ids']}`",
        f"- order matches frozen exploratory_all: `{identity['checks'].get('order_match_exploratory_all')}`",
        f"- processed_task_count / run_status via identity checks: ok if all_required_pass",
        "",
        "## 5. Full / partial / failed",
        f"- `{s['status_counts']}`",
        "",
        "## 6. Auto / manual / final answer metrics",
        f"- auto_gradable_accuracy: `{am['auto_gradable_accuracy']}` (n={am['auto_gradable_n']})",
        f"- final_answer lower/upper: `{am['final_answer_accuracy_lower_bound']}` / `{am['final_answer_accuracy_upper_bound']}`",
        f"- final_answer_accuracy_pending_manual: `{am['final_answer_accuracy_pending_manual']}`",
        f"- manual_required_n: `{am['manual_required_n']}`",
        "",
        "## 7. Common strict answer success",
        f"- `{am['common_strict_answer_success']}`",
        "",
        "## 8. Exact tool-set and operation F1",
        f"- exact_tool_set: `{am['exact_tool_set']}`",
        f"- operation_precision_mean: `{am['operation_precision_mean']}`",
        f"- operation_recall_mean: `{am['operation_recall_mean']}`",
        f"- operation_f1_mean: `{am['operation_f1_mean']}`",
        f"- all_mcp_invocations_successful_rate: `{am['all_mcp_invocations_successful_rate']}`",
        f"- evidence_acceptance_proxy: `{am['evidence_acceptance_proxy_required_output_satisfaction_mean']}`",
        f"- condition-specific E2E (PE-Fin only): `{am['condition_specific_end_to_end_success']}`",
        f"- refusal/clarification: `{am['refusal_clarification']}`",
        "",
        "## 9. C01–C14 funnel",
        "*N/A ≠ pass; not_evaluated ≠ pass; evaluator executed ≠ answer correct.*",
        *funnel_lines,
        "",
        "## 10. Primary blockers",
        f"```json\n{json.dumps(pb, indent=2, ensure_ascii=False)}\n```",
        "",
        "## 11. Domain / difficulty / answerability",
        f"```json\n{json.dumps(grouped, indent=2, ensure_ascii=False)}\n```",
        "",
        "## 12. Retrieval R5",
        f"- r5_runtime_effective: `{r5['r5_runtime_effective']}`",
        f"```json\n{json.dumps(r5, indent=2)}\n```",
        f"- evaluation-only gold metrics: `{json.dumps(retrieval_eval)}`",
        "",
        "## 13. Disclosure / compute / reason / final",
        f"- disclosure: `{json.dumps(disclosure.get('counts'), ensure_ascii=False)}`",
        f"- compute/reason/final: `{json.dumps(crf, ensure_ascii=False)}`",
        "",
        "## 14. Resources",
        f"```json\n{json.dumps(resource, indent=2)}\n```",
        "",
        "## 15. Baseline comparison (descriptive only)",
        f"```json\n{json.dumps(baseline, indent=2)}\n```",
        "",
        "## 16. engineering_valid",
        f"```json\n{json.dumps(eng, indent=2)}\n```",
        "",
        "## 17. Manual scoring required?",
        f"- yes if pending: `{[r['task_id'] for r in kwargs['manual_rows']]}`",
        f"- lower/upper answer accuracy bounds above; do not seal until human_grades filled.",
        "",
        "## 18. Constraints confirmation",
        "- no API",
        "- no rerun",
        "- no raw mutation",
        "- no method change",
        "- no push",
        "",
        "## 19. Stop and wait",
        "Stop and wait for human grades before sealing final answer accuracy.",
        "",
    ]
    return "\n".join(lines)
