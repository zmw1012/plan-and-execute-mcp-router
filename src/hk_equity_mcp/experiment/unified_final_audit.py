"""Offline final audit for HK Unified Operation-Visible exploratory-155 baseline.

Writes ONLY under <run_dir>/final_audit_v2/.
Never modifies predictions.jsonl / run_metadata.json / raw side artifacts.
Never calls LLM / Embedding / MCP / paid APIs.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from hk_equity_mcp.experiment.unified_aggregates import build_grouped_aggregates
from hk_equity_mcp.experiment.unified_error_taxonomy import (
    TAXONOMY_VERSION,
    classify_task_error,
    summarize_taxonomy,
)
from hk_equity_mcp.experiment.unified_evaluator import (
    EVALUATOR_VERSION,
    evaluator_identity,
    extract_invoke_records,
    extract_predicted_operations,
    extract_retrieved_candidates,
    reevaluate_prediction_row,
    summarize_predictions,
)
from hk_equity_mcp.experiment.unified_grading import (
    GRADING_VERSION,
    GRADER_VERSION,
    grade_task,
    grading_identity,
    summarize_grading,
)
from hk_equity_mcp.experiment.unified_runner import atomic_write_json, fsync_file

AUDIT_VERSION = "final_audit_v2"
PROVIDER_SEGMENT_NOTE = (
    "tasks 1–65: Aliyun token-plan host; tasks 66–155: api.zhizengzeng.com relay; "
    "requested model id is deepseek-v4.1-flash for both. Difficulty mix differs by "
    "segment, so accuracy gaps must NOT be interpreted as a provider effect."
)

RAW_FILES = (
    "predictions.jsonl",
    "run_metadata.json",
    "per_task_metrics.csv",
    "manual_grading.csv",
    "retrieval_analysis.json",
)

# Original v1.1 manual_required set (verified against frozen manual_grading.csv).
V11_MANUAL_REQUIRED_EXPECTED = (
    "mkt_dev_003",
    "xd_ht_003",
    "mkt_ht_004",
    "mkt_dev_007",
    "mkt_ht_006",
    "mkt_ht_008",
    "fin_cf_022",
    "mkt_ht_016",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def _stable_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _content_fingerprint(payload: Any) -> str:
    return hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()


def _answer_text(final_answer: Any) -> str:
    if final_answer is None:
        return ""
    if isinstance(final_answer, dict):
        return str(final_answer.get("final_answer") or "")
    return str(final_answer)


def _provider_segment(task_index_1based: int) -> str:
    return "aliyun_token_plan" if task_index_1based <= 65 else "zhizengzeng_relay"


def _load_v11_grades(manual_path: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    if not manual_path.exists():
        return out
    with manual_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            tid = row.get("task_id")
            if not tid:
                continue
            ans = row.get("answer_correct")
            if ans == "True":
                ans_b: Optional[bool] = True
            elif ans == "False":
                ans_b = False
            elif ans in ("", None):
                ans_b = None
            else:
                ans_b = str(ans).lower() == "true"
            out[str(tid)] = {
                "auto_grade_status": row.get("auto_grade_status"),
                "answer_correct": ans_b,
            }
    return out


def _retrieval_hit_at_k(
    gold_ops: Sequence[str], candidates: Sequence[Dict[str, Any]], k: int
) -> bool:
    gold = {str(x) for x in gold_ops if x}
    if not gold:
        return True
    seen = []
    for c in candidates:
        oid = c.get("operation_id") if isinstance(c, dict) else None
        if oid and str(oid) not in seen:
            seen.append(str(oid))
        if len(seen) >= k:
            break
    return bool(gold & set(seen[:k]))


def compute_retrieval_stage_summary(
    enriched: Sequence[Dict[str, Any]],
    retrieval_per_task: Sequence[Dict[str, Any]],
    *,
    top_k: int,
) -> Dict[str, Any]:
    by_id = {str(r.get("task_id")): r for r in retrieval_per_task}
    ks = sorted({1, 5, 10, 20, int(top_k or 20)})
    hits = {k: 0 for k in ks}
    applicable = 0
    selection_given_hit = {"n": 0, "exact": 0, "subset": 0}
    for row in enriched:
        tid = str(row.get("task_id") or row.get("id"))
        gold = [str(x) for x in (row.get("gold_operations") or [])]
        if not gold:
            continue
        applicable += 1
        cands = (by_id.get(tid) or {}).get("retrieved_candidates") or row.get(
            "retrieved_candidates"
        ) or []
        for k in ks:
            if _retrieval_hit_at_k(gold, cands, k):
                hits[k] += 1
        if _retrieval_hit_at_k(gold, cands, int(top_k or 20)):
            selection_given_hit["n"] += 1
            pred = {str(x) for x in (row.get("predicted_operations") or [])}
            gset = set(gold)
            if pred == gset:
                selection_given_hit["exact"] += 1
            if gset.issubset(pred):
                selection_given_hit["subset"] += 1
    n = applicable or 1
    return {
        "applicable_tasks_with_gold_ops": applicable,
        "top_k_config": top_k,
        "recall_at_k": {f"R@{k}": hits[k] / n for k in ks},
        "hit_counts_at_k": {f"@{k}": hits[k] for k in ks},
        "agent_selection_given_retrieval_hit": {
            "n": selection_given_hit["n"],
            "exact_tool_set_rate": (
                selection_given_hit["exact"] / selection_given_hit["n"]
                if selection_given_hit["n"]
                else None
            ),
            "gold_subset_rate": (
                selection_given_hit["subset"] / selection_given_hit["n"]
                if selection_given_hit["n"]
                else None
            ),
        },
    }


def _suggest_manual_row(row: Dict[str, Any], grade: Dict[str, Any]) -> Dict[str, Any]:
    """Cursor-suggested fields only; human_confirmed_* stay blank."""
    answer_correct = grade.get("answer_correct")
    evidence = row.get("evidence") or []
    unsupported = False
    if answer_correct is False and str(row.get("predicted_status") or "").lower() == "answered":
        unsupported = True
    if str(row.get("answerability_class") or "") in {
        "intentionally_infeasible",
        "should_refuse_or_clarify",
        "capability_gap",
    } and str(row.get("predicted_status") or "").lower() == "answered":
        unsupported = True

    evidence_correct: Optional[bool]
    if answer_correct is True and evidence:
        evidence_correct = True
    elif answer_correct is False:
        evidence_correct = False
    else:
        evidence_correct = None

    overall: Optional[bool]
    if answer_correct is True and grade.get("tool_selection_correct") is True:
        overall = True
    elif answer_correct is False:
        overall = False
    else:
        overall = None

    reasons = []
    if grade.get("reason"):
        reasons.append(str(grade.get("reason")))
    if row.get("primary_error_category"):
        reasons.append(f"primary={row.get('primary_error_category')}")
    return {
        "suggested_answer_correct": (
            "" if answer_correct is None else str(bool(answer_correct))
        ),
        "suggested_evidence_correct": (
            "" if evidence_correct is None else str(bool(evidence_correct))
        ),
        "suggested_unsupported_claim": str(bool(unsupported)),
        "suggested_overall_task_success": (
            "" if overall is None else str(bool(overall))
        ),
        "suggestion_reason": "; ".join(reasons),
    }


def build_manual_review_queue(
    enriched: Sequence[Dict[str, Any]],
    grades_by_id: Dict[str, Dict[str, Any]],
    *,
    v11_manual_ids: Sequence[str],
) -> List[Dict[str, Any]]:
    """Queue = current manual_required ∪ original v1.1 manual set (for evidence review)."""
    wanted = set(v11_manual_ids)
    for row in enriched:
        tid = str(row.get("task_id") or "")
        g = grades_by_id.get(tid) or {}
        if g.get("auto_grade_status") == "manual_required":
            wanted.add(tid)
    by_id = {str(r.get("task_id")): r for r in enriched}
    out: List[Dict[str, Any]] = []
    for tid in sorted(wanted, key=lambda t: (0 if t in v11_manual_ids else 1, t)):
        row = by_id.get(tid)
        if not row:
            continue
        grade = grades_by_id.get(tid) or {}
        suggested = _suggest_manual_row(row, grade)
        out.append(
            {
                "task_id": tid,
                "domain": row.get("domain", ""),
                "project_difficulty": row.get("project_difficulty", ""),
                "question": row.get("question", ""),
                "reference_answer": row.get("reference_answer", ""),
                "predicted_answer": _answer_text(row.get("final_answer")),
                "gold_operations": json.dumps(
                    row.get("gold_operations") or [], ensure_ascii=False
                ),
                "predicted_operations": json.dumps(
                    row.get("predicted_operations") or [], ensure_ascii=False
                ),
                **suggested,
                "human_confirmed_answer_correct": "",
                "human_confirmed_evidence_correct": "",
                "human_confirmed_overall_task_success": "",
                "human_notes": "",
                "auto_grade_status_v1_2": grade.get("auto_grade_status", ""),
                "answer_correct_v1_2": grade.get("answer_correct", ""),
                "v11_manual_required": str(tid in set(v11_manual_ids)),
            }
        )
    return out


def _design_requirements_from_taxonomy(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Only emit Proposed Method requirements supported by taxonomy mass."""
    rates = summary.get("primary_rates") or {}
    reqs: List[Dict[str, Any]] = []
    mapping = [
        (
            "retrieval",
            0.05,
            "Domain-aware routing",
            "Retrieval misses concentrate gold ops outside Top-K; domain-conditioned retrieval/routing is supported.",
        ),
        (
            "agent_selection",
            0.05,
            "Explicit evidence-chain planning",
            "Gold ops are often retrieved but not selected as a complete set; planning over evidence chains is supported.",
        ),
        (
            "description_argument_grounding",
            0.03,
            "Operation argument grounding checks",
            "Selected ops still fail on dates/tickers/adjust/period arguments.",
        ),
        (
            "dependency_order",
            0.03,
            "Explicit evidence-chain planning",
            "Multi-step dependency/order failures appear after correct op identity.",
        ),
        (
            "answer_synthesis_calculation",
            0.03,
            "Calculation/synthesis verification",
            "Tool results succeed but final numeric/unit synthesis fails.",
        ),
        (
            "feasibility_refusal",
            0.03,
            "Feasibility and temporal-policy gate",
            "Refusal tasks are answered with substitutes or live/cutoff violations.",
        ),
        (
            "resource_limit",
            0.05,
            "Structured evidence compression",
            "max_llm_turns / reformulations / invocations dominate hard failures; compression/dedup can reduce budget burn.",
        ),
        (
            "resource_limit",
            0.05,
            "Operation deduplication",
            "Repeated find/describe/invoke under budget caps is a supported weakness.",
        ),
        (
            "feasibility_refusal",
            0.03,
            "Refusal validation",
            "Capability-gap / blocked-provider items need post-answer refusal validation.",
        ),
    ]
    seen = set()
    for key, thresh, title, why in mapping:
        if float(rates.get(key) or 0.0) >= thresh and title not in seen:
            seen.add(title)
            reqs.append(
                {
                    "requirement": title,
                    "supported_by_primary_category": key,
                    "primary_rate": rates.get(key),
                    "rationale": why,
                }
            )
    return reqs


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    fsync_file(path)


def _representative_cases(
    enriched: Sequence[Dict[str, Any]], *, success: bool, limit: int = 8
) -> List[Dict[str, Any]]:
    out = []
    for row in enriched:
        ok = row.get("answer_correct") is True
        if ok != success:
            continue
        out.append(
            {
                "task_id": row.get("task_id"),
                "domain": row.get("domain"),
                "project_difficulty": row.get("project_difficulty"),
                "primary_error_category": row.get("primary_error_category"),
                "answer_correct": row.get("answer_correct"),
                "exact_tool_set": row.get("exact_tool_set"),
                "error": row.get("error") or "",
                "predicted_status": row.get("predicted_status"),
                "question": (row.get("question") or "")[:180],
            }
        )
        if len(out) >= limit:
            break
    return out


def render_baseline_report(ctx: Dict[str, Any]) -> str:
    g = ctx["grading_summary_v2"]
    s = ctx["prediction_summary"]
    tax = ctx["error_taxonomy_summary"]
    ret = ctx["retrieval_stage_summary"]
    grouped = ctx["grouped_aggregates_v2"]
    lines: List[str] = []
    lines.append("# HK Unified Operation-Visible Baseline — Final Audit v2")
    lines.append("")
    lines.append("## Experimental identity")
    ident = ctx["evaluation_metadata_v2"]
    lines.append(f"- Baseline git commit: `{ident['baseline_git_commit']}`")
    lines.append(f"- Audit version: `{ident['audit_version']}`")
    lines.append(f"- Evaluator: `{ident['evaluator_version']}` / grader `{ident['grader_version']}`")
    lines.append(f"- Model: `{ident['generation_model']}`")
    lines.append(f"- Prompt hash: `{ident['system_prompt_hash']}`")
    lines.append(f"- Catalog fingerprint: `{ident['catalog_fingerprint']}`")
    lines.append(f"- Provider segment: {PROVIDER_SEGMENT_NOTE}")
    lines.append("")
    lines.append("## Dataset composition")
    lines.append(f"- N tasks: **{ident['question_count']}** (exploratory_all development lock)")
    lines.append(f"- Domain counts: `{json.dumps(ident.get('domain_counts') or {}, sort_keys=True)}`")
    lines.append(
        f"- Difficulty counts: `{json.dumps(ctx.get('difficulty_counts') or {}, sort_keys=True)}`"
    )
    lines.append(
        f"- Answerability counts: `{json.dumps(ctx.get('answerability_counts') or {}, sort_keys=True)}`"
    )
    lines.append("")
    lines.append("## Overall metrics")
    lines.append(
        f"- Auto-gradable: {g.get('auto_gradable_count')} / {g.get('n')}; "
        f"auto-graded accuracy: **{g.get('auto_graded_accuracy')}** "
        f"({g.get('auto_graded_correct')} correct)"
    )
    lines.append(f"- Manual required (answer): **{g.get('manual_required_count')}**")
    lines.append(f"- Exact tool-set rate: **{s.get('exact_tool_set_rate')}**")
    lines.append(
        f"- Mean operation P/R/F1: "
        f"{s.get('mean_operation_precision')} / {s.get('mean_operation_recall')} / {s.get('mean_operation_f1')}"
    )
    lines.append(
        f"- Tool execution success rate: **{s.get('all_tool_invocations_successful_rate')}**"
    )
    lines.append(f"- Recovered-after-tool-error rate: {s.get('recovered_after_tool_error_rate')}")
    lines.append(f"- Mean dependency accuracy: {grouped['overall'].get('mean_dependency_accuracy')}")
    lines.append(f"- Refusal accuracy (auto subset): {grouped['overall'].get('refusal_accuracy')}")
    lines.append(
        f"- Mean latency_ms / llm_calls / total_tokens / mcp_exec: "
        f"{s.get('mean_latency_ms')} / {s.get('mean_llm_calls')} / "
        f"{s.get('mean_total_tokens')} / {s.get('mean_real_mcp_execution_count')}"
    )
    lines.append("")
    lines.append("## Metrics by domain")
    for dom, row in sorted((grouped.get("domain") or {}).items()):
        lines.append(
            f"- **{dom}**: n={row['task_count']}, auto_acc={row.get('auto_graded_accuracy')}, "
            f"exact_tool={row.get('exact_tool_set_rate')}, "
            f"tool_ok={row.get('all_tool_invocations_successful_rate')}"
        )
    lines.append("")
    lines.append("## Metrics by D1–D5")
    for diff, row in sorted((grouped.get("project_difficulty") or {}).items()):
        lines.append(
            f"- **{diff}**: n={row['task_count']}, auto_acc={row.get('auto_graded_accuracy')}, "
            f"exact_tool={row.get('exact_tool_set_rate')}"
        )
    lines.append("")
    lines.append("## Metrics by answerability class")
    for ac, row in sorted((grouped.get("answerability_class") or {}).items()):
        lines.append(
            f"- **{ac}**: n={row['task_count']}, auto_acc={row.get('auto_graded_accuracy')}, "
            f"refusal_acc={row.get('refusal_accuracy')}"
        )
    lines.append("")
    lines.append("## Retrieval Recall@K")
    lines.append(f"- Applicable tasks with gold ops: {ret.get('applicable_tasks_with_gold_ops')}")
    lines.append(f"- Recall@K: `{json.dumps(ret.get('recall_at_k') or {}, sort_keys=True)}`")
    lines.append(
        f"- Agent selection given retrieval hit: "
        f"`{json.dumps(ret.get('agent_selection_given_retrieval_hit') or {}, sort_keys=True)}`"
    )
    lines.append("")
    lines.append("## Agent tool-set exact match / precision / recall / F1")
    lines.append(
        f"- exact_tool_set_rate={s.get('exact_tool_set_rate')}; "
        f"P={s.get('mean_operation_precision')}; R={s.get('mean_operation_recall')}; "
        f"F1={s.get('mean_operation_f1')}"
    )
    lines.append("")
    lines.append("## Dependency accuracy")
    lines.append(f"- mean graph_dependency_accuracy={grouped['overall'].get('mean_dependency_accuracy')}")
    lines.append("")
    lines.append("## Tool execution success")
    lines.append(
        f"- all_tool_invocations_successful_rate={s.get('all_tool_invocations_successful_rate')}; "
        f"recovered_after_tool_error_rate={s.get('recovered_after_tool_error_rate')}"
    )
    lines.append("")
    lines.append("## Final answer accuracy")
    lines.append(
        f"- v1.2 auto-graded accuracy={g.get('auto_graded_accuracy')} "
        f"(v1.1 was {ctx['v11_vs_v12'].get('v1_1_auto_graded_accuracy')})"
    )
    lines.append(f"- Overall accuracy still pending manual confirmation for evidence/overall fields.")
    lines.append("")
    lines.append("## Refusal accuracy")
    lines.append(f"- {grouped['overall'].get('refusal_accuracy')}")
    lines.append("")
    lines.append("## Token / latency / tool-call efficiency")
    lines.append(
        f"- mean_total_tokens={s.get('mean_total_tokens')}, mean_latency_ms={s.get('mean_latency_ms')}, "
        f"mean_llm_calls={s.get('mean_llm_calls')}, mean_real_mcp_execution_count={s.get('mean_real_mcp_execution_count')}"
    )
    lines.append("")
    lines.append("## Failure taxonomy")
    lines.append(f"- Taxonomy version: `{tax.get('taxonomy_version')}`")
    lines.append(f"- Primary counts: `{json.dumps(tax.get('primary_counts') or {}, sort_keys=True)}`")
    lines.append(f"- Primary rates: `{json.dumps(tax.get('primary_rates') or {}, sort_keys=True)}`")
    lines.append("")
    lines.append("## Representative success cases")
    for c in ctx.get("representative_success") or []:
        lines.append(
            f"- `{c['task_id']}` ({c['domain']}/{c['project_difficulty']}): exact_tool={c['exact_tool_set']}"
        )
    lines.append("")
    lines.append("## Representative failure cases")
    for c in ctx.get("representative_failure") or []:
        lines.append(
            f"- `{c['task_id']}` ({c['domain']}/{c['project_difficulty']}): "
            f"primary={c['primary_error_category']}, error={c.get('error') or 'n/a'}"
        )
    lines.append("")
    lines.append("## Operation-Visible strengths")
    lines.append(
        "- Visible find→describe→invoke loop makes retrieval and tool-set metrics auditable."
    )
    lines.append(
        "- Structured refusal channel exists and is measurable via refusal_correctness."
    )
    lines.append(
        f"- Non-trivial retrieval recall and tool execution success "
        f"(R@top_k and tool_ok in artifacts)."
    )
    lines.append("")
    lines.append("## Operation-Visible weaknesses")
    lines.append(
        "- Resource-limit failures (max_llm_turns / reformulations / invocations) truncate answerable tasks."
    )
    lines.append(
        "- Feasibility/refusal leakage: agents often answer with substitutes under capability gaps."
    )
    lines.append(
        "- Even with gold ops retrieved, agent_selection and answer_synthesis remain major residual errors."
    )
    lines.append("")
    lines.append("## Concrete design requirements for the Proposed Method")
    lines.append(
        "Requirements below are emitted only when taxonomy rates support them. "
        "No task-id hard rules."
    )
    for req in ctx.get("proposed_method_requirements") or []:
        lines.append(
            f"- **{req['requirement']}** (from `{req['supported_by_primary_category']}` "
            f"rate={req['primary_rate']}): {req['rationale']}"
        )
    lines.append("")
    lines.append("## Limitations")
    lines.append("- Exploratory dataset / development lock — not an unseen final test set.")
    lines.append("- Provider transition after task 65; do not attribute segment gaps to provider effect.")
    lines.append("- Automatic grading limitations remain (evidence/overall still human).")
    lines.append("- Manual review remaining (see `manual_review_queue.csv`).")
    lines.append("- ETOM external evaluation not yet run.")
    lines.append("")
    lines.append("## v1.1 → v1.2 grading delta")
    delta = ctx["v11_vs_v12"]
    lines.append(f"- `{json.dumps(delta, ensure_ascii=False, sort_keys=True)}`")
    lines.append("")
    lines.append("## Provenance / offline guarantees")
    lines.append("- No LLM / Embedding / MCP / paid API calls during this audit.")
    lines.append("- Raw predictions.jsonl, gold, prompt, and catalog were not modified.")
    lines.append(f"- Raw SHA256: `{json.dumps(ident.get('raw_sha256') or {}, sort_keys=True)}`")
    lines.append("")
    return "\n".join(lines) + "\n"


def run_final_audit(
    run_dir: Path,
    *,
    baseline_git_commit: str,
    audit_outdir: Optional[Path] = None,
) -> Dict[str, Any]:
    run_dir = Path(run_dir)
    out_dir = Path(audit_outdir) if audit_outdir else run_dir / "final_audit_v2"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Protect / fingerprint raw inputs (read-only) ---
    raw_sha = {}
    for name in RAW_FILES:
        path = run_dir / name
        if not path.exists():
            raise FileNotFoundError(f"Missing required raw artifact: {path}")
        raw_sha[name] = sha256_file(path)

    predictions = load_jsonl(run_dir / "predictions.jsonl")
    if len(predictions) != 155:
        raise ValueError(f"Expected 155 predictions, found {len(predictions)}")
    ids = [str(p.get("task_id") or p.get("id")) for p in predictions]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate task_id in predictions.jsonl")

    run_meta = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    if run_meta.get("run_status") != "completed":
        raise ValueError(f"run_status != completed: {run_meta.get('run_status')}")
    if int(run_meta.get("question_count") or 0) != 155:
        raise ValueError("question_count != 155")
    if run_meta.get("generation_model") != "deepseek-v4.1-flash":
        raise ValueError(f"Unexpected model: {run_meta.get('generation_model')}")
    if not run_meta.get("system_prompt_hash"):
        raise ValueError("Missing system_prompt_hash")
    if not run_meta.get("catalog_fingerprint"):
        raise ValueError("Missing catalog_fingerprint")

    v11_grades = _load_v11_grades(run_dir / "manual_grading.csv")
    v11_manual = [
        tid
        for tid, g in v11_grades.items()
        if g.get("auto_grade_status") == "manual_required"
    ]
    if sorted(v11_manual) != sorted(V11_MANUAL_REQUIRED_EXPECTED):
        # Soft check: proceed with actual file list, record mismatch.
        v11_manual_note = {
            "expected": list(V11_MANUAL_REQUIRED_EXPECTED),
            "actual": sorted(v11_manual),
            "match": False,
        }
    else:
        v11_manual_note = {
            "expected": list(V11_MANUAL_REQUIRED_EXPECTED),
            "actual": sorted(v11_manual),
            "match": True,
        }

    retrieval_blob = json.loads(
        (run_dir / "retrieval_analysis.json").read_text(encoding="utf-8")
    )
    retrieval_per_task = retrieval_blob.get("per_task") or []
    retrieval_by_id = {
        str(r.get("task_id")): r for r in retrieval_per_task if r.get("task_id")
    }

    # --- Offline regrade (v1.2) ---
    enriched: List[Dict[str, Any]] = []
    grades: List[Dict[str, Any]] = []
    grades_by_id: Dict[str, Dict[str, Any]] = {}
    audit_rows: List[Dict[str, Any]] = []

    for idx, row in enumerate(predictions, start=1):
        metrics = reevaluate_prediction_row(row)
        # Ensure invoke/retrieval fields present for taxonomy.
        trace = row.get("trace") or []
        if not metrics.get("predicted_operations"):
            metrics["predicted_operations"] = extract_predicted_operations(trace)
        if not metrics.get("retrieved_candidates"):
            metrics["retrieved_candidates"] = extract_retrieved_candidates(trace)
        if not metrics.get("invoke_calls"):
            metrics["invoke_calls"] = extract_invoke_records(trace)

        merged = dict(row)
        merged.update(metrics)
        grade = grade_task(merged, merged)
        grades.append(grade)
        tid = str(merged.get("task_id") or merged.get("id"))
        grades_by_id[tid] = grade
        merged["auto_grade_status"] = grade.get("auto_grade_status")
        merged["answer_correct"] = grade.get("answer_correct")
        merged["auto_grade_reason"] = grade.get("reason")
        merged["evidence_correct"] = grade.get("evidence_correct")
        merged["tool_selection_correct"] = grade.get("tool_selection_correct")
        merged["overall_task_success"] = grade.get("overall_task_success")
        merged["provider_segment"] = _provider_segment(idx)
        merged["task_order_index"] = idx

        v11 = v11_grades.get(tid) or {}
        tax = classify_task_error(
            merged,
            retrieval_row=retrieval_by_id.get(tid),
            v11_answer_correct=v11.get("answer_correct"),
            v11_auto_status=v11.get("auto_grade_status"),
        )
        merged.update(tax)
        enriched.append(merged)

        audit_rows.append(
            {
                "task_id": tid,
                "domain": merged.get("domain"),
                "domain_combination": merged.get("domain_combination"),
                "project_difficulty": merged.get("project_difficulty"),
                "exact_etom_level": merged.get("exact_etom_level"),
                "answerability_class": merged.get("answerability_class"),
                "source_split": merged.get("source_split"),
                "provider_segment": merged.get("provider_segment"),
                "task_order_index": idx,
                "gold_status": merged.get("gold_status"),
                "predicted_status": merged.get("predicted_status"),
                "status": merged.get("status"),
                "error": merged.get("error") or "",
                "error_type": merged.get("error_type") or "",
                "auto_grade_status": grade.get("auto_grade_status"),
                "answer_correct": grade.get("answer_correct"),
                "evidence_correct": grade.get("evidence_correct"),
                "tool_selection_correct": grade.get("tool_selection_correct"),
                "overall_task_success": grade.get("overall_task_success"),
                "auto_grade_reason": grade.get("reason"),
                "v11_auto_grade_status": v11.get("auto_grade_status"),
                "v11_answer_correct": v11.get("answer_correct"),
                "operation_precision": merged.get("operation_precision"),
                "operation_recall": merged.get("operation_recall"),
                "operation_f1": merged.get("operation_f1"),
                "exact_tool_set": merged.get("exact_tool_set"),
                "graph_dependency_accuracy": merged.get("graph_dependency_accuracy"),
                "tool_execution_success": merged.get("tool_execution_success"),
                "all_tool_invocations_successful": merged.get(
                    "all_tool_invocations_successful"
                ),
                "recovered_after_tool_error": merged.get("recovered_after_tool_error"),
                "refusal_correctness": merged.get("refusal_correctness"),
                "wrong_tool_rate": merged.get("wrong_tool_rate"),
                "llm_calls": merged.get("llm_calls"),
                "total_tokens": (merged.get("tokens") or {}).get("total_tokens"),
                "latency_ms": merged.get("latency_ms"),
                "real_mcp_execution_count": merged.get("real_mcp_execution_count"),
                "gold_operations": json.dumps(
                    merged.get("gold_operations") or [], ensure_ascii=False
                ),
                "predicted_operations": json.dumps(
                    merged.get("predicted_operations") or [], ensure_ascii=False
                ),
                "primary_error_category": tax.get("primary_error_category"),
                "secondary_error_categories": json.dumps(
                    tax.get("secondary_error_categories") or [], ensure_ascii=False
                ),
                "error_reason": tax.get("error_reason"),
            }
        )

    prediction_summary = summarize_predictions(enriched)
    grading_summary = summarize_grading(grades)
    grouped = build_grouped_aggregates(enriched)
    taxonomy_summary = summarize_taxonomy(enriched)
    retrieval_summary = compute_retrieval_stage_summary(
        enriched,
        retrieval_per_task,
        top_k=int(run_meta.get("retrieval_top_k") or retrieval_blob.get("top_k") or 20),
    )

    # v1.1 vs v1.2 comparison
    v11_auto = [g for g in v11_grades.values() if g.get("auto_grade_status") == "graded"]
    v11_correct = [g for g in v11_auto if g.get("answer_correct") is True]
    flips_false_to_true = []
    flips_true_to_false = []
    newly_gradable = []
    still_manual = []
    for row, grade in zip(enriched, grades):
        tid = str(row.get("task_id"))
        v11 = v11_grades.get(tid) or {}
        old, new = v11.get("answer_correct"), grade.get("answer_correct")
        if v11.get("auto_grade_status") == "manual_required" and grade.get(
            "auto_grade_status"
        ) == "graded":
            newly_gradable.append(
                {"task_id": tid, "answer_correct_v1_2": new, "reason": grade.get("reason")}
            )
        if grade.get("auto_grade_status") == "manual_required":
            still_manual.append(tid)
        if old is False and new is True:
            flips_false_to_true.append(
                {"task_id": tid, "reason_v1_2": grade.get("reason")}
            )
        if old is True and new is False:
            flips_true_to_false.append(
                {"task_id": tid, "reason_v1_2": grade.get("reason")}
            )

    v11_vs_v12 = {
        "v1_1_auto_gradable_count": len(v11_auto),
        "v1_1_auto_graded_correct": len(v11_correct),
        "v1_1_auto_graded_accuracy": (
            len(v11_correct) / len(v11_auto) if v11_auto else None
        ),
        "v1_1_manual_required_count": len(v11_manual),
        "v1_2_auto_gradable_count": grading_summary.get("auto_gradable_count"),
        "v1_2_auto_graded_correct": grading_summary.get("auto_graded_correct"),
        "v1_2_auto_graded_accuracy": grading_summary.get("auto_graded_accuracy"),
        "v1_2_manual_required_count": grading_summary.get("manual_required_count"),
        "flips_false_to_true_count": len(flips_false_to_true),
        "flips_true_to_false_count": len(flips_true_to_false),
        "flips_false_to_true": flips_false_to_true,
        "flips_true_to_false": flips_true_to_false,
        "newly_auto_gradable_from_manual": newly_gradable,
        "still_manual_required": still_manual,
        "change_explanations": [
            {
                "bucket": "semantic_and_currency_units",
                "explanation": (
                    "v1.1 required literal unit tokens (e.g. 'percent', 'cny'+'million'); "
                    "v1.2 accepts % / RMB↔CNY aliases and semantic units such as decimal_return."
                ),
            },
            {
                "bucket": "unicode_minus_and_scale",
                "explanation": (
                    "v1.2 normalizes Unicode minus and matches percent↔decimal and "
                    "million↔raw/billion scale forms with scaled tolerances."
                ),
            },
            {
                "bucket": "labeled_multi_value",
                "explanation": (
                    "OHLC / unadjusted-qfq-hfq / MA+return dict gold_values are now "
                    "auto-gradable via label binding instead of manual_required."
                ),
            },
            {
                "bucket": "refusal_substitute",
                "explanation": (
                    "Refusal items that answer with numeric substitutes remain incorrect; "
                    "status-only refusals without fabricated targets remain correct."
                ),
            },
        ],
    }

    manual_queue = build_manual_review_queue(
        enriched, grades_by_id, v11_manual_ids=v11_manual_note["actual"]
    )
    proposed_reqs = _design_requirements_from_taxonomy(taxonomy_summary)

    difficulty_counts: Dict[str, int] = defaultdict(int)
    answerability_counts: Dict[str, int] = defaultdict(int)
    for r in enriched:
        difficulty_counts[str(r.get("project_difficulty") or "null")] += 1
        answerability_counts[str(r.get("answerability_class") or "null")] += 1

    # Deterministic timestamp: prefer run completed_at; fall back to hash-derived marker.
    completed_at = run_meta.get("completed_at") or run_meta.get("timestamp")
    audit_timestamp = (
        str(completed_at)
        if completed_at
        else f"deterministic:{_content_fingerprint(raw_sha)[:16]}"
    )

    eval_module = Path(__file__).resolve().parents[0] / "unified_evaluator.py"
    grade_module = Path(__file__).resolve().parents[0] / "unified_grading.py"
    tax_module = Path(__file__).resolve().parents[0] / "unified_error_taxonomy.py"
    audit_module = Path(__file__).resolve()

    evaluation_metadata = {
        "audit_version": AUDIT_VERSION,
        "audit_timestamp": audit_timestamp,
        "baseline_git_commit": baseline_git_commit,
        "baseline_branch_expected": "feature/unified-hk-operation-visible-runner-v1",
        "evaluator_version": EVALUATOR_VERSION,
        "grader_version": GRADER_VERSION,
        "grading_version": GRADING_VERSION,
        "taxonomy_version": TAXONOMY_VERSION,
        "evaluator_source_sha256": hashlib.sha256(eval_module.read_bytes()).hexdigest(),
        "grader_source_sha256": hashlib.sha256(grade_module.read_bytes()).hexdigest(),
        "taxonomy_source_sha256": hashlib.sha256(tax_module.read_bytes()).hexdigest(),
        "audit_source_sha256": hashlib.sha256(audit_module.read_bytes()).hexdigest(),
        "raw_sha256": raw_sha,
        "raw_inputs_immutable": True,
        "question_count": 155,
        "run_status": run_meta.get("run_status"),
        "generation_model": run_meta.get("generation_model"),
        "system_prompt_hash": run_meta.get("system_prompt_hash"),
        "catalog_fingerprint": run_meta.get("catalog_fingerprint"),
        "domain_counts": run_meta.get("domain_counts"),
        "provider_segments": {
            "tasks_1_65": "Aliyun token-plan host (token-plan.cn-beijing.maas.aliyuncs.com)",
            "tasks_66_155": "api.zhizengzeng.com relay",
            "requested_model_id": "deepseek-v4.1-flash",
            "interpretation_constraint": PROVIDER_SEGMENT_NOTE,
        },
        "v11_manual_required_check": v11_manual_note,
        "offline_guarantees": {
            "llm_calls": 0,
            "embedding_calls": 0,
            "mcp_calls": 0,
            "paid_api_calls": 0,
            "predictions_modified": False,
            "gold_modified": False,
            "prompt_modified": False,
            "catalog_modified": False,
        },
        "evaluator": evaluator_identity(),
        "grading": grading_identity(),
        "output_dir": str(out_dir),
    }

    # Report-ready tables
    report_rows = []
    for slice_name, payload in (
        [("overall", {"overall": grouped["overall"]})]
        + [("domain:" + k, {k: v}) for k, v in sorted((grouped.get("domain") or {}).items())]
        + [
            ("difficulty:" + k, {k: v})
            for k, v in sorted((grouped.get("project_difficulty") or {}).items())
        ]
        + [
            ("answerability:" + k, {k: v})
            for k, v in sorted((grouped.get("answerability_class") or {}).items())
        ]
    ):
        for key, row in payload.items():
            report_rows.append(
                {
                    "slice_type": slice_name.split(":")[0],
                    "slice_value": key if slice_name != "overall" else "overall",
                    "task_count": row.get("task_count"),
                    "auto_gradable_count": row.get("auto_gradable_count"),
                    "auto_graded_correct": row.get("auto_graded_correct"),
                    "auto_graded_accuracy": row.get("auto_graded_accuracy"),
                    "manual_required_count": row.get("manual_required_count"),
                    "exact_tool_set_rate": row.get("exact_tool_set_rate"),
                    "mean_operation_precision": row.get("mean_operation_precision"),
                    "mean_operation_recall": row.get("mean_operation_recall"),
                    "mean_operation_f1": row.get("mean_operation_f1"),
                    "mean_dependency_accuracy": row.get("mean_dependency_accuracy"),
                    "all_tool_invocations_successful_rate": row.get(
                        "all_tool_invocations_successful_rate"
                    ),
                    "refusal_accuracy": row.get("refusal_accuracy"),
                    "mean_llm_calls": row.get("mean_llm_calls"),
                    "mean_total_tokens": row.get("mean_total_tokens"),
                    "mean_latency_ms": row.get("mean_latency_ms"),
                }
            )

    taxonomy_rows = [
        {
            "task_id": r.get("task_id"),
            "domain": r.get("domain"),
            "project_difficulty": r.get("project_difficulty"),
            "answerability_class": r.get("answerability_class"),
            "provider_segment": r.get("provider_segment"),
            "primary_error_category": r.get("primary_error_category"),
            "secondary_error_categories": json.dumps(
                r.get("secondary_error_categories") or [], ensure_ascii=False
            ),
            "error_reason": r.get("error_reason"),
            "answer_correct": r.get("answer_correct"),
            "exact_tool_set": r.get("exact_tool_set"),
            "status": r.get("status"),
            "error": r.get("error") or "",
        }
        for r in enriched
    ]

    ctx = {
        "evaluation_metadata_v2": evaluation_metadata,
        "grading_summary_v2": grading_summary,
        "prediction_summary": prediction_summary,
        "grouped_aggregates_v2": grouped,
        "error_taxonomy_summary": taxonomy_summary,
        "retrieval_stage_summary": retrieval_summary,
        "v11_vs_v12": v11_vs_v12,
        "difficulty_counts": dict(sorted(difficulty_counts.items())),
        "answerability_counts": dict(sorted(answerability_counts.items())),
        "representative_success": _representative_cases(enriched, success=True),
        "representative_failure": _representative_cases(enriched, success=False),
        "proposed_method_requirements": proposed_reqs,
    }
    report_md = render_baseline_report(ctx)

    # --- Write artifacts (audit dir only) ---
    atomic_write_json(out_dir / "evaluation_metadata_v2.json", evaluation_metadata)
    atomic_write_json(out_dir / "grading_summary_v2.json", grading_summary)
    atomic_write_json(out_dir / "grouped_aggregates_v2.json", grouped)
    atomic_write_json(out_dir / "error_taxonomy_summary_v1.json", taxonomy_summary)
    atomic_write_json(out_dir / "retrieval_stage_summary_v1.json", retrieval_summary)
    atomic_write_json(
        out_dir / "v11_vs_v12_grading_delta.json",
        v11_vs_v12,
    )
    atomic_write_json(
        out_dir / "proposed_method_requirements_v1.json",
        {
            "note": "Derived only from taxonomy rates; no task-id rules.",
            "requirements": proposed_reqs,
        },
    )

    _write_csv(
        out_dir / "manual_review_queue.csv",
        manual_queue,
        fieldnames=[
            "task_id",
            "domain",
            "project_difficulty",
            "question",
            "reference_answer",
            "predicted_answer",
            "gold_operations",
            "predicted_operations",
            "suggested_answer_correct",
            "suggested_evidence_correct",
            "suggested_unsupported_claim",
            "suggested_overall_task_success",
            "suggestion_reason",
            "human_confirmed_answer_correct",
            "human_confirmed_evidence_correct",
            "human_confirmed_overall_task_success",
            "human_notes",
            "auto_grade_status_v1_2",
            "answer_correct_v1_2",
            "v11_manual_required",
        ],
    )
    _write_csv(
        out_dir / "per_task_audit_v2.csv",
        audit_rows,
        fieldnames=list(audit_rows[0].keys()) if audit_rows else ["task_id"],
    )
    _write_csv(
        out_dir / "error_taxonomy_v1.csv",
        taxonomy_rows,
        fieldnames=[
            "task_id",
            "domain",
            "project_difficulty",
            "answerability_class",
            "provider_segment",
            "primary_error_category",
            "secondary_error_categories",
            "error_reason",
            "answer_correct",
            "exact_tool_set",
            "status",
            "error",
        ],
    )
    _write_csv(
        out_dir / "report_ready_baseline_tables_v1.csv",
        report_rows,
        fieldnames=[
            "slice_type",
            "slice_value",
            "task_count",
            "auto_gradable_count",
            "auto_graded_correct",
            "auto_graded_accuracy",
            "manual_required_count",
            "exact_tool_set_rate",
            "mean_operation_precision",
            "mean_operation_recall",
            "mean_operation_f1",
            "mean_dependency_accuracy",
            "all_tool_invocations_successful_rate",
            "refusal_accuracy",
            "mean_llm_calls",
            "mean_total_tokens",
            "mean_latency_ms",
        ],
    )
    report_path = out_dir / "baseline_error_analysis_v1.md"
    report_path.write_text(report_md, encoding="utf-8")
    fsync_file(report_path)

    # Post-write raw SHA verification (must be unchanged).
    raw_sha_after = {name: sha256_file(run_dir / name) for name in RAW_FILES}
    if raw_sha_after != raw_sha:
        raise RuntimeError("Raw artifact SHA256 changed during audit — aborting")

    result = {
        "out_dir": str(out_dir),
        "n_tasks": len(enriched),
        "grading_summary_v2": grading_summary,
        "v11_vs_v12": {
            k: v11_vs_v12[k]
            for k in v11_vs_v12
            if k
            not in {
                "flips_false_to_true",
                "flips_true_to_false",
                "newly_auto_gradable_from_manual",
            }
        },
        "manual_review_queue_n": len(manual_queue),
        "raw_sha256": raw_sha,
        "raw_sha256_unchanged": True,
        "content_fingerprint": _content_fingerprint(
            {
                "grading_summary_v2": grading_summary,
                "taxonomy_primary_counts": taxonomy_summary.get("primary_counts"),
                "retrieval_recall": retrieval_summary.get("recall_at_k"),
            }
        ),
        "offline_guarantees": evaluation_metadata["offline_guarantees"],
    }
    atomic_write_json(out_dir / "audit_run_result.json", result)
    return result
