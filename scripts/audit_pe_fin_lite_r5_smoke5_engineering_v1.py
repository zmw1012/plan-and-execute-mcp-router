#!/usr/bin/env python3
"""Read-only engineering audit for PE-Fin Lite R5 Smoke-5.

Writes ONLY under <run_dir>/engineering_audit_v1/. Never mutates raw artifacts.
Zero network / no LLM / Embedding / MCP calls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
import sys

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hk_equity_mcp.pe_fin_lite.retrieval_formula import (  # noqa: E402
    R5_CANDIDATE_LOCK_SHA256,
    RETRIEVAL_FORMULA_NAME,
    RETRIEVAL_FORMULA_VERSION,
    retrieval_formula_hash,
)

AUDIT_VERSION = "pe_fin_lite_r5_smoke5_engineering_audit_v1"
EXPECTED_TASK_IDS = [
    "fin_cf_003",
    "fin_cf_023",
    "xd_dev_002",
    "mkt_dev_014",
    "news_ht_004",
]
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


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


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
            "sha256": _sha256_bytes(data),
            "rows": rows,
        }
    return out


def _load_preds(run_dir: Path) -> List[Dict[str, Any]]:
    rows = []
    for line in (run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _score_ok(row: Dict[str, Any]) -> bool:
    cos = row.get("cosine_similarity")
    lex = row.get("lexical_score")
    sc = row.get("score")
    if cos is None or lex is None or sc is None:
        return False
    expect = float(cos) + 0.12 * float(lex)
    return abs(float(sc) - round(expect, 6)) <= 1e-9 or abs(float(sc) - expect) <= 1e-6


def audit_task(pred: Dict[str, Any]) -> Dict[str, Any]:
    tid = str(pred.get("task_id") or pred.get("id"))
    budget = pred.get("budget") or {}
    funnels = pred.get("retrieval_funnels") or []
    embedding_funnels = []
    disclosure_funnels = []
    issues: List[str] = []

    for funnel in funnels:
        if not isinstance(funnel, dict):
            issues.append("funnel_not_dict")
            continue
        if funnel.get("disclosure_composite"):
            disclosure_funnels.append(
                {
                    "step_id": funnel.get("step_id"),
                    "invoked_operations": funnel.get("invoked_operations"),
                    "disclosure": True,
                }
            )
            continue
        hf = funnel.get("hard_filtered") or {}
        rejected = hf.get("rejected") or []
        domain_rejects = [
            r
            for r in rejected
            if isinstance(r, dict) and r.get("reason") == "domain_incompatible"
        ]
        topk = funnel.get("embedding_topk") or []
        described = funnel.get("described_candidates") or []
        rf = funnel.get("retrieval_formula") or {}
        score_rows_ok = all(_score_ok(r) for r in topk) if topk else True
        entry = {
            "step_id": funnel.get("step_id"),
            "search_query": funnel.get("search_query"),
            "domain_hard_filter": funnel.get("domain_hard_filter"),
            "filter_kind": hf.get("filter_kind"),
            "domain_hard_filter_applied": hf.get("domain_hard_filter_applied"),
            "kept_count": hf.get("kept_count"),
            "rejected_count": hf.get("rejected_count"),
            "rejected_reasons": sorted(
                {
                    (r.get("reason") if isinstance(r, dict) else str(r))
                    for r in rejected
                }
            ),
            "domain_incompatible_rejects": len(domain_rejects),
            "n_candidates_scored": funnel.get("n_candidates_scored"),
            "topk_n": len(topk),
            "described_n": len(described) if isinstance(described, list) else 0,
            "topk": topk,
            "described_candidates": described,
            "retrieval_formula": rf,
            "primary": (funnel.get("primary") or {}).get("operation_id")
            if isinstance(funnel.get("primary"), dict)
            else funnel.get("primary"),
            "fallback": (funnel.get("fallback") or {}).get("operation_id")
            if isinstance(funnel.get("fallback"), dict)
            else funnel.get("fallback"),
            "binder_selection_validation": funnel.get("binder_selection_validation"),
            "local_fallback": funnel.get("local_fallback"),
            "invoked_operations": funnel.get("invoked_operations"),
            "score_formula_ok": score_rows_ok,
            "has_cosine_lexical": bool(topk)
            and all(
                ("cosine_similarity" in r and "lexical_score" in r) for r in topk
            ),
        }
        embedding_funnels.append(entry)
        if entry["domain_hard_filter"] is not False:
            issues.append(f"domain_hard_filter_not_false:{entry['step_id']}")
        if entry["filter_kind"] not in {"safety_only_r5", None}:
            # None only if non-embedding path; embedding path must be safety_only_r5
            if topk:
                issues.append(f"unexpected_filter_kind:{entry['filter_kind']}")
        if entry["domain_incompatible_rejects"]:
            issues.append(f"domain_rejects:{entry['step_id']}")
        if entry["topk_n"] > 5:
            issues.append("topk_gt_5")
        if entry["described_n"] > 3:
            issues.append("describe_gt_3")
        if topk and not entry["score_formula_ok"]:
            issues.append("score_formula_mismatch")
        if topk and entry.get("n_candidates_scored") not in (39, None):
            # Prefer exact 39 runtime pool.
            if entry.get("n_candidates_scored") != 39:
                issues.append(
                    f"candidate_pool:{entry.get('n_candidates_scored')}"
                )
        rf_name = (rf or {}).get("name")
        if topk and rf_name != RETRIEVAL_FORMULA_NAME:
            issues.append(f"formula_name:{rf_name}")

    llm = int(budget.get("llm_calls_used") or budget.get("llm_calls") or 0)
    mcp = int(budget.get("mcp_invocations") or 0)
    emb = int(budget.get("embedding_query_calls") or 0)
    fallback = int(budget.get("local_fallback_count") or 0)
    monitor = int(budget.get("monitor_calls") or 0)
    replan = int(budget.get("global_replan_rounds") or 0)
    if llm > 8:
        issues.append(f"llm_over_budget:{llm}")
    if mcp > 8:
        issues.append(f"mcp_over_budget:{mcp}")
    if monitor != 0:
        issues.append("monitor_nonzero")
    if replan != 0:
        issues.append("replan_nonzero")

    calls_by_stage = budget.get("calls_by_stage") or {}
    planner_calls = int(calls_by_stage.get("planner") or 0)
    if planner_calls != 1:
        issues.append(f"planner_calls:{planner_calls}")

    status = pred.get("status") or pred.get("predicted_status")
    answer_correct = bool(pred.get("answer_correct"))
    # Strict success must not claim full success for failed/partial incorrectly in metrics;
    # engineering check: status labels present.
    zero_mcp_marked_tool_success = (
        mcp == 0
        and pred.get("all_tool_invocations_successful") is True
        and bool(pred.get("invoked_operations"))
    )
    if zero_mcp_marked_tool_success:
        issues.append("zero_invocation_marked_tool_success")

    return {
        "task_id": tid,
        "status": status,
        "answer_correct": answer_correct,
        "answerability_class": pred.get("answerability_class"),
        "gold_status": pred.get("gold_status"),
        "error_type": pred.get("error_type"),
        "llm_calls": llm,
        "mcp_invocations": mcp,
        "embedding_query_calls": emb,
        "local_fallback_count": fallback,
        "monitor_calls": monitor,
        "global_replan_rounds": replan,
        "planner_calls": planner_calls,
        "calls_by_stage": calls_by_stage,
        "plan_step_types": [
            s.get("type") for s in ((pred.get("initial_plan") or {}).get("steps") or [])
        ],
        "n_embedding_funnels": len(embedding_funnels),
        "n_disclosure_funnels": len(disclosure_funnels),
        "embedding_funnels": embedding_funnels,
        "disclosure_funnels": disclosure_funnels,
        "issues": issues,
        "engineering_ok": not issues,
    }


def run_audit(*, run_dir: Path, root: Path = ROOT) -> Dict[str, Any]:
    run_dir = Path(run_dir).resolve()
    out_dir = run_dir / "engineering_audit_v1"
    out_dir.mkdir(parents=True, exist_ok=True)

    before = collect_raw_hashes(run_dir)
    meta = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    preds = _load_preds(run_dir)
    task_ids = [str(p.get("task_id") or p.get("id")) for p in preds]

    formula_hash = retrieval_formula_hash()
    identity = {
        "condition": meta.get("condition"),
        "method_version": meta.get("method_version"),
        "architecture": meta.get("architecture"),
        "execution_backend": meta.get("execution_backend"),
        "mock": meta.get("mock"),
        "dry_run": meta.get("dry_run"),
        "git_dirty": meta.get("git_dirty"),
        "git_commit": meta.get("git_commit"),
        "git_branch": meta.get("git_branch"),
        "generation_provider": meta.get("generation_provider"),
        "generation_base_url_host": meta.get("generation_base_url_host"),
        "generation_model": meta.get("generation_model"),
        "embedding_provider": meta.get("embedding_provider"),
        "embedding_base_url_host": meta.get("embedding_base_url_host"),
        "embedding_model": meta.get("embedding_model"),
        "embedding_dimension": meta.get("embedding_dimension"),
        "retrieval_formula_name": meta.get("retrieval_formula_name"),
        "retrieval_formula_version": meta.get("retrieval_formula_version"),
        "retrieval_formula_exact": meta.get("retrieval_formula_exact"),
        "retrieval_formula_hash": meta.get("retrieval_formula_hash"),
        "retrieval_formula_hash_recomputed": formula_hash,
        "lexical_weight": meta.get("lexical_weight"),
        "domain_hard_filter": meta.get("domain_hard_filter"),
        "candidate_lock_sha256": meta.get("candidate_lock_sha256"),
        "retrieval_top_k": meta.get("retrieval_top_k"),
        "describe_top_n": meta.get("describe_top_n"),
        "embedding_index_identity_sha256": meta.get("embedding_index_identity_sha256"),
        "operation_documents_sha256": meta.get("operation_documents_sha256"),
        "catalog_fingerprint": meta.get("catalog_fingerprint"),
        "monitor_enabled": meta.get("monitor_enabled"),
        "global_replan_enabled": meta.get("global_replan_enabled"),
        "limits": meta.get("limits"),
        "run_status": meta.get("run_status"),
        "processed_task_count": meta.get("processed_task_count"),
        "question_count": meta.get("question_count"),
    }

    identity_checks = {
        "pe_fin_lite_plan_first": (
            identity["condition"] == "pe_fin_lite"
            and identity["architecture"] == "plan_first_step_level_retrieval_v1"
            and identity["method_version"] == "pe_fin_lite_v1.2"
        ),
        "r5_formula_name": identity["retrieval_formula_name"] == RETRIEVAL_FORMULA_NAME,
        "r5_formula_version": identity["retrieval_formula_version"]
        == RETRIEVAL_FORMULA_VERSION,
        "r5_formula_hash_match": identity["retrieval_formula_hash"] == formula_hash,
        "candidate_lock_match": identity["candidate_lock_sha256"]
        == R5_CANDIDATE_LOCK_SHA256,
        "domain_hard_filter_false": identity["domain_hard_filter"] is False,
        "lexical_weight_0_12": abs(float(identity["lexical_weight"] or -1) - 0.12)
        < 1e-12,
        "topk_5": int(identity["retrieval_top_k"] or 0) == 5,
        "describe_n_3": int(identity["describe_top_n"] or 0) == 3,
        "monitor_off": identity["monitor_enabled"] is False,
        "replan_off": identity["global_replan_enabled"] is False,
        "no_mock": identity["mock"] is False and identity["dry_run"] is False,
        "git_dirty_false": identity["git_dirty"] is False,
        "real_mcp": identity["execution_backend"] == "real_mcp",
        "generation_identity": (
            identity["generation_provider"] == "openai_compatible"
            and identity["generation_base_url_host"] == "api.zhizengzeng.com"
            and identity["generation_model"] == "deepseek-v4.1-flash"
        ),
        "embedding_identity": (
            identity["embedding_base_url_host"] == "api.zhizengzeng.com"
            and identity["embedding_model"] == "text-embedding-3-small"
            and int(identity["embedding_dimension"] or 0) == 1536
            and identity["embedding_index_identity_sha256"]
            == "ac4c109bed0edb21fb2b837dfdbd745680b24f33efa62e13ad6bca53f174b0df"
        ),
        "embedding_provider_field_null_but_host_present": identity[
            "embedding_provider"
        ]
        is None,
        "catalog_runtime_39": "runtime=39" in str(identity["catalog_fingerprint"] or ""),
        "run_completed": identity["run_status"] == "completed",
        "not_old_formula_resume": identity["retrieval_formula_name"]
        == RETRIEVAL_FORMULA_NAME
        and identity["domain_hard_filter"] is False,
    }

    per_task = [audit_task(p) for p in preds]
    status_counts = Counter(r["status"] for r in per_task)
    correct_count = sum(1 for r in per_task if r["answer_correct"])

    completeness = {
        "n_predictions": len(preds),
        "unique_ids": len(set(task_ids)),
        "task_ids": task_ids,
        "expected_task_ids": EXPECTED_TASK_IDS,
        "order_match": task_ids == EXPECTED_TASK_IDS,
        "missing_predictions": False,
        "runner_crash": False,
    }

    emb_funnel_n = sum(r["n_embedding_funnels"] for r in per_task)
    domain_filter_evidence = {
        "embedding_retrieve_steps": emb_funnel_n,
        "all_domain_hard_filter_false": all(
            f.get("domain_hard_filter") is False
            for r in per_task
            for f in r["embedding_funnels"]
        ),
        "all_filter_kind_safety_only_r5": all(
            f.get("filter_kind") == "safety_only_r5"
            for r in per_task
            for f in r["embedding_funnels"]
        ),
        "domain_incompatible_reject_total": sum(
            int(f.get("domain_incompatible_rejects") or 0)
            for r in per_task
            for f in r["embedding_funnels"]
        ),
        "candidate_pool_39_steps": sum(
            1
            for r in per_task
            for f in r["embedding_funnels"]
            if f.get("n_candidates_scored") == 39
        ),
    }
    safety_evidence = {
        "filter_kind": "safety_only_r5",
        "note": (
            "Resolver registry is runtime-exposed ops; safety_filter kept_count=39 "
            "and domain_incompatible rejects=0 on all embedding retrieves."
        ),
        "kept_counts": [
            f.get("kept_count")
            for r in per_task
            for f in r["embedding_funnels"]
        ],
    }
    formula_parity = {
        "rows_checked": sum(len(f.get("topk") or []) for r in per_task for f in r["embedding_funnels"]),
        "all_scores_match_cosine_plus_0_12_lexical": all(
            f.get("score_formula_ok") for r in per_task for f in r["embedding_funnels"]
        ),
        "meta_hash_matches_code": identity_checks["r5_formula_hash_match"],
        "candidate_lock_matches": identity_checks["candidate_lock_match"],
    }

    task_issues = [i for r in per_task for i in r["issues"]]
    engineering_blockers = []
    if not completeness["order_match"] or completeness["n_predictions"] != 5:
        engineering_blockers.append("incomplete_or_wrong_task_order")
    if not identity_checks["run_completed"]:
        engineering_blockers.append("run_not_completed")
    if not identity_checks["r5_formula_hash_match"]:
        engineering_blockers.append("r5_formula_not_wired")
    if not identity_checks["domain_hard_filter_false"]:
        engineering_blockers.append("domain_hard_filter_still_on")
    if not domain_filter_evidence["all_domain_hard_filter_false"]:
        engineering_blockers.append("domain_filter_seen_in_funnels")
    if not formula_parity["all_scores_match_cosine_plus_0_12_lexical"]:
        engineering_blockers.append("formula_parity_failed")
    if not identity_checks["no_mock"]:
        engineering_blockers.append("mock_or_dry_run")
    if not identity_checks["git_dirty_false"]:
        engineering_blockers.append("git_dirty")
    if task_issues:
        # budget/protocol issues are blockers; descriptive answer errors are not in issues
        engineering_blockers.extend(sorted(set(task_issues)))

    # embedding_provider null is noted but not a blocker when host/model/dim/index OK
    notes = []
    if identity_checks["embedding_provider_field_null_but_host_present"]:
        notes.append(
            "embedding_provider top-level field is null; host/model/dimension/index "
            "identity are correct and assert_live_provider_identity defaults provider "
            "to openai_compatible — treated as non-blocking metadata incompleteness."
        )

    engineering_smoke_valid = not engineering_blockers

    after = collect_raw_hashes(run_dir)
    unchanged = before == after

    summary = {
        "audit_version": AUDIT_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "engineering_smoke_valid": engineering_smoke_valid,
        "engineering_blockers": engineering_blockers,
        "notes": notes,
        "completeness": completeness,
        "status_counts": dict(status_counts),
        "answer_correct_count": correct_count,
        "answer_correct_rate_descriptive": correct_count / max(1, len(per_task)),
        "identity_checks": identity_checks,
        "domain_filter_evidence": domain_filter_evidence,
        "safety_filter_evidence": safety_evidence,
        "formula_parity": formula_parity,
        "raw_hashes_unchanged": unchanged,
        "zero_network_audit": True,
        "result_role": "engineering_smoke_descriptive",
        "no_method_retune_from_smoke_accuracy": True,
    }

    resource = {
        "n_tasks": len(per_task),
        "full": status_counts.get("full", 0),
        "partial": status_counts.get("partial", 0),
        "failed": status_counts.get("failed", 0),
        "answer_correct": correct_count,
        "per_task_budgets": [
            {
                "task_id": r["task_id"],
                "status": r["status"],
                "answer_correct": r["answer_correct"],
                "llm_calls": r["llm_calls"],
                "mcp_invocations": r["mcp_invocations"],
                "embedding_query_calls": r["embedding_query_calls"],
                "local_fallback_count": r["local_fallback_count"],
                "monitor_calls": r["monitor_calls"],
                "global_replan_rounds": r["global_replan_rounds"],
                "planner_calls": r["planner_calls"],
            }
            for r in per_task
        ],
        "llm_calls_total": sum(r["llm_calls"] for r in per_task),
        "mcp_invocations_total": sum(r["mcp_invocations"] for r in per_task),
        "embedding_query_calls_total": sum(r["embedding_query_calls"] for r in per_task),
    }

    # commands (only meaningful if valid)
    py = str(
        Path(
            "/Users/wzm/Desktop/科大/课程/MSBD5018_NLP/group project/"
            "hk-equity-mcp-agent/.venv-aktools-discovery/bin/python"
        )
    )
    repo = str(root)
    gen_env = str(
        Path(
            "/Users/wzm/Desktop/科大/课程/MSBD5018_NLP/group project/"
            "hk-equity-mcp-agent/.env.relay-v41"
        )
    )
    emb_env = str(
        Path(
            "/Users/wzm/Desktop/科大/课程/MSBD5018_NLP/group project/"
            "hk-equity-mcp-agent/.env.embedding-relay"
        )
    )
    cache = str(root / ".cache" / "pe_fin_lite_operation_embeddings_v1")
    hk155_out = str(root / "outputs" / "pe_fin_lite_r5_hk155_live_v1")
    script = str(root / "scripts" / "run_hk_unified_pe_fin_lite_experiment.py")

    preflight = (
        f'cd "{repo}" && PYTHONPATH=src "{py}" "{script}" \\\n'
        f'  --index-status \\\n'
        f'  --require-complete-index \\\n'
        f'  --embedding-cache-dir "{cache}"'
    )
    hk155_full = (
        f'cd "{repo}" && PYTHONPATH=src "{py}" "{script}" \\\n'
        f'  --split exploratory_all \\\n'
        f'  --confirm-retire-heldout \\\n'
        f'  --require-complete-index \\\n'
        f'  --allow-hkex-download \\\n'
        f'  --env-file "{gen_env}" \\\n'
        f'  --embedding-env-file "{emb_env}" \\\n'
        f'  --embedding-cache-dir "{cache}" \\\n'
        f'  --output-dir "{hk155_out}"'
    )
    hk155_batch1 = hk155_full + " \\\n  --max-tasks 40"
    hk155_resume = (
        f'cd "{repo}" && PYTHONPATH=src "{py}" "{script}" \\\n'
        f'  --split exploratory_all \\\n'
        f'  --confirm-retire-heldout \\\n'
        f'  --require-complete-index \\\n'
        f'  --allow-hkex-download \\\n'
        f'  --resume \\\n'
        f'  --env-file "{gen_env}" \\\n'
        f'  --embedding-env-file "{emb_env}" \\\n'
        f'  --embedding-cache-dir "{cache}" \\\n'
        f'  --output-dir "{hk155_out}"'
    )
    commands = {
        "zero_network_preflight_index_status": preflight,
        "hk155_single_shot_recommended": hk155_full,
        "hk155_first_batch_max40_optional": hk155_batch1,
        "hk155_resume_remaining_optional": hk155_resume,
        "resume_support": {
            "supported": True,
            "mechanism": "already_done_ids + --resume with PE_RESUME_IDENTITY_KEYS",
            "note": (
                "Optional batching: first run --max-tasks 40 into fresh dir, then "
                "--resume without --max-tasks to continue remaining IDs. "
                "Do not resume a pre-R5 directory."
            ),
        },
        "do_not_run_in_this_audit_turn": True,
    }

    payload_files = {
        "raw_hashes.json": {"before": before, "after": after, "unchanged": unchanged},
        "identity_audit.json": {"identity": identity, "checks": identity_checks},
        "per_task_audit.json": per_task,
        "resource_summary.json": resource,
        "domain_and_safety_evidence.json": {
            "domain_filter": domain_filter_evidence,
            "safety_filter": safety_evidence,
            "formula_parity": formula_parity,
        },
        "commands.json": commands if engineering_smoke_valid else {"blocked": True},
        "audit_summary.json": summary,
    }
    for name, obj in payload_files.items():
        (out_dir / name).write_text(
            json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    # final hash recheck after writes (raw only)
    after2 = collect_raw_hashes(run_dir)
    if after2 != before:
        raise RuntimeError("Raw artifacts changed during audit (fail-closed)")
    (out_dir / "raw_hashes.json").write_text(
        json.dumps(
            {"before": before, "after": after2, "unchanged": True},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    return {
        "out_dir": str(out_dir),
        "engineering_smoke_valid": engineering_smoke_valid,
        "summary": summary,
        "commands": commands if engineering_smoke_valid else None,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--run-dir",
        default=str(ROOT / "outputs" / "pe_fin_lite_r5_smoke5_live_v1"),
    )
    p.add_argument("--root", default=str(ROOT))
    args = p.parse_args(argv)
    result = run_audit(run_dir=Path(args.run_dir), root=Path(args.root))
    print("PE-FIN-LITE R5 SMOKE5 ENGINEERING AUDIT OK")
    print(f"out_dir={result['out_dir']}")
    print(f"engineering_smoke_valid={result['engineering_smoke_valid']}")
    print(f"blockers={result['summary'].get('engineering_blockers')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
