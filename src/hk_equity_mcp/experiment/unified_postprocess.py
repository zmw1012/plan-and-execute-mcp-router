"""Offline re-evaluation / postprocess for existing unified predictions.

Does NOT call Agent, LLM, Embedding, or MCP.
Does NOT modify predictions.jsonl.
Does NOT overwrite run_metadata identity fields.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from hk_equity_mcp.experiment.unified_evaluator import (
    EVALUATOR_VERSION,
    evaluator_identity,
    reevaluate_prediction_row,
    summarize_predictions,
)
from hk_equity_mcp.experiment.unified_runner import atomic_write_json, fsync_file
from hk_equity_mcp.experiment.unified_schema import (
    MANUAL_GRADING_FIELDS,
    METRIC_FIELD_DEFINITIONS,
    PER_TASK_METRIC_FIELDS,
)


IDENTITY_KEYS = (
    "condition",
    "execution_backend",
    "generation_model",
    "embedding_model",
    "dataset_split",
    "system_prompt_hash",
    "temperature",
    "max_tokens",
    "catalog_fingerprint",
    "git_commit",
    "branch",
    "question_ids",
    "catalog_stats",
)


def load_predictions(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    return rows


def postprocess_run_dir(
    run_dir: Path,
    *,
    write_manual_grading: bool = True,
) -> Dict[str, Any]:
    run_dir = Path(run_dir)
    predictions_path = run_dir / "predictions.jsonl"
    if not predictions_path.exists():
        raise FileNotFoundError(f"Missing predictions.jsonl: {predictions_path}")

    rows = load_predictions(predictions_path)
    enriched: List[Dict[str, Any]] = []
    for row in rows:
        metrics = reevaluate_prediction_row(row)
        # Merge metrics for side artifacts only; predictions.jsonl stays untouched.
        merged = dict(row)
        merged.update(metrics)
        # Preserve original row status/error for summarize.
        enriched.append(merged)

    summary = summarize_predictions(enriched)

    # per_task_metrics.csv
    metrics_path = run_dir / "per_task_metrics.csv"
    with metrics_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(PER_TASK_METRIC_FIELDS))
        writer.writeheader()
        for row in enriched:
            tokens = row.get("tokens") or {}
            writer.writerow(
                {
                    "task_id": row.get("task_id") or row.get("id"),
                    "domain": row.get("domain"),
                    "domain_combination": row.get("domain_combination"),
                    "project_difficulty": row.get("project_difficulty"),
                    "exact_etom_level": row.get("exact_etom_level"),
                    "exact_etom_alignment": row.get("exact_etom_alignment"),
                    "answerability_class": row.get("answerability_class"),
                    "status": row.get("status"),
                    "operation_precision": row.get("operation_precision"),
                    "operation_recall": row.get("operation_recall"),
                    "operation_f1": row.get("operation_f1"),
                    "exact_tool_set": row.get("exact_tool_set"),
                    "graph_dependency_accuracy": row.get("graph_dependency_accuracy"),
                    "tool_execution_success": row.get("tool_execution_success"),
                    "all_tool_invocations_successful": row.get(
                        "all_tool_invocations_successful"
                    ),
                    "successful_invocation_count": row.get(
                        "successful_invocation_count"
                    ),
                    "failed_invocation_count": row.get("failed_invocation_count"),
                    "recovered_after_tool_error": row.get("recovered_after_tool_error"),
                    "protocol_success_rate": row.get("protocol_success_rate"),
                    "wrong_tool_rate": row.get("wrong_tool_rate"),
                    "refusal_correctness": row.get("refusal_correctness"),
                    "llm_calls": row.get("llm_calls"),
                    "input_tokens": tokens.get("input_tokens"),
                    "output_tokens": tokens.get("output_tokens"),
                    "total_tokens": tokens.get("total_tokens"),
                    "embedding_calls": row.get("embedding_calls"),
                    "embedding_tokens": row.get("embedding_tokens"),
                    "latency_ms": row.get("latency_ms"),
                    "real_mcp_execution_count": row.get("real_mcp_execution_count"),
                    "retry_count": row.get("retry_count"),
                    "timeout": row.get("timeout"),
                    "parse_repair_count": row.get("parse_repair_count"),
                }
            )
    fsync_file(metrics_path)

    if write_manual_grading:
        grading_path = run_dir / "manual_grading.csv"
        with grading_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(MANUAL_GRADING_FIELDS))
            writer.writeheader()
            for row in enriched:
                writer.writerow(
                    {
                        "task_id": row.get("task_id") or row.get("id"),
                        "domain": row.get("domain", ""),
                        "domain_combination": row.get("domain_combination", ""),
                        "company": row.get("company", ""),
                        "project_difficulty": row.get("project_difficulty", ""),
                        "exact_etom_level": row.get("exact_etom_level", ""),
                        "exact_etom_alignment": row.get("exact_etom_alignment", ""),
                        "question": row.get("question", ""),
                        "gold_status": row.get("gold_status", ""),
                        "answerability_class": row.get("answerability_class", ""),
                        "gold_operations": json.dumps(
                            row.get("gold_operations") or [], ensure_ascii=False
                        ),
                        "reference_answer": row.get("reference_answer", ""),
                        "predicted_status": row.get("predicted_status", ""),
                        "predicted_answer": row.get("final_answer", ""),
                        "predicted_operations": json.dumps(
                            row.get("predicted_operations") or [], ensure_ascii=False
                        ),
                        "evidence": json.dumps(
                            row.get("evidence") or [], ensure_ascii=False
                        ),
                        "manual_answer_correctness": row.get(
                            "manual_answer_correctness", ""
                        ),
                        "manual_evidence_correctness": row.get(
                            "manual_evidence_correctness", ""
                        ),
                        "manual_unsupported_claim": row.get(
                            "manual_unsupported_claim", ""
                        ),
                        "manual_causal_overclaim": row.get(
                            "manual_causal_overclaim", ""
                        ),
                        "manual_overall_task_success": row.get(
                            "manual_overall_task_success", ""
                        ),
                        "notes": row.get("notes", ""),
                    }
                )
        fsync_file(grading_path)

    cost = {
        "n_tasks": len(enriched),
        "total_input_tokens": sum(
            int((r.get("tokens") or {}).get("input_tokens") or 0) for r in enriched
        ),
        "total_output_tokens": sum(
            int((r.get("tokens") or {}).get("output_tokens") or 0) for r in enriched
        ),
        "total_tokens": sum(
            int((r.get("tokens") or {}).get("total_tokens") or 0) for r in enriched
        ),
        "total_llm_calls": sum(int(r.get("llm_calls") or 0) for r in enriched),
        "total_real_mcp_executions": sum(
            int(r.get("real_mcp_execution_count") or 0) for r in enriched
        ),
        "mean_latency_ms": summary.get("mean_latency_ms"),
        "postprocessed": True,
        "note": "Token totals only; USD cost requires provider price table.",
    }
    atomic_write_json(run_dir / "cost_summary.json", cost)

    meta_path = run_dir / "run_metadata.json"
    run_meta: Dict[str, Any] = {}
    if meta_path.exists():
        run_meta = json.loads(meta_path.read_text(encoding="utf-8"))

    atomic_write_json(
        run_dir / "retrieval_analysis.json",
        {
            "top_k": run_meta.get("retrieval_top_k"),
            "lexical_weight": run_meta.get("lexical_weight"),
            "postprocessed": True,
            "per_task": [
                {
                    "task_id": r.get("task_id") or r.get("id"),
                    "retrieved_candidates": r.get("retrieved_candidates"),
                    "predicted_operations": r.get("predicted_operations"),
                    "gold_operations": r.get("gold_operations"),
                }
                for r in enriched
            ],
        },
    )

    error_payload = {
        "error_counts": summary.get("error_counts") or {},
        "result_status_counts": summary.get("result_status_counts") or {},
        "definitions": {
            "error_counts": METRIC_FIELD_DEFINITIONS["error_counts"],
            "result_status_counts": METRIC_FIELD_DEFINITIONS["result_status_counts"],
            "tool_execution_success": METRIC_FIELD_DEFINITIONS[
                "tool_execution_success"
            ],
            "recovered_after_tool_error": METRIC_FIELD_DEFINITIONS[
                "recovered_after_tool_error"
            ],
        },
        "evaluator": evaluator_identity(),
        "aggregate": summary,
        "postprocessed": True,
        "source_predictions": str(predictions_path),
        "predictions_unchanged": True,
    }
    atomic_write_json(run_dir / "error_summary.json", error_payload)

    eval_meta = {
        **evaluator_identity(),
        "postprocessed_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "n_tasks": len(enriched),
        "predictions_path": str(predictions_path),
        "predictions_unchanged": True,
        "regenerated_artifacts": [
            "per_task_metrics.csv",
            "manual_grading.csv" if write_manual_grading else None,
            "cost_summary.json",
            "retrieval_analysis.json",
            "error_summary.json",
            "evaluation_metadata.json",
        ],
        "preserved_run_metadata_identity_keys": list(IDENTITY_KEYS),
        "run_metadata_identity_snapshot": {
            k: run_meta.get(k) for k in IDENTITY_KEYS if k in run_meta
        },
        "note": (
            "Offline postprocess only. Did not call LLM/Embedding/MCP/Agent. "
            "Did not modify predictions.jsonl or run_metadata identity fields."
        ),
    }
    eval_meta["regenerated_artifacts"] = [
        x for x in eval_meta["regenerated_artifacts"] if x
    ]
    atomic_write_json(run_dir / "evaluation_metadata.json", eval_meta)

    # Intentionally do not rewrite run_metadata.json (identity preserved on disk).

    return {
        "n_tasks": len(enriched),
        "error_counts": error_payload["error_counts"],
        "result_status_counts": error_payload["result_status_counts"],
        "evaluator": evaluator_identity(),
        "run_dir": str(run_dir),
    }
