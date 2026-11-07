"""Offline final audit for HK-155 MCP-Zero exploratory run.

Writes ONLY under <run_dir>/final_audit_mcp_zero_v1/ (plus optional docs mirror).
Never modifies predictions.jsonl / run_metadata.json / other raw side artifacts.
Never calls LLM / Embedding / MCP / paid APIs. Never re-runs tasks.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

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
from hk_equity_mcp.experiment.unified_runner import atomic_write_json
from hk_equity_mcp.mcp_zero import method_identity_sha256
from hk_equity_mcp.mcp_zero.prompt import mcp_zero_prompt_hash

AUDIT_VERSION = "final_audit_mcp_zero_v1"
EXPECTED_PROMPT_HASH = (
    "0879f47079d8ba345cda167829fa3474375aefd647f92efc139014d7a2885ab8"
)
EXPECTED_METHOD_HASH = (
    "77cc3b07a4c2b9af00ad3f6bfb7cba59c82084d94b7592f2c820f46f1d3e8d3f"
)
EXPECTED_N = 155
EXPECTED_DOMAIN = {
    "financial": 60,
    "market": 35,
    "news": 35,
    "cross_domain": 25,
}
EXPECTED_FULL = 130
EXPECTED_FAILED = 25

# User-facing MCP-Zero primary error labels (one per failed/incorrect task).
MCP_ZERO_PRIMARY = (
    "method_resource_limit_failure",
    "protocol_format_failure",
    "llm_provider_transport_failure",
    "mcp_tool_execution_failure",
    "retrieval_failure",
    "incomplete_tool_chain",
    "feasibility_refusal_failure",
    "answer_synthesis_failure",
    "evaluation_manual_required",
    "success",
)

RAW_FILES = (
    "predictions.jsonl",
    "run_metadata.json",
    "per_task_metrics.csv",
    "manual_grading.csv",
    "cost_summary.json",
    "error_summary.json",
    "retrieval_analysis.json",
    "grouped_aggregates.json",
    "grading_summary.json",
)

MCP_FREE_FINAL_ACCURACY = 0.129032  # 20/155
FLAT_FINAL_ACCURACY = 0.567742  # 88/155
FLAT_TOKENS = 9464124
FLAT_MCP_EXECS = 572
FLAT_EXACT_TOOL_SET = 0.232
FLAT_OP_F1 = 0.573


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fingerprint_raw_artifacts(run_dir: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for name in RAW_FILES:
        path = Path(run_dir) / name
        if not path.exists():
            raise FileNotFoundError(f"Missing required raw artifact: {path}")
        raw = path.read_bytes()
        if name.endswith(".jsonl"):
            rows = sum(1 for line in raw.splitlines() if line.strip())
        else:
            rows = raw.count(b"\n") + (
                0 if (not raw or raw.endswith(b"\n")) else 1
            )
        out[name] = {
            "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw),
            "lines_or_rows": rows,
        }
    return out


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _answer_text(final_answer: Any) -> str:
    if final_answer is None:
        return ""
    if isinstance(final_answer, dict):
        return str(final_answer.get("final_answer") or "")
    return str(final_answer)


def _stable_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _content_fingerprint(payload: Any) -> str:
    return hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def _percentile(values: Sequence[float], p: float) -> Optional[float]:
    if not values:
        return None
    xs = sorted(float(v) for v in values)
    if len(xs) == 1:
        return xs[0]
    k = (len(xs) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return xs[int(k)]
    return xs[f] * (c - k) + xs[c] * (k - f)


def load_run_order(root: Path) -> List[str]:
    path = root / "benchmarks" / "manifests" / "hk_unified_exploratory_all_run_order_v1.json"
    blob = json.loads(path.read_text(encoding="utf-8"))
    ids = blob.get("task_ids") or blob.get("run_order") or []
    return [str(x) for x in ids]


def load_reference_answer(root: Path, task_id: str) -> Optional[str]:
    """Read-only lookup for manual queue; never used to alter model outputs."""
    for path in (root / "benchmarks" / "questions").rglob("*.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if task_id not in line:
                continue
            row = json.loads(line)
            if str(row.get("id") or row.get("task_id")) != task_id:
                continue
            ref = row.get("reference_answer")
            return str(ref) if ref is not None else None
    return None


def classify_mcp_zero_primary(
    row: Dict[str, Any],
    *,
    grade: Dict[str, Any],
) -> Dict[str, Any]:
    """One primary MCP-Zero error label + optional secondaries."""
    secondary: List[str] = []
    status = str(row.get("status") or "")
    err = str(row.get("error") or row.get("error_type") or "").strip()
    answer_correct = grade.get("answer_correct")
    auto_status = grade.get("auto_grade_status")
    repair = int(row.get("repair_count") or 0) or (
        int(row.get("protocol_repair_count") or 0)
        + int(row.get("parse_repair_count") or 0)
    )
    inv = int(row.get("real_mcp_execution_count") or 0)
    gold = [str(x) for x in (row.get("gold_operations") or [])]
    pred = [str(x) for x in (row.get("predicted_operations") or [])]
    retrieved = row.get("retrieved_tools") or row.get("retrieved_candidates") or []
    retrieved_ids = {
        str(c.get("operation_id"))
        for c in retrieved
        if isinstance(c, dict) and c.get("operation_id")
    }
    ac = str(row.get("answerability_class") or "")
    should_refuse = ac in {
        "intentionally_infeasible",
        "should_refuse_or_clarify",
        "capability_gap",
        "robustness",
    }

    if auto_status == "manual_required":
        return {
            "mcp_zero_primary_error": "evaluation_manual_required",
            "mcp_zero_secondary_errors": [],
            "mcp_zero_error_reason": "manual_required_gold_values_missing",
        }

    if answer_correct is True and status == "full":
        if err:
            secondary.append("method_resource_limit_failure")
        if gold and not set(gold).issubset(retrieved_ids) and retrieved_ids:
            secondary.append("retrieval_failure")
        if gold and set(pred) != set(gold):
            secondary.append("incomplete_tool_chain")
        return {
            "mcp_zero_primary_error": "success",
            "mcp_zero_secondary_errors": secondary,
            "mcp_zero_error_reason": "answer_correct",
        }

    # Failed / incorrect path — choose primary by specificity.
    if err in {"RuntimeError", "HTTPError", "ConnectionError"} or "runtimeerror" in err.lower():
        primary = "llm_provider_transport_failure"
        reason = f"transport_or_runtime:{err}"
    elif err in {"timeout", "TimeoutError"}:
        primary = "method_resource_limit_failure"
        reason = "timeout"
    elif err in {"max_llm_turns", "max_invocations"}:
        primary = "method_resource_limit_failure"
        reason = err
        if repair >= 3:
            secondary.append("protocol_format_failure")
        if inv == 0 and gold:
            secondary.append("incomplete_tool_chain")
        if gold and retrieved_ids and not (set(gold) & retrieved_ids):
            secondary.append("retrieval_failure")
    elif status == "failed" and repair >= max(1, int(row.get("llm_calls") or 1) // 2):
        primary = "protocol_format_failure"
        reason = "high_repair_without_terminal_contract"
    elif inv > 0 and not bool(row.get("all_tool_invocations_successful", True)):
        primary = "mcp_tool_execution_failure"
        reason = "gateway_or_tool_execution_failed"
    elif gold and (not retrieved_ids or not (set(gold) & retrieved_ids)):
        primary = "retrieval_failure"
        reason = "gold_ops_absent_from_retrieved"
    elif gold and (not pred or not set(gold).issubset(set(pred))):
        primary = "incomplete_tool_chain"
        reason = "gold_tool_chain_not_completed"
        if should_refuse:
            secondary.append("feasibility_refusal_failure")
    elif should_refuse:
        primary = "feasibility_refusal_failure"
        reason = "refusal_or_clarify_class_incorrect"
    else:
        primary = "answer_synthesis_failure"
        reason = "full_or_answered_but_incorrect_or_failed_synthesis"
        if status == "failed":
            secondary.append("method_resource_limit_failure")

    # Deduplicate secondary while preserving order.
    seen = set()
    sec_out = []
    for s in secondary:
        if s not in seen and s != primary:
            seen.add(s)
            sec_out.append(s)
    return {
        "mcp_zero_primary_error": primary,
        "mcp_zero_secondary_errors": sec_out,
        "mcp_zero_error_reason": reason,
    }


def _group_rate(rows: Sequence[Dict[str, Any]], key: str) -> Dict[str, Any]:
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        buckets[str(r.get(key) if r.get(key) is not None else "missing")].append(r)
    out: Dict[str, Any] = {}
    for name, items in sorted(buckets.items()):
        auto = [
            x
            for x in items
            if x.get("auto_grade_status") == "graded"
            and x.get("answer_correct") is not None
        ]
        correct = [x for x in auto if x.get("answer_correct") is True]
        strict_correct = sum(1 for x in items if x.get("strict_answer_correct") is True)
        out[name] = {
            "n": len(items),
            "full": sum(1 for x in items if x.get("status") == "full"),
            "failed": sum(1 for x in items if x.get("status") == "failed"),
            "auto_gradable": len(auto),
            "auto_correct": len(correct),
            "auto_accuracy": (len(correct) / len(auto)) if auto else None,
            "strict_answer_accuracy": strict_correct / len(items) if items else None,
            "exact_tool_set_rate": (
                sum(1 for x in items if x.get("exact_tool_set")) / len(items)
            ),
            "mean_operation_f1": (
                statistics.mean(
                    [
                        float(x["operation_f1"])
                        for x in items
                        if isinstance(x.get("operation_f1"), (int, float))
                    ]
                )
                if any(isinstance(x.get("operation_f1"), (int, float)) for x in items)
                else None
            ),
        }
    return out


def compute_tool_retrieval_protocol(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows) or 1
    attempt = [r for r in rows if int(r.get("real_mcp_execution_count") or 0) > 0]
    zero_fail = [
        r
        for r in rows
        if r.get("status") == "failed" and int(r.get("real_mcp_execution_count") or 0) == 0
    ]
    successful_inv = sum(int(r.get("successful_invocation_count") or 0) for r in rows)
    total_inv = sum(int(r.get("real_mcp_execution_count") or 0) for r in rows)

    def _avg(key: str) -> Optional[float]:
        vals = [float(r[key]) for r in rows if isinstance(r.get(key), (int, float))]
        return (sum(vals) / len(vals)) if vals else None

    return {
        "exact_tool_set_rate": sum(1 for r in rows if r.get("exact_tool_set")) / n,
        "mean_operation_precision": _avg("operation_precision"),
        "mean_operation_recall": _avg("operation_recall"),
        "mean_operation_f1": _avg("operation_f1"),
        "mean_wrong_tool_rate": _avg("wrong_tool_rate"),
        "mean_retrieved_count": statistics.mean(
            [len(r.get("retrieved_tools") or r.get("retrieved_candidates") or []) for r in rows]
        ),
        "mean_selected_count": statistics.mean(
            [len(r.get("selected_operations") or []) for r in rows]
        ),
        "mean_invoked_count": statistics.mean(
            [len(r.get("invoked_operations") or r.get("predicted_operations") or []) for r in rows]
        ),
        "mean_successful_invoked_count": statistics.mean(
            [len(r.get("successful_invoked_operations") or []) for r in rows]
        ),
        "tasks_with_invalid_selection": sum(
            1 for r in rows if (r.get("invalid_selections") or [])
        ),
        "mean_extension_rounds": statistics.mean(
            [float(r.get("number_of_extension_rounds") or 0) for r in rows]
        ),
        "all_tool_invocations_successful_rate": sum(
            1 for r in rows if r.get("all_tool_invocations_successful")
        )
        / n,
        "invocation_attempt_task_count": len(attempt),
        "invocation_conditioned_task_success_rate": (
            sum(1 for r in attempt if r.get("all_tool_invocations_successful"))
            / len(attempt)
            if attempt
            else None
        ),
        "successful_invocations": successful_inv,
        "total_invocations": total_inv,
        "zero_invocation_failed_task_count": len(zero_fail),
        "recovered_after_tool_error_rate": sum(
            1 for r in rows if r.get("recovered_after_tool_error")
        )
        / n,
        "mean_protocol_success_rate": _avg("protocol_success_rate"),
        "task_protocol_success_rate": sum(
            1 for r in rows if r.get("task_protocol_success")
        )
        / n,
        "terminal_contract_valid_rate": sum(
            1 for r in rows if r.get("terminal_contract_valid")
        )
        / n,
        "mean_repair_count": statistics.mean(
            [float(r.get("repair_count") or 0) for r in rows]
        ),
        "mean_protocol_repair_count": statistics.mean(
            [float(r.get("protocol_repair_count") or 0) for r in rows]
        ),
        "field_definitions": {
            "retrieved_tools": "Router Top-K candidates (retrieval Recall@K only)",
            "selected_operations": "Explicit invoke_api operation_ids",
            "invoked_operations": "Selected ops submitted to Gateway",
            "successful_invoked_operations": "Gateway-successful ops",
        },
    }


def compute_resources(
    rows: Sequence[Dict[str, Any]], *, run_meta: Dict[str, Any], cost: Dict[str, Any]
) -> Dict[str, Any]:
    latencies = [
        float(r["latency_ms"])
        for r in rows
        if isinstance(r.get("latency_ms"), (int, float))
    ]
    emb_index = run_meta.get("embedding_index") or {}
    return {
        "n_tasks": len(rows),
        "total_input_tokens": cost.get("total_input_tokens")
        or sum(int((r.get("tokens") or {}).get("input_tokens") or 0) for r in rows),
        "total_output_tokens": cost.get("total_output_tokens")
        or sum(int((r.get("tokens") or {}).get("output_tokens") or 0) for r in rows),
        "total_tokens": cost.get("total_tokens")
        or sum(int((r.get("tokens") or {}).get("total_tokens") or 0) for r in rows),
        "total_llm_calls": cost.get("total_llm_calls")
        or sum(int(r.get("llm_calls") or 0) for r in rows),
        "total_embedding_calls_query_time": cost.get("total_embedding_calls")
        or sum(int(r.get("embedding_calls") or 0) for r in rows),
        "total_embedding_tokens_query_time": cost.get("total_embedding_tokens")
        or sum(int(r.get("embedding_tokens") or 0) for r in rows),
        "embedding_index_build_calls": emb_index.get("build_calls", cost.get("embedding_index_build_calls")),
        "embedding_index_cache_reused": emb_index.get(
            "cache_reused", cost.get("embedding_index_cache_reused")
        ),
        "total_real_mcp_executions": cost.get("total_real_mcp_executions")
        or sum(int(r.get("real_mcp_execution_count") or 0) for r in rows),
        "mean_latency_ms": (sum(latencies) / len(latencies)) if latencies else None,
        "median_latency_ms": statistics.median(latencies) if latencies else None,
        "p95_latency_ms": _percentile(latencies, 95),
        "estimated_usd": None,
        "estimated_usd_note": (
            "No frozen provider price table in-repo; estimated_usd left null."
        ),
    }


def validate_run_identity(
    *,
    predictions: Sequence[Dict[str, Any]],
    run_meta: Dict[str, Any],
    run_order: Sequence[str],
    root: Path,
) -> Dict[str, Any]:
    ids = [str(p.get("task_id") or p.get("id")) for p in predictions]
    if len(predictions) != EXPECTED_N:
        raise ValueError(f"Expected {EXPECTED_N} predictions, found {len(predictions)}")
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate task_id in predictions.jsonl")
    if list(run_order) != ids:
        raise ValueError("predictions task order does not match exploratory run_order manifest")
    if run_meta.get("run_status") != "completed":
        raise ValueError(f"run_status != completed: {run_meta.get('run_status')}")
    if int(run_meta.get("question_count") or 0) != EXPECTED_N:
        raise ValueError("question_count != 155")
    if run_meta.get("condition") != "mcp_zero":
        raise ValueError("condition != mcp_zero")
    if run_meta.get("generation_provider") != "openai_compatible":
        raise ValueError("generation_provider drift")
    if run_meta.get("generation_base_url_host") != "api.zhizengzeng.com":
        raise ValueError("generation host drift")
    if run_meta.get("generation_model") != "deepseek-v4.1-flash":
        raise ValueError("generation_model drift")
    if run_meta.get("embedding_model") != "text-embedding-3-small":
        raise ValueError("embedding_model drift")
    if run_meta.get("system_prompt_hash") != EXPECTED_PROMPT_HASH:
        raise ValueError("system_prompt_hash drift")
    method_hash = run_meta.get("method_card_hash") or run_meta.get("method_identity_sha256")
    if method_hash != EXPECTED_METHOD_HASH:
        raise ValueError("method-card hash drift")
    if mcp_zero_prompt_hash() != EXPECTED_PROMPT_HASH:
        raise ValueError("worktree prompt hash drifted from expected formal hash")
    if method_identity_sha256() != EXPECTED_METHOD_HASH:
        raise ValueError("worktree method hash drifted from expected formal hash")
    domain_counts = Counter(str(r.get("domain")) for r in predictions)
    if dict(domain_counts) != EXPECTED_DOMAIN:
        raise ValueError(f"domain counts mismatch: {dict(domain_counts)}")
    status_counts = Counter(str(r.get("status")) for r in predictions)
    if status_counts.get("full") != EXPECTED_FULL or status_counts.get("failed") != EXPECTED_FAILED:
        raise ValueError(f"status counts mismatch: {dict(status_counts)}")
    catalog = run_meta.get("catalog_stats") or {}
    if int(catalog.get("unique_operation_ids") or 0) != 42:
        raise ValueError("catalog unique_operation_ids != 42")
    if int(catalog.get("runtime_exposed_unique") or catalog.get("mcp_zero_tool_count") or 0) != 39:
        raise ValueError("catalog runtime_exposed_unique != 39")
    emb = run_meta.get("embedding_index") or {}
    build_calls = emb.get("build_calls")
    if emb.get("cache_reused") is not True or build_calls is None or int(build_calls) != 0:
        raise ValueError("expected shared embedding index reuse with build_calls=0")
    return {
        "n": len(predictions),
        "unique_task_ids": len(set(ids)),
        "run_order_match": True,
        "domain_counts": dict(domain_counts),
        "status_counts": dict(status_counts),
        "system_prompt_hash": run_meta.get("system_prompt_hash"),
        "method_card_hash": method_hash,
        "generation_provider": run_meta.get("generation_provider"),
        "generation_base_url_host": run_meta.get("generation_base_url_host"),
        "generation_model": run_meta.get("generation_model"),
        "embedding_model": run_meta.get("embedding_model"),
        "git_commit": run_meta.get("git_commit"),
        "embedding_index": {
            "cache_reused": emb.get("cache_reused"),
            "build_calls": emb.get("build_calls"),
            "identity_sha256": emb.get("identity_sha256"),
            "complete": emb.get("complete"),
        },
        "catalog_stats": {
            "unique_operation_ids": catalog.get("unique_operation_ids"),
            "runtime_exposed_unique": catalog.get("runtime_exposed_unique"),
            "mcp_zero_tool_count": catalog.get("mcp_zero_tool_count"),
        },
    }


def render_markdown_report(ctx: Dict[str, Any]) -> str:
    g = ctx["grading"]
    s = ctx["strict_metrics"]
    res = ctx["resources"]
    tool = ctx["tool_retrieval_protocol"]
    tax = ctx["mcp_zero_error_taxonomy"]
    lines: List[str] = []
    lines.append("# HK-155 MCP-Zero Final Audit v1")
    lines.append("")
    lines.append("Offline audit only. Raw predictions were not modified. No APIs called.")
    lines.append("")
    lines.append("## Identity")
    ident = ctx["identity_validation"]
    lines.append(f"- condition=`mcp_zero`")
    lines.append(f"- provider/host/model=`{ident['generation_provider']}` / `{ident['generation_base_url_host']}` / `{ident['generation_model']}`")
    lines.append(f"- embedding=`{ident['embedding_model']}`")
    lines.append(f"- prompt hash=`{ident['system_prompt_hash']}`")
    lines.append(f"- method-card hash=`{ident['method_card_hash']}`")
    lines.append(f"- git_commit=`{ident['git_commit']}`")
    lines.append(f"- tasks=`{ident['n']}` unique; domains={ident['domain_counts']}")
    lines.append(f"- status={ident['status_counts']}")
    lines.append(
        f"- embedding index reused=`{ident['embedding_index']['cache_reused']}` "
        f"build_calls=`{ident['embedding_index']['build_calls']}`"
    )
    lines.append("")
    lines.append("## Auto / strict grading (v1.2)")
    lines.append(f"- auto_gradable_count=`{g['auto_gradable_count']}`")
    lines.append(f"- auto_graded_correct=`{g['auto_graded_correct']}`")
    lines.append(f"- auto_graded_accuracy=`{g['auto_graded_accuracy']}`")
    lines.append(f"- final_answer_accuracy (auto)=`{s['final_answer_accuracy']}`")
    lines.append(f"- strict_answer_accuracy /155=`{s['strict_answer_accuracy']}`")
    lines.append(f"- strict_overall_task_success /155=`{s['strict_overall_task_success']}`")
    lines.append(f"- manual_required_count=`{g['manual_required_count']}` (fin_cf_022 queue)")
    lines.append("")
    lines.append("## Descriptive baseline comparison (not significance-tested)")
    lines.append(
        f"- MCP-Zero strict answer accuracy ≈ `{s['strict_answer_accuracy']:.4%}` "
        f"vs MCP-Free `{MCP_FREE_FINAL_ACCURACY:.4%}` (20/155) "
        f"vs Flat `{FLAT_FINAL_ACCURACY:.4%}` (88/155)"
    )
    lines.append(
        f"- MCP-Zero tokens=`{res['total_tokens']}` vs Flat `{FLAT_TOKENS}`; "
        f"MCP execs=`{res['total_real_mcp_executions']}` vs Flat `{FLAT_MCP_EXECS}`"
    )
    lines.append(
        f"- MCP-Zero exact_tool_set≈`{tool['exact_tool_set_rate']:.3f}` "
        f"(Flat≈{FLAT_EXACT_TOOL_SET}); "
        f"op F1≈`{(tool['mean_operation_f1'] or 0):.3f}` (Flat≈{FLAT_OP_F1})"
    )
    lines.append(
        "- First-20 vs full-155 differences reflect locked-order sample composition, "
        "not model regression."
    )
    lines.append("")
    lines.append("## Domain / difficulty (strict answer accuracy)")
    for title, key in (
        ("domain", "by_domain"),
        ("project_difficulty", "by_difficulty"),
        ("source_split", "by_source_split"),
        ("answerability_class", "by_answerability"),
        ("status", "by_status"),
    ):
        lines.append(f"### {title}")
        for name, stats in ctx["grouped_strict"][key].items():
            lines.append(
                f"- `{name}`: n={stats['n']} auto_acc="
                f"{None if stats['auto_accuracy'] is None else round(stats['auto_accuracy'], 4)} "
                f"strict={round(stats['strict_answer_accuracy'], 4)}"
            )
        lines.append("")
    lines.append("## MCP-Zero error taxonomy (primary)")
    for name, count in tax["primary_counts"].items():
        lines.append(f"- `{name}`: {count}")
    lines.append("")
    lines.append("## Failed-task error breakdown (raw error field)")
    for name, count in tax["failed_raw_error_counts"].items():
        lines.append(f"- `{name}`: {count}")
    lines.append("")
    lines.append("## Manual review")
    lines.append("- Only `fin_cf_022` is manual_required under grader v1.2.")
    lines.append("- See `manual_review_queue.csv` / `human_grades.json` (human fields blank).")
    lines.append("- Do **not** copy MCP-Free/Flat human grades for this condition.")
    lines.append("")
    lines.append("## Resources")
    lines.append(f"- tokens in/out/total=`{res['total_input_tokens']}`/`{res['total_output_tokens']}`/`{res['total_tokens']}`")
    lines.append(f"- llm_calls=`{res['total_llm_calls']}`")
    lines.append(
        f"- embedding query calls/tokens=`{res['total_embedding_calls_query_time']}`/"
        f"`{res['total_embedding_tokens_query_time']}`"
    )
    lines.append(
        f"- index build_calls=`{res['embedding_index_build_calls']}` "
        f"cache_reused=`{res['embedding_index_cache_reused']}`"
    )
    lines.append(f"- MCP executions=`{res['total_real_mcp_executions']}`")
    lines.append(
        f"- latency mean/median/p95 ms=`{res['mean_latency_ms']}`/"
        f"`{res['median_latency_ms']}`/`{res['p95_latency_ms']}`"
    )
    lines.append(f"- estimated_usd=`{res['estimated_usd']}` ({res['estimated_usd_note']})")
    lines.append("")
    lines.append("## Raw artifact integrity")
    lines.append("- pre/post SHA256 match: see `raw_artifact_fingerprints.json`")
    return "\n".join(lines) + "\n"


def run_mcp_zero_final_audit(
    run_dir: Path,
    *,
    root: Optional[Path] = None,
    audit_outdir: Optional[Path] = None,
    docs_mirror_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    run_dir = Path(run_dir)
    if root is None:
        # Prefer <repo>/outputs/<run> layout.
        cand = run_dir.parent.parent
        root = cand if (cand / "src").is_dir() else Path.cwd()
    else:
        root = Path(root)
    out_dir = Path(audit_outdir) if audit_outdir else run_dir / "final_audit_mcp_zero_v1"
    out_dir.mkdir(parents=True, exist_ok=True)

    pre = fingerprint_raw_artifacts(run_dir)
    atomic_write_json(out_dir / "raw_artifact_fingerprints_pre.json", pre)

    predictions = load_jsonl(run_dir / "predictions.jsonl")
    run_meta = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    cost = json.loads((run_dir / "cost_summary.json").read_text(encoding="utf-8"))
    retrieval_blob = json.loads(
        (run_dir / "retrieval_analysis.json").read_text(encoding="utf-8")
    )
    retrieval_by_id = {
        str(r.get("task_id")): r
        for r in (retrieval_blob.get("per_task") or [])
        if r.get("task_id")
    }
    run_order = load_run_order(root)
    identity = validate_run_identity(
        predictions=predictions, run_meta=run_meta, run_order=run_order, root=root
    )

    enriched: List[Dict[str, Any]] = []
    grades: List[Dict[str, Any]] = []
    grades_by_id: Dict[str, Dict[str, Any]] = {}
    audit_rows: List[Dict[str, Any]] = []
    failed_rows: List[Dict[str, Any]] = []

    for idx, row in enumerate(predictions, start=1):
        metrics = reevaluate_prediction_row(row)
        trace = row.get("trace") or []
        if not metrics.get("predicted_operations"):
            metrics["predicted_operations"] = extract_predicted_operations(trace)
        if not metrics.get("retrieved_candidates"):
            metrics["retrieved_candidates"] = extract_retrieved_candidates(trace)
        if not metrics.get("invoke_calls"):
            metrics["invoke_calls"] = extract_invoke_records(trace)

        merged = dict(row)
        merged.update(metrics)
        # Ensure failed never auto-grades correct (grader already does; reinforce).
        grade = grade_task(merged, merged)
        if str(merged.get("status") or "").lower() == "failed":
            grade = {
                **grade,
                "auto_grade_status": "graded",
                "answer_correct": False,
                "reason": grade.get("reason") or "status_failed",
                "overall_task_success": False,
            }
        grades.append(grade)
        tid = str(merged.get("task_id") or merged.get("id"))
        grades_by_id[tid] = grade
        merged["auto_grade_status"] = grade.get("auto_grade_status")
        merged["answer_correct"] = grade.get("answer_correct")
        merged["auto_grade_reason"] = grade.get("reason")
        merged["tool_selection_correct"] = grade.get("tool_selection_correct")
        merged["overall_task_success"] = grade.get("overall_task_success")
        merged["task_order_index"] = idx
        # Strict labels (manual_required counts as incorrect until human grades).
        merged["strict_answer_correct"] = bool(grade.get("answer_correct") is True)
        merged["strict_overall_task_success"] = bool(
            merged.get("status") == "full"
            and grade.get("answer_correct") is True
            and grade.get("tool_selection_correct") is True
            and bool(merged.get("all_tool_invocations_successful", False))
        )

        mz = classify_mcp_zero_primary(merged, grade=grade)
        merged.update(mz)
        tax = classify_task_error(
            merged,
            retrieval_row=retrieval_by_id.get(tid),
        )
        merged["unified_primary_error_category"] = tax.get("primary_error_category")
        merged["unified_secondary_error_categories"] = tax.get(
            "secondary_error_categories"
        )
        merged["unified_error_reason"] = tax.get("error_reason")
        enriched.append(merged)

        audit_rows.append(
            {
                "task_id": tid,
                "task_order_index": idx,
                "domain": merged.get("domain"),
                "domain_combination": merged.get("domain_combination"),
                "project_difficulty": merged.get("project_difficulty"),
                "answerability_class": merged.get("answerability_class"),
                "source_split": merged.get("source_split"),
                "previously_reserved": merged.get("previously_reserved"),
                "company": merged.get("company"),
                "status": merged.get("status"),
                "error": merged.get("error") or "",
                "error_type": merged.get("error_type") or "",
                "predicted_status": merged.get("predicted_status"),
                "auto_grade_status": grade.get("auto_grade_status"),
                "answer_correct": grade.get("answer_correct"),
                "strict_answer_correct": merged.get("strict_answer_correct"),
                "strict_overall_task_success": merged.get("strict_overall_task_success"),
                "auto_grade_reason": grade.get("reason"),
                "operation_precision": merged.get("operation_precision"),
                "operation_recall": merged.get("operation_recall"),
                "operation_f1": merged.get("operation_f1"),
                "exact_tool_set": merged.get("exact_tool_set"),
                "wrong_tool_rate": merged.get("wrong_tool_rate"),
                "retrieved_n": len(
                    merged.get("retrieved_tools")
                    or merged.get("retrieved_candidates")
                    or []
                ),
                "selected_n": len(merged.get("selected_operations") or []),
                "invoked_n": len(
                    merged.get("invoked_operations")
                    or merged.get("predicted_operations")
                    or []
                ),
                "successful_invoked_n": len(
                    merged.get("successful_invoked_operations") or []
                ),
                "invalid_selections": json.dumps(
                    merged.get("invalid_selections") or [], ensure_ascii=False
                ),
                "number_of_extension_rounds": merged.get("number_of_extension_rounds"),
                "task_protocol_success": merged.get("task_protocol_success"),
                "terminal_contract_valid": merged.get("terminal_contract_valid"),
                "repair_count": merged.get("repair_count"),
                "protocol_repair_count": merged.get("protocol_repair_count"),
                "all_tool_invocations_successful": merged.get(
                    "all_tool_invocations_successful"
                ),
                "recovered_after_tool_error": merged.get("recovered_after_tool_error"),
                "llm_calls": merged.get("llm_calls"),
                "embedding_calls": merged.get("embedding_calls"),
                "total_tokens": (merged.get("tokens") or {}).get("total_tokens"),
                "latency_ms": merged.get("latency_ms"),
                "real_mcp_execution_count": merged.get("real_mcp_execution_count"),
                "gold_operations": json.dumps(
                    merged.get("gold_operations") or [], ensure_ascii=False
                ),
                "predicted_operations": json.dumps(
                    merged.get("predicted_operations") or [], ensure_ascii=False
                ),
                "mcp_zero_primary_error": mz.get("mcp_zero_primary_error"),
                "mcp_zero_secondary_errors": json.dumps(
                    mz.get("mcp_zero_secondary_errors") or [], ensure_ascii=False
                ),
                "mcp_zero_error_reason": mz.get("mcp_zero_error_reason"),
                "unified_primary_error_category": tax.get("primary_error_category"),
            }
        )
        if merged.get("status") == "failed":
            failed_rows.append(audit_rows[-1])

    grading_summary = summarize_grading(grades)
    prediction_summary = summarize_predictions(enriched)
    unified_tax = summarize_taxonomy(
        [
            {
                **r,
                "primary_error_category": r.get("unified_primary_error_category"),
                "secondary_error_categories": r.get("unified_secondary_error_categories")
                or [],
            }
            for r in enriched
        ]
    )
    # Override answer_correct on enriched for grouped aggregates that expect it.
    grouped = build_grouped_aggregates(enriched)

    auto = [g for g in grades if g.get("auto_grade_status") == "graded"]
    auto_correct = [g for g in auto if g.get("answer_correct") is True]
    strict_correct_n = sum(1 for r in enriched if r.get("strict_answer_correct"))
    strict_overall_n = sum(1 for r in enriched if r.get("strict_overall_task_success"))
    strict_metrics = {
        "n": EXPECTED_N,
        "auto_gradable_count": len(auto),
        "auto_graded_correct": len(auto_correct),
        "auto_graded_accuracy": (len(auto_correct) / len(auto)) if auto else None,
        "final_answer_accuracy": (len(auto_correct) / len(auto)) if auto else None,
        "final_answer_accuracy_note": (
            "Equals auto_graded_accuracy over auto-gradable tasks; "
            "failed tasks are forced incorrect; manual_required excluded from auto."
        ),
        "strict_answer_accuracy": strict_correct_n / EXPECTED_N,
        "strict_answer_accuracy_note": (
            "answer_correct is True over all 155 tasks; failed forced False; "
            "manual_required treated as incorrect until human confirmation."
        ),
        "strict_overall_task_success": strict_overall_n / EXPECTED_N,
        "strict_overall_task_success_note": (
            "status=full AND answer_correct AND tool_selection_correct AND "
            "all_tool_invocations_successful; denominator 155."
        ),
        "manual_required_count": grading_summary.get("manual_required_count"),
        "manual_required_task_ids": sorted(
            tid
            for tid, g in grades_by_id.items()
            if g.get("auto_grade_status") == "manual_required"
        ),
    }

    grouped_strict = {
        "by_domain": _group_rate(enriched, "domain"),
        "by_difficulty": _group_rate(enriched, "project_difficulty"),
        "by_source_split": _group_rate(enriched, "source_split"),
        "by_answerability": _group_rate(enriched, "answerability_class"),
        "by_company": _group_rate(enriched, "company"),
        "by_domain_combination": _group_rate(enriched, "domain_combination"),
        "by_status": _group_rate(enriched, "status"),
        "by_previously_reserved": _group_rate(
            [
                {
                    **r,
                    "previously_reserved": (
                        "previously_reserved"
                        if r.get("previously_reserved")
                        else "development_origin"
                    ),
                }
                for r in enriched
            ],
            "previously_reserved",
        ),
    }

    tool_metrics = compute_tool_retrieval_protocol(enriched)
    resources = compute_resources(enriched, run_meta=run_meta, cost=cost)

    primary_counts = Counter(r.get("mcp_zero_primary_error") for r in enriched)
    failed_err = Counter(
        str(r.get("error") or r.get("error_type") or "unknown")
        for r in enriched
        if r.get("status") == "failed"
    )
    mz_tax = {
        "taxonomy_version": "mcp_zero_error_taxonomy_v1",
        "primary_counts": dict(primary_counts),
        "failed_raw_error_counts": dict(failed_err),
        "failed_n": sum(1 for r in enriched if r.get("status") == "failed"),
        "labels": list(MCP_ZERO_PRIMARY),
        "note": (
            "Exactly one mcp_zero_primary_error per task; secondaries optional. "
            "RuntimeError/timeout remain in the 155 denominator."
        ),
    }

    # Manual queue: only current manual_required (expected fin_cf_022).
    manual_ids = strict_metrics["manual_required_task_ids"]
    manual_queue: List[Dict[str, Any]] = []
    for tid in manual_ids:
        row = next(r for r in enriched if r["task_id"] == tid)
        ref = load_reference_answer(root, tid) or row.get("reference_answer") or ""
        evidence = row.get("evidence") or row.get("citations") or []
        manual_queue.append(
            {
                "task_id": tid,
                "domain": row.get("domain"),
                "project_difficulty": row.get("project_difficulty"),
                "company": row.get("company"),
                "answerability_class": row.get("answerability_class"),
                "question": row.get("question"),
                "reference_answer": ref,
                "raw_model_answer": _answer_text(row.get("final_answer")),
                "predicted_status": row.get("predicted_status"),
                "status": row.get("status"),
                "gold_operations": json.dumps(
                    row.get("gold_operations") or [], ensure_ascii=False
                ),
                "predicted_operations": json.dumps(
                    row.get("predicted_operations") or [], ensure_ascii=False
                ),
                "evidence_excerpt": json.dumps(evidence, ensure_ascii=False)[:4000],
                "auto_grade_status": grades_by_id[tid].get("auto_grade_status"),
                "auto_grade_reason": grades_by_id[tid].get("reason"),
                "human_confirmed_answer_correct": "",
                "human_confirmed_evidence_correct": "",
                "human_confirmed_overall_task_success": "",
                "human_notes": "",
                "note": (
                    "Do not copy MCP-Free or Flat human grades; grade this "
                    "condition independently."
                ),
            }
        )

    human_grades_template = {
        "condition": "mcp_zero",
        "grader_version": GRADER_VERSION,
        "grading_version": GRADING_VERSION,
        "tasks": {
            tid: {
                "answer_correct": None,
                "evidence_correct": None,
                "overall_task_success": None,
                "notes": "",
                "graded_by": "",
                "graded_at": "",
            }
            for tid in manual_ids
        },
        "note": "Fill only after independent human review for this condition.",
    }

    first20 = enriched[:20]
    first20_strict = sum(1 for r in first20 if r.get("strict_answer_correct")) / 20.0
    sample_note = {
        "first20_strict_answer_accuracy": first20_strict,
        "full155_strict_answer_accuracy": strict_metrics["strict_answer_accuracy"],
        "interpretation": (
            "Locked-order prefix composition differs from the full 155 mix; "
            "do not interpret the gap as model degradation."
        ),
    }

    descriptive_baselines = {
        "mcp_free_final_accuracy": MCP_FREE_FINAL_ACCURACY,
        "flat_final_accuracy": FLAT_FINAL_ACCURACY,
        "flat_tokens": FLAT_TOKENS,
        "flat_mcp_executions": FLAT_MCP_EXECS,
        "flat_exact_tool_set_approx": FLAT_EXACT_TOOL_SET,
        "flat_operation_f1_approx": FLAT_OP_F1,
        "mcp_zero_strict_answer_accuracy": strict_metrics["strict_answer_accuracy"],
        "mcp_zero_tokens": resources["total_tokens"],
        "mcp_zero_mcp_executions": resources["total_real_mcp_executions"],
        "mcp_zero_exact_tool_set_rate": tool_metrics["exact_tool_set_rate"],
        "mcp_zero_mean_operation_f1": tool_metrics["mean_operation_f1"],
        "note": "Descriptive only; no statistical significance claims.",
    }

    ctx = {
        "audit_version": AUDIT_VERSION,
        "audited_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "identity_validation": identity,
        "grading": grading_summary,
        "strict_metrics": strict_metrics,
        "prediction_summary": prediction_summary,
        "grouped_aggregates": grouped,
        "grouped_strict": grouped_strict,
        "tool_retrieval_protocol": tool_metrics,
        "resources": resources,
        "mcp_zero_error_taxonomy": mz_tax,
        "unified_error_taxonomy": unified_tax,
        "sample_composition_note": sample_note,
        "descriptive_baselines": descriptive_baselines,
        "evaluator": evaluator_identity(),
        "grader": grading_identity(),
        "unified_taxonomy_version": TAXONOMY_VERSION,
        "evaluator_version": EVALUATOR_VERSION,
    }
    report_md = render_markdown_report(ctx)

    # Write artifacts
    atomic_write_json(out_dir / "evaluation_metadata.json", {
        "audit_version": AUDIT_VERSION,
        "identity": identity,
        "evaluator": evaluator_identity(),
        "grader": grading_identity(),
        "expected_prompt_hash": EXPECTED_PROMPT_HASH,
        "expected_method_hash": EXPECTED_METHOD_HASH,
    })
    atomic_write_json(out_dir / "grading_summary.json", {**grading_summary, **strict_metrics})
    atomic_write_json(out_dir / "strict_metrics.json", strict_metrics)
    atomic_write_json(out_dir / "grouped_strict_metrics.json", grouped_strict)
    atomic_write_json(out_dir / "grouped_aggregates.json", grouped)
    atomic_write_json(out_dir / "tool_retrieval_protocol_metrics.json", tool_metrics)
    atomic_write_json(out_dir / "resource_summary.json", resources)
    atomic_write_json(out_dir / "mcp_zero_error_taxonomy_summary.json", mz_tax)
    atomic_write_json(out_dir / "unified_error_taxonomy_summary.json", unified_tax)
    atomic_write_json(out_dir / "sample_composition_note.json", sample_note)
    atomic_write_json(out_dir / "descriptive_baseline_comparison.json", descriptive_baselines)
    atomic_write_json(out_dir / "human_grades.json", human_grades_template)
    atomic_write_json(out_dir / "audit_run_result.json", {
        "ok": True,
        "audit_version": AUDIT_VERSION,
        "n": EXPECTED_N,
        "manual_required": manual_ids,
        "strict_answer_accuracy": strict_metrics["strict_answer_accuracy"],
        "auto_graded_accuracy": grading_summary.get("auto_graded_accuracy"),
        "content_fingerprint": _content_fingerprint(
            {
                "strict": strict_metrics,
                "tax": mz_tax,
                "tool": {
                    k: tool_metrics[k]
                    for k in (
                        "exact_tool_set_rate",
                        "mean_operation_f1",
                        "task_protocol_success_rate",
                    )
                },
            }
        ),
    })

    _write_csv(
        out_dir / "per_task_audit.csv",
        audit_rows,
        fieldnames=list(audit_rows[0].keys()) if audit_rows else ["task_id"],
    )
    _write_csv(
        out_dir / "failed_task_audit.csv",
        failed_rows,
        fieldnames=list(failed_rows[0].keys()) if failed_rows else ["task_id"],
    )
    _write_csv(
        out_dir / "manual_review_queue.csv",
        manual_queue,
        fieldnames=list(manual_queue[0].keys()) if manual_queue else ["task_id"],
    )
    (out_dir / "report.md").write_text(report_md, encoding="utf-8")

    post = fingerprint_raw_artifacts(run_dir)
    atomic_write_json(out_dir / "raw_artifact_fingerprints_post.json", post)
    if pre != post:
        raise RuntimeError("RAW ARTIFACTS CHANGED DURING AUDIT — aborting")
    atomic_write_json(
        out_dir / "raw_artifact_fingerprints.json",
        {"pre": pre, "post": post, "unchanged": True},
    )

    if docs_mirror_dir is not None:
        docs_mirror_dir = Path(docs_mirror_dir)
        docs_mirror_dir.mkdir(parents=True, exist_ok=True)
        for name in (
            "report.md",
            "grading_summary.json",
            "strict_metrics.json",
            "grouped_strict_metrics.json",
            "tool_retrieval_protocol_metrics.json",
            "resource_summary.json",
            "mcp_zero_error_taxonomy_summary.json",
            "descriptive_baseline_comparison.json",
            "sample_composition_note.json",
            "evaluation_metadata.json",
            "audit_run_result.json",
            "raw_artifact_fingerprints.json",
            "manual_review_queue.csv",
            "human_grades.json",
            "per_task_audit.csv",
            "failed_task_audit.csv",
        ):
            src = out_dir / name
            if src.exists():
                (docs_mirror_dir / name).write_bytes(src.read_bytes())
        (docs_mirror_dir / "README.md").write_text(
            "# HK-155 MCP-Zero Final Audit v1\n\n"
            "Offline mirror of `outputs/hk_unified_mcp_zero_exploratory155_v2/"
            "final_audit_mcp_zero_v1/`.\n\n"
            "Reproduce:\n\n"
            "```bash\n"
            "PYTHONPATH=src python3 scripts/run_hk_unified_mcp_zero_final_audit_v1.py \\\n"
            "  --run-dir outputs/hk_unified_mcp_zero_exploratory155_v2\n"
            "```\n\n"
            "Does not modify raw predictions. No LLM/Embedding/MCP calls.\n"
            "Manual review: fill `human_grades.json` / `manual_review_queue.csv` "
            "for `fin_cf_022` only after independent human grading.\n",
            encoding="utf-8",
        )

    return {
        "ok": True,
        "audit_version": AUDIT_VERSION,
        "out_dir": str(out_dir),
        "raw_sha256": {k: v["sha256"] for k, v in pre.items()},
        "strict_answer_accuracy": strict_metrics["strict_answer_accuracy"],
        "auto_graded_accuracy": grading_summary.get("auto_graded_accuracy"),
        "manual_required": manual_ids,
        "failed_n": mz_tax["failed_n"],
        "raw_unchanged": True,
    }
