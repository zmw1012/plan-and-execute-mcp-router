"""Offline final audit for HK-155 MCP-Free (closed-book) exploratory run.

Writes ONLY under <run_dir>/final_audit_mcp_free_v1/ (and optional docs mirror).
Never modifies predictions.jsonl / run_metadata.json / cost / grading raw files.
Never calls LLM / Embedding / MCP / paid APIs.

Condition-aware: catalog_fingerprint MUST be null for mcp_free.
Does not change Operation-Visible final_audit_v2 semantics.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from hk_equity_mcp.experiment.normalize import normalize_final_payload
from hk_equity_mcp.experiment.unified_final_audit import (  # noqa: F401
    load_jsonl,
    sha256_file,
)
from hk_equity_mcp.experiment.unified_grading import (
    GRADING_VERSION,
    GRADER_VERSION,
    grade_task,
    summarize_grading,
)
from hk_equity_mcp.experiment.unified_mcp_free_metrics import (
    TOOL_NA,
    _is_hallucinated_exact_value,
    apply_mcp_free_tool_na,
    build_mcp_free_grouped_aggregates,
    evaluate_mcp_free_task,
)

# Re-export for tests / callers.
__all__ = (
    "run_mcp_free_final_audit",
    "assert_mcp_free_run_identity",
    "try_parse_mcp_free_content",
    "sha256_file",
    "TOOL_NA",
)
from hk_equity_mcp.experiment.unified_mcp_free_prompt import unified_mcp_free_prompt_hash
from hk_equity_mcp.experiment.unified_runner import atomic_write_json, fsync_file

AUDIT_VERSION = "mcp_free_final_audit_v1"
EXPECTED_QUESTION_COUNT = 155
EXPECTED_MODEL = "deepseek-v4.1-flash"
EXPECTED_PROVIDER = "openai_compatible"
EXPECTED_HOST = "api.zhizengzeng.com"
EXPECTED_PROMPT_HASH = "89f91313ed782f85a388d0e212438f376876ed03b039e43b92fc5760b73bc5dc"
EXPECTED_QUESTIONS_HASH = (
    "0c078b977a8d275ed3dc4a2486184305dd59da295da0d491c24fd386372f300e"
)
CATALOG_NOTE = "none — MCP-Free has no operation catalog"

RAW_ARTIFACT_FILES = (
    "predictions.jsonl",
    "run_metadata.json",
    "per_task_metrics.csv",
    "manual_grading.csv",
    "cost_summary.json",
    "error_summary.json",
    "grouped_aggregates.json",
)

MCP_FREE_TAXONOMY = (
    "correct",
    "unavailable_correct",
    "unavailable_when_answerable",
    "incorrect_answer",
    "hallucinated_exact_value",
    "clarification_error",
    "parse_evaluation_failure",
)

# Generic: strip fences then json.loads; then truncated-object field recovery.
_STATUS_RE = re.compile(
    r'"status"\s*:\s*"(answered|unavailable|clarification_needed)"', re.I
)
_FINAL_ANSWER_RE = re.compile(
    r'"final_answer"\s*:\s*"((?:\\.|[^"\\])*)"', re.S
)


def _strip_fences(text: str) -> str:
    text = (text or "").strip()
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def try_parse_mcp_free_content(content: str) -> Dict[str, Any]:
    """Deterministic offline parser (no LLM). Same rules for every task.

    Returns:
      ok, method, predicted_status, final_answer, error
    """
    raw = content if content is not None else ""
    text = _strip_fences(raw)
    if not text.strip():
        return {
            "ok": False,
            "method": None,
            "predicted_status": None,
            "final_answer": "",
            "error": "empty_content",
            "parsed_obj": None,
        }
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            status, answer = normalize_final_payload(obj)
            return {
                "ok": True,
                "method": "json.loads",
                "predicted_status": status,
                "final_answer": answer,
                "error": None,
                "parsed_obj": obj,
            }
    except Exception as exc:  # noqa: BLE001
        json_err = type(exc).__name__
    else:
        json_err = "non_object_json"

    # Truncated JSON recovery: only extract completed quoted fields already present.
    # Does not invent missing characters inside unterminated strings.
    status_m = _STATUS_RE.search(text)
    answer_m = _FINAL_ANSWER_RE.search(text)
    if status_m and answer_m:
        status = status_m.group(1).lower()
        try:
            answer = json.loads(f'"{answer_m.group(1)}"')
        except Exception:  # noqa: BLE001
            answer = answer_m.group(1).encode("utf-8").decode("unicode_escape")
        return {
            "ok": True,
            "method": "truncated_completed_fields",
            "predicted_status": status,
            "final_answer": answer,
            "error": None,
            "parsed_obj": {"status": status, "final_answer": answer},
            "json_loads_error": json_err,
        }

    return {
        "ok": False,
        "method": None,
        "predicted_status": None,
        "final_answer": text,
        "error": json_err or "unparseable",
        "parsed_obj": None,
    }


def audit_parse_from_trace(row: Dict[str, Any]) -> Dict[str, Any]:
    """Inspect saved trace contents only (no re-query)."""
    trace = row.get("trace") or []
    turns: List[Dict[str, Any]] = []
    best: Optional[Dict[str, Any]] = None
    for event in trace:
        if not isinstance(event, dict) or event.get("type") != "llm_turn":
            continue
        content = event.get("content")
        # Prefer string content; never invent when missing.
        if content is None:
            parsed = {
                "ok": False,
                "method": None,
                "predicted_status": None,
                "final_answer": "",
                "error": "missing_raw_content",
                "parsed_obj": None,
            }
        else:
            parsed = try_parse_mcp_free_content(str(content))
        turn_info = {
            "turn": event.get("turn"),
            "parse_repair": bool(event.get("parse_repair")),
            "recorded_parse_error": event.get("parse_error"),
            "content_len": len(str(content or "")),
            "content_saved": content is not None,
            **{k: parsed[k] for k in ("ok", "method", "error", "predicted_status")},
            "final_answer_preview": (parsed.get("final_answer") or "")[:160],
        }
        turns.append(turn_info)
        if parsed.get("ok"):
            best = parsed
    run_error = str(row.get("error") or "")
    is_parse_failure = run_error.startswith("parse_error") or (
        str(row.get("status") or "").lower() == "failed"
        and "parse" in run_error.lower()
    )
    return {
        "task_id": row.get("task_id") or row.get("id"),
        "domain": row.get("domain"),
        "run_status": row.get("status"),
        "run_error": row.get("error"),
        "is_parse_failure": bool(is_parse_failure or run_error.startswith("parse_error")),
        "raw_response_saved": any(t.get("content_saved") for t in turns),
        "error_stage": "parse" if run_error.startswith("parse_error") else (
            "runtime" if row.get("error") else "none"
        ),
        "turns": turns,
        "offline_recoverable": bool(best and best.get("ok")),
        "recovery_method": (best or {}).get("method"),
        "recovered_predicted_status": (best or {}).get("predicted_status"),
        "recovered_final_answer": (best or {}).get("final_answer"),
        "needs_manual_scoring": False,  # filled by caller when appropriate
        "formal_treatment_if_unrecoverable": "incorrect_evaluation_failure",
    }


def classify_mcp_free_error(row: Dict[str, Any]) -> Dict[str, Any]:
    """MCP-Free answer/refusal taxonomy. Never labels zero tools as tool failure."""
    status = str(row.get("status") or "").lower()
    err = str(row.get("error") or "")
    pred = str(row.get("predicted_status") or "").lower()
    gold = str(row.get("gold_status") or "").lower()
    auto = row.get("auto_grade_status")
    correct = row.get("answer_correct")
    reason = str(row.get("auto_grade_reason") or row.get("reason") or "")

    if status == "failed" or err.startswith("parse_error"):
        primary = "parse_evaluation_failure"
    elif auto == "manual_required":
        primary = "manual_required"
    elif correct is True:
        if pred == "unavailable" and gold in {"unavailable", "clarification_needed"}:
            primary = "unavailable_correct"
        else:
            primary = "correct"
    elif _is_hallucinated_exact_value(row) or "fabricat" in reason.lower():
        primary = "hallucinated_exact_value"
    elif pred == "unavailable" and gold not in {
        "unavailable",
        "clarification_needed",
        "",
    }:
        # Answerable gold but model refused.
        if gold in {"answered", "partial"} or (
            row.get("answerability_class")
            and "unanswerable" not in str(row.get("answerability_class")).lower()
            and "infeasible" not in str(row.get("answerability_class")).lower()
            and "refuse" not in str(row.get("answerability_class")).lower()
        ):
            primary = "unavailable_when_answerable"
        else:
            primary = "incorrect_answer"
    elif pred == "clarification_needed" and gold not in {
        "clarification_needed",
        "unavailable",
    }:
        primary = "clarification_error"
    elif pred == "clarification_needed" and correct is False:
        primary = "clarification_error"
    else:
        primary = "incorrect_answer"

    return {
        "error_taxonomy": primary,
        "error_taxonomy_version": "mcp_free_error_taxonomy_v1",
        "tool_failure": False,
        "tool_failure_note": "MCP-Free: zero tool calls are not tool failures",
    }


def _percentile(sorted_vals: Sequence[float], p: float) -> Optional[float]:
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = int(math.floor(k))
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return float(sorted_vals[f])
    return float(sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f))


def _tool_zero_checks(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    violations: List[str] = []
    for r in rows:
        tid = r.get("task_id") or r.get("id")
        for key in (
            "embedding_calls",
            "embedding_tokens",
            "real_mcp_execution_count",
            "underlying_operation_calls",
        ):
            if int(r.get(key) or 0) != 0:
                violations.append(f"{tid}:{key}={r.get(key)}")
        resolver = r.get("resolver_calls") or {}
        if isinstance(resolver, dict):
            for k, v in resolver.items():
                if int(v or 0) != 0:
                    violations.append(f"{tid}:resolver.{k}={v}")
        for key in (
            "operation_precision",
            "operation_recall",
            "operation_f1",
            "exact_tool_set",
            "tool_execution_success",
            "all_tool_invocations_successful",
        ):
            val = r.get(key)
            if val is True or val == 1.0 or val == 1:
                violations.append(f"{tid}:{key}=vacuous_true({val!r})")
            elif val not in (TOOL_NA, "N/A", None) and not (
                isinstance(val, str) and val.upper() == "N/A"
            ):
                # Allow only N/A for tool accuracy fields.
                if key.startswith("operation_") or key in {
                    "exact_tool_set",
                    "tool_execution_success",
                    "all_tool_invocations_successful",
                }:
                    if val not in (TOOL_NA, "N/A"):
                        violations.append(f"{tid}:{key}={val!r}")
    return {
        "ok": not violations,
        "violations": violations[:50],
        "violation_count": len(violations),
        "tool_metrics": TOOL_NA,
        "embedding_calls": 0,
        "resolver_calls": 0,
        "real_mcp_executions": 0,
        "tool_calls": 0,
    }


def assert_mcp_free_run_identity(run_meta: Dict[str, Any]) -> None:
    if run_meta.get("condition") != "mcp_free":
        raise ValueError(f"condition != mcp_free: {run_meta.get('condition')}")
    if run_meta.get("execution_backend") != "none":
        raise ValueError(
            f"execution_backend != none: {run_meta.get('execution_backend')}"
        )
    if run_meta.get("catalog_fingerprint") is not None:
        raise ValueError(
            "MCP-Free requires catalog_fingerprint=null; "
            f"got {run_meta.get('catalog_fingerprint')!r}"
        )
    note = run_meta.get("catalog_fingerprint_note") or ""
    if "MCP-Free" not in str(note) and "no operation catalog" not in str(note).lower():
        # Soft: record but do not hard-fail older notes if fingerprint null.
        pass
    if run_meta.get("embedding_model") not in (None, "", "null"):
        raise ValueError(f"embedding_model must be null: {run_meta.get('embedding_model')}")
    if run_meta.get("generation_model") != EXPECTED_MODEL:
        raise ValueError(f"Unexpected model: {run_meta.get('generation_model')}")
    if run_meta.get("generation_provider") != EXPECTED_PROVIDER:
        raise ValueError(
            f"Unexpected provider: {run_meta.get('generation_provider')}"
        )
    host = run_meta.get("provider_host") or run_meta.get("generation_base_url_host")
    if host != EXPECTED_HOST:
        raise ValueError(f"Unexpected provider_host: {host}")
    if int(run_meta.get("question_count") or 0) != EXPECTED_QUESTION_COUNT:
        raise ValueError("question_count != 155")
    if run_meta.get("run_status") != "completed":
        raise ValueError(f"run_status != completed: {run_meta.get('run_status')}")
    if run_meta.get("system_prompt_hash") != EXPECTED_PROMPT_HASH:
        # Allow match to live frozen hash function as soft check.
        live_hash = unified_mcp_free_prompt_hash()
        if run_meta.get("system_prompt_hash") != live_hash:
            raise ValueError(
                f"system_prompt_hash mismatch: {run_meta.get('system_prompt_hash')}"
            )
    qh = run_meta.get("questions_identity_hash")
    if qh and qh != EXPECTED_QUESTIONS_HASH:
        raise ValueError(f"questions_identity_hash mismatch: {qh}")


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    fsync_file(path)


def _md_escape(text: Any) -> str:
    return str(text).replace("|", "\\|")


def build_audit_report_md(summary: Dict[str, Any]) -> str:
    m = summary.get("metrics") or {}
    parse = summary.get("parse_error_audit") or {}
    lines = [
        "# HK-155 MCP-Free Final Audit v1",
        "",
        f"- Audit version: `{AUDIT_VERSION}`",
        f"- Generated (UTC): `{summary.get('generated_at')}`",
        f"- Run dir: `{summary.get('run_dir')}`",
        f"- Condition: `mcp_free` / execution_backend=`none`",
        f"- Provider: `{summary.get('generation_provider')}`",
        f"- Provider host: `{summary.get('provider_host')}`",
        f"- Model: `{summary.get('generation_model')}`",
        f"- Catalog fingerprint: `null` ({CATALOG_NOTE})",
        "",
        "## Accuracy",
        "",
        f"- Strict accuracy (denominator 155): `{m.get('strict_accuracy')}`",
        f"- Auto-gradable accuracy (denominator 154): `{m.get('auto_gradable_accuracy')}`",
        f"- Auto correct / auto gradable: `{m.get('auto_graded_correct')}/{m.get('auto_gradable_count')}`",
        f"- Manual required: `{m.get('manual_required_count')}`",
        f"- Overall accuracy (manual pending → null): `{m.get('overall_accuracy')}`",
        f"- Conservative lower bound correct_confirmed/155: `{m.get('conservative_lower_bound_accuracy')}`",
        "",
        "## Status / failure rates",
        "",
        f"- Unavailable rate: `{m.get('unavailable_rate')}`",
        f"- Answered rate: `{m.get('answered_rate')}`",
        f"- Clarification rate: `{m.get('clarification_rate')}`",
        f"- Hallucinated exact-value rate: `{m.get('hallucinated_exact_value_rate')}`",
        f"- Parse failure rate: `{m.get('parse_failure_rate')}`",
        "",
        "## Parse-error task",
        "",
        f"- task_id: `{parse.get('task_id')}`",
        f"- domain: `{parse.get('domain')}`",
        f"- raw response saved: `{parse.get('raw_response_saved')}`",
        f"- error stage: `{parse.get('error_stage')}`",
        f"- offline recoverable: `{parse.get('offline_recoverable')}`",
        f"- recovery method: `{parse.get('recovery_method')}`",
        f"- needs manual scoring: `{parse.get('needs_manual_scoring')}`",
        f"- formal treatment: `{parse.get('formal_treatment')}`",
        "",
        "## Cost / latency",
        "",
        f"- Input tokens: `{m.get('total_input_tokens')}`",
        f"- Output tokens: `{m.get('total_output_tokens')}`",
        f"- Total tokens: `{m.get('total_tokens')}`",
        f"- LLM calls: `{m.get('total_llm_calls')}`",
        f"- Mean / median / p95 latency_ms: "
        f"`{m.get('mean_latency_ms')}` / `{m.get('median_latency_ms')}` / `{m.get('p95_latency_ms')}`",
        f"- Estimated USD cost: `{m.get('estimated_usd_cost')}` "
        f"({m.get('estimated_usd_cost_note')})",
        "",
        "## Tool / embedding / MCP",
        "",
        "- tool metrics: **N/A** (not vacuous 1.0)",
        "- embedding calls: **0**",
        "- resolver calls: **0**",
        "- real MCP executions: **0**",
        "- tool calls: **0**",
        "",
        "## Notes",
        "",
        "- No LLM / Embedding / MCP called during this audit.",
        "- Raw predictions.jsonl and run artifacts were not modified.",
        "- Empty tool sets are not scored as tool accuracy 1.0.",
        "",
    ]
    return "\n".join(lines)


def run_mcp_free_final_audit(
    run_dir: Path,
    *,
    audit_outdir: Optional[Path] = None,
    docs_mirror: Optional[Path] = None,
) -> Dict[str, Any]:
    run_dir = Path(run_dir)
    out_dir = Path(audit_outdir) if audit_outdir else run_dir / "final_audit_mcp_free_v1"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Protect raw artifacts ---
    raw_sha_before: Dict[str, str] = {}
    for name in RAW_ARTIFACT_FILES:
        path = run_dir / name
        if not path.exists():
            raise FileNotFoundError(f"Missing required raw artifact: {path}")
        raw_sha_before[name] = sha256_file(path)

    predictions = load_jsonl(run_dir / "predictions.jsonl")
    if len(predictions) != EXPECTED_QUESTION_COUNT:
        raise ValueError(f"Expected 155 predictions, found {len(predictions)}")
    ids = [str(p.get("task_id") or p.get("id")) for p in predictions]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate task_id in predictions.jsonl")

    run_meta = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert_mcp_free_run_identity(run_meta)

    # Single host/model across all tasks.
    providers = set()
    models = set()
    for row in predictions:
        pm = row.get("provider_metadata") or {}
        providers.add(pm.get("llm_provider") or run_meta.get("generation_provider"))
        models.add(pm.get("llm_model") or run_meta.get("generation_model"))
    if providers - {EXPECTED_PROVIDER, None}:
        raise ValueError(f"Mixed/unexpected providers in predictions: {providers}")
    if models - {EXPECTED_MODEL, None}:
        raise ValueError(f"Mixed/unexpected models in predictions: {models}")

    tool_check = _tool_zero_checks(predictions)
    if not tool_check["ok"]:
        raise ValueError(
            "MCP-Free tool/embedding invariants violated: "
            + "; ".join(tool_check["violations"][:10])
        )

    cost = json.loads((run_dir / "cost_summary.json").read_text(encoding="utf-8"))

    # --- Offline re-grade (read-only overlay; does not write predictions) ---
    audited_rows: List[Dict[str, Any]] = []
    taxonomy_rows: List[Dict[str, Any]] = []
    parse_audits: List[Dict[str, Any]] = []
    parser_delta: List[Dict[str, Any]] = []

    for idx, row in enumerate(predictions, start=1):
        tid = str(row.get("task_id") or row.get("id"))
        # Re-evaluate with MCP-Free N/A overlay using frozen row fields as item/result.
        item = {
            "task_id": tid,
            "domain": row.get("domain"),
            "domain_combination": row.get("domain_combination"),
            "project_difficulty": row.get("project_difficulty"),
            "exact_etom_level": row.get("exact_etom_level"),
            "exact_etom_alignment": row.get("exact_etom_alignment"),
            "answerability_class": row.get("answerability_class"),
            "gold_status": row.get("gold_status"),
            "gold_values": row.get("gold_values"),
            "gold_operations": row.get("gold_operations") or [],
            "reference_answer": row.get("reference_answer"),
            "company": row.get("company"),
            "source_split": row.get("source_split"),
            "previously_reserved": row.get("previously_reserved"),
            "dependency_graph": row.get("dependency_graph") or [],
        }
        result = {
            "final_answer": row.get("final_answer"),
            "predicted_status": row.get("predicted_status"),
            "status": row.get("status"),
            "error": row.get("error"),
            "trace": row.get("trace") or [],
            "tokens": row.get("tokens") or {},
            "llm_calls": row.get("llm_calls"),
            "parse_repair_count": row.get("parse_repair_count"),
            "latency_ms": row.get("latency_ms"),
            "embedding_calls": 0,
            "embedding_tokens": 0,
            "underlying_operation_calls": 0,
            "real_mcp_execution_count": 0,
            "resolver_calls": row.get("resolver_calls")
            or {"find_operations": 0, "describe_operation": 0},
        }
        metrics = evaluate_mcp_free_task(item=item, result=result)
        metrics = apply_mcp_free_tool_na(metrics)
        grade = grade_task(item, {**result, **metrics, "task_id": tid})

        # Prefer frozen postprocess grade when present (audit consistency with user numbers).
        frozen_auto = row.get("auto_grade_status")
        frozen_correct = row.get("answer_correct")
        frozen_reason = row.get("auto_grade_reason")
        use_auto = frozen_auto if frozen_auto is not None else grade.get("auto_grade_status")
        use_correct = (
            frozen_correct if frozen_correct is not None else grade.get("answer_correct")
        )
        use_reason = frozen_reason if frozen_reason is not None else grade.get("reason")

        merged = dict(row)
        merged.update(metrics)
        merged["auto_grade_status"] = use_auto
        merged["answer_correct"] = use_correct
        merged["auto_grade_reason"] = use_reason
        merged["reeval_auto_grade_status"] = grade.get("auto_grade_status")
        merged["reeval_answer_correct"] = grade.get("answer_correct")
        merged["reeval_auto_grade_reason"] = grade.get("reason")
        merged["task_order_index"] = idx

        parse_info = audit_parse_from_trace(row)
        parse_audits.append(parse_info)

        # Optional offline recovered grade (report-only; does not alter formal metrics).
        recovered_grade = None
        if parse_info.get("offline_recoverable") and (
            parse_info.get("is_parse_failure")
            or str(row.get("error") or "").startswith("parse_error")
        ):
            rec_result = dict(result)
            rec_result["predicted_status"] = parse_info.get("recovered_predicted_status")
            rec_result["final_answer"] = parse_info.get("recovered_final_answer")
            rec_result["status"] = "full"
            rec_result["error"] = None
            rec_metrics = evaluate_mcp_free_task(item=item, result=rec_result)
            recovered_grade = grade_task(
                item, {**rec_result, **rec_metrics, "task_id": tid}
            )
            parser_delta.append(
                {
                    "task_id": tid,
                    "baseline_auto_grade_status": use_auto,
                    "baseline_answer_correct": use_correct,
                    "recovered_auto_grade_status": recovered_grade.get("auto_grade_status"),
                    "recovered_answer_correct": recovered_grade.get("answer_correct"),
                    "recovery_method": parse_info.get("recovery_method"),
                    "answer_correct_changed": use_correct
                    != recovered_grade.get("answer_correct"),
                }
            )

        tax = classify_mcp_free_error(merged)
        # Parse failures always taxonomy parse_evaluation_failure for formal label.
        if str(row.get("error") or "").startswith("parse_error"):
            tax["error_taxonomy"] = "parse_evaluation_failure"
            if not parse_info.get("offline_recoverable"):
                parse_info["needs_manual_scoring"] = False
                parse_info["formal_treatment"] = "incorrect_evaluation_failure"
            else:
                # Recoverable offline but formal score still uses frozen failed row
                # unless operator adopts parser delta (not applied to formal metrics).
                parse_info["needs_manual_scoring"] = False
                parse_info["formal_treatment"] = (
                    "incorrect_evaluation_failure_formal;"
                    "offline_parser_delta_reported_only"
                )
        elif use_auto == "manual_required":
            parse_info["needs_manual_scoring"] = parse_info.get("needs_manual_scoring")

        merged.update(tax)
        merged["parse_offline_recoverable"] = parse_info.get("offline_recoverable")
        merged["parse_recovery_method"] = parse_info.get("recovery_method")
        if recovered_grade is not None:
            merged["recovered_answer_correct"] = recovered_grade.get("answer_correct")
            merged["recovered_auto_grade_status"] = recovered_grade.get(
                "auto_grade_status"
            )

        audited_rows.append(merged)
        taxonomy_rows.append(
            {
                "task_id": tid,
                "domain": merged.get("domain"),
                "project_difficulty": merged.get("project_difficulty"),
                "source_split": merged.get("source_split"),
                "gold_status": merged.get("gold_status"),
                "predicted_status": merged.get("predicted_status"),
                "status": merged.get("status"),
                "error": merged.get("error") or "",
                "auto_grade_status": use_auto,
                "answer_correct": use_correct,
                "auto_grade_reason": use_reason,
                "error_taxonomy": tax["error_taxonomy"],
                "tool_failure": False,
                "llm_calls": merged.get("llm_calls"),
                "latency_ms": merged.get("latency_ms"),
                "input_tokens": (merged.get("tokens") or {}).get("input_tokens"),
                "output_tokens": (merged.get("tokens") or {}).get("output_tokens"),
                "total_tokens": (merged.get("tokens") or {}).get("total_tokens"),
            }
        )

    # --- Formal metrics (frozen grades; 155 / 154 denominators) ---
    n = len(audited_rows)
    auto = [r for r in audited_rows if r.get("auto_grade_status") == "graded"]
    manual = [r for r in audited_rows if r.get("auto_grade_status") == "manual_required"]
    auto_correct = [r for r in auto if r.get("answer_correct") is True]
    confirmed_correct = [r for r in audited_rows if r.get("answer_correct") is True]
    parse_fail = [
        r
        for r in audited_rows
        if str(r.get("error") or "").startswith("parse_error")
        or (
            str(r.get("status") or "").lower() == "failed"
            and "parse" in str(r.get("error") or "").lower()
        )
    ]
    unavailable = sum(
        1
        for r in audited_rows
        if str(r.get("predicted_status") or "").lower() == "unavailable"
    )
    answered = sum(
        1
        for r in audited_rows
        if str(r.get("predicted_status") or "").lower() == "answered"
    )
    clarification = sum(
        1
        for r in audited_rows
        if str(r.get("predicted_status") or "").lower() == "clarification_needed"
    )
    halluc = sum(1 for r in audited_rows if _is_hallucinated_exact_value(r))

    lat = sorted(
        float(r["latency_ms"])
        for r in audited_rows
        if isinstance(r.get("latency_ms"), (int, float))
    )
    total_in = sum(
        int((r.get("tokens") or {}).get("input_tokens") or 0) for r in audited_rows
    )
    total_out = sum(
        int((r.get("tokens") or {}).get("output_tokens") or 0) for r in audited_rows
    )
    total_tok = sum(
        int((r.get("tokens") or {}).get("total_tokens") or 0) for r in audited_rows
    )
    total_llm = sum(int(r.get("llm_calls") or 0) for r in audited_rows)

    # No frozen price table in-repo for zhizengzeng relay → do not guess USD.
    estimated_usd = None
    estimated_note = (
        "No frozen provider price table for openai_compatible/"
        f"{EXPECTED_HOST}; USD cost not estimated."
    )

    manual_complete = all(
        str(r.get("overall_task_success_0_or_1") or r.get("manual_overall_task_success") or "")
        .strip()
        not in {"", "None", "null"}
        for r in manual
    ) if manual else True
    overall_accuracy = None
    if manual_complete and not manual:
        overall_accuracy = len(confirmed_correct) / n if n else None
    elif manual_complete and manual:
        # If manual fields filled, would compute; currently expected null.
        vals = []
        for r in audited_rows:
            raw = r.get("overall_task_success_0_or_1")
            if raw in ("", None):
                raw = r.get("manual_overall_task_success")
            if raw in ("", None) and r.get("answer_correct") is True:
                vals.append(1)
            elif raw in ("", None) and r.get("answer_correct") is False:
                vals.append(0)
            elif raw in ("", None):
                overall_accuracy = None
                vals = []
                break
            else:
                try:
                    vals.append(int(raw))
                except Exception:
                    overall_accuracy = None
                    vals = []
                    break
        if vals and len(vals) == n:
            overall_accuracy = sum(vals) / n

    metrics = {
        "n_tasks": n,
        "strict_accuracy": len(confirmed_correct) / n,  # treats non-True as not correct
        "strict_accuracy_note": (
            "correct_confirmed / 155; manual_required and incorrect count as not correct"
        ),
        "auto_gradable_count": len(auto),
        "auto_graded_correct": len(auto_correct),
        "auto_gradable_accuracy": (len(auto_correct) / len(auto)) if auto else None,
        "manual_required_count": len(manual),
        "manual_required_ids": [r.get("task_id") for r in manual],
        "overall_accuracy": overall_accuracy,
        "overall_accuracy_note": (
            "null until manual_required items are human-confirmed"
            if manual and overall_accuracy is None
            else "computed after manual fields"
        ),
        "conservative_lower_bound_accuracy": len(confirmed_correct) / n,
        "unavailable_rate": unavailable / n,
        "answered_rate": answered / n,
        "clarification_rate": clarification / n,
        "hallucinated_exact_value_rate": halluc / n,
        "hallucinated_exact_value_count": halluc,
        "parse_failure_count": len(parse_fail),
        "parse_failure_rate": len(parse_fail) / n,
        "total_input_tokens": total_in,
        "total_output_tokens": total_out,
        "total_tokens": total_tok,
        "total_llm_calls": total_llm,
        "mean_latency_ms": statistics.mean(lat) if lat else None,
        "median_latency_ms": statistics.median(lat) if lat else None,
        "p95_latency_ms": _percentile(lat, 95),
        "estimated_usd_cost": estimated_usd,
        "estimated_usd_cost_note": estimated_note,
        "cost_summary_echo": {
            "total_input_tokens": cost.get("total_input_tokens"),
            "total_output_tokens": cost.get("total_output_tokens"),
            "total_tokens": cost.get("total_tokens"),
            "total_llm_calls": cost.get("total_llm_calls"),
            "mean_latency_ms": cost.get("mean_latency_ms"),
        },
        "tool_metrics": TOOL_NA,
        "tool_calls": 0,
        "embedding_calls": 0,
        "resolver_calls": 0,
        "real_mcp_executions": 0,
        "predicted_status_counts": dict(
            Counter(str(r.get("predicted_status")) for r in audited_rows)
        ),
        "result_status_counts": dict(
            Counter(str(r.get("status")) for r in audited_rows)
        ),
        "error_taxonomy_counts": dict(
            Counter(r.get("error_taxonomy") for r in audited_rows)
        ),
        "grader_version": GRADER_VERSION,
        "grading_version": GRADING_VERSION,
    }

    grouped = build_mcp_free_grouped_aggregates(audited_rows)

    # Unique parse-error report
    parse_failures = [p for p in parse_audits if p.get("is_parse_failure")]
    primary_parse = parse_failures[0] if parse_failures else None
    if primary_parse is not None and "formal_treatment" not in primary_parse:
        if primary_parse.get("offline_recoverable"):
            primary_parse["formal_treatment"] = (
                "incorrect_evaluation_failure_formal;"
                "offline_parser_delta_reported_only"
            )
        else:
            primary_parse["formal_treatment"] = "incorrect_evaluation_failure"

    # Manual review queue
    manual_queue: List[Dict[str, Any]] = []
    for r in audited_rows:
        if r.get("auto_grade_status") != "manual_required" and not str(
            r.get("error") or ""
        ).startswith("parse_error"):
            continue
        manual_queue.append(
            {
                "task_id": r.get("task_id"),
                "domain": r.get("domain"),
                "project_difficulty": r.get("project_difficulty"),
                "source_split": r.get("source_split"),
                "gold_status": r.get("gold_status"),
                "predicted_status": r.get("predicted_status"),
                "status": r.get("status"),
                "error": r.get("error") or "",
                "auto_grade_status": r.get("auto_grade_status"),
                "answer_correct": r.get("answer_correct"),
                "auto_grade_reason": r.get("auto_grade_reason"),
                "question": r.get("question"),
                "final_answer": r.get("final_answer"),
                "error_taxonomy": r.get("error_taxonomy"),
                "human_confirmed_answer_correct": "",
                "human_confirmed_overall_task_success": "",
                "human_notes": "",
                "queue_reason": (
                    "parse_evaluation_failure"
                    if str(r.get("error") or "").startswith("parse_error")
                    else "manual_required"
                ),
            }
        )

    per_task_fields = [
        "task_id",
        "domain",
        "domain_combination",
        "project_difficulty",
        "exact_etom_level",
        "answerability_class",
        "source_split",
        "company",
        "gold_status",
        "predicted_status",
        "status",
        "error",
        "auto_grade_status",
        "answer_correct",
        "auto_grade_reason",
        "error_taxonomy",
        "operation_precision",
        "operation_recall",
        "operation_f1",
        "exact_tool_set",
        "tool_execution_success",
        "embedding_calls",
        "real_mcp_execution_count",
        "llm_calls",
        "latency_ms",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "parse_offline_recoverable",
        "parse_recovery_method",
    ]
    per_task_csv_rows = []
    for r in audited_rows:
        tokens = r.get("tokens") or {}
        per_task_csv_rows.append(
            {
                **{k: r.get(k) for k in per_task_fields},
                "input_tokens": tokens.get("input_tokens"),
                "output_tokens": tokens.get("output_tokens"),
                "total_tokens": tokens.get("total_tokens"),
            }
        )

    generated_at = datetime.now(timezone.utc).isoformat()
    summary: Dict[str, Any] = {
        "audit_version": AUDIT_VERSION,
        "generated_at": generated_at,
        "run_dir": str(run_dir),
        "condition": "mcp_free",
        "execution_backend": "none",
        "generation_provider": run_meta.get("generation_provider"),
        "provider_host": run_meta.get("provider_host")
        or run_meta.get("generation_base_url_host"),
        "generation_model": run_meta.get("generation_model"),
        "catalog_fingerprint": None,
        "catalog_fingerprint_note": run_meta.get("catalog_fingerprint_note")
        or CATALOG_NOTE,
        "embedding_model": None,
        "system_prompt_hash": run_meta.get("system_prompt_hash"),
        "questions_identity_hash": run_meta.get("questions_identity_hash"),
        "run_order_hash": run_meta.get("run_order_hash"),
        "exploratory_manifest_hash": run_meta.get("exploratory_manifest_hash"),
        "metrics": metrics,
        "tool_invariant_check": tool_check,
        "parse_error_audit": primary_parse,
        "parse_error_task_id": (primary_parse or {}).get("task_id"),
        "parse_error_count": len(parse_failures),
        "parser_delta": {
            "n_changed_answer_correct": sum(
                1 for d in parser_delta if d.get("answer_correct_changed")
            ),
            "rows": parser_delta,
            "note": (
                "Offline deterministic parser applied to saved trace contents only; "
                "formal metrics use frozen predictions grades (no raw mutation)."
            ),
        },
        "grading_summary": summarize_grading(
            [
                {
                    "auto_grade_status": r.get("auto_grade_status"),
                    "answer_correct": r.get("answer_correct"),
                }
                for r in audited_rows
            ]
        ),
        "no_llm_embedding_mcp": True,
        "raw_predictions_modified": False,
        "tasks_rerun": False,
    }

    # --- Write artifacts ---
    atomic_write_json(out_dir / "audit_summary.json", summary)
    atomic_write_json(out_dir / "grouped_metrics.json", grouped)
    (out_dir / "audit_report.md").write_text(
        build_audit_report_md(summary), encoding="utf-8"
    )
    fsync_file(out_dir / "audit_report.md")

    _write_csv(out_dir / "per_task_audited.csv", per_task_csv_rows, per_task_fields)
    _write_csv(
        out_dir / "manual_review_queue.csv",
        manual_queue,
        [
            "task_id",
            "domain",
            "project_difficulty",
            "source_split",
            "gold_status",
            "predicted_status",
            "status",
            "error",
            "auto_grade_status",
            "answer_correct",
            "auto_grade_reason",
            "error_taxonomy",
            "queue_reason",
            "question",
            "final_answer",
            "human_confirmed_answer_correct",
            "human_confirmed_overall_task_success",
            "human_notes",
        ],
    )

    with (out_dir / "error_taxonomy.jsonl").open("w", encoding="utf-8") as handle:
        for row in taxonomy_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        import os

        os.fsync(handle.fileno())

    atomic_write_json(
        out_dir / "parse_error_detail.json",
        {
            "parse_failures": parse_failures,
            "parser_delta": parser_delta,
            "parser_rules": [
                "json.loads after optional markdown fence strip",
                "truncated_completed_fields: extract completed status+final_answer quotes only",
                "never invent missing characters; never call LLM",
                "same rules applied to all 155 tasks' saved trace contents",
            ],
        },
    )

    # Post hashes must match pre hashes for raw artifacts.
    raw_sha_after = {name: sha256_file(run_dir / name) for name in RAW_ARTIFACT_FILES}
    if raw_sha_before != raw_sha_after:
        raise RuntimeError(
            "Raw artifact hash drift detected during audit: "
            + str(
                {
                    k: {"before": raw_sha_before[k], "after": raw_sha_after[k]}
                    for k in RAW_ARTIFACT_FILES
                    if raw_sha_before[k] != raw_sha_after[k]
                }
            )
        )

    hash_payload = {
        "before": raw_sha_before,
        "after": raw_sha_after,
        "unchanged": True,
        "files": list(RAW_ARTIFACT_FILES),
        "audit_version": AUDIT_VERSION,
        "generated_at": generated_at,
    }
    atomic_write_json(out_dir / "raw_artifact_hashes.json", hash_payload)

    if docs_mirror is not None:
        docs_mirror = Path(docs_mirror)
        docs_mirror.mkdir(parents=True, exist_ok=True)
        for name in (
            "audit_report.md",
            "audit_summary.json",
            "per_task_audited.csv",
            "manual_review_queue.csv",
            "grouped_metrics.json",
            "error_taxonomy.jsonl",
            "raw_artifact_hashes.json",
            "parse_error_detail.json",
        ):
            src = out_dir / name
            dst = docs_mirror / name
            dst.write_bytes(src.read_bytes())
            fsync_file(dst)
        readme = docs_mirror / "README.md"
        readme.write_text(
            "\n".join(
                [
                    "# HK-155 MCP-Free Final Audit v1",
                    "",
                    "Tracked mirror of "
                    "`outputs/hk_unified_exploratory155_mcp_free_live_v1/final_audit_mcp_free_v1/`.",
                    "",
                    "Offline only. No LLM / Embedding / MCP. Raw predictions unchanged.",
                    "",
                    "```bash",
                    "PYTHONPATH=src python3 scripts/run_hk_unified_mcp_free_final_audit_v1.py \\",
                    "  --run-dir outputs/hk_unified_exploratory155_mcp_free_live_v1",
                    "```",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        fsync_file(readme)

    return {
        "audit_version": AUDIT_VERSION,
        "out_dir": str(out_dir),
        "docs_mirror": str(docs_mirror) if docs_mirror else None,
        "raw_sha256": raw_sha_before,
        "metrics": metrics,
        "parse_error_task_id": (primary_parse or {}).get("task_id"),
        "raw_unchanged": True,
    }
