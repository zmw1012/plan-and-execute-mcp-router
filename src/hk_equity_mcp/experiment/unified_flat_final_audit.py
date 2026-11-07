"""Offline final audit for HK-155 Flat All Tools exploratory live run.

Writes ONLY under <run_dir>/final_audit_flat_v1/ (and optional docs mirror).
Never modifies predictions.jsonl / run_metadata.json / raw side artifacts.
Never calls LLM / Embedding / MCP / paid APIs.

Flat-aware: retrieval metrics remain N/A; does not force OV catalog/retrieval
assertions. Failed/partial tasks stay in the strict denominator.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from hk_equity_mcp.experiment.unified_evaluator import (
    EVALUATOR_VERSION,
    REFUSAL_CLASSES,
    REFUSAL_STATUSES,
    evaluator_identity,
    extract_invoke_records,
    extract_predicted_operations,
    reevaluate_prediction_row,
)
from hk_equity_mcp.experiment.unified_final_audit import load_jsonl, sha256_file
from hk_equity_mcp.experiment.unified_flat_metrics import (
    RETRIEVAL_NA,
    apply_flat_retrieval_na,
    evaluate_flat_task,
)
from hk_equity_mcp.experiment.unified_flat_prompt import flat_prompt_rules_hash
from hk_equity_mcp.experiment.unified_grading import (
    GRADING_VERSION,
    GRADER_VERSION,
    grade_task,
    grading_identity,
    summarize_grading,
)
from hk_equity_mcp.experiment.unified_runner import atomic_write_json, fsync_file

AUDIT_VERSION = "flat_final_audit_v1"
EXPECTED_QUESTION_COUNT = 155
EXPECTED_CONDITION = "flat"
EXPECTED_ARCHITECTURE = "flat_all_tools"
EXPECTED_BACKEND = "real_mcp"
EXPECTED_PROVIDER = "openai_compatible"
EXPECTED_HOST = "api.zhizengzeng.com"
EXPECTED_MODEL = "deepseek-v4.1-flash"
EXPECTED_PROMPT_HASH = (
    "fdecb85f16b506c6ed0e5e2c8b2c248929bd7342c68d11a2e6ac7f604ed1198c"
)
EXPECTED_IMPL_COMMIT = "9c4c904311d72c78ea169c1ac9e452966366ca66"
EXPECTED_MANIFEST_HASH = (
    "f2257e3de89189c5d5784e3f58824c10928294d9454bcf68b3d061dca37f92c5"
)
EXPECTED_RUN_ORDER_HASH = (
    "14aa3c253eea155b038b70fe11b8a69ee51331b5599790bb2827e72d4e321254"
)
EXPECTED_SCHEMA_BUNDLE = (
    "1e2181d01c2a60e74f026dbeb249d96019a3a52e1e343bb3526e4f914555bc53"
)
EXPECTED_ACTION_MAP = (
    "1381662c108561c7a24920e8dd07e063ebf15d092aa1f47adb2cd89d5b0d850a"
)
EXPECTED_MANUAL_TASKS = ("fin_cf_022",)

RAW_ARTIFACT_FILES = (
    "predictions.jsonl",
    "run_metadata.json",
    "per_task_metrics.csv",
    "manual_grading.csv",
    "cost_summary.json",
    "error_summary.json",
    "grouped_aggregates.json",
    "retrieval_analysis.json",
)

# Known pre-check hashes (cross-check only; audit recomputes).
KNOWN_RAW_SHA256 = {
    "predictions.jsonl": (
        "41bdc25238a3016ffa086bfc567bdae53c71abad7d70372087435107286f4e51"
    ),
    "run_metadata.json": (
        "4e08b6858cec8f46d8df330f17e3aabbd99f367280231a8d7715c3516ecbd336"
    ),
    "error_summary.json": (
        "c3b0d0b4eede32c435cd0621e5c93cfcecf05f34ac81ca45c20f77a757b3363d"
    ),
    "cost_summary.json": (
        "6d11eb41a92712fc9c9845af6e25989acb21d1fd0d7777267666ded1a2cfd45b"
    ),
    "grouped_aggregates.json": (
        "31aaaf095c6ecb221e63eb35ea43efe48d57d4f8ad1c5f28506b3703a812d4bb"
    ),
    "retrieval_analysis.json": (
        "738744c7446f5be27716fea6fef2afd83ffdf2d65db91d9ed181f2c95b02c74e"
    ),
}

FLAT_TAXONOMY = (
    "success",
    "resource_limit_max_invocations",
    "resource_limit_max_llm_turns",
    "protocol_or_parse_failure",
    "wrong_tool_selection",
    "redundant_tool_selection",
    "incomplete_tool_chain",
    "argument_error",
    "dependency_error",
    "tool_execution_failure",
    "answer_synthesis",
    "refusal_or_feasibility_error",
    "evidence_failure",
    "evaluation_failure",
)

TAXONOMY_VERSION = "flat_error_taxonomy_v1"


def _as_set(values: Optional[Sequence[Any]]) -> Set[str]:
    return {str(v) for v in (values or []) if v is not None and str(v).strip()}


def _safe_div(num: float, den: float) -> Optional[float]:
    if den == 0:
        return None
    return num / den


def _mean(vals: Sequence[float]) -> Optional[float]:
    return (sum(vals) / len(vals)) if vals else None


def fingerprint_raw_artifacts(run_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Compute SHA256 / bytes / mtime / row counts for raw artifacts."""
    out: Dict[str, Dict[str, Any]] = {}
    for name in RAW_ARTIFACT_FILES:
        path = Path(run_dir) / name
        if not path.exists():
            raise FileNotFoundError(f"Missing required raw artifact: {path}")
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        st = path.stat()
        info: Dict[str, Any] = {
            "path": str(path),
            "sha256": digest,
            "bytes": len(data),
            "mtime_utc": datetime.fromtimestamp(
                st.st_mtime, tz=timezone.utc
            ).isoformat(),
        }
        if name.endswith(".jsonl"):
            lines = [ln for ln in data.decode("utf-8").splitlines() if ln.strip()]
            ids = []
            for ln in lines:
                row = json.loads(ln)
                ids.append(str(row.get("task_id") or row.get("id")))
            info["row_count"] = len(lines)
            info["unique_task_ids"] = len(set(ids))
        elif name.endswith(".csv"):
            text = data.decode("utf-8")
            rows = list(csv.DictReader(text.splitlines()))
            info["row_count"] = len(rows)
            if rows and "task_id" in rows[0]:
                info["unique_task_ids"] = len(
                    {str(r.get("task_id")) for r in rows if r.get("task_id")}
                )
        known = KNOWN_RAW_SHA256.get(name)
        if known is not None:
            info["known_expected_sha256"] = known
            info["matches_known_precheck"] = known == digest
        out[name] = info
    return out


def assert_known_raw_hashes(fingerprints: Dict[str, Dict[str, Any]]) -> None:
    mismatches = []
    for name, expected in KNOWN_RAW_SHA256.items():
        actual = (fingerprints.get(name) or {}).get("sha256")
        if actual != expected:
            mismatches.append(f"{name}: expected={expected} actual={actual}")
    if mismatches:
        raise RuntimeError(
            "Raw artifact hash mismatch vs known pre-check — STOP. "
            + "; ".join(mismatches)
        )


def assert_flat_run_identity(run_meta: Dict[str, Any]) -> Dict[str, Any]:
    """Fail closed on Flat formal live identity (no OV retrieval assertions)."""
    checks: Dict[str, Any] = {}

    def _eq(key: str, expected: Any) -> None:
        actual = run_meta.get(key)
        ok = actual == expected
        checks[key] = {"expected": expected, "actual": actual, "ok": ok}
        if not ok:
            raise ValueError(f"Flat run identity failed for {key}: {actual!r} != {expected!r}")

    _eq("condition", EXPECTED_CONDITION)
    _eq("architecture", EXPECTED_ARCHITECTURE)
    _eq("execution_backend", EXPECTED_BACKEND)
    _eq("generation_provider", EXPECTED_PROVIDER)
    host = run_meta.get("provider_host") or run_meta.get("generation_base_url_host")
    if host != EXPECTED_HOST:
        raise ValueError(f"Unexpected provider host: {host!r}")
    checks["provider_host"] = {
        "expected": EXPECTED_HOST,
        "actual": host,
        "ok": True,
    }
    _eq("generation_model", EXPECTED_MODEL)
    _eq("embedding_enabled", False)
    _eq("retrieval_enabled", False)
    _eq("system_prompt_hash", EXPECTED_PROMPT_HASH)
    _eq("git_commit", EXPECTED_IMPL_COMMIT)
    _eq("git_dirty", False)
    _eq("run_status", "completed")
    _eq("question_count", EXPECTED_QUESTION_COUNT)
    _eq("exploratory_manifest_hash", EXPECTED_MANIFEST_HASH)
    _eq("run_order_hash", EXPECTED_RUN_ORDER_HASH)
    _eq("schema_bundle_sha256", EXPECTED_SCHEMA_BUNDLE)
    _eq("action_mapping_fingerprint", EXPECTED_ACTION_MAP)
    if not run_meta.get("catalog_fingerprint"):
        raise ValueError("Missing catalog_fingerprint")
    checks["catalog_fingerprint"] = {
        "actual": run_meta.get("catalog_fingerprint"),
        "ok": True,
    }
    return checks


def _selected_ops(row: Dict[str, Any]) -> List[str]:
    ops = row.get("selected_operations")
    if isinstance(ops, list) and ops:
        return [str(x) for x in ops]
    return list(extract_predicted_operations(row.get("trace") or []))


def _invoked_ops(row: Dict[str, Any]) -> List[str]:
    ops = row.get("invoked_operations")
    if isinstance(ops, list) and ops:
        return [str(x) for x in ops]
    recs = extract_invoke_records(row.get("trace") or [])
    out: List[str] = []
    seen: Set[str] = set()
    for r in recs:
        oid = r.get("operation_id")
        if oid and str(oid) not in seen:
            seen.add(str(oid))
            out.append(str(oid))
    return out


def _successful_ops(row: Dict[str, Any]) -> List[str]:
    ops = row.get("successful_invoked_operations")
    if isinstance(ops, list):
        return [str(x) for x in ops]
    recs = extract_invoke_records(row.get("trace") or [])
    return [str(r["operation_id"]) for r in recs if r.get("success") and r.get("operation_id")]


def _invoke_recs(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    recs = row.get("invoke_calls")
    if isinstance(recs, list) and recs:
        return [r for r in recs if isinstance(r, dict)]
    return extract_invoke_records(row.get("trace") or [])


def _has_argument_error(row: Dict[str, Any]) -> bool:
    for item in row.get("invalid_selections") or []:
        if not isinstance(item, dict):
            continue
        et = str(item.get("error_type") or "").lower()
        if "validation" in et or "argument" in et:
            return True
    for r in _invoke_recs(row):
        if r.get("success"):
            continue
        err = str(r.get("error") or r.get("error_type") or "").lower()
        if any(k in err for k in ("validation", "argument", "schema", "missing required")):
            return True
    return False


def _has_tool_execution_failure(row: Dict[str, Any]) -> bool:
    recs = _invoke_recs(row)
    return any(not r.get("success") for r in recs)


def _dependency_issue(row: Dict[str, Any]) -> bool:
    acc = row.get("graph_dependency_accuracy")
    if isinstance(acc, (int, float)) and acc < 1.0:
        return True
    graph = row.get("dependency_graph") or []
    if not graph:
        return False
    pred = _as_set(_invoked_ops(row) or _selected_ops(row))
    for edge in graph:
        if not isinstance(edge, dict):
            continue
        producer = edge.get("producer_operation") or edge.get("from_operation")
        consumer = edge.get("consumer_operation") or edge.get("to_operation")
        if producer and consumer:
            if str(producer) not in pred or str(consumer) not in pred:
                return True
    return False


def _should_refuse(row: Dict[str, Any]) -> bool:
    ac = str(row.get("answerability_class") or "").strip()
    gs = str(row.get("gold_status") or "").strip().lower()
    gold = _as_set(row.get("gold_operations"))
    return (
        ac in REFUSAL_CLASSES
        or gs in {"unavailable", "clarification_needed", "refuse"}
        or (not gold and ac not in {"structured_answerable", "evidence_answerable"})
    )


def classify_flat_error(row: Dict[str, Any]) -> Dict[str, Any]:
    """Evidence-driven Flat taxonomy. No task_id special cases.

    Priority: resource limits first; then protocol; then refusal; then tools;
    then answer synthesis / success.
    """
    secondary: List[str] = []
    status = str(row.get("status") or "")
    error = str(row.get("error") or "").strip()
    auto = str(row.get("auto_grade_status") or "")
    answer_correct = row.get("answer_correct")
    gold = _as_set(row.get("gold_operations"))
    selected = _as_set(_selected_ops(row))
    invoked = _as_set(_invoked_ops(row))
    pred_status = str(row.get("predicted_status") or "").lower()

    # 1) Resource limits (highest priority when present).
    if error == "max_invocations":
        primary = "resource_limit_max_invocations"
        if gold and not (gold & (selected or invoked)):
            secondary.append("wrong_tool_selection")
        elif gold and gold.issubset(selected | invoked) and (selected | invoked) - gold:
            secondary.append("redundant_tool_selection")
        elif gold and not gold.issubset(selected | invoked):
            secondary.append("incomplete_tool_chain")
        return {
            "primary_error_category": primary,
            "secondary_error_categories": secondary,
            "error_taxonomy_version": TAXONOMY_VERSION,
        }
    if error == "max_llm_turns":
        return {
            "primary_error_category": "resource_limit_max_llm_turns",
            "secondary_error_categories": secondary,
            "error_taxonomy_version": TAXONOMY_VERSION,
        }

    # 2) Protocol / parse failures.
    if error.startswith("parse_error") or "parse" in error.lower():
        return {
            "primary_error_category": "protocol_or_parse_failure",
            "secondary_error_categories": secondary,
            "error_taxonomy_version": TAXONOMY_VERSION,
        }
    if status == "partial":
        # Incomplete terminal contract / mid-trajectory stop without final answer object.
        primary = "protocol_or_parse_failure"
        if gold and not gold.issubset(selected | invoked):
            secondary.append("incomplete_tool_chain")
        return {
            "primary_error_category": primary,
            "secondary_error_categories": secondary,
            "error_taxonomy_version": TAXONOMY_VERSION,
            "partial_note": (
                "status=partial: no final_answer tool-free turn; "
                "strict answer correctness treated as incorrect"
            ),
        }

    # 3) Evaluation / manual.
    if auto == "manual_required":
        return {
            "primary_error_category": "evaluation_failure",
            "secondary_error_categories": secondary,
            "error_taxonomy_version": TAXONOMY_VERSION,
        }

    # 4) Refusal / feasibility.
    if _should_refuse(row):
        if pred_status in REFUSAL_STATUSES and answer_correct is True:
            return {
                "primary_error_category": "success",
                "secondary_error_categories": secondary,
                "error_taxonomy_version": TAXONOMY_VERSION,
            }
        # Should refuse but answered / wrong refusal / tools used incorrectly.
        return {
            "primary_error_category": "refusal_or_feasibility_error",
            "secondary_error_categories": secondary,
            "error_taxonomy_version": TAXONOMY_VERSION,
        }

    # Correct path.
    if answer_correct is True and status == "full":
        if row.get("exact_tool_set") is False and gold:
            secondary.append(
                "redundant_tool_selection"
                if gold.issubset(selected | invoked)
                else "incomplete_tool_chain"
            )
        return {
            "primary_error_category": "success",
            "secondary_error_categories": secondary,
            "error_taxonomy_version": TAXONOMY_VERSION,
        }

    # Tool-path diagnosis for incorrect / incomplete answers.
    if _has_argument_error(row):
        primary = "argument_error"
    elif _dependency_issue(row):
        primary = "dependency_error"
    elif gold and not (gold & (selected | invoked)):
        primary = "wrong_tool_selection"
    elif gold and not gold.issubset(selected | invoked):
        primary = "incomplete_tool_chain"
    elif gold and (selected | invoked) - gold and gold.issubset(selected | invoked):
        primary = "redundant_tool_selection"
    elif _has_tool_execution_failure(row):
        primary = "tool_execution_failure"
    elif answer_correct is False and (
        row.get("exact_tool_set") is True
        or (gold and gold.issubset(invoked or selected))
    ):
        primary = "answer_synthesis"
    elif answer_correct is False:
        # Evidence-heavy failures without clear tool miss.
        primary = "evidence_failure" if str(
            row.get("answerability_class") or ""
        ) == "evidence_answerable" else "answer_synthesis"
    else:
        primary = "evaluation_failure"

    if primary != "tool_execution_failure" and _has_tool_execution_failure(row):
        secondary.append("tool_execution_failure")
    if primary != "redundant_tool_selection" and gold and (selected | invoked) - gold:
        if gold.issubset(selected | invoked):
            secondary.append("redundant_tool_selection")
    if primary != "wrong_tool_selection" and gold and (selected | invoked) - gold:
        if not gold.issubset(selected | invoked) and (selected | invoked) - gold:
            secondary.append("wrong_tool_selection")

    return {
        "primary_error_category": primary,
        "secondary_error_categories": sorted(set(secondary)),
        "error_taxonomy_version": TAXONOMY_VERSION,
    }


def _parse_human_flag(value: Any) -> Optional[int]:
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.lower() in {"none", "null", "nan"}:
        return None
    if text.lower() in {"1", "true", "yes", "y"}:
        return 1
    if text.lower() in {"0", "false", "no", "n"}:
        return 0
    try:
        return int(text)
    except Exception:
        return None


def load_human_grades(paths: Sequence[Path]) -> Dict[str, Dict[str, Any]]:
    """Load human confirmations from JSON (and optional CSV). No task-id hardcoding."""
    out: Dict[str, Dict[str, Any]] = {}
    for path in paths:
        if path is None or not Path(path).exists():
            continue
        path = Path(path)
        if path.suffix.lower() == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            rows: Any
            if isinstance(payload, dict) and isinstance(payload.get("by_task_id"), dict):
                rows = [
                    {"task_id": tid, **(grade or {})}
                    for tid, grade in payload["by_task_id"].items()
                ]
            elif isinstance(payload, dict) and isinstance(payload.get("grades"), list):
                rows = payload["grades"]
            elif isinstance(payload, list):
                rows = payload
            else:
                continue
            for row in rows:
                tid = str(row.get("task_id") or "").strip()
                if not tid:
                    continue
                out[tid] = {
                    "human_confirmed_answer_correct": _parse_human_flag(
                        row.get("human_confirmed_answer_correct")
                    ),
                    "human_confirmed_evidence_correct": _parse_human_flag(
                        row.get("human_confirmed_evidence_correct")
                    ),
                    "human_confirmed_overall_success": _parse_human_flag(
                        row.get("human_confirmed_overall_success")
                        if row.get("human_confirmed_overall_success") is not None
                        else row.get("human_confirmed_overall_task_success")
                    ),
                    "reviewer_notes": row.get("reviewer_notes")
                    or row.get("human_notes")
                    or "",
                    "source_file": path.name,
                }
        elif path.suffix.lower() == ".csv":
            with path.open("r", encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    tid = str(row.get("task_id") or "").strip()
                    if not tid:
                        continue
                    ans = _parse_human_flag(row.get("human_confirmed_answer_correct"))
                    ev = _parse_human_flag(row.get("human_confirmed_evidence_correct"))
                    overall = _parse_human_flag(
                        row.get("human_confirmed_overall_success")
                    )
                    notes = row.get("reviewer_notes") or ""
                    if (
                        ans is None
                        and ev is None
                        and overall is None
                        and not str(notes).strip()
                    ):
                        continue
                    out[tid] = {
                        "human_confirmed_answer_correct": ans,
                        "human_confirmed_evidence_correct": ev,
                        "human_confirmed_overall_success": overall,
                        "reviewer_notes": notes,
                        "source_file": path.name,
                    }
    return out


def apply_human_grades(
    rows: List[Dict[str, Any]], human_grades: Dict[str, Dict[str, Any]]
) -> None:
    """Annotate rows in-place with final_* fields from human confirmations only.

    Auto-graded rows take frozen answer_correct. Manual-required rows take values
    only from the human_grades mapping (no task-id hardcoding).
    """
    for row in rows:
        tid = str(row.get("task_id") or row.get("id") or "")
        human = human_grades.get(tid) or {}
        row["human_confirmed_answer_correct"] = human.get(
            "human_confirmed_answer_correct"
        )
        row["human_confirmed_evidence_correct"] = human.get(
            "human_confirmed_evidence_correct"
        )
        row["human_confirmed_overall_success"] = human.get(
            "human_confirmed_overall_success"
        )
        row["reviewer_notes"] = human.get("reviewer_notes") or ""

        auto_status = row.get("auto_grade_status")
        auto_correct = row.get("answer_correct")
        human_ans = human.get("human_confirmed_answer_correct")
        human_ev = human.get("human_confirmed_evidence_correct")
        human_overall = human.get("human_confirmed_overall_success")

        if auto_status == "manual_required":
            if human_ans is None:
                row["final_answer_correct"] = None
                row["final_evidence_correct"] = None
                row["final_overall_task_success"] = None
                continue
            final_correct = bool(int(human_ans) == 1)
            if human_ev is not None:
                final_evidence: Optional[bool] = bool(int(human_ev) == 1)
            else:
                final_evidence = final_correct
            if human_overall is not None:
                final_overall = int(human_overall)
            else:
                final_overall = int(human_ans)
            row["final_answer_correct"] = final_correct
            row["final_evidence_correct"] = final_evidence
            row["final_overall_task_success"] = final_overall
            if final_correct:
                row["primary_error_category"] = "success"
                row["secondary_error_categories"] = sorted(
                    set(
                        list(row.get("secondary_error_categories") or [])
                        + ["evaluation_failure_resolved_by_human"]
                    )
                )
            else:
                row["primary_error_category"] = "evaluation_failure"
            continue

        final_correct = bool(auto_correct is True)
        row["final_answer_correct"] = final_correct
        row["final_evidence_correct"] = None
        if human_overall is not None:
            row["final_overall_task_success"] = int(human_overall)
        elif final_correct and row.get("status") == "full":
            row["final_overall_task_success"] = 1
        else:
            row["final_overall_task_success"] = 0


def strict_answer_correct(row: Dict[str, Any]) -> Optional[bool]:
    """Strict completion policy: failed/partial cannot be answer-correct.

    When final_answer_correct is set (after human grades), that value is used for
    full-status tasks. Unresolved manual_required remains None.
    """
    if row.get("status") in {"failed", "partial"}:
        return False
    if row.get("final_answer_correct") is True:
        return True
    if row.get("final_answer_correct") is False:
        return False
    if row.get("auto_grade_status") == "manual_required":
        return None
    val = row.get("answer_correct")
    if val is True:
        return True
    if val is False:
        return False
    return None


def strict_overall_task_success(row: Dict[str, Any]) -> bool:
    """Strict overall: status=full and final/auto overall success."""
    if row.get("status") != "full":
        return False
    if row.get("final_overall_task_success") is not None:
        return int(row.get("final_overall_task_success") or 0) == 1
    return (
        row.get("answer_correct") is True
        and row.get("auto_grade_status") == "graded"
    )


def _group_answer_stats(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    auto = [r for r in rows if r.get("auto_grade_status") == "graded"]
    auto_correct = [r for r in auto if r.get("answer_correct") is True]
    manual = [r for r in rows if r.get("auto_grade_status") == "manual_required"]
    has_final = bool(rows) and all(
        "final_answer_correct" in r and r.get("final_answer_correct") is not None
        for r in rows
    )
    final_true = [r for r in rows if r.get("final_answer_correct") is True]
    strict_true = [r for r in rows if strict_answer_correct(r) is True]
    overall_true = [r for r in rows if strict_overall_task_success(r)]
    status_counts = Counter(str(r.get("status") or "missing") for r in rows)
    error_counts = Counter(
        str(r.get("error"))
        for r in rows
        if r.get("error") not in (None, "", "None", "null")
    )
    tax_counts = Counter(
        str(r.get("primary_error_category") or "unknown") for r in rows
    )

    def _avg_key(key: str) -> Optional[float]:
        vals = [float(r[key]) for r in rows if isinstance(r.get(key), (int, float))]
        return _mean(vals)

    return {
        "task_count": n,
        "auto_gradable_count": len(auto),
        "auto_graded_correct": len(auto_correct),
        "auto_accuracy": _safe_div(len(auto_correct), len(auto)),
        "manual_required_count": len(manual),
        "manual_task_ids": [str(r.get("task_id")) for r in manual],
        "final_answer_correct_count": len(final_true) if has_final else None,
        "final_answer_accuracy": (
            _safe_div(len(final_true), n) if has_final and n else None
        ),
        "strict_answer_correct_count": len(strict_true),
        "strict_answer_accuracy": _safe_div(len(strict_true), n) if n else None,
        "strict_overall_task_success_count": len(overall_true),
        "strict_overall_task_success_rate": _safe_div(len(overall_true), n)
        if n
        else None,
        "exact_tool_set_rate": _safe_div(
            sum(1 for r in rows if r.get("exact_tool_set") is True), n
        ),
        "mean_operation_precision": _avg_key("operation_precision"),
        "mean_operation_recall": _avg_key("operation_recall"),
        "mean_operation_f1": _avg_key("operation_f1"),
        "mean_wrong_tool_rate": _avg_key("wrong_tool_rate"),
        "all_tool_invocations_successful_rate": _safe_div(
            sum(1 for r in rows if r.get("all_tool_invocations_successful") is True), n
        ),
        "recovered_after_tool_error_rate": _safe_div(
            sum(1 for r in rows if r.get("recovered_after_tool_error") is True), n
        ),
        "mean_llm_calls": _avg_key("llm_calls"),
        "mean_real_mcp_execution_count": _avg_key("real_mcp_execution_count"),
        "mean_latency_ms": _avg_key("latency_ms"),
        "total_llm_calls": sum(int(r.get("llm_calls") or 0) for r in rows),
        "total_real_mcp_executions": sum(
            int(r.get("real_mcp_execution_count") or 0) for r in rows
        ),
        "total_embedding_calls": sum(int(r.get("embedding_calls") or 0) for r in rows),
        "total_retrieval_calls": sum(int(r.get("retrieval_calls") or 0) for r in rows),
        "status_counts": dict(status_counts),
        "error_counts": dict(error_counts),
        "primary_error_counts": dict(tax_counts),
        "retrieval_metrics": RETRIEVAL_NA,
    }


def build_flat_grouped_audit(
    rows: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    out: Dict[str, Any] = {"overall": _group_answer_stats(rows)}
    for key in (
        "domain",
        "project_difficulty",
        "answerability_class",
        "domain_combination",
        "company",
        "source_split",
        "status",
        "primary_error_category",
    ):
        buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in rows:
            val = row.get(key)
            name = "null" if val is None or val == "" else str(val)
            buckets[name].append(row)
        out[key] = {k: _group_answer_stats(v) for k, v in sorted(buckets.items())}

    # Development origin: development vs previously_reserved.
    origin: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("previously_reserved") is True:
            origin["previously_reserved"].append(row)
        else:
            origin["development"].append(row)
    out["development_origin"] = {
        k: _group_answer_stats(v) for k, v in sorted(origin.items())
    }

    # Exact ETOM-aligned subset only.
    etom = [r for r in rows if r.get("exact_etom_alignment") is True]
    out["exact_etom_aligned_subset"] = _group_answer_stats(etom)
    return out


def _tool_selection_analysis(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    exact = sum(1 for r in rows if r.get("exact_tool_set") is True)
    redundant = 0
    missing = 0
    wrong = 0
    for r in rows:
        gold = _as_set(r.get("gold_operations"))
        selected = _as_set(_selected_ops(r))
        if not gold:
            continue
        if selected - gold and gold.issubset(selected):
            redundant += 1
        if gold - selected:
            missing += 1
        if selected - gold and not gold.issubset(selected):
            wrong += 1
    out: Dict[str, Any] = {
        "n_tasks": n,
        "exact_tool_set_rate": exact / n if n else None,
        "mean_operation_precision": _mean(
            [float(r["operation_precision"]) for r in rows if isinstance(r.get("operation_precision"), (int, float))]
        ),
        "mean_operation_recall": _mean(
            [float(r["operation_recall"]) for r in rows if isinstance(r.get("operation_recall"), (int, float))]
        ),
        "mean_operation_f1": _mean(
            [float(r["operation_f1"]) for r in rows if isinstance(r.get("operation_f1"), (int, float))]
        ),
        "mean_wrong_tool_rate": _mean(
            [float(r["wrong_tool_rate"]) for r in rows if isinstance(r.get("wrong_tool_rate"), (int, float))]
        ),
        "all_tool_invocations_successful_rate": _safe_div(
            sum(1 for r in rows if r.get("all_tool_invocations_successful") is True), n
        ),
        "recovered_after_tool_error_rate": _safe_div(
            sum(1 for r in rows if r.get("recovered_after_tool_error") is True), n
        ),
        "redundant_selection_count": redundant,
        "missing_gold_count": missing,
        "wrong_and_incomplete_count": wrong,
        "invocation_failure_tasks": sum(
            1 for r in rows if int(r.get("failed_invocation_count") or 0) > 0
        ),
        "vacuous_execution_success_note": (
            "all_tool_invocations_successful is vacuous True when zero invokes; "
            "this is NOT exact tool-set success."
        ),
        "retrieval_metrics": RETRIEVAL_NA,
        "by_domain": {},
        "by_difficulty": {},
    }
    domain_buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        domain_buckets[str(r.get("domain"))].append(r)
    for d, rs in sorted(domain_buckets.items()):
        out["by_domain"][d] = {
            "n": len(rs),
            "exact_tool_set_rate": _safe_div(
                sum(1 for r in rs if r.get("exact_tool_set") is True), len(rs)
            ),
            "total_real_mcp_executions": sum(
                int(r.get("real_mcp_execution_count") or 0) for r in rs
            ),
            "mean_wrong_tool_rate": _mean(
                [
                    float(r["wrong_tool_rate"])
                    for r in rs
                    if isinstance(r.get("wrong_tool_rate"), (int, float))
                ]
            ),
        }
    diff_buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        diff_buckets[str(r.get("project_difficulty"))].append(r)
    for d, rs in sorted(diff_buckets.items(), key=lambda kv: (kv[0] is None, kv[0])):
        out["by_difficulty"][d] = {
            "n": len(rs),
            "exact_tool_set_rate": _safe_div(
                sum(1 for r in rs if r.get("exact_tool_set") is True), len(rs)
            ),
            "total_real_mcp_executions": sum(
                int(r.get("real_mcp_execution_count") or 0) for r in rs
            ),
        }
    return out


def _resource_analysis(
    rows: Sequence[Dict[str, Any]],
    *,
    run_meta: Dict[str, Any],
    cost: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "total_input_tokens": int(cost.get("total_input_tokens") or 0),
        "total_output_tokens": int(cost.get("total_output_tokens") or 0),
        "total_tokens": int(cost.get("total_tokens") or 0),
        "total_llm_calls": int(cost.get("total_llm_calls") or 0),
        "total_real_mcp_executions": int(cost.get("total_real_mcp_executions") or 0),
        "total_embedding_calls": int(cost.get("total_embedding_calls") or 0),
        "total_retrieval_calls": int(cost.get("total_retrieval_calls") or 0),
        "total_find_operations": int(cost.get("total_find_operations") or 0),
        "total_describe_operation": int(cost.get("total_describe_operation") or 0),
        "mean_latency_ms": cost.get("mean_latency_ms"),
        "schema_bundle_char_count": run_meta.get("serialized_schema_char_count")
        or cost.get("schema_bundle_char_count"),
        "schema_bundle_token_estimate": run_meta.get("schema_bundle_token_estimate")
        or cost.get("schema_bundle_token_estimate"),
        "estimated_usd": None,
        "estimated_usd_note": (
            "No frozen provider price table for openai_compatible/"
            f"{EXPECTED_HOST}; USD cost not estimated."
        ),
        "max_invocations_failures": sum(
            1 for r in rows if r.get("error") == "max_invocations"
        ),
        "max_llm_turns_failures": sum(
            1 for r in rows if r.get("error") == "max_llm_turns"
        ),
        "max_invocations_by_domain": dict(
            Counter(
                str(r.get("domain"))
                for r in rows
                if r.get("error") == "max_invocations"
            )
        ),
        "limits": run_meta.get("limits") or {},
    }


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    fsync_file(path)


def _manual_queue_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for r in rows:
        if r.get("auto_grade_status") != "manual_required":
            continue
        out.append(
            {
                "task_id": r.get("task_id"),
                "domain": r.get("domain"),
                "project_difficulty": r.get("project_difficulty"),
                "answerability_class": r.get("answerability_class"),
                "question": r.get("question"),
                "final_answer": r.get("final_answer"),
                "reference_answer": r.get("reference_answer") or "",
                "gold_specification": json.dumps(
                    {
                        "gold_status": r.get("gold_status"),
                        "gold_values": r.get("gold_values"),
                        "gold_operations": r.get("gold_operations"),
                    },
                    ensure_ascii=False,
                ),
                "evidence": json.dumps(r.get("evidence") or [], ensure_ascii=False),
                "selected_operations": json.dumps(
                    _selected_ops(r), ensure_ascii=False
                ),
                "invoked_operations": json.dumps(_invoked_ops(r), ensure_ascii=False),
                "run_status": r.get("status"),
                "predicted_status": r.get("predicted_status"),
                "error": r.get("error") or "",
                "auto_reason": r.get("auto_grade_reason") or "",
                "human_confirmed_answer_correct": (
                    ""
                    if r.get("human_confirmed_answer_correct") is None
                    else r.get("human_confirmed_answer_correct")
                ),
                "human_confirmed_evidence_correct": (
                    ""
                    if r.get("human_confirmed_evidence_correct") is None
                    else r.get("human_confirmed_evidence_correct")
                ),
                "human_confirmed_overall_success": (
                    ""
                    if r.get("human_confirmed_overall_success") is None
                    else r.get("human_confirmed_overall_success")
                ),
                "reviewer_notes": r.get("reviewer_notes") or "",
            }
        )
    return out


def _build_report_md(ctx: Dict[str, Any]) -> str:
    m = ctx["final_metrics"]
    g = ctx["grouped_metrics"]
    tax = ctx["error_taxonomy"]
    human_applied = bool(m.get("human_grades_applied"))
    lines = [
        "# HK-155 Flat All Tools Final Audit v1",
        "",
        "Offline audit only. No LLM / Embedding / MCP / paid API. "
        "Raw predictions and run_metadata were not modified.",
        "",
        "## Identity",
        f"- Condition / architecture: `{m.get('condition')}` / `{m.get('architecture')}`",
        f"- Provider / host / model: `{m.get('generation_provider')}` / "
        f"`{m.get('provider_host')}` / `{m.get('generation_model')}`",
        f"- Implementation commit: `{m.get('git_commit')}`",
        f"- system_prompt_hash: `{m.get('system_prompt_hash')}`",
        f"- N tasks: **{m.get('n_tasks')}**",
        f"- human_grades_applied: `{human_applied}`",
        "",
        "## Status",
        f"- full / partial / failed: `{m.get('status_counts')}`",
        f"- explicit errors: `{m.get('error_counts')}`",
        "",
        "## Answer scoring",
        f"- auto_gradable: `{m.get('auto_gradable')}`",
        f"- auto correct: `{m.get('auto_correct')}`",
        f"- auto accuracy (do NOT treat as final overall): `{m.get('auto_accuracy')}`",
        f"- manual_required: `{m.get('manual_required_count')}` → `{m.get('manual_task_ids')}`",
        f"- final answer accuracy: `{m.get('final_answer_correct_count')}` / "
        f"`{m.get('n_tasks')}` = `{m.get('final_answer_accuracy')}`",
        f"- overall_accuracy (= final answer accuracy when human closed): "
        f"`{m.get('overall_accuracy')}`",
        f"- conservative lower bound (auto-only correct / 155): "
        f"`{m.get('conservative_lower_bound_accuracy')}`",
        f"- upper bound if manual correct: `{m.get('upper_bound_if_manual_correct')}`",
        f"- strict answer accuracy (failed/partial forced false): "
        f"`{m.get('strict_answer_correct_count')}` / `{m.get('n_tasks')}` = "
        f"`{m.get('strict_answer_accuracy')}`",
        f"- strict overall task success: "
        f"`{m.get('strict_overall_task_success_count')}` / `{m.get('n_tasks')}` = "
        f"`{m.get('strict_overall_task_success_rate')}`",
        "",
        "## Retrieval",
        f"- retrieval_metrics: `{RETRIEVAL_NA}` (not 1.0)",
        f"- embedding / find / describe calls: `0`",
        "",
        "## Tool selection / execution",
        f"- exact tool-set: `{m.get('exact_tool_set_rate')}`",
        f"- operation P/R/F1: `{m.get('mean_operation_precision')}` / "
        f"`{m.get('mean_operation_recall')}` / `{m.get('mean_operation_f1')}`",
        f"- wrong-tool rate: `{m.get('mean_wrong_tool_rate')}`",
        f"- all invocations successful rate: "
        f"`{m.get('all_tool_invocations_successful_rate')}`",
        f"- recovered after tool error rate: "
        f"`{m.get('recovered_after_tool_error_rate')}`",
        "",
        "## Domain final / auto accuracy",
    ]
    for dom, stats in (g.get("domain") or {}).items():
        lines.append(
            f"- {dom}: final={stats.get('final_answer_correct_count')}/"
            f"{stats.get('task_count')}={stats.get('final_answer_accuracy')}; "
            f"auto={stats.get('auto_graded_correct')}/{stats.get('auto_gradable_count')} "
            f"= {stats.get('auto_accuracy')}; manual={stats.get('manual_required_count')}"
        )
    lines.extend(["", "## Difficulty final / auto accuracy"])
    for d, stats in (g.get("project_difficulty") or {}).items():
        lines.append(
            f"- {d}: final={stats.get('final_answer_correct_count')}/"
            f"{stats.get('task_count')}={stats.get('final_answer_accuracy')}; "
            f"auto={stats.get('auto_graded_correct')}/{stats.get('auto_gradable_count')} "
            f"= {stats.get('auto_accuracy')}"
        )
    lines.extend(["", "## Answerability final / auto accuracy"])
    for a, stats in (g.get("answerability_class") or {}).items():
        lines.append(
            f"- {a}: final={stats.get('final_answer_correct_count')}/"
            f"{stats.get('task_count')}={stats.get('final_answer_accuracy')}; "
            f"auto={stats.get('auto_graded_correct')}/{stats.get('auto_gradable_count')} "
            f"= {stats.get('auto_accuracy')}"
        )
    lines.extend(
        [
            "",
            "## Error taxonomy (primary)",
            f"- counts: `{tax.get('primary_counts')}`",
            "",
            "## Resources",
            f"- tokens in/out/total: `{m.get('total_input_tokens')}` / "
            f"`{m.get('total_output_tokens')}` / `{m.get('total_tokens')}`",
            f"- LLM calls / real MCP: `{m.get('total_llm_calls')}` / "
            f"`{m.get('total_real_mcp_executions')}`",
            f"- mean latency ms: `{m.get('mean_latency_ms')}`",
            f"- estimated_usd: `null`",
            "",
            "## Partial task",
            f"- `{m.get('partial_task')}`",
            "",
        ]
    )
    if human_applied:
        lines.append(
            "Human grades closed from `human_grades.json` "
            "(no task-id scoring hardcode)."
        )
    else:
        lines.append(
            "Stop: waiting for human confirmation via `human_grades.json`."
        )
    return "\n".join(lines) + "\n"


def run_flat_final_audit(
    run_dir: Path,
    *,
    audit_outdir: Optional[Path] = None,
    docs_mirror: Optional[Path] = None,
    human_grades_paths: Optional[Sequence[Path]] = None,
) -> Dict[str, Any]:
    run_dir = Path(run_dir)
    out_dir = Path(audit_outdir) if audit_outdir else run_dir / "final_audit_flat_v1"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Protect raw artifacts ---
    fingerprints_before = fingerprint_raw_artifacts(run_dir)
    assert_known_raw_hashes(fingerprints_before)
    raw_sha_before = {k: v["sha256"] for k, v in fingerprints_before.items()}

    predictions = load_jsonl(run_dir / "predictions.jsonl")
    if len(predictions) != EXPECTED_QUESTION_COUNT:
        raise ValueError(f"Expected 155 predictions, found {len(predictions)}")
    ids = [str(p.get("task_id") or p.get("id")) for p in predictions]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate task_id in predictions.jsonl")

    run_meta = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    identity = assert_flat_run_identity(run_meta)
    if list(run_meta.get("question_ids") or []) != ids:
        raise ValueError("prediction task_id order != run_metadata.question_ids")

    cost = json.loads((run_dir / "cost_summary.json").read_text(encoding="utf-8"))
    retrieval_blob = json.loads(
        (run_dir / "retrieval_analysis.json").read_text(encoding="utf-8")
    )
    if retrieval_blob.get("retrieval_metrics") not in {RETRIEVAL_NA, "N/A", None}:
        # Accept missing with note, but never treat as success float.
        if retrieval_blob.get("retrieval_metrics") == 1.0:
            raise ValueError("Flat retrieval_metrics must not be 1.0")

    audited: List[Dict[str, Any]] = []
    for idx, row in enumerate(predictions, start=1):
        tid = str(row.get("task_id") or row.get("id"))
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
            "question": row.get("question"),
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
            "protocol_repair_count": row.get("protocol_repair_count"),
            "latency_ms": row.get("latency_ms"),
            "embedding_calls": int(row.get("embedding_calls") or 0),
            "embedding_tokens": int(row.get("embedding_tokens") or 0),
            "retrieval_calls": int(row.get("retrieval_calls") or 0),
            "underlying_operation_calls": row.get("underlying_operation_calls"),
            "real_mcp_execution_count": row.get("real_mcp_execution_count"),
            "resolver_calls": row.get("resolver_calls")
            or {"find_operations": 0, "describe_operation": 0},
            "selected_operations": row.get("selected_operations"),
            "invoked_operations": row.get("invoked_operations"),
            "successful_invoked_operations": row.get("successful_invoked_operations"),
            "invalid_selections": row.get("invalid_selections"),
        }
        metrics = evaluate_flat_task(item=item, result=result)
        metrics = apply_flat_retrieval_na(metrics)
        grade = grade_task(item, {**result, **metrics, "task_id": tid})

        # Prefer frozen prediction grades for formal auto accuracy (matches live sheet).
        use_auto = row.get("auto_grade_status")
        use_correct = row.get("answer_correct")
        use_reason = row.get("auto_grade_reason")
        if use_auto is None:
            use_auto = grade.get("auto_grade_status")
            use_correct = grade.get("answer_correct")
            use_reason = grade.get("reason")

        merged = dict(row)
        merged.update(metrics)
        # Restore frozen formal answer fields after metrics update.
        merged["auto_grade_status"] = use_auto
        merged["answer_correct"] = use_correct
        merged["auto_grade_reason"] = use_reason
        merged["reeval_auto_grade_status"] = grade.get("auto_grade_status")
        merged["reeval_answer_correct"] = grade.get("answer_correct")
        merged["reeval_auto_grade_reason"] = grade.get("reason")
        merged["evidence_correct"] = grade.get("evidence_correct")
        merged["tool_selection_correct"] = grade.get("tool_selection_correct")
        merged["overall_task_success"] = grade.get("overall_task_success")
        merged["task_order_index"] = idx
        merged["selected_operations"] = _selected_ops(merged)
        merged["invoked_operations"] = _invoked_ops(merged)
        merged["successful_invoked_operations"] = _successful_ops(merged)
        merged["retrieval_metrics"] = RETRIEVAL_NA
        merged["embedding_calls"] = 0
        merged["retrieval_calls"] = 0
        resolver = merged.get("resolver_calls") or {}
        if int(resolver.get("find_operations") or 0) != 0 or int(
            resolver.get("describe_operation") or 0
        ) != 0:
            raise ValueError(f"Flat invariant violated: find/describe non-zero on {tid}")

        tax = classify_flat_error(merged)
        merged.update(tax)
        merged["answer_correct_strict"] = strict_answer_correct(merged)
        merged["strict_overall_task_success"] = strict_overall_task_success(merged)

        audited.append(merged)

    # Discover / load human grades (file-driven; no task-id scoring hardcode).
    grade_paths: List[Path] = []
    if human_grades_paths:
        grade_paths.extend(Path(p) for p in human_grades_paths)
    else:
        for candidate in (
            out_dir / "human_grades.json",
            Path(docs_mirror) / "human_grades.json" if docs_mirror else None,
            run_dir / "final_audit_flat_v1" / "human_grades.json",
        ):
            if candidate is not None and candidate.exists():
                grade_paths.append(candidate)
    seen_paths: Set[str] = set()
    unique_paths: List[Path] = []
    for p in grade_paths:
        key = str(p.resolve()) if p.exists() else str(p)
        if key in seen_paths:
            continue
        seen_paths.add(key)
        unique_paths.append(p)
    human_grades = load_human_grades(unique_paths)
    human_grades_applied = bool(human_grades)
    if human_grades_applied:
        apply_human_grades(audited, human_grades)
        for merged in audited:
            merged["answer_correct_strict"] = strict_answer_correct(merged)
            merged["strict_overall_task_success"] = strict_overall_task_success(merged)
        # Persist canonical human_grades.json into audit out-dir for formal mirror.
        dest = out_dir / "human_grades.json"
        if not dest.exists():
            for src in unique_paths:
                if src.exists() and src.name == "human_grades.json":
                    dest.write_bytes(src.read_bytes())
                    break
            else:
                payload = {
                    "audit_version": AUDIT_VERSION,
                    "source": "human_review",
                    "grades": [
                        {
                            "task_id": tid,
                            **{
                                k: v
                                for k, v in grade.items()
                                if k != "source_file"
                            },
                        }
                        for tid, grade in sorted(human_grades.items())
                    ],
                }
                atomic_write_json(dest, payload)

    taxonomy_rows: List[Dict[str, Any]] = []
    for merged in audited:
        tid = str(merged.get("task_id") or merged.get("id"))
        taxonomy_rows.append(
            {
                "task_id": tid,
                "domain": merged.get("domain"),
                "domain_combination": merged.get("domain_combination"),
                "project_difficulty": merged.get("project_difficulty"),
                "answerability_class": merged.get("answerability_class"),
                "source_split": merged.get("source_split"),
                "previously_reserved": merged.get("previously_reserved"),
                "exact_etom_alignment": merged.get("exact_etom_alignment"),
                "exact_etom_level": merged.get("exact_etom_level"),
                "company": merged.get("company"),
                "gold_status": merged.get("gold_status"),
                "predicted_status": merged.get("predicted_status"),
                "status": merged.get("status"),
                "error": merged.get("error") or "",
                "auto_grade_status": merged.get("auto_grade_status"),
                "answer_correct": merged.get("answer_correct"),
                "answer_correct_strict": merged.get("answer_correct_strict"),
                "final_answer_correct": merged.get("final_answer_correct"),
                "final_overall_task_success": merged.get("final_overall_task_success"),
                "human_confirmed_answer_correct": merged.get(
                    "human_confirmed_answer_correct"
                ),
                "auto_grade_reason": merged.get("auto_grade_reason"),
                "primary_error_category": merged.get("primary_error_category"),
                "secondary_error_categories": json.dumps(
                    merged.get("secondary_error_categories") or [], ensure_ascii=False
                ),
                "selected_operations": json.dumps(
                    merged.get("selected_operations") or [], ensure_ascii=False
                ),
                "invoked_operations": json.dumps(
                    merged.get("invoked_operations") or [], ensure_ascii=False
                ),
                "successful_invoked_operations": json.dumps(
                    merged.get("successful_invoked_operations") or [],
                    ensure_ascii=False,
                ),
                "gold_operations": json.dumps(
                    merged.get("gold_operations") or [], ensure_ascii=False
                ),
                "exact_tool_set": merged.get("exact_tool_set"),
                "operation_precision": merged.get("operation_precision"),
                "operation_recall": merged.get("operation_recall"),
                "operation_f1": merged.get("operation_f1"),
                "wrong_tool_rate": merged.get("wrong_tool_rate"),
                "all_tool_invocations_successful": merged.get(
                    "all_tool_invocations_successful"
                ),
                "recovered_after_tool_error": merged.get("recovered_after_tool_error"),
                "refusal_correctness": merged.get("refusal_correctness"),
                "graph_dependency_accuracy": merged.get("graph_dependency_accuracy"),
                "llm_calls": merged.get("llm_calls"),
                "real_mcp_execution_count": merged.get("real_mcp_execution_count"),
                "embedding_calls": 0,
                "retrieval_calls": 0,
                "latency_ms": merged.get("latency_ms"),
                "input_tokens": (merged.get("tokens") or {}).get("input_tokens"),
                "output_tokens": (merged.get("tokens") or {}).get("output_tokens"),
                "total_tokens": (merged.get("tokens") or {}).get("total_tokens"),
                "final_answer": merged.get("final_answer"),
            }
        )

    # Invariant: Flat tool/retrieval zeros.
    if any(int(r.get("embedding_calls") or 0) for r in audited):
        raise ValueError("Non-zero embedding_calls in Flat audit rows")
    if any(int(r.get("retrieval_calls") or 0) for r in audited):
        raise ValueError("Non-zero retrieval_calls in Flat audit rows")

    status_counts = Counter(str(r.get("status") or "missing") for r in audited)
    error_counts = Counter(
        str(r.get("error"))
        for r in audited
        if r.get("error") not in (None, "", "None", "null")
    )
    auto = [r for r in audited if r.get("auto_grade_status") == "graded"]
    auto_correct = [r for r in auto if r.get("answer_correct") is True]
    manual = [r for r in audited if r.get("auto_grade_status") == "manual_required"]
    manual_ids = [str(r.get("task_id")) for r in manual]
    if sorted(manual_ids) != sorted(EXPECTED_MANUAL_TASKS):
        raise ValueError(
            f"Unexpected manual queue: {manual_ids} != {list(EXPECTED_MANUAL_TASKS)}"
        )

    n = len(audited)
    lower = len(auto_correct) / n
    upper = (len(auto_correct) + len(manual)) / n
    strict_true = sum(1 for r in audited if r.get("answer_correct_strict") is True)
    strict_overall_true = sum(
        1 for r in audited if r.get("strict_overall_task_success") is True
    )
    has_final = bool(audited) and all(
        "final_answer_correct" in r and r.get("final_answer_correct") is not None
        for r in audited
    )
    final_true_count = (
        sum(1 for r in audited if r.get("final_answer_correct") is True)
        if has_final
        else None
    )
    final_answer_accuracy = (final_true_count / n) if has_final and n else None
    overall_accuracy = final_answer_accuracy if has_final else None

    partial_rows = [r for r in audited if r.get("status") == "partial"]
    partial_task = None
    if partial_rows:
        p = partial_rows[0]
        partial_task = {
            "task_id": p.get("task_id"),
            "status": p.get("status"),
            "error": p.get("error"),
            "predicted_status": p.get("predicted_status"),
            "final_answer": p.get("final_answer"),
            "answer_correct": p.get("answer_correct"),
            "answer_correct_strict": p.get("answer_correct_strict"),
            "primary_error_category": p.get("primary_error_category"),
            "selected_operations": p.get("selected_operations"),
            "invoked_operations": p.get("invoked_operations"),
            "llm_calls": p.get("llm_calls"),
            "real_mcp_execution_count": p.get("real_mcp_execution_count"),
            "treatment": (
                "strict incorrect; remains in denominator; no rerun; "
                "classified as protocol_or_parse_failure"
            ),
        }

    grouped = build_flat_grouped_audit(audited)
    tool_sel = _tool_selection_analysis(audited)
    resources = _resource_analysis(audited, run_meta=run_meta, cost=cost)
    primary_counts = Counter(
        str(r.get("primary_error_category")) for r in audited
    )
    error_taxonomy = {
        "taxonomy_version": TAXONOMY_VERSION,
        "categories": list(FLAT_TAXONOMY),
        "primary_counts": dict(primary_counts),
        "notes": [
            "max_invocations / max_llm_turns take priority over answer errors.",
            "Redundant vs missing gold tools are separated.",
            "Tool execution failure ≠ wrong tool selection.",
            "No task_id special-case scoring rules; human grades come from file.",
            "Retrieval metrics are N/A for Flat.",
        ],
        "max_invocations_financial_note": (
            "Financial accounts for most max_invocations failures; "
            "full 39-tool exposure + multi-step filing paths likely drive over-selection "
            "and repeated invokes before a final answer."
        ),
        "should_refuse_zero_note": (
            "should_refuse_or_clarify auto accuracy is 0/6 under frozen grades: "
            "agents often answered with tools or refused without matching gold refusal "
            "contract (answer_correct False even when refusal_correctness is 1.0 on "
            "some rows)."
        ),
        "evidence_answerable_note": (
            "evidence_answerable accuracy is low: multi-doc chains + max_invocations "
            "and synthesis failures dominate."
        ),
        "human_grades_note": (
            "Manual-required final scores are taken only from human_grades.json; "
            "auto accuracy 87/154 is never used as final overall accuracy."
            if human_grades_applied
            else "Awaiting human_grades.json for manual_required closure."
        ),
    }

    final_metrics = {
        "audit_version": AUDIT_VERSION,
        "evaluator_version": EVALUATOR_VERSION,
        "grading_version": GRADING_VERSION,
        "grader_version": GRADER_VERSION,
        "condition": EXPECTED_CONDITION,
        "architecture": EXPECTED_ARCHITECTURE,
        "generation_provider": EXPECTED_PROVIDER,
        "provider_host": EXPECTED_HOST,
        "generation_model": EXPECTED_MODEL,
        "git_commit": EXPECTED_IMPL_COMMIT,
        "system_prompt_hash": EXPECTED_PROMPT_HASH,
        "n_tasks": n,
        "status_counts": dict(status_counts),
        "error_counts": dict(error_counts),
        "auto_gradable": len(auto),
        "auto_correct": len(auto_correct),
        "auto_accuracy": len(auto_correct) / len(auto) if auto else None,
        "manual_required_count": len(manual),
        "manual_task_ids": manual_ids,
        "final_answer_correct_count": final_true_count,
        "final_answer_accuracy": final_answer_accuracy,
        "overall_accuracy": overall_accuracy,
        "conservative_lower_bound_accuracy": lower,
        "upper_bound_if_manual_correct": upper,
        "strict_answer_correct_count": strict_true,
        "strict_answer_accuracy": strict_true / n if n else None,
        "strict_overall_task_success_count": strict_overall_true,
        "strict_overall_task_success_rate": strict_overall_true / n if n else None,
        "exact_tool_set_rate": tool_sel["exact_tool_set_rate"],
        "mean_operation_precision": tool_sel["mean_operation_precision"],
        "mean_operation_recall": tool_sel["mean_operation_recall"],
        "mean_operation_f1": tool_sel["mean_operation_f1"],
        "mean_wrong_tool_rate": tool_sel["mean_wrong_tool_rate"],
        "all_tool_invocations_successful_rate": tool_sel[
            "all_tool_invocations_successful_rate"
        ],
        "recovered_after_tool_error_rate": tool_sel[
            "recovered_after_tool_error_rate"
        ],
        "retrieval_metrics": RETRIEVAL_NA,
        "total_input_tokens": resources["total_input_tokens"],
        "total_output_tokens": resources["total_output_tokens"],
        "total_tokens": resources["total_tokens"],
        "total_llm_calls": resources["total_llm_calls"],
        "total_real_mcp_executions": resources["total_real_mcp_executions"],
        "total_embedding_calls": 0,
        "total_retrieval_calls": 0,
        "mean_latency_ms": resources["mean_latency_ms"],
        "estimated_usd": None,
        "partial_task": partial_task,
        "failed_remain_in_denominator": True,
        "human_grades_applied": human_grades_applied,
        "human_grades_paths": [str(p) for p in unique_paths if p.exists()],
        "human_graded_task_ids": sorted(human_grades.keys()),
    }

    # Write artifacts
    atomic_write_json(out_dir / "raw_artifact_hashes.json", {
        "before": fingerprints_before,
        "known_precheck": KNOWN_RAW_SHA256,
        "note": "Audit must not modify these files; after hashes recomputed at end.",
    })
    atomic_write_json(out_dir / "final_metrics.json", final_metrics)
    atomic_write_json(out_dir / "grouped_metrics.json", grouped)
    atomic_write_json(out_dir / "error_taxonomy.json", error_taxonomy)
    atomic_write_json(out_dir / "resource_analysis.json", resources)
    atomic_write_json(out_dir / "tool_selection_analysis.json", tool_sel)
    atomic_write_json(
        out_dir / "evaluation_identity.json",
        {
            "evaluator": evaluator_identity(),
            "grading": grading_identity(),
            "audit_version": AUDIT_VERSION,
            "taxonomy_version": TAXONOMY_VERSION,
            "flat_prompt_rules_hash": flat_prompt_rules_hash(),
            "run_identity_checks": identity,
            "human_grades_applied": human_grades_applied,
        },
    )

    per_task_fields = list(taxonomy_rows[0].keys()) if taxonomy_rows else ["task_id"]
    _write_csv(out_dir / "per_task_error_analysis.csv", per_task_fields, taxonomy_rows)

    manual_rows = _manual_queue_rows(audited)
    manual_fields = [
        "task_id",
        "domain",
        "project_difficulty",
        "answerability_class",
        "question",
        "final_answer",
        "reference_answer",
        "gold_specification",
        "evidence",
        "selected_operations",
        "invoked_operations",
        "run_status",
        "predicted_status",
        "error",
        "auto_reason",
        "human_confirmed_answer_correct",
        "human_confirmed_evidence_correct",
        "human_confirmed_overall_success",
        "reviewer_notes",
    ]
    _write_csv(out_dir / "manual_review_queue.csv", manual_fields, manual_rows)

    ctx = {
        "final_metrics": final_metrics,
        "grouped_metrics": grouped,
        "error_taxonomy": error_taxonomy,
    }
    report = _build_report_md(ctx)
    (out_dir / "report.md").write_text(report, encoding="utf-8")
    (out_dir / "README.md").write_text(report, encoding="utf-8")

    audit_summary = {
        "audit_version": AUDIT_VERSION,
        "run_dir": str(run_dir),
        "out_dir": str(out_dir),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "identity": identity,
        "final_metrics": final_metrics,
        "primary_error_counts": dict(primary_counts),
        "manual_review_queue": manual_ids,
        "guarantees": {
            "no_task_rerun": True,
            "no_llm": True,
            "no_embedding": True,
            "no_mcp_network": True,
            "no_paid_api": True,
            "raw_predictions_unchanged": True,
            "prompt_catalog_gold_unchanged": True,
            "human_grades_from_file_only": True,
            "human_grades_applied": human_grades_applied,
        },
    }
    atomic_write_json(out_dir / "audit_summary.json", audit_summary)

    # Docs mirror
    if docs_mirror is not None:
        docs_mirror = Path(docs_mirror)
        docs_mirror.mkdir(parents=True, exist_ok=True)
        for name in (
            "audit_summary.json",
            "raw_artifact_hashes.json",
            "final_metrics.json",
            "grouped_metrics.json",
            "error_taxonomy.json",
            "resource_analysis.json",
            "tool_selection_analysis.json",
            "evaluation_identity.json",
            "per_task_error_analysis.csv",
            "manual_review_queue.csv",
            "human_grades.json",
            "report.md",
            "README.md",
        ):
            src = out_dir / name
            if src.exists():
                (docs_mirror / name).write_bytes(src.read_bytes())

    # --- Re-verify raw hashes unchanged ---
    fingerprints_after = fingerprint_raw_artifacts(run_dir)
    raw_sha_after = {k: v["sha256"] for k, v in fingerprints_after.items()}
    if raw_sha_after != raw_sha_before:
        raise RuntimeError("Raw artifacts changed during Flat final audit — STOP")
    assert_known_raw_hashes(fingerprints_after)
    atomic_write_json(
        out_dir / "raw_artifact_hashes.json",
        {
            "before": fingerprints_before,
            "after": fingerprints_after,
            "unchanged": raw_sha_after == raw_sha_before,
            "known_precheck": KNOWN_RAW_SHA256,
        },
    )
    if docs_mirror is not None:
        (Path(docs_mirror) / "raw_artifact_hashes.json").write_bytes(
            (out_dir / "raw_artifact_hashes.json").read_bytes()
        )

    return {
        "ok": True,
        "out_dir": str(out_dir),
        "docs_mirror": str(docs_mirror) if docs_mirror else None,
        "n_tasks": n,
        "auto_accuracy": final_metrics["auto_accuracy"],
        "final_answer_accuracy": final_metrics["final_answer_accuracy"],
        "strict_answer_accuracy": final_metrics["strict_answer_accuracy"],
        "strict_overall_task_success_rate": final_metrics[
            "strict_overall_task_success_rate"
        ],
        "lower_bound": lower,
        "upper_bound": upper,
        "manual_task_ids": manual_ids,
        "human_grades_applied": human_grades_applied,
        "raw_unchanged": True,
        "audit_summary": audit_summary,
    }
