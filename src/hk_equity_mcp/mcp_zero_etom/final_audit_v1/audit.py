"""Deterministic offline Formal-100 final audit for MCP-Zero × ETOM."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from hk_equity_mcp.etom.evaluator import evaluate_prediction
from hk_equity_mcp.etom.formal_metrics import dependency_accuracy
from hk_equity_mcp.etom.formal_sampler import load_formal_100_lock
from hk_equity_mcp.mcp_zero import method_identity_sha256
from hk_equity_mcp.mcp_zero_etom import (
    EXPECTED_FORMAL_SAMPLE_SHA256,
    EXPECTED_HK_METHOD_CARD_SHA256,
    EXPECTED_ORDERED_IDS_SHA256,
    etom_method_identity_sha256,
)
from hk_equity_mcp.mcp_zero_etom.l4_metrics import (
    efs_covering_node_set,
    evaluate_l4_separate,
)
from hk_equity_mcp.mcp_zero_etom.prompt import etom_prompt_hash

AUDIT_VERSION = "etom_mcp_zero_formal100_final_audit_v1"

RAW_ARTIFACT_NAMES: Tuple[str, ...] = (
    "predictions.jsonl",
    "run_metadata.json",
    "grouped_results.json",
    "cost_summary.json",
    "retrieval_analysis.json",
    "protocol_summary.json",
    "error_summary.json",
    "source_manifest.json",
    "index_status.json",
    "index_build.json",
    "per_task_metrics.csv",
    "preflight_report.json",
)

EXPECTED = {
    "sample_sha256": EXPECTED_FORMAL_SAMPLE_SHA256,
    "ordered_task_ids_sha256": EXPECTED_ORDERED_IDS_SHA256,
    "source_commit": "790f28727f567b88918420d163c524d46644a694",
    "generation_model": "deepseek-v4.1-flash",
    "embedding_model": "text-embedding-3-small",
    "provider_host": "api.zhizengzeng.com",
    "server_top_k": 3,
    "tool_top_k": 5,
    "hk_method_identity_sha256": EXPECTED_HK_METHOD_CARD_SHA256,
    "etom_method_identity_sha256": (
        "0c14b7aec5b6bdf1156b9d1ab0e330c3e0bbf8264e2db0950d2efbdce09e05f6"
    ),
    "system_prompt_sha256": (
        "8ebac6ce045af0176c1fb9e580bad483b7ca4564e3521adc433c09476db285d7"
    ),
    "registry_servers": 491,
    "registry_tools": 2375,
    "embedding_units": 3357,
    "strict_overall": {"successes": 49, "denominator": 100, "rate": 0.49},
    "strict_per_level": {
        "L1": {"successes": 17, "denominator": 20, "rate": 0.85},
        "L2": {"successes": 10, "denominator": 20, "rate": 0.50},
        "L3": {"successes": 9, "denominator": 20, "rate": 0.45},
        "L4": {"successes": 2, "denominator": 20, "rate": 0.10},
        "L5": {"successes": 11, "denominator": 20, "rate": 0.55},
    },
    "completed_only_overall": {"successes": 49, "denominator": 84, "rate": 49 / 84},
}

# Frozen Operation-Visible Formal-100 descriptive comparison (no re-run).
OV_COMPARISON_FROZEN = {
    "source": "operation_visible_formal100_canonical_audit_frozen_numbers",
    "note": (
        "Descriptive comparison only. No statistical significance claims. "
        "L4 uses complete_route_success under the same definition."
    ),
    "operation_visible_strict": {
        "L1": 0.85,
        "L2": 0.60,
        "L3": 0.65,
        "L4_complete_route": 0.25,
        "L5": 0.95,
        "overall": {"successes": 66, "denominator": 100, "rate": 0.66},
        "completion": {"completed": 92, "denominator": 100, "rate": 0.92},
    },
    "mcp_zero_strict": {
        "L1": 0.85,
        "L2": 0.50,
        "L3": 0.45,
        "L4_complete_route": 0.10,
        "L5": 0.55,
        "overall": {"successes": 49, "denominator": 100, "rate": 0.49},
        "completion": {"completed": 84, "denominator": 100, "rate": 0.84},
    },
}

PRIMARY_LABELS = (
    "exact_route_success",
    "server_retrieval_miss",
    "tool_retrieval_miss",
    "gold_retrieved_but_not_selected",
    "incomplete_tool_chain",
    "dependency_order_error",
    "unnecessary_tool_or_l5_incorrect_rejection",
    "protocol_parse_failure",
    "max_turn_resource_limit",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _rate(successes: int, denominator: int) -> Optional[float]:
    if denominator <= 0:
        return None
    return successes / denominator


def _mean(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return float(statistics.mean(values))


def _pct(values: Sequence[float], q: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def _retrieved_server_ids(row: Dict[str, Any]) -> Set[str]:
    out: Set[str] = set()
    for item in row.get("retrieved_servers") or []:
        if isinstance(item, dict):
            sid = item.get("server_id") or item.get("server_name")
            if sid:
                out.add(str(sid))
        elif item:
            out.add(str(item))
    return out


def _predicted_ops(row: Dict[str, Any]) -> Set[str]:
    ops: Set[str] = set()
    for node in row.get("predicted_nodes") or []:
        if not isinstance(node, dict):
            continue
        server = node.get("server_name") or node.get("server")
        tool = node.get("tool_name")
        if server and tool:
            ops.add(f"{server}::{tool}")
    for tool in row.get("predicted_tools") or []:
        if not isinstance(tool, dict):
            continue
        server = tool.get("server_name") or tool.get("server")
        name = tool.get("tool_name")
        if server and name:
            ops.add(f"{server}::{name}")
    for op in row.get("selected_operations") or []:
        if op:
            ops.add(str(op))
    return ops


def _gold_ops(row: Dict[str, Any]) -> Set[str]:
    return {str(x) for x in (row.get("gold_operation_ids") or []) if x}


def _gold_servers(row: Dict[str, Any]) -> Set[str]:
    return {str(x) for x in (row.get("gold_server_names") or []) if x}


def _atr_rounds(row: Dict[str, Any]) -> int:
    trace = row.get("mcp_zero_trace") or {}
    rounds = trace.get("rounds") if isinstance(trace, dict) else None
    if isinstance(rounds, list):
        return len(rounds)
    return int(row.get("number_of_extension_rounds") or 0)


def _llm_turns(row: Dict[str, Any]) -> int:
    contents = row.get("raw_assistant_contents")
    if isinstance(contents, list) and contents:
        return len(contents)
    return int(row.get("llm_calls") or 0)


def hash_raw_artifacts(run_dir: Path) -> Dict[str, Any]:
    files: Dict[str, Any] = {}
    missing: List[str] = []
    for name in RAW_ARTIFACT_NAMES:
        path = run_dir / name
        if not path.exists():
            missing.append(name)
            continue
        files[name] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return {
        "run_dir": str(run_dir),
        "files": files,
        "missing": missing,
        "aggregate_sha256": sha256_bytes(
            json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ),
    }


def assert_raw_untouched(
    before: Dict[str, Any], after: Dict[str, Any]
) -> Dict[str, Any]:
    mismatches = []
    for name in RAW_ARTIFACT_NAMES:
        b = (before.get("files") or {}).get(name)
        a = (after.get("files") or {}).get(name)
        if b != a:
            mismatches.append({"file": name, "before": b, "after": a})
    return {
        "ok": not mismatches and before.get("missing") == after.get("missing"),
        "mismatches": mismatches,
        "before_aggregate_sha256": before.get("aggregate_sha256"),
        "after_aggregate_sha256": after.get("aggregate_sha256"),
    }


def audit_identity(
    *,
    root: Path,
    run_dir: Path,
    rows: Sequence[Dict[str, Any]],
    meta: Dict[str, Any],
) -> Dict[str, Any]:
    lock = load_formal_100_lock(root)
    ordered_ids = list(lock.get("ordered_task_ids") or [])
    pred_ids = [r["task_id"] for r in rows]
    level_counts = Counter(int(r.get("level") or 0) for r in rows)
    checks = {
        "predictions_rows_100": len(rows) == 100,
        "unique_task_ids_100": len(set(pred_ids)) == 100,
        "level_quotas_20": all(level_counts.get(i) == 20 for i in range(1, 6)),
        "task_order_matches_lock": pred_ids == ordered_ids,
        "sample_sha256": meta.get("sample_sha256") == EXPECTED["sample_sha256"],
        "ordered_task_ids_sha256": (
            meta.get("ordered_task_ids_sha256") == EXPECTED["ordered_task_ids_sha256"]
        ),
        "source_commit": meta.get("source_commit") == EXPECTED["source_commit"],
        "generation_model": meta.get("generation_model") == EXPECTED["generation_model"],
        "embedding_model": meta.get("embedding_model") == EXPECTED["embedding_model"],
        "provider_host": meta.get("generation_base_url_host") == EXPECTED["provider_host"],
        "server_top_k": int(meta.get("server_top_k") or 0) == EXPECTED["server_top_k"],
        "tool_top_k": int(meta.get("tool_top_k") or 0) == EXPECTED["tool_top_k"],
        "hk_method_identity": (
            meta.get("hk_method_identity_sha256")
            == EXPECTED["hk_method_identity_sha256"]
            == method_identity_sha256()
        ),
        "etom_method_identity": (
            meta.get("etom_method_identity_sha256")
            == EXPECTED["etom_method_identity_sha256"]
            == etom_method_identity_sha256()
        ),
        "prompt_hash": (
            meta.get("system_prompt_sha256")
            == EXPECTED["system_prompt_sha256"]
            == etom_prompt_hash()
        ),
        "registry_counts": (
            int(meta.get("registry_server_count") or 0) == EXPECTED["registry_servers"]
            and int(meta.get("registry_tool_count") or 0) == EXPECTED["registry_tools"]
        ),
        "embedding_index_complete": (
            bool((meta.get("index_build") or {}).get("final_complete"))
            and int((meta.get("index_build") or {}).get("completed_units") or 0)
            == EXPECTED["embedding_units"]
            and bool((meta.get("index_build") or {}).get("cache_reused"))
        ),
        "real_tool_executions_zero": int(meta.get("real_tool_executions", -1)) == 0,
    }
    return {
        "ok": all(checks.values()),
        "checks": checks,
        "level_counts": {f"L{k}": level_counts[k] for k in range(1, 6)},
        "predictions_rows": len(rows),
        "unique_task_ids": len(set(pred_ids)),
        "ordered_task_ids_head": ordered_ids[:3],
        "ordered_task_ids_tail": ordered_ids[-3:],
        "sample_sha256": meta.get("sample_sha256"),
        "ordered_task_ids_sha256": meta.get("ordered_task_ids_sha256"),
        "source_commit": meta.get("source_commit"),
        "generation_model": meta.get("generation_model"),
        "embedding_model": meta.get("embedding_model"),
        "generation_base_url_host": meta.get("generation_base_url_host"),
        "embedding_base_url_host": meta.get("embedding_base_url_host"),
        "server_top_k": meta.get("server_top_k"),
        "tool_top_k": meta.get("tool_top_k"),
        "hk_method_identity_sha256": meta.get("hk_method_identity_sha256"),
        "etom_method_identity_sha256": meta.get("etom_method_identity_sha256"),
        "system_prompt_sha256": meta.get("system_prompt_sha256"),
        "registry_sha256": meta.get("registry_sha256"),
        "registry_fingerprint": meta.get("registry_fingerprint"),
        "registry_server_count": meta.get("registry_server_count"),
        "registry_tool_count": meta.get("registry_tool_count"),
        "embedding_index": {
            "identity_sha256": (meta.get("index_build") or {}).get("identity_sha256"),
            "completed_units": (meta.get("index_build") or {}).get("completed_units"),
            "expected_units": (meta.get("index_build") or {}).get("expected_units"),
            "final_complete": (meta.get("index_build") or {}).get("final_complete"),
            "cache_reused": (meta.get("index_build") or {}).get("cache_reused"),
            "build_calls": (meta.get("index_build") or {}).get("build_calls"),
        },
        "real_tool_executions": meta.get("real_tool_executions"),
        "routing_only": meta.get("routing_only"),
        "lock_sample_sha256": lock.get("sample_sha256"),
        "lock_ordered_task_ids_sha256": lock.get("ordered_task_ids_sha256"),
        "lock_source_commit": lock.get("source_commit"),
        "grouped_results_warning": (
            "raw grouped_results.json pooled_100.exact_match_rate is completed-only "
            "(49/84); official headline must use strict 49/100."
        ),
    }


def _strict_bool_rate(
    rows: Sequence[Dict[str, Any]], *, level: int, predicate
) -> Dict[str, Any]:
    group = [r for r in rows if int(r.get("level") or 0) == level]
    successes = sum(1 for r in group if predicate(r))
    return {
        "successes": successes,
        "strict_denominator": 20,
        "strict_rate": _rate(successes, 20),
        "completed": sum(1 for r in group if r.get("status") == "completed"),
        "failed_scored_zero": sum(1 for r in group if r.get("status") != "completed"),
    }


def compute_official_metrics(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Recompute official ETOM metrics with explicit strict vs completed-only labels."""
    by_level: Dict[str, List[Dict[str, Any]]] = {
        f"L{i}": [r for r in rows if int(r.get("level") or 0) == i] for i in range(1, 6)
    }

    def completed(group: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [r for r in group if r.get("status") == "completed"]

    def exact_ok(row: Dict[str, Any]) -> bool:
        return row.get("status") == "completed" and row.get("exact_match") is True

    # --- L1 ---
    l1 = by_level["L1"]
    l1_strict = _strict_bool_rate(rows, level=1, predicate=exact_ok)
    l1_completed = completed(l1)
    l1_block = {
        "exact_match": {
            "strict": l1_strict,
            "completed_only": {
                "successes": sum(1 for r in l1_completed if r.get("exact_match") is True),
                "denominator": len(l1_completed),
                "rate": _rate(
                    sum(1 for r in l1_completed if r.get("exact_match") is True),
                    len(l1_completed),
                ),
            },
        }
    }

    # --- L2 ---
    l2 = by_level["L2"]
    l2_single = [r for r in l2 if len(r.get("gold_route") or []) == 1]
    l2_multi = [r for r in l2 if len(r.get("gold_route") or []) > 1]

    def efs_cover_ok(row: Dict[str, Any]) -> bool:
        if row.get("status") != "completed":
            return False
        return efs_covering_node_set(row.get("gold_route"), row.get("predicted_tools") or [])

    def _subset_metrics(subset: Sequence[Dict[str, Any]], *, denom_mode: str) -> Dict[str, Any]:
        if denom_mode == "strict_level":
            # Contribute to level-20 strict only via caller; here report subset denom.
            denom = len(subset)
        else:
            denom = len(subset)
        em = sum(1 for r in subset if exact_ok(r))
        cover = sum(1 for r in subset if efs_cover_ok(r))
        return {
            "tasks": len(subset),
            "exact_match": {
                "successes": em,
                "denominator": denom,
                "rate": _rate(em, denom),
            },
            "any_equivalent_tool_coverage": {
                "successes": cover,
                "denominator": denom,
                "rate": _rate(cover, denom),
                "definition": (
                    "Every gold Equal-Function-Set group has >=1 predicted member."
                ),
            },
        }

    l2_block = {
        "equal_function_set_exact_match": {
            "strict": _strict_bool_rate(rows, level=2, predicate=exact_ok),
            "completed_only": {
                "successes": sum(1 for r in completed(l2) if r.get("exact_match") is True),
                "denominator": len(completed(l2)),
                "rate": _rate(
                    sum(1 for r in completed(l2) if r.get("exact_match") is True),
                    len(completed(l2)),
                ),
            },
        },
        "any_equivalent_tool_coverage": {
            "strict": _strict_bool_rate(rows, level=2, predicate=efs_cover_ok),
            "completed_only": {
                "successes": sum(1 for r in completed(l2) if efs_cover_ok(r)),
                "denominator": len(completed(l2)),
                "rate": _rate(
                    sum(1 for r in completed(l2) if efs_cover_ok(r)),
                    len(completed(l2)),
                ),
            },
        },
        "single_step": _subset_metrics(l2_single, denom_mode="subset"),
        "multi_step": _subset_metrics(l2_multi, denom_mode="subset"),
        "note": (
            "L2 official primary metric is Equal Function Set exact match. "
            "any_equivalent_tool_coverage is diagnostic coverage, not a substitute."
        ),
    }

    # --- L3 ---
    def l3_node_set_ok(row: Dict[str, Any]) -> bool:
        if row.get("status") != "completed":
            return False
        eval_row = evaluate_prediction(3, row.get("gold_route"), row.get("predicted_tools") or [])
        return bool(eval_row.get("node_set_match"))

    def l3_dep_ok(row: Dict[str, Any]) -> bool:
        if row.get("status") != "completed":
            return False
        dep = dependency_accuracy(
            level=3,
            ground_truth_tools=row.get("gold_route"),
            predicted_nodes=row.get("predicted_nodes") or [],
        )
        acc = dep.get("dependency_edge_accuracy")
        return acc is not None and float(acc) >= 1.0 - 1e-12

    def l3_ordered_dep_ok(row: Dict[str, Any]) -> bool:
        if row.get("status") != "completed":
            return False
        dep = dependency_accuracy(
            level=3,
            ground_truth_tools=row.get("gold_route"),
            predicted_nodes=row.get("predicted_nodes") or [],
        )
        acc = dep.get("ordered_dependency_accuracy")
        return acc is not None and float(acc) >= 1.0 - 1e-12

    def l3_complete_ok(row: Dict[str, Any]) -> bool:
        return exact_ok(row) and l3_dep_ok(row)

    def _prf_strict(level: int) -> Dict[str, Any]:
        group = by_level[f"L{level}"]
        vals_p, vals_r, vals_f = [], [], []
        for row in group:
            if row.get("status") == "completed":
                vals_p.append(float(row.get("precision") or 0.0))
                vals_r.append(float(row.get("recall") or 0.0))
                vals_f.append(float(row.get("f1") or 0.0))
            else:
                vals_p.append(0.0)
                vals_r.append(0.0)
                vals_f.append(0.0)
        return {
            "strict_denominator": 20,
            "mean_precision": _mean(vals_p),
            "mean_recall": _mean(vals_r),
            "mean_f1": _mean(vals_f),
            "note": "Failures contribute 0 to means (strict /20).",
        }

    def _prf_completed(level: int) -> Dict[str, Any]:
        group = completed(by_level[f"L{level}"])
        return {
            "denominator": len(group),
            "mean_precision": _mean([float(r.get("precision") or 0.0) for r in group]),
            "mean_recall": _mean([float(r.get("recall") or 0.0) for r in group]),
            "mean_f1": _mean([float(r.get("f1") or 0.0) for r in group]),
            "note": "Completed-only means; not a substitute for strict.",
        }

    l3_block = {
        "node_set_exact_match": {
            "strict": _strict_bool_rate(rows, level=3, predicate=l3_node_set_ok),
        },
        "exact_match_route": {
            "strict": _strict_bool_rate(rows, level=3, predicate=exact_ok),
            "completed_only": {
                "successes": sum(1 for r in completed(by_level["L3"]) if r.get("exact_match")),
                "denominator": len(completed(by_level["L3"])),
                "rate": _rate(
                    sum(1 for r in completed(by_level["L3"]) if r.get("exact_match")),
                    len(completed(by_level["L3"])),
                ),
            },
        },
        "precision_recall_f1": {
            "strict": _prf_strict(3),
            "completed_only": _prf_completed(3),
        },
        "dependency_correctness": {
            "strict": _strict_bool_rate(rows, level=3, predicate=l3_dep_ok),
        },
        "ordered_dependency_correctness": {
            "strict": _strict_bool_rate(rows, level=3, predicate=l3_ordered_dep_ok),
        },
        "complete_route_success": {
            "strict": _strict_bool_rate(rows, level=3, predicate=l3_complete_ok),
            "definition": "exact_match AND dependency_edge_accuracy==1 (failures score 0).",
        },
    }

    # --- L4 (four metrics must stay distinct) ---
    def l4_flag(row: Dict[str, Any], key: str) -> bool:
        if row.get("status") != "completed":
            return False
        # Prefer stored flags; recompute if missing.
        if key in row and row.get(key) is not None:
            return bool(row.get(key))
        recomputed = evaluate_l4_separate(
            ground_truth_tools=row.get("gold_route"),
            predicted_tools=row.get("predicted_tools") or [],
            predicted_nodes=row.get("predicted_nodes") or [],
            status=str(row.get("status") or "failed"),
        )
        return bool(recomputed.get(key))

    l4_block = {
        "official_union_node_set_em": {
            "strict": _strict_bool_rate(
                rows, level=4, predicate=lambda r: l4_flag(r, "official_union_node_set_em")
            ),
            "label": "official_union_node_set_em",
        },
        "efs_covering_node_set": {
            "strict": _strict_bool_rate(
                rows, level=4, predicate=lambda r: l4_flag(r, "efs_covering_node_set")
            ),
            "label": "efs_covering_node_set",
            "note": "NOT cross_server_dependency_correctness.",
        },
        "precision_recall_f1": {
            "strict": _prf_strict(4),
            "completed_only": _prf_completed(4),
        },
        "cross_server_dependency_correctness": {
            "strict": _strict_bool_rate(
                rows,
                level=4,
                predicate=lambda r: l4_flag(r, "cross_server_dependency_correctness"),
            ),
            "label": "cross_server_dependency_correctness",
            "note": "NOT efs_covering_node_set. Do not conflate.",
        },
        "complete_route_success": {
            "strict": _strict_bool_rate(
                rows, level=4, predicate=lambda r: l4_flag(r, "complete_route_success")
            ),
            "label": "complete_route_success",
            "definition": "exact_match AND cross_server_dependency_correctness.",
        },
        "exact_match_route_alias": {
            "strict": _strict_bool_rate(rows, level=4, predicate=exact_ok),
            "note": (
                "Stored exact_match for L4; primary comparable OV column uses "
                "complete_route_success."
            ),
        },
        "label_guard": {
            "efs_covering_is_not_cross_server_dependency": True,
            "official_union_em_is_not_complete_route": True,
        },
    }

    # --- L5 ---
    def l5_rejection_ok(row: Dict[str, Any]) -> bool:
        return exact_ok(row)

    def l5_false_positive(row: Dict[str, Any]) -> bool:
        if int(row.get("level") or 0) != 5:
            return False
        if row.get("status") != "completed":
            return False
        preds = row.get("predicted_tools") or []
        if row.get("exact_match") is True:
            return False
        # Completed but selected tools / failed rejection.
        return bool(preds) or row.get("predicted_rejection") is False

    l5_block = {
        "exact_rejection_match": {
            "strict": _strict_bool_rate(rows, level=5, predicate=l5_rejection_ok),
            "completed_only": {
                "successes": sum(1 for r in completed(by_level["L5"]) if r.get("exact_match")),
                "denominator": len(completed(by_level["L5"])),
                "rate": _rate(
                    sum(1 for r in completed(by_level["L5"]) if r.get("exact_match")),
                    len(completed(by_level["L5"])),
                ),
            },
        },
        "false_positive_tool_selection": {
            "completed_count": sum(1 for r in by_level["L5"] if l5_false_positive(r)),
            "failed_tasks_strict_zero": sum(
                1 for r in by_level["L5"] if r.get("status") != "completed"
            ),
            "note": (
                "False-positive = completed L5 with non-empty tools / non-rejection "
                "and exact_match=false. Failed tasks remain in strict denom with score 0."
            ),
        },
    }

    # Overall aggregates
    strict_successes = sum(1 for r in rows if exact_ok(r))
    completed_rows = [r for r in rows if r.get("status") == "completed"]
    completed_successes = sum(1 for r in completed_rows if r.get("exact_match") is True)
    per_level_strict = {
        f"L{i}": _strict_bool_rate(rows, level=i, predicate=exact_ok) for i in range(1, 6)
    }
    macro_strict = _mean(
        [float(per_level_strict[f"L{i}"]["strict_rate"] or 0.0) for i in range(1, 6)]
    )
    # micro = pooled strict /100
    micro_strict = _rate(strict_successes, 100)
    # completed-only micro /84
    micro_completed = _rate(completed_successes, len(completed_rows))
    macro_completed = _mean(
        [
            (
                sum(
                    1
                    for r in completed(by_level[f"L{i}"])
                    if r.get("exact_match") is True
                )
                / len(completed(by_level[f"L{i}"]))
            )
            if completed(by_level[f"L{i}"])
            else 0.0
            for i in range(1, 6)
        ]
    )

    return {
        "denominator_policy": {
            "strict": "failures score 0; per-level denom=20; overall denom=100",
            "completed_only": "diagnostic only; overall denom=84; must not replace strict",
            "macro": "unweighted mean of L1–L5 rates under the chosen policy",
            "micro_pooled": "pooled successes / pooled denominator",
        },
        "L1": l1_block,
        "L2": l2_block,
        "L3": l3_block,
        "L4": l4_block,
        "L5": l5_block,
        "overall": {
            "strict_exact_route": {
                "successes": strict_successes,
                "denominator": 100,
                "rate": micro_strict,
                "label": "strict_exact_route_/100",
            },
            "completed_only_exact_route": {
                "successes": completed_successes,
                "denominator": len(completed_rows),
                "rate": micro_completed,
                "label": "completed_only_exact_route_/84",
                "warning": (
                    "Raw grouped_results.json pooled_100.exact_match_rate uses this "
                    "completed-only denominator. Official headline is strict 49/100."
                ),
            },
            "macro_across_levels_strict": {
                "rate": macro_strict,
                "denominator_per_level": 20,
                "levels": 5,
            },
            "macro_across_levels_completed_only": {
                "rate": macro_completed,
                "note": "Unweighted mean of completed-only per-level EM rates.",
            },
            "micro_pooled_strict": {
                "successes": strict_successes,
                "denominator": 100,
                "rate": micro_strict,
            },
            "micro_pooled_completed_only": {
                "successes": completed_successes,
                "denominator": len(completed_rows),
                "rate": micro_completed,
            },
            "completion": {
                "completed": len(completed_rows),
                "failed": 100 - len(completed_rows),
                "denominator": 100,
                "rate": _rate(len(completed_rows), 100),
            },
        },
        "per_level_strict_exact_route": per_level_strict,
        "regression_targets": EXPECTED["strict_per_level"],
    }


def classify_task(row: Dict[str, Any]) -> Dict[str, Any]:
    """Exclusive primary taxonomy; optional secondary list."""
    secondary: List[str] = []
    gold_servers = _gold_servers(row)
    retrieved_servers = _retrieved_server_ids(row)
    gold_ops = _gold_ops(row)
    candidates = {str(x) for x in (row.get("candidate_operation_ids") or [])}
    predicted = _predicted_ops(row)
    level = int(row.get("level") or 0)

    if row.get("status") == "completed" and row.get("exact_match") is True:
        primary = "exact_route_success"
    elif row.get("status") != "completed" and row.get("error_type") == "max_llm_turns":
        primary = "max_turn_resource_limit"
        if gold_ops and not (gold_ops & candidates):
            secondary.append("tool_retrieval_miss")
        elif gold_servers and not (gold_servers & retrieved_servers):
            secondary.append("server_retrieval_miss")
        elif gold_ops & candidates and not (gold_ops & predicted):
            secondary.append("gold_retrieved_but_not_selected")
        elif gold_ops & predicted and (gold_ops - predicted):
            secondary.append("incomplete_tool_chain")
    elif level == 5 and row.get("status") == "completed" and row.get("exact_match") is not True:
        primary = "unnecessary_tool_or_l5_incorrect_rejection"
    elif gold_servers and not (gold_servers & retrieved_servers):
        primary = "server_retrieval_miss"
    elif gold_ops and not (gold_ops & candidates):
        primary = "tool_retrieval_miss"
    elif gold_ops & candidates and not (gold_ops & predicted):
        primary = "gold_retrieved_but_not_selected"
    elif gold_ops & predicted and (gold_ops - predicted):
        primary = "incomplete_tool_chain"
    elif level in (3, 4) and row.get("status") == "completed":
        dep = dependency_accuracy(
            level=level,
            ground_truth_tools=row.get("gold_route"),
            predicted_nodes=row.get("predicted_nodes") or [],
        )
        node_ok = bool(
            evaluate_prediction(
                level, row.get("gold_route"), row.get("predicted_tools") or []
            ).get("node_set_match")
        )
        dep_acc = dep.get("dependency_edge_accuracy")
        if node_ok and dep_acc is not None and float(dep_acc) < 1.0 - 1e-12:
            primary = "dependency_order_error"
        elif row.get("task_protocol_success") is False:
            primary = "protocol_parse_failure"
        else:
            primary = "gold_retrieved_but_not_selected"
            secondary.append("dependency_order_error")
    elif row.get("task_protocol_success") is False:
        primary = "protocol_parse_failure"
    else:
        primary = "gold_retrieved_but_not_selected"

    return {
        "task_id": row["task_id"],
        "level": level,
        "status": row.get("status"),
        "exact_match": bool(row.get("exact_match")),
        "primary": primary,
        "secondary": secondary,
        "gold_in_candidates": bool(gold_ops & candidates) if gold_ops else None,
        "gold_server_retrieved": bool(gold_servers & retrieved_servers)
        if gold_servers
        else None,
    }


def build_error_taxonomy(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    classified = [classify_task(r) for r in rows]
    primary_counts = Counter(c["primary"] for c in classified)
    return {
        "version": "mcp_zero_etom_formal100_taxonomy_v1",
        "exclusive_primary": True,
        "labels": list(PRIMARY_LABELS),
        "primary_counts": {k: primary_counts.get(k, 0) for k in PRIMARY_LABELS},
        "primary_counts_observed": dict(primary_counts),
        "tasks": classified,
        "sum_primary": sum(primary_counts.values()),
    }


def audit_protocol_failures(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    failed = [r for r in rows if r.get("status") != "completed"]
    completed_n = sum(1 for r in rows if r.get("status") == "completed")
    all_max_turns = all(r.get("error_type") == "max_llm_turns" for r in failed)
    details = []
    recoverable_exact_if_accepted = 0
    for row in failed:
        gold_ops = _gold_ops(row)
        candidates = {str(x) for x in (row.get("candidate_operation_ids") or [])}
        predicted = row.get("predicted_tools") or []
        eval_row = None
        would_exact = False
        if predicted:
            eval_row = evaluate_prediction(
                int(row["level"]), row.get("gold_route"), predicted
            )
            would_exact = bool(eval_row.get("exact_match"))
            if would_exact:
                recoverable_exact_if_accepted += 1
        last_state = {
            "predicted_tools_n": len(predicted),
            "predicted_nodes_n": len(row.get("predicted_nodes") or []),
            "selected_operations_n": len(row.get("selected_operations") or []),
            "termination_reason": (row.get("mcp_zero_trace") or {}).get(
                "termination_reason"
            ),
            "task_protocol_success": row.get("task_protocol_success"),
        }
        details.append(
            {
                "task_id": row["task_id"],
                "level": int(row["level"]),
                "error_type": row.get("error_type"),
                "failure_stage": row.get("failure_stage") or "max_llm_turns",
                "llm_turns": _llm_turns(row),
                "llm_calls": row.get("llm_calls"),
                "atr_rounds": _atr_rounds(row),
                "extension_rounds": row.get("number_of_extension_rounds"),
                "retrieved_servers": sorted(_retrieved_server_ids(row)),
                "retrieved_servers_n": len(_retrieved_server_ids(row)),
                "candidate_tools_n": len(candidates),
                "gold_in_candidates": bool(gold_ops & candidates) if gold_ops else None,
                "gold_operation_ids": sorted(gold_ops),
                "last_valid_state": last_state,
                "existing_predicted_tools_exact_match_if_accepted": would_exact,
                "parser_recovery_note": (
                    "Offline check only: whether already-emitted predicted_tools "
                    "would score exact_match. No new parser, no re-run, no guessed route."
                ),
                "eval_status_if_scored": (eval_row or {}).get("status"),
            }
        )
    return {
        "completed": completed_n,
        "failed": len(failed),
        "all_failures_are_max_llm_turns": all_max_turns,
        "parse_repairs": sum(int(r.get("parse_repairs") or 0) for r in rows),
        "protocol_repairs": sum(int(r.get("protocol_repair_count") or 0) for r in rows),
        "task_protocol_success_rate": _rate(
            sum(1 for r in rows if r.get("task_protocol_success")), len(rows)
        ),
        "raw_content_preserved_tasks": sum(
            1
            for r in rows
            if r.get("raw_assistant_contents")
            or any(
                isinstance(e, dict) and e.get("raw_content") is not None
                for e in (r.get("trace") or [])
            )
        ),
        "parser_v2_1": {
            "built": False,
            "reason": (
                "All 16 failures are max_llm_turns resource limits, not isolated "
                "parse failures. No task-id-special parser was introduced."
            ),
            "score_delta_if_built": None,
            "existing_partial_routes_that_would_exact_match": recoverable_exact_if_accepted,
        },
        "failed_tasks": details,
    }


def audit_retrieval(rows: Sequence[Dict[str, Any]], stored: Dict[str, Any]) -> Dict[str, Any]:
    server_hits = server_total = 0
    tool_hits = tool_total = 0
    gold_in_candidates = 0
    routed_gold_not_selected = 0
    selected_not_retrieved = 0
    query_emb_calls = 0
    query_emb_tokens = 0
    for row in rows:
        gold_servers = _gold_servers(row)
        retrieved = _retrieved_server_ids(row)
        gold_ops = _gold_ops(row)
        candidates = {str(x) for x in (row.get("candidate_operation_ids") or [])}
        predicted = _predicted_ops(row)
        if gold_servers:
            server_total += 1
            if gold_servers & retrieved:
                server_hits += 1
        if gold_ops:
            tool_total += 1
            if gold_ops & candidates:
                tool_hits += 1
                gold_in_candidates += 1
            if gold_ops - predicted:
                routed_gold_not_selected += 1
        if predicted - candidates:
            selected_not_retrieved += 1
        query_emb_calls += int(row.get("query_embedding_calls") or 0)
        query_emb_tokens += int(row.get("query_embedding_tokens") or 0)
    recomputed = {
        "server_recall": _rate(server_hits, server_total),
        "tool_recall": _rate(tool_hits, tool_total),
        "gold_in_candidates_tasks": gold_in_candidates,
        "routed_gold_not_selected_tasks": routed_gold_not_selected,
        "selected_not_retrieved_tasks": selected_not_retrieved,
        "query_embedding_calls": query_emb_calls,
        "query_embedding_tokens": query_emb_tokens,
        "server_hits": server_hits,
        "server_total": server_total,
        "tool_hits": tool_hits,
        "tool_total": tool_total,
    }
    return {
        "stored": stored,
        "recomputed": recomputed,
        "matches_expected": {
            "server_recall": recomputed["server_recall"] == 0.775,
            "tool_recall": recomputed["tool_recall"] == 0.8625,
            "gold_in_candidates_tasks": gold_in_candidates == 69,
            "routed_gold_not_selected_tasks": routed_gold_not_selected == 37,
            "selected_not_retrieved_tasks": selected_not_retrieved == 0,
            "query_embedding_calls": query_emb_calls == 308,
            "query_embedding_tokens": query_emb_tokens == 5868,
        },
        "note": (
            "Post-hoc retrieval analysis from prediction fields. Gold never entered "
            "the agent prompt. Routing-only; real_tool_executions=0."
        ),
    }


def audit_cost(rows: Sequence[Dict[str, Any]], cost: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    latencies = [float(r["latency_ms"]) for r in rows if r.get("latency_ms") is not None]
    return {
        "llm_calls": cost.get("llm_calls"),
        "input_tokens": cost.get("input_tokens"),
        "output_tokens": cost.get("output_tokens"),
        "total_tokens": cost.get("total_tokens"),
        "query_embedding_calls": cost.get("query_embedding_calls"),
        "query_embedding_tokens": cost.get("query_embedding_tokens"),
        "catalog_index": cost.get("one_time_catalog_index"),
        "latency_ms": {
            "stored": cost.get("latency_ms"),
            "recomputed": {
                "mean": round(_mean(latencies) or 0.0, 5),
                "median": round(_pct(latencies, 0.5) or 0.0, 5),
                "p95": round(_pct(latencies, 0.95) or 0.0, 5),
                "n": len(latencies),
            },
        },
        "this_run_estimated_usd": cost.get("this_run_estimated_usd"),
        "pricing": {
            "source": "frozen pricing block in configs/mcp_zero_etom_v1.yaml / cost_summary",
            "llm_input_usd_per_million": (cost.get("projection_uncertainty") or {}).get(
                "llm_input_usd_per_million"
            ),
            "llm_output_usd_per_million": (cost.get("projection_uncertainty") or {}).get(
                "llm_output_usd_per_million"
            ),
            "embedding_usd_per_million": (cost.get("projection_uncertainty") or {}).get(
                "embedding_usd_per_million"
            ),
            "pricing_sha256": meta.get("pricing_sha256"),
        },
        "retry_count": cost.get("retry_count"),
        "timeouts": cost.get("timeouts"),
        "real_tool_executions": cost.get("real_tool_executions"),
        "checks": {
            "llm_calls_451": cost.get("llm_calls") == 451,
            "input_tokens": cost.get("input_tokens") == 1277207,
            "output_tokens": cost.get("output_tokens") == 142939,
            "total_tokens": cost.get("total_tokens") == 1420146,
            "query_embedding_calls_308": cost.get("query_embedding_calls") == 308,
            "query_embedding_tokens_5868": cost.get("query_embedding_tokens") == 5868,
            "catalog_build_calls_0": (cost.get("one_time_catalog_index") or {}).get("calls")
            == 0,
            "cache_reused": bool((cost.get("one_time_catalog_index") or {}).get("cache_reused")),
            "this_run_estimated_usd": cost.get("this_run_estimated_usd") == 0.554806,
            "retry_count_0": cost.get("retry_count") == 0,
            "timeouts_0": cost.get("timeouts") == 0,
        },
    }


def build_ov_comparison(official: Dict[str, Any]) -> Dict[str, Any]:
    mz_l4 = official["L4"]["complete_route_success"]["strict"]["strict_rate"]
    return {
        **OV_COMPARISON_FROZEN,
        "aligned_columns": {
            "metric": "strict rates; L4 column = complete_route_success",
            "mcp_zero_observed_l4_complete_route": mz_l4,
            "matches_frozen_mcp_zero_l4": mz_l4 == 0.10,
        },
        "deltas_mcp_zero_minus_ov": {
            "L1": 0.85 - 0.85,
            "L2": 0.50 - 0.60,
            "L3": 0.45 - 0.65,
            "L4_complete_route": 0.10 - 0.25,
            "L5": 0.55 - 0.95,
            "overall": 0.49 - 0.66,
            "completion": 0.84 - 0.92,
        },
    }


def render_final_report(payloads: Dict[str, Any]) -> str:
    identity = payloads["identity_audit"]
    strict = payloads["strict_metrics"]
    overall = strict["overall"]["strict_exact_route"]
    completed = strict["overall"]["completed_only_exact_route"]
    tax = payloads["error_taxonomy"]["primary_counts"]
    cost = payloads["cost_analysis"]
    raw_sha = payloads["raw_artifact_sha256"]
    integrity_ok = (raw_sha.get("integrity") or {}).get("ok")
    lines = [
        "# MCP-Zero × ETOM Formal-100 Final Audit v1",
        "",
        f"- Audit version: `{AUDIT_VERSION}`",
        f"- Generated (UTC): `{payloads['audit_summary']['generated_at_utc']}`",
        f"- Raw aggregate SHA256: `{raw_sha['before']['aggregate_sha256']}`",
        f"- Raw SHA unchanged after audit: `{integrity_ok}`",
        "",
        "## Official headline (strict denominator)",
        "",
        f"- Overall strict exact route: **{overall['successes']}/{overall['denominator']} = {overall['rate']:.2f}**",
        f"- Completed-only diagnostic: {completed['successes']}/{completed['denominator']} = {completed['rate']:.4f} (must not replace strict)",
        f"- Completion: {strict['overall']['completion']['completed']}/100",
        "",
        "| Level | Strict EM | Completed-only EM |",
        "|---|---:|---:|",
    ]
    for level in ("L1", "L2", "L3", "L4", "L5"):
        block = strict["per_level_strict_exact_route"][level]
        # completed-only from official blocks
        if level == "L1":
            co = strict["L1"]["exact_match"]["completed_only"]["rate"]
        elif level == "L2":
            co = strict["L2"]["equal_function_set_exact_match"]["completed_only"]["rate"]
        elif level == "L3":
            co = strict["L3"]["exact_match_route"]["completed_only"]["rate"]
        elif level == "L4":
            co = None
            # use exact_match completed only via successes/completed
            co = (
                block["successes"] / block["completed"]
                if block["completed"]
                else None
            )
        else:
            co = strict["L5"]["exact_rejection_match"]["completed_only"]["rate"]
        lines.append(
            f"| {level} | {block['successes']}/20 = {block['strict_rate']:.2f} | "
            f"{'' if co is None else f'{co:.4f}'} |"
        )
    lines.extend(
        [
            "",
            "### L4 separate metrics (strict /20)",
            "",
            f"- official_union_node_set_em: {strict['L4']['official_union_node_set_em']['strict']['successes']}/20",
            f"- efs_covering_node_set: {strict['L4']['efs_covering_node_set']['strict']['successes']}/20",
            f"- cross_server_dependency_correctness: {strict['L4']['cross_server_dependency_correctness']['strict']['successes']}/20",
            f"- complete_route_success: {strict['L4']['complete_route_success']['strict']['successes']}/20",
            "",
            "## Identity",
            "",
            f"- Identity checks ok: `{identity['ok']}`",
            f"- sample_sha256: `{identity['sample_sha256']}`",
            f"- ordered_task_ids_sha256: `{identity['ordered_task_ids_sha256']}`",
            f"- source_commit: `{identity['source_commit']}`",
            f"- models: `{identity['generation_model']}` / `{identity['embedding_model']}`",
            f"- host: `{identity['generation_base_url_host']}`",
            f"- top-k: server={identity['server_top_k']} tool={identity['tool_top_k']}",
            f"- catalog: {identity['registry_server_count']} servers / {identity['registry_tool_count']} tools",
            f"- embedding index: {identity['embedding_index']}",
            "",
            "## Retrieval",
            "",
            f"- server_recall={payloads['retrieval_analysis']['recomputed']['server_recall']}",
            f"- tool_recall={payloads['retrieval_analysis']['recomputed']['tool_recall']}",
            f"- gold_in_candidates_tasks={payloads['retrieval_analysis']['recomputed']['gold_in_candidates_tasks']}",
            f"- routed_gold_not_selected_tasks={payloads['retrieval_analysis']['recomputed']['routed_gold_not_selected_tasks']}",
            f"- selected_not_retrieved_tasks={payloads['retrieval_analysis']['recomputed']['selected_not_retrieved_tasks']}",
            "",
            "## Protocol / failures",
            "",
            f"- completed={payloads['protocol_failure_analysis']['completed']} failed={payloads['protocol_failure_analysis']['failed']}",
            f"- all failures max_llm_turns: {payloads['protocol_failure_analysis']['all_failures_are_max_llm_turns']}",
            f"- parse_repairs={payloads['protocol_failure_analysis']['parse_repairs']} protocol_repairs={payloads['protocol_failure_analysis']['protocol_repairs']}",
            "",
            "## Error taxonomy (exclusive primary)",
            "",
        ]
    )
    for label in PRIMARY_LABELS:
        lines.append(f"- {label}: {tax.get(label, 0)}")
    lines.extend(
        [
            "",
            "## Cost",
            "",
            f"- LLM calls={cost['llm_calls']}; tokens in/out/total="
            f"{cost['input_tokens']}/{cost['output_tokens']}/{cost['total_tokens']}",
            f"- query embeddings={cost['query_embedding_calls']}/{cost['query_embedding_tokens']}",
            f"- this_run_estimated_usd={cost['this_run_estimated_usd']}",
            f"- retry_count={cost['retry_count']}; timeouts={cost['timeouts']}",
            "",
            "## Operation-Visible descriptive comparison",
            "",
            "- OV overall strict 66/100=0.66; completion 92/100",
            "- MCP-Zero overall strict 49/100=0.49; completion 84/100",
            "- L4 column aligned on complete_route_success (OV 0.25 vs MCP-Zero 0.10)",
            "- No statistical significance claims.",
            "",
            "## Notes",
            "",
            "- Route exact match is not final-answer accuracy.",
            "- Raw grouped_results.json pooled exact_match_rate is completed-only; corrected in this audit.",
            "- No raw artifacts were modified; no APIs were called.",
            "",
        ]
    )
    return "\n".join(lines)


def run_final_audit(
    *,
    root: Path,
    run_dir: Path,
    audit_dir: Optional[Path] = None,
    docs_mirror_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    root = Path(root).resolve()
    run_dir = Path(run_dir).resolve()
    audit_dir = Path(audit_dir) if audit_dir else run_dir / "final_audit_v1"
    docs_mirror_dir = (
        Path(docs_mirror_dir)
        if docs_mirror_dir
        else root / "docs" / "etom_mcp_zero_formal100_final_audit_v1"
    )

    before = hash_raw_artifacts(run_dir)
    if before["missing"]:
        raise FileNotFoundError(f"Missing raw artifacts: {before['missing']}")

    rows = load_jsonl(run_dir / "predictions.jsonl")
    meta = load_json(run_dir / "run_metadata.json")
    cost = load_json(run_dir / "cost_summary.json")
    retrieval_stored = load_json(run_dir / "retrieval_analysis.json")
    # grouped_results is read for labeling critique only; never overwritten.
    grouped_raw = load_json(run_dir / "grouped_results.json")

    identity = audit_identity(root=root, run_dir=run_dir, rows=rows, meta=meta)
    official = compute_official_metrics(rows)
    completed_only = {
        "label": "completed_only_diagnostic",
        "must_not_replace_strict": True,
        "overall": official["overall"]["completed_only_exact_route"],
        "macro_across_levels": official["overall"]["macro_across_levels_completed_only"],
        "raw_grouped_results_pooled_exact_match_rate": (grouped_raw.get("pooled_100") or {}).get(
            "exact_match_rate"
        ),
        "raw_grouped_results_pooled_is_completed_only": True,
        "corrected_label": "completed_only_exact_route_rate_84",
        "official_headline": "strict_exact_route_49_100",
    }
    per_level = {
        "L1": official["L1"],
        "L2": official["L2"],
        "L3": official["L3"],
        "L4": official["L4"],
        "L5": official["L5"],
        "per_level_strict_exact_route": official["per_level_strict_exact_route"],
    }
    retrieval = audit_retrieval(rows, retrieval_stored)
    protocol = audit_protocol_failures(rows)
    taxonomy = build_error_taxonomy(rows)
    cost_analysis = audit_cost(rows, cost, meta)
    ov_cmp = build_ov_comparison(official)

    summary = {
        "audit_version": AUDIT_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "audit_dir": str(audit_dir),
        "identity_ok": identity["ok"],
        "strict_overall": official["overall"]["strict_exact_route"],
        "completed_only_overall": official["overall"]["completed_only_exact_route"],
        "completion": official["overall"]["completion"],
        "taxonomy_primary_counts": taxonomy["primary_counts"],
        "raw_sha_unchanged": None,  # filled after write
        "constraints": {
            "no_rerun": True,
            "no_api_calls": True,
            "raw_artifacts_immutable": True,
            "no_router_retune": True,
        },
    }

    payloads = {
        "audit_summary": summary,
        "identity_audit": identity,
        "strict_metrics": official,
        "completed_only_metrics": completed_only,
        "per_level_metrics": per_level,
        "retrieval_analysis": retrieval,
        "protocol_failure_analysis": protocol,
        "error_taxonomy": taxonomy,
        "cost_analysis": cost_analysis,
        "operation_visible_comparison": ov_cmp,
        "raw_artifact_sha256": {"before": before},
    }
    payloads["final_report_md"] = render_final_report(payloads)

    audit_dir.mkdir(parents=True, exist_ok=True)
    for name, key in [
        ("audit_summary.json", "audit_summary"),
        ("identity_audit.json", "identity_audit"),
        ("strict_metrics.json", "strict_metrics"),
        ("completed_only_metrics.json", "completed_only_metrics"),
        ("per_level_metrics.json", "per_level_metrics"),
        ("retrieval_analysis.json", "retrieval_analysis"),
        ("protocol_failure_analysis.json", "protocol_failure_analysis"),
        ("error_taxonomy.json", "error_taxonomy"),
        ("cost_analysis.json", "cost_analysis"),
        ("operation_visible_comparison.json", "operation_visible_comparison"),
    ]:
        atomic_write_json(audit_dir / name, payloads[key])
    atomic_write_text(audit_dir / "final_report.md", payloads["final_report_md"])

    after = hash_raw_artifacts(run_dir)
    integrity = assert_raw_untouched(before, after)
    raw_payload = {"before": before, "after": after, "integrity": integrity}
    atomic_write_json(audit_dir / "raw_artifact_sha256.json", raw_payload)
    summary["raw_sha_unchanged"] = integrity["ok"]
    atomic_write_json(audit_dir / "audit_summary.json", summary)
    payloads["raw_artifact_sha256"] = raw_payload
    payloads["audit_summary"] = summary

    # Refresh report with integrity bit.
    payloads["final_report_md"] = render_final_report(payloads)
    atomic_write_text(audit_dir / "final_report.md", payloads["final_report_md"])

    # Docs mirror (committable)
    if docs_mirror_dir is not None:
        docs_mirror_dir.mkdir(parents=True, exist_ok=True)
        for path in audit_dir.iterdir():
            if path.is_file():
                shutil.copy2(path, docs_mirror_dir / path.name)

    if not integrity["ok"]:
        raise RuntimeError("Raw artifact SHA changed during audit; refuse to seal.")
    if not identity["ok"]:
        raise RuntimeError(f"Identity audit failed: {identity['checks']}")

    # Regression guard on headline numbers
    assert official["overall"]["strict_exact_route"]["successes"] == 49
    assert official["overall"]["strict_exact_route"]["denominator"] == 100
    assert official["overall"]["completed_only_exact_route"]["denominator"] == 84
    assert official["per_level_strict_exact_route"]["L1"]["successes"] == 17
    assert official["per_level_strict_exact_route"]["L2"]["successes"] == 10
    assert official["per_level_strict_exact_route"]["L3"]["successes"] == 9
    assert official["per_level_strict_exact_route"]["L4"]["successes"] == 2
    assert official["per_level_strict_exact_route"]["L5"]["successes"] == 11

    return {
        "audit_dir": str(audit_dir),
        "docs_mirror_dir": str(docs_mirror_dir),
        "summary": summary,
        "integrity": integrity,
    }
