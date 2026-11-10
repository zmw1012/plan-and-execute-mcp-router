"""Frozen Planner protocol Diagnostic-10 (engineering only; not accuracy)."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

from hk_equity_mcp.experiment.pe_fin_lite_retrieval_audit.identity import (
    ordered_task_ids_hash,
    sha256_text,
    stable_json_hash,
)

DIAGNOSTIC10_MANIFEST_REL = (
    "benchmarks/manifests/hk_pe_fin_lite_planner_diagnostic10_v1.json"
)
DIAGNOSTIC10_ROLE = "engineering_protocol_diagnostic_only"
DIAGNOSTIC10_CACHE_KIND = "planner_only_diagnostic10_cache"
ALL155_CACHE_KIND = "planner_only_cache"
SOURCE_PLANNER_V1_CACHE_SHA256 = (
    "002d60972d13b1e3728642ef7d63eeb1ada543c19bd25129b54ae67be3a312d5"
)
EXPECTED_N = 10

# Fields allowed through to Planner runtime for diagnostic subset.
_RUNTIME_TASK_KEEP = (
    "task_id",
    "id",
    "question",
    "domain",
    "source_split",
    "project_difficulty_level",
    "difficulty",
    "answerability_class",
)

_GOLD_BAN = (
    "gold_operations",
    "acceptable_tool_sets",
    "reference_answer",
    "gold_values",
    "gold_domains",
    "gold_server_ids",
)


def classify_planner_error(error: Optional[str]) -> str:
    e = str(error or "")
    if not e:
        return "none"
    if "missing required property 'depends_on'" in e and "steps[0]" in e:
        return "missing_depends_on_first_step"
    if "missing required property 'depends_on'" in e:
        return "missing_depends_on_later_step"
    if e.startswith("invalid JSON"):
        return "invalid_json"
    if e in {"empty content", "null content"}:
        return "empty_or_null_content"
    if "truncated" in e or "no JSON object found" in e:
        return "truncated_or_non_json"
    if "not in enum" in e and "domain" in e:
        return "invalid_domain_enum"
    if "maxLength" in e:
        return "maxLength_violation"
    if "missing required property" in e:
        return "other_missing_field"
    return "other"


def _content_sha_excluding_manifest_sha(payload: Dict[str, Any]) -> str:
    core = {k: v for k, v in payload.items() if k != "manifest_sha256"}
    blob = json.dumps(core, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def load_planner_diagnostic10_manifest(root: Path) -> Dict[str, Any]:
    path = Path(root) / DIAGNOSTIC10_MANIFEST_REL
    if not path.exists():
        raise FileNotFoundError(f"Missing Diagnostic-10 manifest: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("role") != DIAGNOSTIC10_ROLE:
        raise RuntimeError("Diagnostic-10 manifest role mismatch")
    if int(payload.get("n_tasks") or 0) != EXPECTED_N:
        raise RuntimeError(f"Diagnostic-10 expects n_tasks={EXPECTED_N}")
    ordered = list(payload.get("ordered_task_ids") or [])
    if len(ordered) != EXPECTED_N or len(set(ordered)) != EXPECTED_N:
        raise RuntimeError("Diagnostic-10 ordered_task_ids must be 10 unique IDs")
    expected_order_hash = ordered_task_ids_hash(ordered)
    if payload.get("ordered_task_ids_hash") != expected_order_hash:
        raise RuntimeError("Diagnostic-10 ordered_task_ids_hash mismatch (fail-closed)")
    expected_manifest_sha = _content_sha_excluding_manifest_sha(payload)
    if payload.get("manifest_sha256") != expected_manifest_sha:
        raise RuntimeError("Diagnostic-10 manifest_sha256 mismatch (fail-closed)")
    if payload.get("source_planner_v1_cache_sha256") != SOURCE_PLANNER_V1_CACHE_SHA256:
        raise RuntimeError("Diagnostic-10 source planner v1 cache SHA mismatch")
    blob = json.dumps(payload, ensure_ascii=False)
    for key in _GOLD_BAN:
        if key in blob:
            raise RuntimeError(f"Diagnostic-10 manifest must not contain {key}")
    return {
        "path": str(path.relative_to(Path(root))),
        "absolute_path": str(path),
        "payload": payload,
        "ordered_task_ids": ordered,
        "manifest_sha256": payload["manifest_sha256"],
        "ordered_task_ids_hash": payload["ordered_task_ids_hash"],
        "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "distribution": payload.get("distribution") or {},
        "role": DIAGNOSTIC10_ROLE,
    }


def strip_task_for_diagnostic_runtime(task: Dict[str, Any]) -> Dict[str, Any]:
    """Drop gold/reference fields before Planner runtime."""
    out = {k: task.get(k) for k in _RUNTIME_TASK_KEEP if k in task}
    tid = str(task.get("task_id") or task.get("id") or "")
    out["task_id"] = tid
    out["id"] = tid
    blob = json.dumps(out, ensure_ascii=False)
    for key in _GOLD_BAN:
        if key in blob:
            raise RuntimeError(f"Diagnostic runtime task leaked {key}")
    return out


def select_diagnostic10_tasks(
    *,
    root: Path,
    all_tasks: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    manifest = load_planner_diagnostic10_manifest(root)
    by_id = {str(t.get("task_id") or t.get("id")): t for t in all_tasks}
    missing = [tid for tid in manifest["ordered_task_ids"] if tid not in by_id]
    if missing:
        raise RuntimeError(f"Diagnostic-10 IDs missing from HK-155: {missing}")
    tasks = [
        strip_task_for_diagnostic_runtime(by_id[tid])
        for tid in manifest["ordered_task_ids"]
    ]
    meta_by_id = {
        row["task_id"]: row for row in (manifest["payload"].get("tasks") or [])
    }
    return {
        "manifest": manifest,
        "tasks": tasks,
        "manifest_task_meta_by_id": meta_by_id,
    }


def planner_diagnostic10_cache_identity(
    *,
    diagnostic_manifest: Dict[str, Any],
    planner_prompt_hash: str,
    planner_schema_hash: str,
    plan_canonicalizer_hash: str,
    capability_profile_hash: Optional[str],
    generation_provider: str,
    generation_host: str,
    generation_model: str,
    planner_only_protocol_version: str,
    temperature: float = 0.0,
    max_tokens: int = 2048,
) -> Dict[str, Any]:
    payload = {
        "kind": DIAGNOSTIC10_CACHE_KIND,
        "role": DIAGNOSTIC10_ROLE,
        "explicitly_not_formal_accuracy": True,
        "diagnostic_manifest_path": DIAGNOSTIC10_MANIFEST_REL,
        "manifest_sha256": diagnostic_manifest["manifest_sha256"],
        "ordered_task_ids_hash": diagnostic_manifest["ordered_task_ids_hash"],
        "ordered_task_ids": list(diagnostic_manifest["ordered_task_ids"]),
        "n_tasks": EXPECTED_N,
        "source_planner_v1_cache_sha256": SOURCE_PLANNER_V1_CACHE_SHA256,
        "planner_prompt_hash": planner_prompt_hash,
        "planner_schema_hash": planner_schema_hash,
        "plan_canonicalizer_hash": plan_canonicalizer_hash,
        "capability_profile_hash": capability_profile_hash,
        "generation_provider": generation_provider,
        "generation_host": generation_host,
        "generation_model": generation_model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "planner_only_protocol_version": planner_only_protocol_version,
        "no_binder": True,
        "no_mcp": True,
        "no_embedding_in_planner_stage": True,
        "no_gold_in_planner": True,
        "cannot_resume_all155": True,
        "cannot_formal_lock": True,
    }
    return {**payload, "identity_sha256": stable_json_hash(payload)}


def assert_not_cross_resume(identity: Dict[str, Any], previous: Dict[str, Any]) -> None:
    """Refuse diagnostic↔all155 resume even if other hashes somehow collide."""
    kind_a = identity.get("kind")
    kind_b = previous.get("kind")
    if kind_a != kind_b:
        raise RuntimeError(
            f"Planner cache kind mismatch (fail-closed): current={kind_a} previous={kind_b}"
        )


def build_planner_protocol_diagnostic(
    *,
    rows: Sequence[Dict[str, Any]],
    manifest: Dict[str, Any],
    manifest_task_meta_by_id: Dict[str, Dict[str, Any]],
    cache_identity: Dict[str, Any],
) -> Dict[str, Any]:
    ordered = list(manifest["ordered_task_ids"])
    ids = [str(r.get("task_id") or "") for r in rows]
    unique = sorted({x for x in ids if x})
    dup_counts = Counter(ids)
    duplicates = sorted(tid for tid, n in dup_counts.items() if n > 1)
    ok_ids = [str(r["task_id"]) for r in rows if r.get("ok")]
    failed = [r for r in rows if not r.get("ok")]
    repair_counts = [int(r.get("repair_count") or 0) for r in rows]
    n_repair = sum(1 for c in repair_counts if c > 0)
    error_cats = Counter(
        classify_planner_error(None if r.get("ok") else r.get("error")) for r in rows
    )
    # Systematic schema categories that Diagnostic-10 aims to eliminate.
    systematic = {
        "missing_depends_on_first_step",
        "invalid_domain_enum",
        "other_missing_field",
        "missing_depends_on_later_step",
    }
    systematic_schema_errors = sum(error_cats.get(c, 0) for c in systematic)

    per_task = []
    by_domain: Dict[str, Dict[str, int]] = {}
    for r in rows:
        tid = str(r.get("task_id"))
        meta = manifest_task_meta_by_id.get(tid) or {}
        v1_cat = meta.get("v1_engineering_error_category")
        v2_ok = bool(r.get("ok"))
        v2_cat = classify_planner_error(None if v2_ok else r.get("error"))
        recurred = (not v2_ok) and (v2_cat == v1_cat)
        domain = str(meta.get("domain") or "unknown")
        bucket = by_domain.setdefault(domain, {"n": 0, "ok": 0, "failed": 0})
        bucket["n"] += 1
        if v2_ok:
            bucket["ok"] += 1
        else:
            bucket["failed"] += 1
        per_task.append(
            {
                "task_id": tid,
                "domain": domain,
                "difficulty": meta.get("difficulty"),
                "source_split": meta.get("source_split"),
                "v1_error_category": v1_cat,
                "v2_outcome": "ok" if v2_ok else "failed",
                "v2_error_category": v2_cat,
                "whether_original_failure_recurred": recurred,
                "repair_count": int(r.get("repair_count") or 0),
                "llm_calls": r.get("llm_calls"),
                "tokens_in": r.get("tokens_in"),
                "tokens_out": r.get("tokens_out"),
                "tokens_total": r.get("tokens_total"),
                "latency_ms": r.get("latency_ms"),
                "error": r.get("error"),
            }
        )

    n = len(rows)
    repair_rate = (n_repair / n) if n else None
    complete = (
        n == EXPECTED_N
        and len(unique) == EXPECTED_N
        and not duplicates
        and set(unique) == set(ordered)
    )
    pass_gates = {
        "rows_10_unique": complete,
        "all_10_planner_valid": complete and len(ok_ids) == EXPECTED_N,
        "duplicates_zero": len(duplicates) == 0,
        "systematic_schema_errors_zero": systematic_schema_errors == 0,
        "repair_rate_le_0_30": (repair_rate is not None and repair_rate <= 0.30),
        "no_task_specific_branches": True,
        "no_gold_leakage": True,
    }
    engineering_pass = all(pass_gates.values())

    llm_calls = [int(r.get("llm_calls") or 0) for r in rows]
    tokens_in = [int(r["tokens_in"]) for r in rows if r.get("tokens_in") is not None]
    tokens_out = [int(r["tokens_out"]) for r in rows if r.get("tokens_out") is not None]
    tokens_total = [
        int(r["tokens_total"]) for r in rows if r.get("tokens_total") is not None
    ]
    latency = [float(r["latency_ms"]) for r in rows if r.get("latency_ms") is not None]

    return {
        "role": DIAGNOSTIC10_ROLE,
        "explicitly_not_formal_accuracy": True,
        "expected": EXPECTED_N,
        "processed": n,
        "unique": len(unique),
        "duplicates": duplicates,
        "ok": len(ok_ids),
        "failed": len(failed),
        "repair_count_tasks": n_repair,
        "repair_rate": repair_rate,
        "error_categories": dict(error_cats),
        "systematic_schema_errors": systematic_schema_errors,
        "llm_calls_total": int(sum(llm_calls)),
        "tokens_in_total": int(sum(tokens_in)) if tokens_in else 0,
        "tokens_out_total": int(sum(tokens_out)) if tokens_out else 0,
        "tokens_total": int(sum(tokens_total)) if tokens_total else 0,
        "latency_ms_sum": float(sum(latency)) if latency else 0.0,
        "per_domain": by_domain,
        "per_task": per_task,
        "complete_for_diagnostic10": complete,
        "engineering_pass_criteria": pass_gates,
        "engineering_pass": engineering_pass,
        "note": (
            "engineering_pass is a protocol diagnostic gate only; "
            "it does not authorize formal formula lock, all155 replacement, "
            "accuracy claims, or reserved85 claims."
        ),
        "cannot_formal_lock": True,
        "cannot_replace_all155_planner_cache": True,
        "cannot_run_step_embeddings_as_formal": True,
        "identity_sha256": cache_identity.get("identity_sha256"),
        "manifest_sha256": manifest.get("manifest_sha256"),
    }
