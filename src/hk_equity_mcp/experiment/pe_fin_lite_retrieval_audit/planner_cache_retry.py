"""Retry only failed Planner-only rows; freeze attempt snapshot; atomic promote.

Generic: failed task IDs are derived from the cache (ok=false), never hardcoded.
Does not modify PE-Fin Lite method, prompt, schema, or retrieval formulas.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from hk_equity_mcp.experiment.pe_fin_lite_retrieval_audit import (
    PLANNER_FAILED_RETRY_PROTOCOL_VERSION,
    PLANNER_ONLY_PROTOCOL_VERSION,
)
from hk_equity_mcp.experiment.pe_fin_lite_retrieval_audit.identity import (
    assert_identity_match,
    sha256_text,
)
from hk_equity_mcp.experiment.pe_fin_lite_retrieval_audit.planner_cache_gate import (
    IncompletePlannerCacheError,
    summarize_planner_rows,
)
from hk_equity_mcp.experiment.pe_fin_lite_retrieval_audit.planner_only import (
    _assert_no_gold_leakage,
    mock_plan_for_question,
    extract_retrieve_steps,
    run_single_live_planner_task,
)


class PlannerCacheRetryError(RuntimeError):
    """Fail-closed retry / merge / promote errors."""


ATTEMPT_SNAPSHOT_FILES = (
    "planner_outputs.jsonl",
    "planner_cache_identity.json",
    "planner_only_summary.json",
    "error_summary.json",
    "run_metadata.json",
)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_planner_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def rows_by_task_id(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    dups: List[str] = []
    for row in rows:
        tid = str(row.get("task_id") or "")
        if not tid:
            raise PlannerCacheRetryError("planner row missing task_id")
        if tid in out:
            dups.append(tid)
        out[tid] = row
    if dups:
        raise PlannerCacheRetryError(
            f"duplicate task IDs in planner cache (fail-closed): {sorted(set(dups))}"
        )
    return out


def derive_failed_task_ids(
    rows: Sequence[Dict[str, Any]],
    *,
    ordered_task_ids: Optional[Sequence[str]] = None,
) -> List[str]:
    """Auto-derive failed IDs from ok=false; order by frozen task order when given."""
    failed = {str(r.get("task_id")) for r in rows if not r.get("ok") and r.get("task_id")}
    if ordered_task_ids:
        return [tid for tid in ordered_task_ids if tid in failed]
    # Stable fallback: appearance order in cache.
    seen = set()
    out: List[str] = []
    for r in rows:
        tid = str(r.get("task_id") or "")
        if tid in failed and tid not in seen:
            out.append(tid)
            seen.add(tid)
    return out


def aggregate_row_costs(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    calls = 0
    tokens_in = 0
    tokens_out = 0
    tokens_total = 0
    for r in rows:
        calls += int(r.get("llm_calls") or 0)
        tokens_in += int(r.get("tokens_in") or 0)
        tokens_out += int(r.get("tokens_out") or 0)
        tokens_total += int(r.get("tokens_total") or 0)
    return {
        "llm_calls": calls,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "tokens_total": tokens_total,
        "n_rows": len(rows),
    }


def freeze_attempt_snapshot(
    *,
    planner_cache_dir: Path,
    attempt_dir: Path,
    expected_outputs_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    """Copy current planner_cache into a frozen attempt snapshot (no overwrite mutate)."""
    src = Path(planner_cache_dir)
    dst = Path(attempt_dir)
    outputs = src / "planner_outputs.jsonl"
    if not outputs.exists():
        raise PlannerCacheRetryError(f"missing planner outputs: {outputs}")
    src_sha = file_sha256(outputs)
    if expected_outputs_sha256 and src_sha != expected_outputs_sha256:
        raise PlannerCacheRetryError(
            f"planner_outputs SHA mismatch before snapshot: "
            f"expected={expected_outputs_sha256} observed={src_sha}"
        )

    if dst.exists():
        prev_hashes_path = dst / "raw_hashes.json"
        if not prev_hashes_path.exists():
            raise PlannerCacheRetryError(
                f"attempt snapshot exists without raw_hashes.json: {dst} (fail-closed)"
            )
        prev = json.loads(prev_hashes_path.read_text(encoding="utf-8"))
        prev_sha = (prev.get("files") or {}).get("planner_outputs.jsonl", {}).get(
            "sha256"
        )
        if prev_sha != src_sha:
            raise PlannerCacheRetryError(
                f"attempt snapshot {dst.name} already frozen with different "
                f"planner_outputs sha={prev_sha}; refusing overwrite"
            )
        # Idempotent: same snapshot already frozen.
        return prev

    dst.mkdir(parents=True, exist_ok=False)
    hashes: Dict[str, Any] = {}
    for name in ATTEMPT_SNAPSHOT_FILES:
        sp = src / name
        if not sp.exists():
            continue
        shutil.copy2(sp, dst / name)
        # Ensure snapshot copy is not later mutated via shared inode surprises.
        blob = (dst / name).read_bytes()
        hashes[name] = {"sha256": hashlib.sha256(blob).hexdigest(), "bytes": len(blob)}

    if "planner_outputs.jsonl" not in hashes:
        raise PlannerCacheRetryError("snapshot missing planner_outputs.jsonl")

    payload = {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_dir": str(src),
        "attempt_dir": str(dst),
        "files": hashes,
        "planner_outputs_sha256": hashes["planner_outputs.jsonl"]["sha256"],
        "immutable": True,
        "note": "Read-only attempt archive; do not delete or overwrite.",
    }
    (dst / "raw_hashes.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    # Best-effort chmod read-only on copied artifacts.
    for name in list(hashes) + ["raw_hashes.json"]:
        p = dst / name
        if p.exists():
            try:
                os.chmod(p, 0o444)
            except OSError:
                pass
    return payload


def assert_source_identity_matches_expected(
    *,
    source_identity: Dict[str, Any],
    expected_identity: Dict[str, Any],
) -> None:
    assert_identity_match(
        expected_identity, source_identity, label="planner_cache_retry_source"
    )
    # Extra prompt/schema fail-closed checks beyond digest.
    for key in (
        "planner_prompt_hash",
        "planner_schema_hash",
        "plan_canonicalizer_hash",
        "planner_only_protocol_version",
        "generation_provider",
        "generation_host",
        "generation_model",
        "temperature",
        "max_tokens",
    ):
        if key not in expected_identity:
            continue
        if source_identity.get(key) != expected_identity.get(key):
            raise PlannerCacheRetryError(
                f"identity field mismatch for {key}: "
                f"source={source_identity.get(key)!r} "
                f"expected={expected_identity.get(key)!r}"
            )


def merge_canonical_rows(
    *,
    ordered_task_ids: Sequence[str],
    original_rows_by_id: Dict[str, Dict[str, Any]],
    retry_success_by_id: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Rebuild exactly one row per ordered task ID; replace only successful retries."""
    if len(ordered_task_ids) != len(set(ordered_task_ids)):
        raise PlannerCacheRetryError("ordered_task_ids contain duplicates (fail-closed)")
    if len(ordered_task_ids) != 155:
        raise PlannerCacheRetryError(
            f"ordered_task_ids must be 155, got {len(ordered_task_ids)}"
        )
    missing_orig = [tid for tid in ordered_task_ids if tid not in original_rows_by_id]
    if missing_orig:
        raise PlannerCacheRetryError(
            f"original cache missing task IDs: {missing_orig[:20]}"
        )
    extra_orig = sorted(set(original_rows_by_id) - set(ordered_task_ids))
    if extra_orig:
        raise PlannerCacheRetryError(
            f"original cache has unexpected task IDs: {extra_orig[:20]}"
        )

    merged: List[Dict[str, Any]] = []
    for tid in ordered_task_ids:
        if tid in retry_success_by_id:
            row = dict(retry_success_by_id[tid])
            if not row.get("ok"):
                raise PlannerCacheRetryError(
                    f"refuse to merge non-ok retry row for {tid}"
                )
            merged.append(row)
        else:
            merged.append(dict(original_rows_by_id[tid]))

    summary = summarize_planner_rows(merged)
    if not summary["complete"]:
        raise PlannerCacheRetryError(
            "merged cache incomplete after promotion check: "
            f"rows={summary['n_rows']} unique={summary['n_unique_task_ids']} "
            f"failed={summary['n_failed']} dupes={summary.get('duplicate_task_ids')}"
        )
    if len(merged) != 155:
        raise PlannerCacheRetryError(f"merged physical rows != 155 ({len(merged)})")
    return merged


def atomic_write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> str:
    """Write JSONL via temp file + os.replace; return sha256 of final bytes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    payload = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    data = payload.encode("utf-8")
    with tmp.open("wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    # fsync directory entry best-effort
    try:
        dir_fd = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except OSError:
        pass
    return hashlib.sha256(data).hexdigest()


def atomic_write_json(path: Path, obj: Any) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    data = (json.dumps(obj, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    with tmp.open("wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    return hashlib.sha256(data).hexdigest()


def _mock_success_row(task: Dict[str, Any]) -> Dict[str, Any]:
    tid = str(task.get("task_id") or task.get("id"))
    plan = mock_plan_for_question(task)
    retrieve_steps = extract_retrieve_steps(plan)
    row = {
        "task_id": tid,
        "ok": True,
        "mock": True,
        "live": False,
        "plan": plan,
        "retrieve_steps": retrieve_steps,
        "raw_planner_response": json.dumps(plan, ensure_ascii=False),
        "repair_raw_content": None,
        "validated_plan": plan,
        "error": None,
        "error_type": None,
        "repair_count": 0,
        "usage": {"calls": 1, "input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        "latency_ms": 1.0,
        "llm_calls": 1,
        "tokens_in": 1,
        "tokens_out": 1,
        "tokens_total": 2,
        "planner_max_tokens": 4096,
        "call_observability": [],
        "planner_only_protocol_version": PLANNER_ONLY_PROTOCOL_VERSION,
        "no_embedding": True,
        "no_mcp": True,
        "no_binder": True,
        "retry": True,
    }
    _assert_no_gold_leakage(row)
    return row


def retry_failed_planner_cache(
    *,
    output_dir: Path,
    tasks_by_id: Dict[str, Dict[str, Any]],
    ordered_task_ids: Sequence[str],
    expected_identity: Dict[str, Any],
    llm: Any = None,
    capability_profile: Optional[Dict[str, Any]] = None,
    mock: bool = False,
    dry_run: bool = False,
    expected_source_outputs_sha256: Optional[str] = None,
    attempt_name: str = "planner_cache_attempt1",
    task_runner: Optional[Callable[..., Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Freeze incomplete cache, retry failed rows only, promote iff all retries ok.

    ``mock``/``dry_run``/fake ``task_runner`` are for tests. Live CLI must pass a real llm.
    """
    out = Path(output_dir)
    cache_dir = out / "planner_cache"
    outputs_path = cache_dir / "planner_outputs.jsonl"
    identity_path = cache_dir / "planner_cache_identity.json"
    if not outputs_path.exists() or not identity_path.exists():
        raise PlannerCacheRetryError(
            "planner_cache/planner_outputs.jsonl and planner_cache_identity.json required"
        )

    source_identity = json.loads(identity_path.read_text(encoding="utf-8"))
    assert_source_identity_matches_expected(
        source_identity=source_identity, expected_identity=expected_identity
    )

    source_rows = load_planner_rows(outputs_path)
    source_by_id = rows_by_task_id(source_rows)
    source_sha = file_sha256(outputs_path)
    if expected_source_outputs_sha256 and source_sha != expected_source_outputs_sha256:
        raise PlannerCacheRetryError(
            f"source planner_outputs SHA mismatch: expected="
            f"{expected_source_outputs_sha256} observed={source_sha}"
        )

    failed_ids = derive_failed_task_ids(
        source_rows, ordered_task_ids=ordered_task_ids
    )
    if not failed_ids:
        raise PlannerCacheRetryError(
            "no failed planner rows (ok=false); nothing to retry"
        )

    attempt_dir = out / attempt_name
    snapshot = freeze_attempt_snapshot(
        planner_cache_dir=cache_dir,
        attempt_dir=attempt_dir,
        expected_outputs_sha256=source_sha,
    )
    # Re-verify attempt snapshot still matches source (immutable).
    snap_sha = snapshot["planner_outputs_sha256"]
    if snap_sha != source_sha:
        raise PlannerCacheRetryError("snapshot sha diverged from source (fail-closed)")

    original_costs = aggregate_row_costs(source_rows)
    provenance_base = {
        "retry_protocol_version": PLANNER_FAILED_RETRY_PROTOCOL_VERSION,
        "source_planner_cache_sha256": source_sha,
        "source_identity_sha256": source_identity.get("identity_sha256"),
        "source_identity": {
            k: source_identity.get(k)
            for k in (
                "planner_prompt_hash",
                "planner_schema_hash",
                "plan_canonicalizer_hash",
                "capability_profile_hash",
                "generation_provider",
                "generation_host",
                "generation_model",
                "temperature",
                "max_tokens",
                "planner_only_protocol_version",
            )
        },
        "failed_ids_derived_from_source_cache": list(failed_ids),
        "retry_ids": list(failed_ids),
        "attempt_snapshot": attempt_name,
        "attempt_raw_hashes": snapshot.get("files"),
        "original_raw_hashes_unchanged": True,
        "no_task_id_hardcode": True,
        "no_gold_use": True,
        "no_embedding": True,
        "no_mcp": True,
        "no_binder": True,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    if dry_run:
        report = {
            **provenance_base,
            "dry_run": True,
            "promoted": False,
            "retry_result": "dry_run_no_llm",
            "original_costs": original_costs,
            "note": "Dry-run only listed failed IDs and froze/verified snapshot; no LLM.",
        }
        atomic_write_json(cache_dir / "retry_provenance.json", report)
        return report

    # Resolve tasks for failed IDs from frozen dataset map (by ID, not hardcode).
    retry_tasks: List[Dict[str, Any]] = []
    for tid in failed_ids:
        task = tasks_by_id.get(tid)
        if task is None:
            raise PlannerCacheRetryError(
                f"failed task_id {tid!r} not found in frozen HK-155 dataset"
            )
        retry_tasks.append(task)

    retry_rows: List[Dict[str, Any]] = []
    runner = task_runner
    for task in retry_tasks:
        if runner is not None:
            row = runner(task=task)
        elif mock:
            row = _mock_success_row(task)
        else:
            if llm is None:
                raise PlannerCacheRetryError(
                    "live retry requires llm (fail-closed; no silent mock)"
                )
            row = run_single_live_planner_task(
                task=task,
                llm=llm,
                capability_profile=capability_profile or {},
            )
        row = dict(row)
        row["retry"] = True
        row["retry_protocol_version"] = PLANNER_FAILED_RETRY_PROTOCOL_VERSION
        row["retry_source_planner_cache_sha256"] = source_sha
        _assert_no_gold_leakage(row)
        retry_rows.append(row)

    # Persist retry attempt artifacts (never silently drop).
    retry_art_dir = out / "planner_cache_retry" / "latest"
    retry_art_dir.mkdir(parents=True, exist_ok=True)
    retry_outputs_sha = atomic_write_jsonl(
        retry_art_dir / "retry_outputs.jsonl", retry_rows
    )
    atomic_write_json(
        retry_art_dir / "retry_tasks.json",
        {
            "retry_ids": failed_ids,
            "n": len(failed_ids),
            "derived_from_source_sha256": source_sha,
        },
    )

    retry_costs = aggregate_row_costs(retry_rows)
    success_by_id = {
        str(r["task_id"]): r for r in retry_rows if r.get("ok") and r.get("task_id")
    }
    still_failed = [tid for tid in failed_ids if tid not in success_by_id]

    if still_failed:
        # Fail closed: leave canonical planner_cache untouched.
        still_sha = file_sha256(outputs_path)
        if still_sha != source_sha:
            raise PlannerCacheRetryError(
                "CRITICAL: canonical planner_outputs changed despite failed retry "
                f"(before={source_sha} after={still_sha})"
            )
        report = {
            **provenance_base,
            "promoted": False,
            "retry_result": "failed",
            "still_failed_ids": still_failed,
            "retry_outputs_sha256": retry_outputs_sha,
            "retry_costs": retry_costs,
            "original_costs": original_costs,
            "combined_costs": {
                "llm_calls": original_costs["llm_calls"] + retry_costs["llm_calls"],
                "tokens_in": original_costs["tokens_in"] + retry_costs["tokens_in"],
                "tokens_out": original_costs["tokens_out"] + retry_costs["tokens_out"],
                "tokens_total": original_costs["tokens_total"]
                + retry_costs["tokens_total"],
            },
            "canonical_planner_outputs_sha256_unchanged": source_sha,
            "formal_step_pipeline_allowed": False,
        }
        atomic_write_json(cache_dir / "retry_provenance.json", report)
        atomic_write_json(retry_art_dir / "retry_report.json", report)
        raise IncompletePlannerCacheError(
            "Planner failed-row retry did not complete: "
            f"still_failed={still_failed}. Canonical cache unchanged "
            f"(sha={source_sha}). Step embedding / formal lock remain forbidden."
        )

    merged = merge_canonical_rows(
        ordered_task_ids=ordered_task_ids,
        original_rows_by_id=source_by_id,
        retry_success_by_id=success_by_id,
    )
    # Semantic identity: unchanged success rows must equal source rows.
    for i, tid in enumerate(ordered_task_ids):
        if tid in success_by_id:
            continue
        if merged[i] != source_by_id[tid]:
            raise PlannerCacheRetryError(
                f"non-retry success row mutated during merge for {tid}"
            )

    merged_sha = atomic_write_jsonl(outputs_path, merged)
    # Confirm attempt1 raw still intact.
    attempt_outputs = attempt_dir / "planner_outputs.jsonl"
    if file_sha256(attempt_outputs) != source_sha:
        raise PlannerCacheRetryError(
            "attempt snapshot planner_outputs mutated after promote (fail-closed)"
        )

    combined_costs = {
        "llm_calls": original_costs["llm_calls"] + retry_costs["llm_calls"],
        "tokens_in": original_costs["tokens_in"] + retry_costs["tokens_in"],
        "tokens_out": original_costs["tokens_out"] + retry_costs["tokens_out"],
        "tokens_total": original_costs["tokens_total"] + retry_costs["tokens_total"],
        "note": (
            "combined = original cache costs (including prior failed attempts) "
            "+ retry attempt costs; not replaced by 155 alone"
        ),
    }

    # Refresh summary / error summary / metadata cost fields; keep identity file.
    summary = {
        "n_tasks": 155,
        "n_rows": 155,
        "n_errors": 0,
        "planner_cache_sha256": merged_sha,
        "identity": source_identity,
        "output_path": str(outputs_path),
        "errors": [],
        "retry_promoted": True,
        "original_costs": original_costs,
        "retry_costs": retry_costs,
        "combined_costs": combined_costs,
        "source_planner_cache_sha256": source_sha,
        "attempt_snapshot": attempt_name,
    }
    atomic_write_json(cache_dir / "planner_only_summary.json", summary)
    atomic_write_json(cache_dir / "error_summary.json", {"errors": []})

    # Preserve prior run_metadata if present; annotate costs.
    meta_path = cache_dir / "run_metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    else:
        meta = {}
    meta["planner_cache_retry"] = {
        "promoted": True,
        "source_sha256": source_sha,
        "merged_sha256": merged_sha,
        "retry_protocol_version": PLANNER_FAILED_RETRY_PROTOCOL_VERSION,
        "original_costs": original_costs,
        "retry_costs": retry_costs,
        "combined_costs": combined_costs,
    }
    meta["planner_only_summary"] = {
        **(meta.get("planner_only_summary") or {}),
        "planner_cache_sha256": merged_sha,
        "n_errors": 0,
        "original_costs": original_costs,
        "retry_costs": retry_costs,
        "combined_costs": combined_costs,
    }
    atomic_write_json(meta_path, meta)

    report = {
        **provenance_base,
        "promoted": True,
        "retry_result": "success",
        "still_failed_ids": [],
        "retry_outputs_sha256": retry_outputs_sha,
        "merged_planner_outputs_sha256": merged_sha,
        "retry_costs": retry_costs,
        "original_costs": original_costs,
        "combined_costs": combined_costs,
        "formal_step_pipeline_allowed": True,
        "complete_155_ok": True,
    }
    atomic_write_json(cache_dir / "retry_provenance.json", report)
    atomic_write_json(retry_art_dir / "retry_report.json", report)
    return report
