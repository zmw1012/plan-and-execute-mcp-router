"""Offline zero-network Checkpoint Audit runner (read-only raw artifacts)."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from hk_equity_mcp.experiment.pe_fin_lite_checkpoint_audit import (
    AUDIT_VERSION,
    CHECKPOINT_IDS,
    CHECKPOINT_TRACE_CONTRACT_VERSION,
    DEFAULT_AUDIT_SUBDIR,
    EXPLICITLY_NOT_FORMAL_ACCURACY,
    RESULT_ROLE,
)
from hk_equity_mcp.experiment.pe_fin_lite_checkpoint_audit.engineering_pass import (
    assess_engineering_pass,
    funnel_counts,
)
from hk_equity_mcp.experiment.pe_fin_lite_checkpoint_audit.evaluate import (
    evaluate_task_checkpoints,
)
from hk_equity_mcp.pe_fin_lite.checkpoint_observability import (
    checkpoint_observability_hash,
)

RAW_CANDIDATES = (
    "predictions.jsonl",
    "run_metadata.json",
    "error_summary.json",
    "cost_summary.json",
    "grading_summary.json",
    "per_task_metrics.csv",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_predictions(path: Path) -> List[Dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def collect_raw_hashes(run_dir: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for name in RAW_CANDIDATES:
        p = run_dir / name
        if p.exists() and p.is_file():
            out[str(p.relative_to(run_dir))] = {
                "sha256": sha256_file(p),
                "bytes": p.stat().st_size,
            }
    # Also hash any *.jsonl under run_dir root only (not audit subdir).
    for p in sorted(run_dir.glob("*.jsonl")):
        rel = str(p.relative_to(run_dir))
        if rel not in out:
            out[rel] = {"sha256": sha256_file(p), "bytes": p.stat().st_size}
    return out


def _slice_analysis(
    per_task: Sequence[Dict[str, Any]], checkpoint: str
) -> Dict[str, Any]:
    rows = []
    for t in per_task:
        c = (t.get("checkpoints") or {}).get(checkpoint) or {}
        rows.append(
            {
                "task_id": t.get("task_id"),
                "status": c.get("status"),
                "detail": c.get("detail"),
                "primary_blocking": t.get("primary_blocking_checkpoint") == checkpoint,
            }
        )
    cnt = Counter(r["status"] for r in rows)
    return {
        "checkpoint": checkpoint,
        "counts": dict(cnt),
        "tasks": rows,
    }


def run_checkpoint_audit(
    *,
    run_dir: Path,
    output_dir: Optional[Path] = None,
    expected_n: int = 12,
    docs_mirror: Optional[Path] = None,
    allow_missing_predictions: bool = False,
) -> Dict[str, Any]:
    """Read-only audit. Writes only under output_dir (default run_dir/final_checkpoint_audit_v1)."""
    run_dir = Path(run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir not found: {run_dir}")
    out_dir = Path(output_dir) if output_dir else (run_dir / DEFAULT_AUDIT_SUBDIR)
    if out_dir.resolve() == run_dir.resolve():
        raise RuntimeError("Refuse to write audit into run_dir root (would mix with raw)")
    # Never write into raw artifact filenames.
    out_dir.mkdir(parents=True, exist_ok=True)

    before = collect_raw_hashes(run_dir)
    pred_path = run_dir / "predictions.jsonl"
    if not pred_path.exists():
        if not allow_missing_predictions:
            raise FileNotFoundError(f"missing predictions.jsonl under {run_dir}")
        predictions: List[Dict[str, Any]] = []
    else:
        predictions = _load_predictions(pred_path)

    meta_path = run_dir / "run_metadata.json"
    run_metadata = _load_json(meta_path) if meta_path.exists() else {}

    per_task = [evaluate_task_checkpoints(row) for row in predictions]
    task_ids = [str(t.get("task_id") or "") for t in per_task]
    funnel = funnel_counts(per_task)

    # After hashes — must match before (audit writes only to out_dir).
    after = collect_raw_hashes(run_dir)
    unchanged = before == after

    eng = assess_engineering_pass(
        task_ids=task_ids,
        expected_n=expected_n,
        run_metadata=run_metadata,
        per_task=per_task,
        raw_hashes_unchanged=unchanged,
        mock=bool(run_metadata.get("mock")),
        dry_run=bool(run_metadata.get("dry_run")),
    )

    primary_counts = Counter(
        t.get("primary_blocking_checkpoint") or "none_or_all_non_fail" for t in per_task
    )
    taxonomy = {
        "primary_blocking_checkpoint_counts": dict(primary_counts),
        "secondary_failure_counts": dict(
            Counter(
                s for t in per_task for s in (t.get("secondary_failures") or [])
            )
        ),
        "agent_error_types": dict(
            Counter(str(t.get("agent_error_type") or "none") for t in per_task)
        ),
        "note": "Accuracy failures are not remapped to upstream pass.",
    }

    resource = {
        "n_tasks": len(per_task),
        "sum_llm_calls": sum(int((predictions[i].get("llm_calls") or 0)) for i in range(len(predictions))),
        "sum_tokens": sum(
            int(((predictions[i].get("tokens") or {}).get("total_tokens") or 0))
            for i in range(len(predictions))
        ),
        "sum_mcp": sum(
            int(predictions[i].get("real_mcp_execution_count") or 0)
            for i in range(len(predictions))
        ),
        "sum_embedding_calls": sum(
            int(predictions[i].get("embedding_calls") or 0) for i in range(len(predictions))
        ),
    }

    case_studies = []
    for t in per_task:
        if t.get("primary_blocking_checkpoint"):
            case_studies.append(
                {
                    "task_id": t.get("task_id"),
                    "primary": t.get("primary_blocking_checkpoint"),
                    "secondary": t.get("secondary_failures"),
                    "answer_correct": t.get("answer_correct"),
                }
            )

    summary = {
        "audit_version": AUDIT_VERSION,
        "result_role": RESULT_ROLE,
        "explicitly_not_formal_accuracy": EXPLICITLY_NOT_FORMAL_ACCURACY,
        "checkpoint_trace_contract_version": CHECKPOINT_TRACE_CONTRACT_VERSION,
        "observability_hash": checkpoint_observability_hash(),
        "run_dir": str(run_dir),
        "n_predictions": len(predictions),
        "unique_task_ids": len(set(task_ids)),
        "ordered_task_ids": task_ids,
        "engineering_pass": eng.get("engineering_pass"),
        "engineering": eng,
        "funnel": funnel,
        "raw_hashes_unchanged": unchanged,
    }

    # Write artifacts
    _write_json(out_dir / "raw_hashes.json", {"before": before, "after": after, "unchanged": unchanged})
    _write_json(out_dir / "checkpoint_summary.json", summary)
    _write_json(out_dir / "checkpoint_funnel.json", funnel)
    _write_json(out_dir / "error_taxonomy.json", taxonomy)
    _write_json(out_dir / "resource_summary.json", resource)
    _write_json(out_dir / "case_study_index.json", {"cases": case_studies})
    _write_json(out_dir / "filter_analysis.json", _slice_analysis(per_task, "C02_filter"))
    _write_json(out_dir / "retrieval_analysis.json", {
        "C03": _slice_analysis(per_task, "C03_embedding"),
        "C04": _slice_analysis(per_task, "C04_describe"),
    })
    _write_json(
        out_dir / "binder_argument_analysis.json",
        {
            "C05": _slice_analysis(per_task, "C05_binder"),
            "C06": _slice_analysis(per_task, "C06_arguments"),
        },
    )
    _write_json(out_dir / "mcp_analysis.json", _slice_analysis(per_task, "C07_mcp"))
    _write_json(out_dir / "normalization_analysis.json", _slice_analysis(per_task, "C08_normalize"))
    _write_json(
        out_dir / "evidence_acceptance_analysis.json",
        _slice_analysis(per_task, "C09_evidence_acceptance"),
    )
    _write_json(
        out_dir / "fallback_unavailable_analysis.json",
        _slice_analysis(per_task, "C10_fallback_unavailable"),
    )
    _write_json(out_dir / "compute_analysis.json", _slice_analysis(per_task, "C11_compute"))
    _write_json(out_dir / "reason_analysis.json", _slice_analysis(per_task, "C12_reason"))
    _write_json(
        out_dir / "final_evaluator_analysis.json",
        {
            "C13": _slice_analysis(per_task, "C13_final"),
            "C14": _slice_analysis(per_task, "C14_evaluator"),
        },
    )

    jsonl_path = out_dir / "per_task_checkpoints.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for t in per_task:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    csv_path = out_dir / "per_task_checkpoints.csv"
    fieldnames = ["task_id", "domain", "primary_blocking_checkpoint", "answer_correct", "status"] + list(
        CHECKPOINT_IDS
    )
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for t in per_task:
            row = {
                "task_id": t.get("task_id"),
                "domain": t.get("domain"),
                "primary_blocking_checkpoint": t.get("primary_blocking_checkpoint"),
                "answer_correct": t.get("answer_correct"),
                "status": t.get("status"),
            }
            for cid in CHECKPOINT_IDS:
                row[cid] = ((t.get("checkpoints") or {}).get(cid) or {}).get("status")
            w.writerow(row)

    report = _render_report(summary, funnel, taxonomy, eng)
    (out_dir / "audit_report.md").write_text(report, encoding="utf-8")

    # Re-check hashes after all writes
    after2 = collect_raw_hashes(run_dir)
    if after2 != before:
        raise RuntimeError("FATAL: checkpoint audit mutated raw run artifacts")

    if docs_mirror:
        docs_mirror = Path(docs_mirror)
        docs_mirror.mkdir(parents=True, exist_ok=True)
        for p in out_dir.iterdir():
            if p.is_file():
                shutil.copy2(p, docs_mirror / p.name)

    return {
        "output_dir": str(out_dir),
        "engineering_pass": eng.get("engineering_pass"),
        "raw_hashes_unchanged": True,
        "summary": summary,
    }


def _render_report(
    summary: Dict[str, Any],
    funnel: Dict[str, Any],
    taxonomy: Dict[str, Any],
    eng: Dict[str, Any],
) -> str:
    lines = [
        "# PE-Fin Lite End-to-End Checkpoint Audit v1",
        "",
        f"- audit_version: `{summary.get('audit_version')}`",
        f"- result_role: `{summary.get('result_role')}`",
        f"- explicitly_not_formal_accuracy: `{summary.get('explicitly_not_formal_accuracy')}`",
        f"- engineering_pass: `{eng.get('engineering_pass')}`",
        f"- n_predictions: {summary.get('n_predictions')}",
        f"- unique_task_ids: {summary.get('unique_task_ids')}",
        f"- raw_hashes_unchanged: `{summary.get('raw_hashes_unchanged')}`",
        f"- checkpoint_trace_contract_version: `{summary.get('checkpoint_trace_contract_version')}`",
        "",
        "## Engineering pass (not accuracy)",
        "",
        "```json",
        json.dumps(eng, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Checkpoint funnel",
        "",
    ]
    for cid, stats in funnel.items():
        lines.append(
            f"- {cid}: pass={stats.get('pass')} fail={stats.get('fail')} "
            f"N/A={stats.get('N/A')} unknown={stats.get('unknown')} "
            f"(pass_rate_excl_NA={stats.get('pass_rate_excluding_na'):.3f})"
        )
    lines.extend(
        [
            "",
            "## Primary blocking taxonomy",
            "",
            "```json",
            json.dumps(taxonomy.get("primary_blocking_checkpoint_counts"), indent=2),
            "```",
            "",
            "## Notes",
            "",
            "- N/A is not counted as pass.",
            "- Upstream failures remain failures even if final answer happens to look correct.",
            "- Evaluator (C14) is separated from agent/method errors.",
            "- Zero MCP invocation is never recorded as tool success.",
            "",
        ]
    )
    return "\n".join(lines) + "\n"
