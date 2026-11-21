#!/usr/bin/env python3
"""ETOM Empty-Route Control (fully offline).

This is NOT an intelligent MCP-Free agent. Every task predicts an empty
operation set. No LLM / Embedding / MCP / network calls.

Reads the frozen formal_100_v1.lock.json and scores with the ETOM formal
evaluator imported from the sibling ETOM worktree (read-only).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCK = (
    ROOT.parent
    / "hk-equity-mcp-agent-etom"
    / "benchmarks"
    / "etom"
    / "formal_100_v1.lock.json"
)
DEFAULT_ETOM_ROOT = ROOT.parent / "hk-equity-mcp-agent-etom"
CONTROL_NAME = "ETOM Empty-Route Control"


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    data = json.dumps(payload, ensure_ascii=False, indent=2)
    tmp.write_text(data, encoding="utf-8")
    tmp.replace(path)


def _import_etom_evaluator(etom_root: Path):
    etom_src = (etom_root / "src").resolve()
    if not etom_src.is_dir():
        raise SystemExit(f"ETOM src not found: {etom_src}")
    if str(etom_src) not in sys.path:
        sys.path.insert(0, str(etom_src))
    from hk_equity_mcp.etom.evaluator import evaluate_prediction  # noqa: WPS433

    return evaluate_prediction


def _load_ground_truth_from_etom(
    etom_root: Path, lock: Dict[str, Any]
) -> Dict[str, Any]:
    """Optional: load official GT via ETOM loader (still offline / local files)."""
    etom_src = (etom_root / "src").resolve()
    if str(etom_src) not in sys.path:
        sys.path.insert(0, str(etom_src))
    try:
        from hk_equity_mcp.etom.formal_sampler import select_formal_locked_tasks
        from hk_equity_mcp.etom.loader import load_official_tasks
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": type(exc).__name__, "by_id": {}}

    try:
        tasks = load_official_tasks(etom_root, verify=False)
        selected = select_formal_locked_tasks(tasks, lock, max_tasks=None)
        by_id = {t.task_id: t.ground_truth_tools for t in selected}
        return {"ok": True, "error": None, "by_id": by_id}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc), "by_id": {}}


def _reconstruct_gt_from_lock_row(row: Dict[str, Any]) -> Any:
    """Fallback GT reconstruction from lock metadata (offline, no source parse)."""
    level = int(row["level"])
    if level == 5 or row.get("infeasible"):
        return []
    servers = list(row.get("server_names") or [])
    tools = list(row.get("tool_names") or [])
    if not tools and row.get("operation_ids"):
        # operation_id format: "Server::tool"
        for oid in row["operation_ids"]:
            if "::" in str(oid):
                s, t = str(oid).split("::", 1)
                servers.append(s)
                tools.append(t)
    if not tools:
        # Non-empty placeholder so empty prediction is not vacuously exact on L3/L4.
        return [[{"server_name": "_lock_missing", "tool_name": "_lock_missing"}]]

    if level in (1, 2):
        # ETOM L1/L2: one step with optional EFS alternatives.
        return [
            [
                {
                    "server_name": servers[i] if i < len(servers) else "",
                    "tool_name": tools[i],
                    "dependencies": [],
                }
                for i in range(len(tools))
            ]
        ]

    # L3/L4: sequential steps (dependencies approximate; empty-pred scoring only
    # needs non-empty GT flat list).
    steps = []
    for i, tool_name in enumerate(tools):
        steps.append(
            {
                "server_name": servers[i] if i < len(servers) else "",
                "tool_name": tool_name,
                "dependencies": [i - 1] if i > 0 else [],
            }
        )
    return steps


def _mean(vals: Sequence[float]) -> Optional[float]:
    if not vals:
        return None
    return sum(vals) / len(vals)


def evaluate_empty_route(
    *,
    lock_path: Path,
    etom_root: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    evaluate_prediction = _import_etom_evaluator(etom_root)
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    tasks = list(lock.get("tasks") or [])
    if len(tasks) != 100:
        raise SystemExit(
            f"Expected 100 locked tasks, found {len(tasks)} in {lock_path}"
        )

    gt_bundle = _load_ground_truth_from_etom(etom_root, lock)
    gt_by_id = gt_bundle.get("by_id") or {}

    rows: List[Dict[str, Any]] = []
    by_level: Dict[int, List[Dict[str, Any]]] = defaultdict(list)

    for row in tasks:
        task_id = row["task_id"]
        level = int(row["level"])
        if task_id in gt_by_id:
            ground_truth = gt_by_id[task_id]
            gt_source = "etom_official_loader"
        else:
            ground_truth = _reconstruct_gt_from_lock_row(row)
            gt_source = "lock_metadata_reconstruction"

        # Empty-route prediction: no operations.
        prediction: List[Any] = []
        metrics = evaluate_prediction(level, ground_truth, prediction)
        record = {
            "task_id": task_id,
            "level": level,
            "control": CONTROL_NAME,
            "predicted_operations": [],
            "ground_truth_source": gt_source,
            "exact_match": bool(metrics.get("exact_match")),
            "precision": metrics.get("precision"),
            "recall": metrics.get("recall"),
            "f1": metrics.get("f1"),
            "status": metrics.get("status"),
            "infeasible": bool(row.get("infeasible")),
            "network_calls": 0,
            "llm_calls": 0,
            "embedding_calls": 0,
            "mcp_calls": 0,
        }
        rows.append(record)
        by_level[level].append(record)

    def level_summary(level_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "n": len(level_rows),
            "exact_match": _mean(
                [1.0 if r["exact_match"] else 0.0 for r in level_rows]
            ),
            "precision": _mean(
                [
                    float(r["precision"])
                    for r in level_rows
                    if isinstance(r.get("precision"), (int, float))
                ]
            ),
            "recall": _mean(
                [
                    float(r["recall"])
                    for r in level_rows
                    if isinstance(r.get("recall"), (int, float))
                ]
            ),
            "f1": _mean(
                [
                    float(r["f1"])
                    for r in level_rows
                    if isinstance(r.get("f1"), (int, float))
                ]
            ),
        }

    per_level = {f"L{lv}": level_summary(by_level[lv]) for lv in range(1, 6)}
    overall = level_summary(rows)

    # Expectation checks (report only; do not hard-code scores).
    expectations = {
        "L1_L4_exact_match_expected": 0.0,
        "L5_exact_match_expected": 1.0,
    }
    anomalies: List[str] = []
    for lv in (1, 2, 3, 4):
        got = per_level[f"L{lv}"]["exact_match"]
        if got is None or abs(float(got) - 0.0) > 1e-9:
            anomalies.append(
                f"L{lv} exact_match={got} (expected 0.0). "
                "Likely schema/evaluator/GT wiring issue — do not hard-code."
            )
    l5 = per_level["L5"]["exact_match"]
    if l5 is None or abs(float(l5) - 1.0) > 1e-9:
        anomalies.append(
            f"L5 exact_match={l5} (expected 1.0 for empty-route). "
            "Likely schema/evaluator/GT wiring issue — do not hard-code."
        )

    report = {
        "control_name": CONTROL_NAME,
        "not_an_mcp_free_agent": True,
        "description": (
            "Fully offline empty-operation-set control against ETOM formal_100. "
            "No LLM/Embedding/MCP/network."
        ),
        "lock_path": str(lock_path),
        "etom_root": str(etom_root),
        "sample_size": len(rows),
        "seed": lock.get("seed"),
        "formal_sample_sha256": lock.get("sample_sha256")
        or lock.get("ordered_task_ids_sha256"),
        "ground_truth_load": {
            "official_loader_ok": bool(gt_bundle.get("ok")),
            "error": gt_bundle.get("error"),
            "official_loaded": len(gt_by_id),
        },
        "network_calls": 0,
        "llm_calls": 0,
        "embedding_calls": 0,
        "mcp_calls": 0,
        "overall": overall,
        "per_level": per_level,
        "expectations": expectations,
        "expectation_anomalies": anomalies,
        "expectation_ok": len(anomalies) == 0,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    pred_path = output_dir / "empty_route_predictions.jsonl"
    with pred_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    _atomic_write_json(output_dir / "empty_route_summary.json", report)

    print(CONTROL_NAME)
    print(f"tasks={len(rows)} network_calls=0")
    print(f"overall={json.dumps(overall)}")
    for lv in range(1, 6):
        print(f"L{lv}={json.dumps(per_level[f'L{lv}'])}")
    if anomalies:
        print("EXPECTATION ANOMALIES (schema/evaluator — not hard-coded):")
        for a in anomalies:
            print(f"  - {a}")
    else:
        print("expectation_ok=true (L1-L4 exact=0, L5 exact=1)")
    print(f"summary={output_dir / 'empty_route_summary.json'}")
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=CONTROL_NAME)
    p.add_argument("--lock", default=str(DEFAULT_LOCK))
    p.add_argument("--etom-root", default=str(DEFAULT_ETOM_ROOT))
    p.add_argument(
        "--output-dir",
        default=str(ROOT / "outputs" / "etom_empty_route_control_v1"),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    evaluate_empty_route(
        lock_path=Path(args.lock).resolve(),
        etom_root=Path(args.etom_root).resolve(),
        output_dir=Path(args.output_dir).resolve(),
    )


if __name__ == "__main__":
    main()
