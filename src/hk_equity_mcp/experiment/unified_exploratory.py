"""HK-155 exploratory_all benchmark (pointer-only; never mutates frozen JSONL).

Role: method development + difficulty audit.
After the first real (non-mock, non-dry-run) exploratory_all Agent run, the
previously reserved 85 items cease to be unseen held-out / final-test material.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from hk_equity_mcp.experiment.runner import file_sha256
from hk_equity_mcp.experiment.unified_dataset import load_json, load_jsonl, normalize_task

EXPLORATORY_MANIFEST = "benchmarks/manifests/hk_unified_exploratory_all_v1.json"
EXPLORATORY_RUN_ORDER = "benchmarks/manifests/hk_unified_exploratory_all_run_order_v1.json"
RETIREMENT_POLICY_VERSION = "retirement_policy_v1"
BENCHMARK_ROLE = "method_development_and_difficulty_audit"
RUN_ORDER_SEED = 20261104
RUN_ORDER_POLICY = "interleave_domain_difficulty_company_v1"

SOURCE_SPECS: Tuple[Dict[str, Any], ...] = (
    {
        "domain": "financial",
        "source_split": "development",
        "path": "benchmarks/questions/financial_real_mcp_coverage_v1_3.jsonl",
        "expected_count": 30,
    },
    {
        "domain": "financial",
        "source_split": "previously_reserved",
        "path": "benchmarks/heldout/financial_test_30_v1_3.jsonl",
        "expected_count": 30,
    },
    {
        "domain": "market",
        "source_split": "development",
        "path": "benchmarks/questions/market_development_15_v1_1.jsonl",
        "expected_count": 15,
    },
    {
        "domain": "market",
        "source_split": "previously_reserved",
        "path": "benchmarks/heldout/market_test_20_v1_1.jsonl",
        "expected_count": 20,
    },
    {
        "domain": "news",
        "source_split": "development",
        "path": "benchmarks/questions/news_development_15_v1.jsonl",
        "expected_count": 15,
    },
    {
        "domain": "news",
        "source_split": "previously_reserved",
        "path": "benchmarks/heldout/news_test_20_v1.jsonl",
        "expected_count": 20,
    },
    {
        "domain": "cross_domain",
        "source_split": "development",
        "path": "benchmarks/questions/cross_domain_development_10_v1.jsonl",
        "expected_count": 10,
    },
    {
        "domain": "cross_domain",
        "source_split": "previously_reserved",
        "path": "benchmarks/heldout/cross_domain_test_15_v1.jsonl",
        "expected_count": 15,
    },
)


def _stable_json_hash(obj: Any) -> str:
    payload = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def exploratory_manifest_path(root: Path) -> Path:
    return (root / EXPLORATORY_MANIFEST).resolve()


def exploratory_run_order_path(root: Path) -> Path:
    return (root / EXPLORATORY_RUN_ORDER).resolve()


def normalize_exploratory_task(
    row: Dict[str, Any],
    *,
    domain: str,
    source_split: str,
) -> Dict[str, Any]:
    """In-memory normalization only — never writes back to frozen JSONL."""
    out = normalize_task(row, domain=domain, split=source_split)
    out["dataset_split"] = "exploratory_all"
    out["source_split"] = source_split
    out["benchmark_role"] = BENCHMARK_ROLE
    out["previously_reserved"] = source_split == "previously_reserved"
    # Preserve original file provenance for audits.
    out["original_dataset_split"] = row.get("dataset_split")
    return out


def load_exploratory_tasks(root: Path) -> List[Dict[str, Any]]:
    tasks: List[Dict[str, Any]] = []
    for spec in SOURCE_SPECS:
        path = (root / spec["path"]).resolve()
        rows = load_jsonl(path)
        if len(rows) != int(spec["expected_count"]):
            raise ValueError(
                f"exploratory_all {spec['domain']}/{spec['source_split']}: "
                f"expected {spec['expected_count']} in {spec['path']}, got {len(rows)}"
            )
        for row in rows:
            tasks.append(
                normalize_exploratory_task(
                    row,
                    domain=str(spec["domain"]),
                    source_split=str(spec["source_split"]),
                )
            )
    ids = [t["task_id"] for t in tasks]
    if len(ids) != 155:
        raise ValueError(f"exploratory_all expected 155 tasks, got {len(ids)}")
    if len(set(ids)) != 155:
        raise ValueError("Duplicate task_id in exploratory_all")
    return tasks


def build_run_order_ids(tasks: Sequence[Dict[str, Any]], *, seed: int = RUN_ORDER_SEED) -> List[str]:
    """Deterministic interleave across domain / difficulty / company."""
    rng = random.Random(seed)
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for t in tasks:
        buckets[str(t.get("domain"))].append(t)
    for domain in buckets:
        buckets[domain].sort(
            key=lambda x: (
                str(x.get("project_difficulty") or ""),
                str(x.get("company") or ""),
                str(x["task_id"]),
            )
        )
        # Light within-bucket shuffle that remains deterministic via rng.
        rng.shuffle(buckets[domain])
        buckets[domain].sort(
            key=lambda x: (
                # Keep difficulty bands visible while mixing companies.
                str(x.get("project_difficulty") or "Z"),
                str(x.get("company") or ""),
                hashlib.sha256(
                    f"{seed}:{x['task_id']}".encode("utf-8")
                ).hexdigest(),
            )
        )

    domain_cycle = ["financial", "market", "news", "cross_domain"]
    pointers = {d: 0 for d in domain_cycle}
    ordered: List[str] = []
    while len(ordered) < len(tasks):
        progressed = False
        for domain in domain_cycle:
            idx = pointers[domain]
            bucket = buckets.get(domain) or []
            if idx >= len(bucket):
                continue
            ordered.append(bucket[idx]["task_id"])
            pointers[domain] = idx + 1
            progressed = True
        if not progressed:
            break
    if len(ordered) != len(tasks) or len(set(ordered)) != len(tasks):
        raise RuntimeError("Failed to build complete exploratory run order")
    return ordered


def build_exploratory_artifacts(
    root: Path,
    *,
    base_commit: Optional[str] = None,
) -> Dict[str, Any]:
    """Construct locked manifest + run-order payloads (caller may write files)."""
    tasks = load_exploratory_tasks(root)
    sources = []
    for spec in SOURCE_SPECS:
        path = (root / spec["path"]).resolve()
        sources.append(
            {
                **spec,
                "sha256": file_sha256(path),
            }
        )
    order_ids = build_run_order_ids(tasks, seed=RUN_ORDER_SEED)
    run_order = {
        "run_order_id": "hk_unified_exploratory_all_run_order_v1",
        "benchmark_id": "hk_unified_exploratory_all_v1",
        "seed": RUN_ORDER_SEED,
        "policy": RUN_ORDER_POLICY,
        "expected_count": 155,
        "task_ids": order_ids,
    }
    run_order["run_order_hash"] = _stable_json_hash(
        {
            "seed": run_order["seed"],
            "policy": run_order["policy"],
            "task_ids": run_order["task_ids"],
        }
    )

    domain_counts: Dict[str, int] = defaultdict(int)
    source_counts: Dict[str, int] = defaultdict(int)
    for t in tasks:
        domain_counts[str(t["domain"])] += 1
        source_counts[str(t["source_split"])] += 1

    manifest = {
        "benchmark_id": "hk_unified_exploratory_all_v1",
        "role": BENCHMARK_ROLE,
        "total_count": 155,
        "development_source_count": 70,
        "previously_reserved_source_count": 85,
        "contains_previously_reserved_items": True,
        "previously_reserved_items_cease_to_be_heldout_after_first_real_agent_run": True,
        "not_for_unseen_final_claims": True,
        "no_final_test_claim": True,
        "retirement_policy_version": RETIREMENT_POLICY_VERSION,
        "cutoff_date": "2026-08-12",
        "timezone": "Asia/Hong_Kong",
        "base_commit": base_commit,
        "sources": sources,
        "domain_counts": dict(sorted(domain_counts.items())),
        "source_split_counts": dict(sorted(source_counts.items())),
        "run_order_path": EXPLORATORY_RUN_ORDER,
        "run_order_hash": run_order["run_order_hash"],
        "notes": [
            "Pointer-only manifest; frozen JSONL files are not copied or rewritten.",
            "Dry-run/mock must not be recorded as the first real held-out exposure.",
            "After first real exploratory_all Agent run, previously reserved 85 are retired.",
        ],
    }
    manifest["manifest_hash"] = _stable_json_hash(
        {k: v for k, v in manifest.items() if k != "manifest_hash"}
    )
    return {"manifest": manifest, "run_order": run_order, "tasks": tasks}


def load_exploratory_manifest(root: Path) -> Dict[str, Any]:
    path = exploratory_manifest_path(root)
    if not path.exists():
        raise FileNotFoundError(f"Missing exploratory manifest: {path}")
    return load_json(path)


def load_exploratory_run_order(root: Path) -> Dict[str, Any]:
    path = exploratory_run_order_path(root)
    if not path.exists():
        raise FileNotFoundError(f"Missing exploratory run-order: {path}")
    return load_json(path)


def load_exploratory_tasks_in_locked_order(root: Path) -> List[Dict[str, Any]]:
    tasks = load_exploratory_tasks(root)
    order = load_exploratory_run_order(root)
    ids = list(order["task_ids"])
    if len(ids) != 155 or len(set(ids)) != 155:
        raise ValueError("Locked exploratory run-order must contain 155 unique IDs")
    by_id = {t["task_id"]: t for t in tasks}
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise ValueError(f"Run-order IDs missing from tasks: {missing[:5]}")
    extra = set(by_id) - set(ids)
    if extra:
        raise ValueError(f"Tasks missing from run-order: {sorted(extra)[:5]}")
    # Verify hash stability against locked payload.
    expected = _stable_json_hash(
        {"seed": order["seed"], "policy": order["policy"], "task_ids": ids}
    )
    if order.get("run_order_hash") != expected:
        raise ValueError(
            "exploratory run-order hash mismatch "
            f"(file={order.get('run_order_hash')} computed={expected})"
        )
    return [by_id[i] for i in ids]


def assert_exploratory_unlocked(
    *,
    split: str,
    confirm_retire_heldout: bool,
) -> Dict[str, Any]:
    status = {
        "split": split,
        "confirm_retire_heldout": bool(confirm_retire_heldout),
        "allowed": False,
        "reason": "",
        "retirement_policy_version": RETIREMENT_POLICY_VERSION,
    }
    if split != "exploratory_all":
        status["allowed"] = True
        status["reason"] = "not_exploratory_all"
        return status
    if not confirm_retire_heldout:
        status["reason"] = (
            "exploratory_all is locked. Pass --split exploratory_all and "
            "--confirm-retire-heldout together. This retires previously reserved "
            "items from unseen-final-test status after the first real Agent run."
        )
        raise RuntimeError(status["reason"])
    status["allowed"] = True
    status["reason"] = "explicit_retire_heldout_confirmation"
    return status


def build_retirement_metadata(
    *,
    dry_run: bool,
    mock: bool,
    confirm_retire_heldout: bool,
    git_commit: Optional[str],
    prompt_hash: str,
    catalog_fingerprint: str,
    manifest_hash: str,
    run_order_hash: str,
    timestamp: str,
    existing_first_real: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Retirement is effective only for real (non-mock, non-dry-run) runs."""
    real = (not dry_run) and (not mock)
    meta: Dict[str, Any] = {
        "retirement_policy_version": RETIREMENT_POLICY_VERSION,
        "confirm_retire_heldout": bool(confirm_retire_heldout),
        "benchmark_role": BENCHMARK_ROLE,
        "unseen_final_claim_allowed": False,
        "not_for_unseen_final_claims": True,
        "no_final_test_claim": True,
        "dry_run": dry_run,
        "mock": mock,
        "manifest_hash": manifest_hash,
        "run_order_hash": run_order_hash,
    }
    if real:
        meta["heldout_status"] = "retired_into_exploratory_development"
        meta["retirement_effective"] = True
        first = existing_first_real or {
            "timestamp": timestamp,
            "git_commit": git_commit,
            "system_prompt_hash": prompt_hash,
            "catalog_fingerprint": catalog_fingerprint,
            "manifest_hash": manifest_hash,
            "run_order_hash": run_order_hash,
        }
        meta["first_real_exploratory_run"] = first
    else:
        meta["heldout_status"] = "pending_real_exploratory_run"
        meta["retirement_effective"] = False
        meta["note"] = (
            "dry-run/mock show retirement policy but do not count as first real exposure"
        )
    return meta
