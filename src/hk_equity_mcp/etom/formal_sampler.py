"""Deterministic equal-level stratified ETOM-100 external formal sampler.

This is not a population-proportional draw. Quotas are 20 tasks per level.
Cost-pilot ids are always excluded. The lock file is authoritative after freeze.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set

from hk_equity_mcp.etom import (
    FORMAL_100_MAX_TASKS,
    FORMAL_100_QUOTAS,
    FORMAL_100_SAMPLER,
    FORMAL_100_SEED,
)
from hk_equity_mcp.etom.loader import EtomTask
from hk_equity_mcp.etom.sampler import SamplerError, excluded_task_ids, load_lock
from hk_equity_mcp.etom.source import (
    file_sha256,
    formal_100_lock_path,
    load_pin,
    lock_path,
    pin_path,
)

MAX_SERVER_FREQUENCY = 2


def _eligible_pool(level: int, tasks: Sequence[EtomTask], banned: Set[str]) -> List[EtomTask]:
    pool = [
        task
        for task in tasks
        if task.level == level and task.task_id not in banned
    ]
    if level == 1:
        pool = [task for task in pool if task.step_count == 1]
    elif level == 3:
        pool = [task for task in pool if task.single_server_sequential]
    elif level == 4:
        pool = [task for task in pool if task.cross_server_dependency]
    elif level == 5:
        pool = [
            task
            for task in pool
            if task.infeasible and not task.server_names and not task.operation_ids
        ]
    pool.sort(key=lambda task: task.task_id)
    return pool


def _would_exceed_server_cap(
    task: EtomTask, server_counts: Counter, *, cap: int = MAX_SERVER_FREQUENCY
) -> bool:
    if not task.server_names:
        return False
    return any(server_counts[server] >= cap for server in task.server_names)


def sample_formal_100(
    tasks: Sequence[EtomTask],
    *,
    seed: int = FORMAL_100_SEED,
    banned_ids: Optional[Iterable[str]] = None,
) -> List[EtomTask]:
    if dict(FORMAL_100_QUOTAS) != {1: 20, 2: 20, 3: 20, 4: 20, 5: 20}:
        raise SamplerError("Formal-100 quotas must be exactly 20 per level.")
    if sum(FORMAL_100_QUOTAS.values()) != FORMAL_100_MAX_TASKS:
        raise SamplerError("Formal-100 quotas must sum to 100.")
    banned = set(banned_ids or [])
    rng = random.Random(seed)
    selected: List[EtomTask] = []
    chosen: Set[str] = set()
    used_servers: Set[str] = set()
    server_counts: Counter = Counter()

    for level, quota in FORMAL_100_QUOTAS.items():
        pool = _eligible_pool(level, tasks, banned)
        if len(pool) < quota:
            raise SamplerError(
                f"Formal-100 L{level} eligible pool has {len(pool)} tasks; need {quota}. "
                "Fail closed; quotas and constraints were not relaxed."
            )
        level_selected = 0
        while level_selected < quota:
            remaining = [
                task
                for task in pool
                if task.task_id not in chosen
                and not _would_exceed_server_cap(task, server_counts)
            ]
            if not remaining:
                raise SamplerError(
                    f"Formal-100 L{level} cannot fill quota={quota} under "
                    f"max_server_frequency={MAX_SERVER_FREQUENCY} without relaxing "
                    "constraints. Fail closed."
                )
            weights = [
                1 + len(task.server_names - used_servers) for task in remaining
            ]
            pick = rng.choices(remaining, weights=weights, k=1)[0]
            chosen.add(pick.task_id)
            used_servers.update(pick.server_names)
            for server in pick.server_names:
                server_counts[server] += 1
            selected.append(pick)
            level_selected += 1

    if len(selected) != FORMAL_100_MAX_TASKS:
        raise SamplerError(f"Formal sampler returned {len(selected)} tasks, expected 100.")
    if len({task.task_id for task in selected}) != FORMAL_100_MAX_TASKS:
        raise SamplerError("Formal sampler produced duplicate task ids.")
    overlap = sorted(banned & {task.task_id for task in selected})
    if overlap:
        raise SamplerError(f"Formal sample overlaps cost-pilot ids: {overlap}")
    return selected


def ordered_task_ids_sha256(task_ids: Sequence[str]) -> str:
    blob = "\n".join(task_ids).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def sample_document_sha256(payload: dict) -> str:
    """Stable hash over the frozen sample identity (not pretty-printed whitespace)."""
    identity = {
        "dataset_split": payload.get("dataset_split"),
        "benchmark_role": payload.get("benchmark_role"),
        "sample_size": payload.get("sample_size"),
        "seed": payload.get("seed"),
        "sampler_version": payload.get("sampler_version"),
        "source_commit": payload.get("source_commit"),
        "excluded_task_ids": payload.get("excluded_task_ids"),
        "quotas": payload.get("quotas"),
        "ordered_task_ids": payload.get("ordered_task_ids"),
        "tasks": [
            {
                "task_id": row["task_id"],
                "level": row["level"],
                "query_id": row["query_id"],
                "source_file": row["source_file"],
                "query_sha256": row["query_sha256"],
                "ground_truth_sha256": row["ground_truth_sha256"],
                "selected_order": row["selected_order"],
            }
            for row in payload.get("tasks") or []
        ],
    }
    blob = json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def build_formal_100_lock(
    root: Path,
    tasks: Sequence[EtomTask],
    *,
    seed: int = FORMAL_100_SEED,
) -> dict:
    pin = load_pin(root)
    pilot_lock = load_lock(root)
    banned = list(pilot_lock["task_ids"])
    selected = sample_formal_100(tasks, seed=seed, banned_ids=banned)
    task_rows = [
        task.formal_lock_row(selected_order=index)
        for index, task in enumerate(selected, start=1)
    ]
    ordered_ids = [task.task_id for task in selected]
    payload = {
        "dataset_split": "external_formal",
        "benchmark_role": "external_validation",
        "sample_size": FORMAL_100_MAX_TASKS,
        "seed": seed,
        "sampler_version": FORMAL_100_SAMPLER,
        "source_repository": pin["repository_url"],
        "source_commit": pin["commit_sha"],
        "source_pin_path": str(pin_path(root).relative_to(root)),
        "source_pin_sha256": file_sha256(pin_path(root)),
        "excluded_cost_pilot_lock_path": str(lock_path(root).relative_to(root)),
        "excluded_cost_pilot_lock_sha256": file_sha256(lock_path(root)),
        "excluded_task_ids": banned,
        "quotas": {f"L{level}": quota for level, quota in FORMAL_100_QUOTAS.items()},
        "constraints": {
            "equal_level_stratified": True,
            "not_population_proportional": True,
            "exclude_cost_pilot_ids": True,
            "L1_single_step_only": True,
            "L3_all_single_server_sequential": True,
            "L4_all_cross_server_dependency": True,
            "L5_official_infeasible_empty_route_only": True,
            "max_server_frequency_per_level": MAX_SERVER_FREQUENCY,
            "server_coverage": (
                "seeded weights prefer servers not yet selected; "
                "hard cap max_server_frequency=2 within each level"
            ),
        },
        "ordered_task_ids": ordered_ids,
        "ordered_task_ids_sha256": ordered_task_ids_sha256(ordered_ids),
        "tasks": task_rows,
        "reporting_contract": {
            "macro_average_across_five_levels": True,
            "per_level_results": True,
            "pooled_100_task_result": True,
            "pooled_is_not_etom_natural_distribution_estimate": True,
        },
        "anti_leakage": {
            "cost_pilot_exposed": True,
            "cost_pilot_overlap_must_be_zero": True,
            "do_not_tune_proposed_method_on_etom_100": True,
            "proposed_method_primary_dev_signal": "HK-155 development error taxonomy",
            "etom_100_role": (
                "external comparison of Operation-Visible vs the final frozen method"
            ),
        },
        "note": (
            "Equal-level stratified external formal sample (20 per L1–L5). "
            "Not a population-weighted ETOM draw. Cost-pilot ids are excluded."
        ),
    }
    payload["sample_sha256"] = sample_document_sha256(payload)
    return payload


def write_formal_100_lock(root: Path, payload: dict) -> Path:
    path = formal_100_lock_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return path


def load_formal_100_lock(root: Path) -> dict:
    path = formal_100_lock_path(root)
    if not path.exists():
        raise SamplerError(f"Missing formal-100 lock: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def assert_formal_100_lock_matches(root: Path, tasks: Sequence[EtomTask]) -> dict:
    payload = load_formal_100_lock(root)
    rebuilt = build_formal_100_lock(root, tasks, seed=int(payload["seed"]))
    if rebuilt["ordered_task_ids"] != payload["ordered_task_ids"]:
        raise SamplerError(
            "Formal-100 lock does not match the deterministic sampler. "
            "Refusing to use a drifted sample."
        )
    if rebuilt["sample_sha256"] != payload["sample_sha256"]:
        raise SamplerError("Formal-100 sample_sha256 drifted from the sampler.")
    if payload.get("dataset_split") != "external_formal":
        raise SamplerError("Formal-100 lock dataset_split must be external_formal.")
    if int(payload.get("sample_size") or 0) != FORMAL_100_MAX_TASKS:
        raise SamplerError("Formal-100 lock sample_size must be 100.")
    banned = set(payload.get("excluded_task_ids") or [])
    overlap = banned.intersection(payload["ordered_task_ids"])
    if overlap:
        raise SamplerError(f"Formal-100 lock overlaps cost-pilot ids: {sorted(overlap)}")
    return payload


def select_formal_locked_tasks(
    tasks: Sequence[EtomTask], payload: dict, *, max_tasks: Optional[int] = None
) -> List[EtomTask]:
    by_id = {task.task_id: task for task in tasks}
    ordered = []
    for task_id in payload["ordered_task_ids"]:
        if task_id not in by_id:
            raise SamplerError(f"Locked formal task id {task_id} missing from official data.")
        ordered.append(by_id[task_id])
    if max_tasks is None:
        return ordered
    limit = int(max_tasks)
    if limit < 1 or limit > len(ordered):
        raise SamplerError(
            f"max_tasks must be in 1..{len(ordered)} for the frozen formal sample."
        )
    return ordered[:limit]


def summarize_formal_sample(tasks: Sequence[EtomTask], payload: dict) -> dict:
    selected = select_formal_locked_tasks(tasks, payload)
    server_counts = Counter(
        server for task in selected for server in task.server_names
    )
    step_counts = Counter(task.step_count for task in selected)
    level_counts = Counter(task.level for task in selected)
    l5_subtypes = Counter(
        task.infeasible_subtype or "unknown" for task in selected if task.level == 5
    )
    return {
        "sample_size": len(selected),
        "level_counts": {f"L{level}": level_counts[level] for level in range(1, 6)},
        "unique_servers": len(server_counts),
        "top_server_frequencies": server_counts.most_common(15),
        "step_count_distribution": dict(sorted(step_counts.items())),
        "dependency": {
            "has_dependency": sum(1 for task in selected if task.has_dependency),
            "single_server_sequential": sum(
                1 for task in selected if task.single_server_sequential
            ),
            "cross_server_dependency": sum(
                1 for task in selected if task.cross_server_dependency
            ),
            "L3_single_server_sequential": sum(
                1 for task in selected if task.level == 3 and task.single_server_sequential
            ),
            "L4_cross_server_dependency": sum(
                1 for task in selected if task.level == 4 and task.cross_server_dependency
            ),
        },
        "l5_subtype_distribution": dict(l5_subtypes),
        "pilot_overlap": sorted(
            set(payload.get("excluded_task_ids") or []).intersection(
                payload.get("ordered_task_ids") or []
            )
        ),
        "exact_duplicate_task_ids": [
            task_id
            for task_id, count in Counter(payload.get("ordered_task_ids") or []).items()
            if count > 1
        ],
    }
