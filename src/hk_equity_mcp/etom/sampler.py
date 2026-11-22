"""Seeded 10-task ETOM cost-pilot sampler.

The lock file is authoritative. A later 100/200 draw must reject these ids.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from hk_equity_mcp.etom import COST_PILOT_SEED, PILOT_MAX_TASKS
from hk_equity_mcp.etom.loader import EtomTask
from hk_equity_mcp.etom.source import lock_path

QUOTAS = {1: 2, 2: 2, 3: 2, 4: 2, 5: 2}
SAMPLER_ID = "seeded_weighted_server_coverage_v1"


class SamplerError(RuntimeError):
    pass


def sample_cost_pilot(
    tasks: Sequence[EtomTask], seed: int = COST_PILOT_SEED
) -> List[EtomTask]:
    if sum(QUOTAS.values()) != PILOT_MAX_TASKS:
        raise SamplerError("Cost-pilot quotas must sum to 10.")
    by_level: Dict[int, List[EtomTask]] = {level: [] for level in QUOTAS}
    for task in tasks:
        if task.level in by_level:
            by_level[task.level].append(task)
    rng = random.Random(seed)
    selected: List[EtomTask] = []
    chosen = set()
    used_servers = set()

    def pick(pool: Sequence[EtomTask]) -> EtomTask:
        eligible = [task for task in pool if task.task_id not in chosen]
        if not eligible:
            raise SamplerError("Cost-pilot sampler exhausted an eligible pool.")
        eligible.sort(key=lambda task: task.task_id)
        weights = [1 + len(task.server_names - used_servers) for task in eligible]
        return rng.choices(eligible, weights=weights, k=1)[0]

    def take(task: EtomTask) -> None:
        chosen.add(task.task_id)
        used_servers.update(task.server_names)
        selected.append(task)

    for level, quota in QUOTAS.items():
        pool = list(by_level[level])
        if level == 3:
            constrained = [task for task in pool if task.single_server_sequential]
            if not constrained:
                raise SamplerError("No L3 single-server sequential dependency task.")
            take(pick(constrained))
        elif level == 4:
            constrained = [task for task in pool if task.cross_server_dependency]
            if not constrained:
                raise SamplerError("No L4 cross-server dependency task.")
            take(pick(constrained))
        elif level == 5:
            pool = [task for task in pool if task.infeasible and not task.server_names]
            if len(pool) < quota:
                raise SamplerError("L5 official infeasible pool is smaller than 2.")
        while sum(1 for task in selected if task.level == level) < quota:
            take(pick(pool))

    if len(selected) != PILOT_MAX_TASKS:
        raise SamplerError(f"Sampler returned {len(selected)} tasks, expected 10.")
    return selected


def lock_document(tasks: Sequence[EtomTask], seed: int = COST_PILOT_SEED) -> dict:
    return {
        "dataset_split": "cost_pilot",
        "exclude_from_final_etom_sample": True,
        "seed": seed,
        "sampler": SAMPLER_ID,
        "max_tasks": PILOT_MAX_TASKS,
        "quotas": {f"L{level}": quota for level, quota in QUOTAS.items()},
        "constraints": {
            "L3_at_least_one_single_server_sequential_dependency": True,
            "L4_at_least_one_cross_server_dependency": True,
            "L5_official_infeasible_only": True,
            "server_coverage": "seeded weights prefer servers not yet selected",
        },
        "task_ids": [task.task_id for task in tasks],
        "tasks": [task.public_lock_row() for task in tasks],
        "note": (
            "These ids are a cost pilot only. Final 100/200 ETOM samples must "
            "exclude every task_id in this file."
        ),
    }


def write_lock(root: Path, tasks: Sequence[EtomTask], seed: int = COST_PILOT_SEED) -> Path:
    path = lock_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(lock_document(tasks, seed), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def load_lock(root: Path) -> dict:
    path = lock_path(root)
    if not path.exists():
        raise SamplerError(f"Missing cost-pilot lock: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def excluded_task_ids(root: Path) -> List[str]:
    payload = load_lock(root)
    return list(payload["task_ids"])


def assert_not_excluded(root: Path, task_ids: Iterable[str]) -> None:
    banned = set(excluded_task_ids(root))
    overlap = [task_id for task_id in task_ids if task_id in banned]
    if overlap:
        raise SamplerError(
            "Refusing ids already used by the ETOM cost pilot "
            f"(exclude_from_final_etom_sample): {overlap}"
        )


def assert_lock_matches(root: Path, tasks: Sequence[EtomTask]) -> dict:
    payload = load_lock(root)
    sampled = [task.task_id for task in sample_cost_pilot(tasks, int(payload["seed"]))]
    if sampled != list(payload["task_ids"]):
        raise SamplerError(
            "Cost-pilot lock does not match the deterministic sampler. "
            "Refusing to run a different 10 tasks."
        )
    if len(payload["task_ids"]) != PILOT_MAX_TASKS:
        raise SamplerError("Cost-pilot lock must contain exactly 10 task ids.")
    flags = payload.get("exclude_from_final_etom_sample")
    if flags is not True or payload.get("dataset_split") != "cost_pilot":
        raise SamplerError("Cost-pilot lock is missing exclusion flags.")
    return payload


def select_locked_tasks(tasks: Sequence[EtomTask], payload: dict) -> List[EtomTask]:
    by_id = {task.task_id: task for task in tasks}
    chosen = []
    for task_id in payload["task_ids"]:
        if task_id not in by_id:
            raise SamplerError(f"Locked task id {task_id} is not in the official data.")
        chosen.append(by_id[task_id])
    return chosen
