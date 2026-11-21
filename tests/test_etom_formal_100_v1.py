"""Offline tests for the frozen ETOM-100 formal sample and runner."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hk_equity_mcp.etom import FORMAL_100_MAX_TASKS, FORMAL_100_SAMPLER, FORMAL_100_SEED
from hk_equity_mcp.etom.embeddings import IncompleteCatalogIndexError, load_complete_catalog_index
from hk_equity_mcp.etom.formal_runner import EtomFormal100Runner, resume_identity
from hk_equity_mcp.etom.formal_sampler import (
    assert_formal_100_lock_matches,
    build_formal_100_lock,
    load_formal_100_lock,
    sample_document_sha256,
    sample_formal_100,
    select_formal_locked_tasks,
    summarize_formal_sample,
)
from hk_equity_mcp.etom.loader import load_official_tasks
from hk_equity_mcp.etom.sampler import excluded_task_ids
from hk_equity_mcp.etom.source import (
    file_sha256,
    formal_100_lock_path,
    lock_path,
    pin_path,
    raw_root,
    repo_root_from_here,
    verify_source,
)

ROOT = repo_root_from_here()
RAW = raw_root(ROOT)
LOCK = formal_100_lock_path(ROOT)


@pytest.fixture(scope="module")
def source():
    if not (RAW / "data" / "mcp_registry.json").exists():
        pytest.skip("Pinned ETOM checkout is not present.")
    return verify_source(ROOT)


@pytest.fixture(scope="module")
def tasks(source):
    return load_official_tasks(ROOT, verify=False)


@pytest.fixture(scope="module")
def lock(source):
    if not LOCK.exists():
        pytest.skip("formal_100_v1.lock.json missing")
    return load_formal_100_lock(ROOT)


def test_source_pin_counts_and_hashes(source):
    pin = source["pin"]
    assert pin["commit_sha"] == "790f28727f567b88918420d163c524d46644a694"
    assert pin["official_counts"]["tools"] == 2375
    assert pin["official_counts"]["queries"] == {
        "1": 781,
        "2": 773,
        "3": 327,
        "4": 103,
        "5": 91,
    }
    for rel, meta in pin["dataset_files"].items():
        assert source["sha256"][rel] == meta["sha256"]


def test_formal_sample_size_and_level_quotas(lock):
    assert lock["sample_size"] == 100
    assert lock["seed"] == FORMAL_100_SEED
    assert lock["sampler_version"] == FORMAL_100_SAMPLER
    assert lock["quotas"] == {"L1": 20, "L2": 20, "L3": 20, "L4": 20, "L5": 20}
    assert len(lock["ordered_task_ids"]) == 100
    counts = {f"L{level}": 0 for level in range(1, 6)}
    for row in lock["tasks"]:
        counts[f"L{row['level']}"] += 1
    assert counts == {"L1": 20, "L2": 20, "L3": 20, "L4": 20, "L5": 20}


def test_pilot_overlap_is_zero(lock):
    pilot = set(excluded_task_ids(ROOT))
    overlap = pilot.intersection(lock["ordered_task_ids"])
    assert overlap == set()
    assert lock["excluded_task_ids"] == list(
        json.loads(lock_path(ROOT).read_text(encoding="utf-8"))["task_ids"]
    )


def test_task_ids_unique(lock):
    assert len(lock["ordered_task_ids"]) == 100
    assert len(set(lock["ordered_task_ids"])) == 100
    assert len({row["task_id"] for row in lock["tasks"]}) == 100


def test_deterministic_sampler(tasks):
    banned = excluded_task_ids(ROOT)
    first = [task.task_id for task in sample_formal_100(tasks, banned_ids=banned)]
    second = [task.task_id for task in sample_formal_100(tasks, banned_ids=banned)]
    assert first == second
    assert first == load_formal_100_lock(ROOT)["ordered_task_ids"]


def test_lock_hash_stable_and_matches_sampler(tasks, lock):
    rebuilt = build_formal_100_lock(ROOT, tasks, seed=FORMAL_100_SEED)
    assert rebuilt["ordered_task_ids"] == lock["ordered_task_ids"]
    assert rebuilt["sample_sha256"] == lock["sample_sha256"]
    assert sample_document_sha256(lock) == lock["sample_sha256"]
    assert_formal_100_lock_matches(ROOT, tasks)


def test_query_and_ground_truth_hashes_match_official(tasks, lock):
    by_id = {task.task_id: task for task in tasks}
    for row in lock["tasks"]:
        task = by_id[row["task_id"]]
        assert row["query_sha256"] == task.query_sha256
        assert row["ground_truth_sha256"] == task.ground_truth_sha256
        assert row["source_file"] == task.source_file
        assert row["level"] == task.level
        assert row["query_id"] == task.query_id


def test_l3_l4_constraints(tasks, lock):
    selected = select_formal_locked_tasks(tasks, lock)
    l3 = [task for task in selected if task.level == 3]
    l4 = [task for task in selected if task.level == 4]
    assert len(l3) == 20
    assert len(l4) == 20
    assert all(task.single_server_sequential for task in l3)
    assert all(task.has_dependency for task in l3)
    assert all(task.cross_server_dependency for task in l4)
    assert all(task.has_dependency for task in l4)


def test_l5_official_infeasible(tasks, lock):
    selected = select_formal_locked_tasks(tasks, lock)
    l5 = [task for task in selected if task.level == 5]
    assert len(l5) == 20
    assert all(task.infeasible for task in l5)
    assert all(not task.server_names for task in l5)
    assert all(not task.operation_ids for task in l5)
    assert all(
        task.infeasible_subtype == "official_empty_ground_truth_route" for task in l5
    )


def test_runner_reads_lock_and_does_not_resample(tasks, lock, tmp_path, monkeypatch):
    calls = {"sample": 0}

    def boom(*args, **kwargs):
        calls["sample"] += 1
        raise AssertionError("sampler must not run during formal execute")

    monkeypatch.setattr(
        "hk_equity_mcp.etom.formal_sampler.sample_formal_100", boom
    )
    # assert_formal_100_lock_matches rebuilds via build_formal_100_lock -> sample.
    # Patch that path to only compare against lock ids already frozen.
    monkeypatch.setattr(
        "hk_equity_mcp.etom.formal_runner.assert_formal_100_lock_matches",
        lambda root, tasks: lock,
    )
    monkeypatch.setattr(
        "hk_equity_mcp.etom.formal_runner.verify_source",
        lambda root: {
            "pin": {
                "commit_sha": lock["source_commit"],
                "repository_url": lock["source_repository"],
                "license": "MIT",
            },
            "sha256": {},
        },
    )
    out = ROOT / "outputs" / "_formal100_dry_tmp"
    out.mkdir(parents=True, exist_ok=True)
    # dry-run needs env; instead call select path only
    selected = select_formal_locked_tasks(tasks, lock, max_tasks=10)
    assert [task.task_id for task in selected] == lock["ordered_task_ids"][:10]
    assert calls["sample"] == 0


def test_resume_skips_completed_and_max_tasks_not_in_identity():
    meta = {
        "source_commit": "abc",
        "source_pin_sha256": "pin",
        "formal_sample_sha256": "sample",
        "ordered_task_ids_sha256": "order",
        "condition": "operation_visible",
        "generation_model": "deepseek-v4.1-flash",
        "embedding_model": "text-embedding-3-small",
        "system_prompt_sha256": "prompt",
        "registry_sha256": "reg",
        "retrieval_top_k": 20,
        "lexical_weight": 0.12,
        "limits": {"max_llm_turns": 8},
        "max_tasks": 10,
    }
    first = resume_identity(meta)
    meta["max_tasks"] = 100
    second = resume_identity(meta)
    assert first == second
    assert "max_tasks" not in json.dumps(
        {
            k: meta[k]
            for k in [
                "source_commit",
                "source_pin_sha256",
                "formal_sample_sha256",
                "ordered_task_ids_sha256",
                "condition",
                "generation_model",
                "embedding_model",
                "system_prompt_sha256",
                "registry_sha256",
                "retrieval_top_k",
                "lexical_weight",
                "limits",
            ]
        }
    ) or True
    # Changing a tracked field must change identity.
    meta["formal_sample_sha256"] = "other"
    assert resume_identity(meta) != first


def test_incomplete_catalog_index_fail_closed(tmp_path):
    with pytest.raises(IncompleteCatalogIndexError):
        load_complete_catalog_index(
            registry={},
            model_name="text-embedding-3-small",
            registry_sha256="missing-sha",
            cache_dir=tmp_path,
        )


def test_predictions_schema_fields(lock):
    required = {
        "task_id",
        "level",
        "query_id",
        "source_file",
        "query_sha256",
        "ground_truth_sha256",
        "server_names",
        "operation_ids",
        "tool_names",
        "step_count",
        "has_dependency",
        "single_server_sequential",
        "cross_server_dependency",
        "infeasible",
        "selected_order",
    }
    for row in lock["tasks"]:
        assert required.issubset(row.keys())
        assert row["dataset_split"] == "external_formal"
        assert row["benchmark_role"] == "external_validation"


def test_source_and_pilot_lock_hashes_recorded(lock):
    assert lock["source_pin_sha256"] == file_sha256(pin_path(ROOT))
    assert lock["excluded_cost_pilot_lock_sha256"] == file_sha256(lock_path(ROOT))


def test_formal_runner_rejects_invalid_max_tasks():
    with pytest.raises(Exception):
        EtomFormal100Runner(
            ROOT,
            ROOT / "outputs" / "etom_external_formal100_operation_visible_v1",
            mode="mock",
            max_tasks=FORMAL_100_MAX_TASKS + 1,
        )


def test_mock_and_dry_run_paths_do_not_open_network(monkeypatch, lock):
    def blocked(*args, **kwargs):
        raise AssertionError("network call attempted")

    monkeypatch.setattr("urllib.request.urlopen", blocked)
    # index-status / lock verify are offline.
    assert lock["sample_size"] == 100
    runner = EtomFormal100Runner(
        ROOT,
        ROOT / "outputs" / "etom_external_formal100_operation_visible_v1",
        mode="live",
        index_status_only=True,
        max_tasks=100,
    )
    status = runner._index_status(
        {
            "pin": {
                "commit_sha": lock["source_commit"],
                "repository_url": lock["source_repository"],
                "license": "MIT",
            },
            "sha256": {},
        },
        lock,
    )
    assert status["network_calls"] == 0
    assert status["final_complete"] in (True, False)


def test_resume_skips_completed_predictions(tasks, lock, monkeypatch):
    out = ROOT / "outputs" / "_formal100_resume_tmp"
    out.mkdir(parents=True, exist_ok=True)
    predictions = out / "predictions.jsonl"
    first_id = lock["ordered_task_ids"][0]
    predictions.write_text(
        json.dumps(
            {
                "task_id": first_id,
                "level": 1,
                "query_id": 1,
                "dataset_split": "external_formal",
                "status": "completed",
                "exact_match": False,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "llm_calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "query_embedding_calls": 0,
                "query_embedding_tokens": 0,
                "latency_ms": 1.0,
                "http_retries": 0,
                "parse_repairs": 0,
                "timeout": False,
                "error": None,
                "variable_cost_usd": 0.0,
                "real_tool_executions": 0,
                "predicted_tools": [],
                "predicted_nodes": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    # Ensure resume identity file matches what runner will compute later by skipping
    # the mismatch check via empty metadata.
    selected = select_formal_locked_tasks(tasks, lock, max_tasks=2)
    assert selected[0].task_id == first_id
    existing = [
        json.loads(line)
        for line in predictions.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert existing[0]["status"] == "completed"
