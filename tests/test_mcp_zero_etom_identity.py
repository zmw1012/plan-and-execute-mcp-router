"""Offline identity checks for MCP-Zero × ETOM freeze."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from hk_equity_mcp.etom.formal_sampler import ordered_task_ids_sha256, sample_document_sha256
from hk_equity_mcp.etom.sampler import load_lock
from hk_equity_mcp.etom.formal_sampler import load_formal_100_lock
from hk_equity_mcp.etom.source import file_sha256, load_pin
from hk_equity_mcp.mcp_zero import method_identity_sha256
from hk_equity_mcp.mcp_zero_etom import (
    EXPECTED_FORMAL_SAMPLE_SHA256,
    EXPECTED_HK_METHOD_CARD_SHA256,
    EXPECTED_ORDERED_IDS_SHA256,
    ETOM_SOURCE_COMMIT,
    assert_shared_knobs_match_hk,
    etom_method_identity_sha256,
)
from hk_equity_mcp.mcp_zero_etom.prompt import etom_prompt_hash
from hk_equity_mcp.mcp_zero_etom.source import assert_formal_lock_identity

ROOT = Path(__file__).resolve().parents[1]


def test_hk_method_card_unchanged():
    assert method_identity_sha256() == EXPECTED_HK_METHOD_CARD_SHA256
    assert_shared_knobs_match_hk()


def test_formal_lock_identity():
    info = assert_formal_lock_identity(ROOT)
    assert info["sample_sha256"] == EXPECTED_FORMAL_SAMPLE_SHA256
    assert info["ordered_task_ids_sha256"] == EXPECTED_ORDERED_IDS_SHA256
    assert info["pilot_overlap"] == []
    assert info["level_counts"] == {"L1": 20, "L2": 20, "L3": 20, "L4": 20, "L5": 20}


def test_100_tasks_quotas_and_recompute():
    lock = load_formal_100_lock(ROOT)
    assert len(lock["ordered_task_ids"]) == 100
    assert Counter(tid.split("-")[0] for tid in lock["ordered_task_ids"]) == {
        "L1": 20, "L2": 20, "L3": 20, "L4": 20, "L5": 20
    }
    assert ordered_task_ids_sha256(lock["ordered_task_ids"]) == EXPECTED_ORDERED_IDS_SHA256
    assert sample_document_sha256(lock) == EXPECTED_FORMAL_SAMPLE_SHA256
    assert lock["seed"] == 20261106
    assert lock["source_commit"] == ETOM_SOURCE_COMMIT


def test_pilot_overlap_zero():
    formal = load_formal_100_lock(ROOT)
    pilot = load_lock(ROOT)
    pilot_ids = {t["task_id"] for t in pilot["tasks"]}
    assert pilot_ids.isdisjoint(formal["ordered_task_ids"])


def test_source_pin_counts():
    pin = load_pin(ROOT)
    assert pin["commit_sha"] == ETOM_SOURCE_COMMIT
    assert pin["official_counts"]["tools"] == 2375
    queries = pin["official_counts"]["queries"]
    assert queries == {"1": 781, "2": 773, "3": 327, "4": 103, "5": 91}
    assert sum(queries.values()) == 2075


def test_import_manifest_present():
    path = ROOT / "benchmarks" / "etom" / "import_manifest_mcp_zero_v1.json"
    data = json.loads(path.read_text())
    assert data["identity"]["sample_sha256"] == EXPECTED_FORMAL_SAMPLE_SHA256
    assert data["identity"]["pilot_overlap"] == []
    assert data["etom_source_commit"] == ETOM_SOURCE_COMMIT


def test_etom_prompt_hash_stable_and_distinct():
    h1 = etom_prompt_hash()
    h2 = etom_prompt_hash()
    assert h1 == h2
    assert len(h1) == 64
    # ETOM method hash exists and differs from HK card
    eh = etom_method_identity_sha256()
    assert eh != EXPECTED_HK_METHOD_CARD_SHA256
