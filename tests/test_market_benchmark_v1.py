"""v1 Market benchmarks retained as superseded_before_any_agent_run trail."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEV = ROOT / "benchmarks" / "questions" / "market_development_15_v1.jsonl"
HELD = ROOT / "benchmarks" / "heldout" / "market_test_20_v1.jsonl"


def test_v1_files_retained_as_trail():
    assert DEV.exists() and HELD.exists()
    assert len(DEV.read_text(encoding="utf-8").splitlines()) == 15
    assert len(HELD.read_text(encoding="utf-8").splitlines()) == 20
    # Active evaluation path is v1.1
    assert (ROOT / "benchmarks" / "questions" / "market_development_15_v1_1.jsonl").exists()
    assert (ROOT / "benchmarks" / "heldout" / "market_test_20_v1_1.jsonl").exists()


def test_v1_heldout_still_locked_membership():
    rows = [
        json.loads(line)
        for line in HELD.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert all(r.get("locked") for r in rows)
    assert all(r.get("dataset_split") == "heldout" for r in rows)
