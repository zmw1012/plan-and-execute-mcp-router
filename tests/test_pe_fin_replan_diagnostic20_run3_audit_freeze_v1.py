"""Offline freeze checks for RePlan Diagnostic-20 run3 final audit (no network)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN3 = ROOT / "outputs" / "pe_fin_replan_diagnostic20_live_v1_4_1_run3"
AUD = ROOT / "outputs" / "pe_fin_replan_diagnostic20_live_v1_4_1_run3_audit"
MAN = ROOT / "benchmarks" / "manifests" / "hk_pe_fin_lite_eea_diagnostic20_v1.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_run3_raw_hashes_frozen_before_equals_after():
    before = json.loads((AUD / "raw_artifact_sha256_before.json").read_text(encoding="utf-8"))
    after = json.loads((AUD / "raw_artifact_sha256_after.json").read_text(encoding="utf-8"))
    assert before == after
    for name, expected in before.items():
        assert _sha(RUN3 / name) == expected


def test_run3_valid_marker_and_identity():
    marker = (AUD / "VALID.md").read_text(encoding="utf-8")
    assert "valid_posthoc_exploratory_diagnostic" in marker
    assert "explicitly_not_formal_accuracy" in marker
    bundle = json.loads((AUD / "final_audit_bundle.json").read_text(encoding="utf-8"))
    assert bundle["validity"]["cutoff_clean"] is True
    assert bundle["validity"]["explicitly_not_formal_accuracy"] is True
    assert bundle["run3_summary"]["n"] == 20
    man = json.loads(MAN.read_text(encoding="utf-8"))
    ids = list(man.get("ordered_task_ids") or [t["task_id"] for t in man["tasks"]])
    rows = [
        json.loads(line)
        for line in (RUN3 / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [r["task_id"] for r in rows] == ids


def test_run2_accuracy_marked_invalid_in_audit():
    bundle = json.loads((AUD / "final_audit_bundle.json").read_text(encoding="utf-8"))
    assert bundle["run2_summary_engineering_only"]["accuracy_invalid"] is True
    assert "cutoff" in bundle["run2_summary_engineering_only"]["reason"]
