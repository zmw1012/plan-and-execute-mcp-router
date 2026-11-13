"""Boundary tests for ETOM required_outputs maxLength 64→128 (no silent truncate)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from hk_equity_mcp.pe_fin_lite.schemas import SchemaValidationError
from hk_equity_mcp.pe_fin_replan_etom.identity import resume_identity_hash
from hk_equity_mcp.pe_fin_replan_etom.prompts import (
    PLANNER_SYSTEM_PROMPT,
    PROMPT_SET_VERSION,
    pe_fin_replan_etom_prompt_hash,
    render_planner_user,
)
from hk_equity_mcp.pe_fin_replan_etom.schemas import (
    PLAN_SCHEMA_VERSION,
    REQUIRED_OUTPUT_ITEM_MAX_LENGTH,
    SCHEMA_SET_VERSION,
    format_repair_user_message,
    parse_and_validate,
    schema_set_hash,
    validate_plan_payload,
)

ROOT = Path(__file__).resolve().parents[1]
SMOKE_V4 = ROOT / "outputs" / "pe_fin_replan_etom_smoke5_v4"

# Frozen before this engineering fix; must remain unchanged.
V4_RAW = {
    "predictions.jsonl": (
        "17fa41944929c3767074a0a1b6803a0bb68fde1a93946efe24f1b19974282f4d",
        59468,
        5,
    ),
    "run_metadata.json": (
        "fd615acfd0111f789c776ccf98d6752eb8974afdd682ef3e25e640bfd600f062",
        127828,
        None,
    ),
}

# Pre-fix identity (ca38871 / schema_set_v1 / prompt_set_v1 / maxLength=64).
PRE_FIX_PROMPT_HASH = "ad55224d319003b9e0770ff7d54505109b29f64e0c83ce5b3be7ff642a7c7a22"
PRE_FIX_SCHEMA_SET_HASH = "84d9fb0bcea9dada38c45ea57cb147f901ced75c69b563a43d3b2756c7bb38bb"


def _step(required_outputs):
    return {
        "id": "s1",
        "type": "retrieve",
        "domain": "search",
        "objective": "look up the requested information",
        "required_outputs": list(required_outputs),
        "depends_on": [],
    }


def _plan(required_outputs):
    return {"steps": [_step(required_outputs)]}


def test_01_required_output_max_length_is_128():
    assert REQUIRED_OUTPUT_ITEM_MAX_LENGTH == 128
    assert PLAN_SCHEMA_VERSION == "pe_fin_replan_etom_plan_v2"
    assert SCHEMA_SET_VERSION == "pe_fin_replan_etom_schema_set_v2"
    assert PROMPT_SET_VERSION == "pe_fin_replan_etom_prompt_set_v2"


@pytest.mark.parametrize("n", [64, 65, 128])
def test_02_required_output_length_passes(n):
    label = "x" * n
    out = validate_plan_payload(_plan([label]))
    assert out["steps"][0]["required_outputs"][0] == label
    assert len(out["steps"][0]["required_outputs"][0]) == n


def test_03_required_output_129_fails_closed():
    label = "y" * 129
    with pytest.raises(SchemaValidationError) as exc:
        validate_plan_payload(_plan([label]))
    msg = str(exc.value)
    assert "required_outputs[0]" in msg
    assert "allowed=128" in msg
    assert "actual=129" in msg
    assert "maxLength 128" in msg


def test_04_no_silent_truncate_on_overlength():
    label = "z" * 129
    with pytest.raises(SchemaValidationError):
        validate_plan_payload(_plan([label]))
    # Validator must not mutate the caller's list by truncating.
    assert len(label) == 129
    assert label == "z" * 129


def test_05_repair_revalidates_strict_schema():
    bad = json.dumps(_plan(["a" * 70]))  # would fail under old 64; passes under 128
    ok_payload, rc0, err0 = parse_and_validate(bad, "plan")
    assert err0 is None
    assert rc0 == 0
    assert ok_payload is not None
    assert len(ok_payload["steps"][0]["required_outputs"][0]) == 70

    still_bad = json.dumps(_plan(["b" * 129]))
    payload, rc, err = parse_and_validate(
        still_bad,
        "plan",
        repair_content=still_bad,  # repair did not shorten — still fails
    )
    assert payload is None
    assert rc == 1
    assert err is not None
    assert "allowed=128" in err
    assert "actual=129" in err


def test_06_repair_message_includes_bound_and_no_truncate():
    err = (
        "$.steps[0].required_outputs[1]: string length 70 exceeds maxLength 64 "
        "(allowed=64, actual=70)"
    )
    msg = format_repair_user_message("plan", err, previous_content="{}")
    assert "Error: " + err in msg
    assert "no truncation of overlength strings" in msg
    assert str(REQUIRED_OUTPUT_ITEM_MAX_LENGTH) in msg
    assert "short noun phrases" in msg


def test_07_task_id_and_gold_not_in_planner_prompt():
    user = render_planner_user(
        question="find a calendar event tomorrow",
        capability_profile={"domains": ["search", "calendar"]},
        limits={
            "max_total_llm_calls_per_task": 16,
            "max_total_mcp_invocations": 8,
            "max_execution_rounds": 7,
        },
        max_plan_steps=6,
    )
    blob = PLANNER_SYSTEM_PROMPT + "\n" + user
    for banned in (
        "L5-q68",
        "task_id",
        "ground_truth",
        "gold_operations",
        "accepted_tools",
        "GOLD_SENTINEL",
    ):
        assert banned not in blob
    assert f"max {REQUIRED_OUTPUT_ITEM_MAX_LENGTH} characters" in blob


def test_08_identity_hashes_changed_from_v4():
    assert pe_fin_replan_etom_prompt_hash() != PRE_FIX_PROMPT_HASH
    assert schema_set_hash() != PRE_FIX_SCHEMA_SET_HASH


def test_09_v4_v5_resume_identity_fail_closed():
    base = {
        "condition": "pe_fin_replan_etom",
        "method_version": "pe_fin_replan_etom_v1",
        "protocol_version": "planner_executor_replanner_etom_seven_round_v1",
        "dataset_sample_sha256": "a",
        "dataset_manifest_sha256": "b",
        "ordered_task_ids_sha256": "c",
        "capability_profile_hash": "d",
        "prompt_hash": PRE_FIX_PROMPT_HASH,
        "schema_set_hash": PRE_FIX_SCHEMA_SET_HASH,
        "replanner_schema_hash": "g",
        "grounding_contract_hash": "h",
        "cli_output_contract_version": "pe_fin_replan_etom_cli_compact_v1",
        "catalog_registry_sha256": "i",
        "operation_documents_sha256": "j",
        "operation_index_identity": "k",
        "embedding_cache_namespace": "pe_fin_replan_etom_operation_embeddings_v1",
        "generation_provider_profile_id": "pe_fin_replan_etom_zhizengzeng_v1",
        "generation_provider_profile_hash": "m",
        "generation_provider": "openai_compatible",
        "generation_model": "deepseek-v4.1-flash",
        "generation_host": "api.zhizengzeng.com",
        "embedding_model": "text-embedding-3-small",
        "embedding_host": "x",
        "budgets": {},
        "evaluator_identity": {},
        "run_mode": "live",
        "run_role": "engineering_live_path_smoke_only",
        "git_commit": "ca38871",
        "git_dirty": False,
        "fixture_run": False,
    }
    v5 = dict(base)
    v5["prompt_hash"] = pe_fin_replan_etom_prompt_hash()
    v5["schema_set_hash"] = schema_set_hash()
    assert resume_identity_hash(base) != resume_identity_hash(v5)


def test_10_smoke_v4_raw_frozen_and_invalidated():
    for name, (want, nbytes, rows) in V4_RAW.items():
        path = SMOKE_V4 / name
        data = path.read_bytes()
        assert len(data) == nbytes
        assert hashlib.sha256(data).hexdigest() == want
        if rows is not None:
            assert sum(1 for line in data.decode().splitlines() if line.strip()) == rows
    text = (SMOKE_V4 / "INVALIDATED.md").read_text(encoding="utf-8")
    assert (
        "invalidated_engineering_smoke_due_to_arbitrary_required_output_length_boundary"
        in text
    )
    assert "outputs/pe_fin_replan_etom_smoke5_v5" in text
    # Predictions unchanged: L5 still records the old 64-bound failure.
    rows = [
        json.loads(line)
        for line in (SMOKE_V4 / "predictions.jsonl").read_text().splitlines()
        if line.strip()
    ]
    l5 = next(r for r in rows if r["task_id"] == "L5-q68")
    assert l5["stopping_reason"] == "planner_invalid_after_repair"
    assert "allowed=64" in l5["error"]
    assert "actual=70" in l5["error"]
