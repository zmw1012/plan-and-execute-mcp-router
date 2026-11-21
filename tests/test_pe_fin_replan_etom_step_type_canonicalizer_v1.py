"""Strict ETOM step-type allowlist canonicalizer (type≠domain) offline tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from artifact_gates import etom_artifact_root, require_etom_artifact

from hk_equity_mcp.pe_fin_lite.schemas import SchemaValidationError
from hk_equity_mcp.pe_fin_replan_etom.identity import resume_identity_hash
from hk_equity_mcp.pe_fin_replan_etom.prompts import (
    PLANNER_SYSTEM_PROMPT,
    PROMPT_SET_VERSION,
    REPLANNER_SYSTEM_PROMPT,
    pe_fin_replan_etom_prompt_hash,
    render_planner_user,
)
from hk_equity_mcp.pe_fin_replan_etom.schemas import (
    REQUIRED_OUTPUT_ITEM_MAX_LENGTH,
    STEP_TYPE_CANONICALIZER_VERSION,
    _TYPE_ALIASES,
    canonicalize_step,
    step_canonicalizer_hash,
    validate_plan_payload,
)

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = etom_artifact_root()
SMOKE_V5 = ARTIFACT_ROOT / "outputs" / "pe_fin_replan_etom_smoke5_v5"

V5_RAW = {
    "predictions.jsonl": (
        "e5b14b2899ad4e180d233f1c35c0a3b8678af7a40482538f45ed0200a2a6afa6",
        65461,
        5,
    ),
    "run_metadata.json": (
        "fb6c7b194bf06d3e8f17e2554410861510a391020d820a78e44f52c9373213d6",
        127823,
        None,
    ),
}

PRE_FIX_PROMPT_HASH = "e16de9f5d6169562354baa5504560a03ce1d56a2ae8a80659711fa99de38fc45"
PRE_FIX_SCHEMA_SET_HASH = "84227e688363fff6f7577d88f511f89fe76b04563062445c91b4b78cf1f26554"


def _step(*, stype: str, domain: str = "search", objective: str = "do the work", outs=None):
    return {
        "id": "s1",
        "type": stype,
        "domain": domain,
        "objective": objective,
        "required_outputs": list(outs if outs is not None else ["route"]),
        "depends_on": [],
    }


def _plan(step):
    return {"steps": [step]}


def test_01_canonicalizer_version():
    assert STEP_TYPE_CANONICALIZER_VERSION == "pe_fin_replan_etom_step_type_canonicalizer_v1"
    assert REQUIRED_OUTPUT_ITEM_MAX_LENGTH == 128
    assert PROMPT_SET_VERSION == "pe_fin_replan_etom_prompt_set_v3"
    assert "general" not in _TYPE_ALIASES


@pytest.mark.parametrize(
    "alias,want",
    [
        ("write", "retrieve"),
        ("search", "retrieve"),
        ("read", "retrieve"),
        ("list", "retrieve"),
        ("calendar", "retrieve"),
        ("communication", "retrieve"),
        ("code", "retrieve"),
        ("docs", "retrieve"),
        ("data", "retrieve"),
        ("tool", "retrieve"),
        ("invoke", "retrieve"),
        ("action", "retrieve"),
        ("external_tool", "retrieve"),
        ("analysis", "reason"),
        ("reasoning", "reason"),
        ("calculation", "compute"),
        ("arithmetic", "compute"),
    ],
)
def test_02_allowlist_aliases(alias, want):
    fixed, rec = canonicalize_step(_step(stype=alias, domain="calendar" if alias == "write" else "search"))
    assert fixed["type"] == want
    assert rec is not None
    assert rec["original_step_type"] == alias
    assert rec["canonical_step_type"] == want
    assert rec["canonicalization_rule"] == f"type_alias_{alias}_to_{want}"
    assert rec["canonicalizer_version"] == STEP_TYPE_CANONICALIZER_VERSION
    assert rec["canonicalizer_hash"] == step_canonicalizer_hash()


@pytest.mark.parametrize("legal", ["retrieve", "compute", "reason"])
def test_03_legal_types_unchanged(legal):
    domain = "compute" if legal == "compute" else ("reason" if legal == "reason" else "search")
    step = _step(stype=legal, domain=domain)
    if legal == "compute":
        step["compute_spec"] = {
            "expression": "a+b",
            "operand_refs": {"a": "x", "b": "y"},
            "output_name": "sum",
        }
        step["depends_on"] = ["s0"]
        step["required_outputs"] = ["sum"]
    fixed, rec = canonicalize_step(step)
    assert fixed["type"] == legal
    assert rec is None


def test_04_write_maps_then_strict_validation():
    raw = _plan(_step(stype="write", domain="calendar", objective="Create the requested calendar event", outs=["event creation route"]))
    out = validate_plan_payload(raw)
    assert out["steps"][0]["type"] == "retrieve"
    assert out["steps"][0]["domain"] == "calendar"
    assert out["steps"][0]["objective"] == "Create the requested calendar event"
    assert out["steps"][0]["required_outputs"] == ["event creation route"]
    recs = out.get("step_type_canonicalizations") or []
    assert len(recs) == 1
    assert recs[0]["original_step_type"] == "write"
    assert recs[0]["canonical_step_type"] == "retrieve"


def test_05_unknown_and_general_fail_closed():
    with pytest.raises(SchemaValidationError) as exc:
        validate_plan_payload(_plan(_step(stype="unknown_xyz")))
    assert "not in enum" in str(exc.value)
    with pytest.raises(SchemaValidationError) as exc2:
        validate_plan_payload(_plan(_step(stype="general", domain="general")))
    assert "not in enum" in str(exc2.value)
    # Canonicalizer itself must not rewrite unknown/general.
    fixed_u, rec_u = canonicalize_step(_step(stype="unknown_xyz"))
    assert fixed_u["type"] == "unknown_xyz" and rec_u is None
    fixed_g, rec_g = canonicalize_step(_step(stype="general", domain="general"))
    assert fixed_g["type"] == "general" and rec_g is None


def test_06_no_rewrite_of_objective_outputs_domain():
    step = _step(
        stype="write",
        domain="write",
        objective="persist the draft under the given path",
        outs=["write acknowledgement"],
    )
    step["depends_on"] = ["s0"]
    fixed, _ = canonicalize_step(step)
    assert fixed["type"] == "retrieve"
    assert fixed["domain"] == "write"
    assert fixed["objective"] == "persist the draft under the given path"
    assert fixed["required_outputs"] == ["write acknowledgement"]
    assert fixed["depends_on"] == ["s0"]


def test_07_no_task_id_or_gold_in_prompts_or_aliases():
    blob = PLANNER_SYSTEM_PROMPT + "\n" + REPLANNER_SYSTEM_PROMPT
    user = render_planner_user(
        question="schedule a meeting tomorrow afternoon",
        capability_profile={"domains": list(_TYPE_ALIASES)[:3]},
        limits={
            "max_total_llm_calls_per_task": 16,
            "max_total_mcp_invocations": 8,
            "max_execution_rounds": 7,
        },
    )
    blob = blob + "\n" + user
    for banned in ("L2-q766", "Outlook", "ground_truth", "gold_operations", "accepted_tools", "GOLD_SENTINEL"):
        assert banned not in blob
    # type/domain distinction present
    assert '`type` describes execution mechanism' in PLANNER_SYSTEM_PROMPT
    assert 'type="retrieve"' in PLANNER_SYSTEM_PROMPT
    assert '"domain": "calendar"' in PLANNER_SYSTEM_PROMPT
    assert "Create the requested calendar event" in PLANNER_SYSTEM_PROMPT
    assert '`type` describes execution mechanism' in REPLANNER_SYSTEM_PROMPT


def test_08_v5_v6_resume_identity_fail_closed():
    from hk_equity_mcp.pe_fin_replan_etom import schemas as S

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
        "step_canonicalizer_hash": "old_canon_hash",
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
        "git_commit": "fcb1720",
        "git_dirty": False,
        "fixture_run": False,
    }
    v6 = dict(base)
    v6["prompt_hash"] = pe_fin_replan_etom_prompt_hash()
    v6["step_canonicalizer_hash"] = S.step_canonicalizer_hash()
    assert resume_identity_hash(base) != resume_identity_hash(v6)
    assert pe_fin_replan_etom_prompt_hash() != PRE_FIX_PROMPT_HASH


def test_09_smoke_v5_raw_frozen_and_invalidated():
    require_etom_artifact("outputs/pe_fin_replan_etom_smoke5_v5/predictions.jsonl")
    for name, (want, nbytes, rows) in V5_RAW.items():
        path = SMOKE_V5 / name
        data = path.read_bytes()
        assert len(data) == nbytes
        assert hashlib.sha256(data).hexdigest() == want
        if rows is not None:
            assert sum(1 for line in data.decode().splitlines() if line.strip()) == rows
    text = (SMOKE_V5 / "INVALIDATED.md").read_text(encoding="utf-8")
    assert "invalidated_engineering_smoke_due_to_planner_step_type_domain_confusion" in text
    assert "outputs/pe_fin_replan_etom_smoke5_v6" in text
    rows = [
        json.loads(line)
        for line in (SMOKE_V5 / "predictions.jsonl").read_text().splitlines()
        if line.strip()
    ]
    l2 = next(r for r in rows if r["task_id"] == "L2-q766")
    assert l2["stopping_reason"] == "planner_invalid_after_repair"
    assert "value 'write' not in enum" in l2["error"]
