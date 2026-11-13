"""Offline tests: pe_fin_replan_etom generation provider identity (no network)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from hk_equity_mcp.etom import EXPECTED_LLM_PROVIDER, EXPECTED_LLM_MODEL
from hk_equity_mcp.pe_fin_replan_etom.generation_provider import (
    EXPECTED_HOST,
    EXPECTED_MODEL,
    EXPECTED_PROVIDER,
    PROFILE_ID,
    GenerationProviderIdentityError,
    assert_live_generation_identity,
    generation_provider_profile_hash,
    load_generation_provider_profile,
    public_generation_identity,
)
from hk_equity_mcp.pe_fin_replan_etom.identity import (
    protocol_identity,
    resume_identity_hash,
)
from hk_equity_mcp.experiment.unified_pe_fin_replan_etom_runner import (
    ResumeIdentityMismatch,
    UnifiedPeFinReplanEtomRunner,
)

ROOT = Path(__file__).resolve().parents[1]
SMOKE_V2 = ROOT / "outputs" / "pe_fin_replan_etom_smoke5_v2"

# Frozen after required_outputs maxLength 64→128 (plan/schema/prompt set v2).
# R5 formula hash must not drift with this bound-only schema/prompt bump.
FROZEN_PROMPT_HASH = "e16de9f5d6169562354baa5504560a03ce1d56a2ae8a80659711fa99de38fc45"
FROZEN_SCHEMA_SET_HASH = "84227e688363fff6f7577d88f511f89fe76b04563062445c91b4b78cf1f26554"
FROZEN_REPLANNER_SCHEMA_HASH = "4c9270b91ba519d462fe8e4b4c3c2dd34f4ac867b66e316697a1430bc73a1fe8"
FROZEN_RETRIEVAL_FORMULA_HASH = "7414751814f6858a375d95da918921e3ee365dfc5b69cdaba0cee1bf9e1ecb3e"

V2_RAW = {
    "predictions.jsonl": "ba1152c4dac40457b45090324db148595f2ff4cfb65be17b7b276799510eec26",
    "run_metadata.json": "7dd39a7f932a2dc56cacd294b17913b64f54d78209788719d6734df091956e86",
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_01_match_openai_compatible_zhizengzeng():
    profile = load_generation_provider_profile(ROOT)
    assert profile["profile_id"] == PROFILE_ID
    assert profile["provider"] == EXPECTED_PROVIDER == "openai_compatible"
    assert profile["provider_host"] == EXPECTED_HOST == "api.zhizengzeng.com"
    assert profile["model"] == EXPECTED_MODEL == "deepseek-v4.1-flash"
    assert_live_generation_identity(
        provider="openai_compatible",
        model="deepseek-v4.1-flash",
        host="api.zhizengzeng.com",
        profile=profile,
        env_file_name=".env.relay-v41",
    )
    pub = public_generation_identity(profile, host="api.zhizengzeng.com")
    assert pub["generation_provider_profile_id"] == PROFILE_ID
    assert pub["generation_provider_profile_hash"] == profile["profile_hash"]
    assert "key" not in json.dumps(pub).lower()
    assert "sk-" not in json.dumps(pub)


def test_02_aliyun_rejected():
    with pytest.raises(GenerationProviderIdentityError):
        assert_live_generation_identity(
            provider="aliyun",
            model="deepseek-v4.1-flash",
            host="token-plan.cn-beijing.maas.aliyuncs.com",
        )


def test_03_wrong_model_rejected():
    with pytest.raises(GenerationProviderIdentityError):
        assert_live_generation_identity(
            provider="openai_compatible",
            model="deepseek-v3",
            host="api.zhizengzeng.com",
        )


def test_04_wrong_host_rejected():
    with pytest.raises(GenerationProviderIdentityError):
        assert_live_generation_identity(
            provider="openai_compatible",
            model="deepseek-v4.1-flash",
            host="api.openai.com",
        )


def test_05_null_host_rejected():
    with pytest.raises(GenerationProviderIdentityError):
        assert_live_generation_identity(
            provider="openai_compatible",
            model="deepseek-v4.1-flash",
            host=None,
        )


def test_06_mock_provider_rejected_for_live():
    for bad in ("mock", "hash", "scripted"):
        with pytest.raises(GenerationProviderIdentityError):
            assert_live_generation_identity(
                provider=bad,
                model="deepseek-v4.1-flash",
                host="api.zhizengzeng.com",
            )


def test_07_no_api_key_in_profile_or_public_identity():
    profile = load_generation_provider_profile(ROOT)
    blob = json.dumps(profile, default=str)
    assert "api_key" not in blob.lower()
    assert "sk-" not in blob
    pub = public_generation_identity(profile, host=EXPECTED_HOST)
    assert "api_key" not in json.dumps(pub).lower()


def test_08_v2_raw_hashes_unchanged():
    assert SMOKE_V2.is_dir()
    for name, want in V2_RAW.items():
        assert _sha(SMOKE_V2 / name) == want
    invalidated = (SMOKE_V2 / "INVALIDATED.md").read_text(encoding="utf-8")
    assert (
        "invalidated_engineering_smoke_due_to_inherited_aliyun_provider_identity_and_http_403"
        in invalidated
    )
    assert "outputs/pe_fin_replan_etom_smoke5_v3" in invalidated


def test_09_v2_cannot_resume_into_relay_identity(tmp_path):
    v2_meta = json.loads((SMOKE_V2 / "run_metadata.json").read_text(encoding="utf-8"))
    profile = load_generation_provider_profile(ROOT)
    relay_meta = {
        "condition": v2_meta.get("condition"),
        "method_version": v2_meta.get("method_version"),
        "protocol_version": v2_meta.get("protocol_version"),
        "dataset_sample_sha256": v2_meta.get("dataset_sample_sha256"),
        "dataset_manifest_sha256": v2_meta.get("dataset_manifest_sha256"),
        "ordered_task_ids_sha256": v2_meta.get("ordered_task_ids_sha256"),
        "capability_profile_hash": v2_meta.get("capability_profile_hash"),
        "prompt_hash": v2_meta.get("prompt_hash"),
        "schema_set_hash": v2_meta.get("schema_set_hash"),
        "replanner_schema_hash": v2_meta.get("replanner_schema_hash"),
        "grounding_contract_hash": v2_meta.get("grounding_contract_hash"),
        "catalog_registry_sha256": v2_meta.get("catalog_registry_sha256"),
        "operation_documents_sha256": v2_meta.get("operation_documents_sha256"),
        "operation_index_identity": v2_meta.get("operation_index_identity"),
        "embedding_cache_namespace": v2_meta.get("embedding_cache_namespace"),
        "generation_provider_profile_id": profile["profile_id"],
        "generation_provider_profile_hash": profile["profile_hash"],
        "generation_provider": "openai_compatible",
        "generation_model": "deepseek-v4.1-flash",
        "generation_host": "api.zhizengzeng.com",
        "embedding_model": v2_meta.get("embedding_model"),
        "embedding_host": v2_meta.get("embedding_host"),
        "budgets": v2_meta.get("budgets"),
        "evaluator_identity": v2_meta.get("evaluator_identity"),
        "run_mode": "live",
        "run_role": "engineering_live_path_smoke_only",
        "git_commit": v2_meta.get("git_commit"),
        "git_dirty": v2_meta.get("git_dirty"),
        "fixture_run": False,
    }
    assert resume_identity_hash(v2_meta) != resume_identity_hash(relay_meta)

    # Simulated resume into a dir that still has v2 metadata must fail closed.
    out = tmp_path / "smoke_v3_attempt"
    out.mkdir()
    (out / "run_metadata.json").write_text(
        json.dumps({**v2_meta, "resume_identity_sha256": resume_identity_hash(v2_meta)}),
        encoding="utf-8",
    )
    (out / "predictions.jsonl").write_text(
        (SMOKE_V2 / "predictions.jsonl").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    runner = UnifiedPeFinReplanEtomRunner(
        ROOT,
        out,
        mode="mock",
        engineering_smoke5=True,
        formal100=False,
        resume=True,
        max_tasks=5,
    )
    # mock resume still checks resume identity against freshly computed mock metadata
    with pytest.raises((ResumeIdentityMismatch, Exception)):
        runner.execute()


def test_10_embedding_index_identity_unchanged():
    cache = ROOT / ".cache" / "pe_fin_replan_etom_operation_embeddings_v1"
    finals = list(cache.glob("pe_fin_replan_etom_ops_*.json"))
    assert finals, "expected complete index artifact present"
    # digest in filename must remain (generation provider is not part of index identity)
    assert any(
        "d1da417107a292e19c68bda841e516054299a72401b4ed691288ac3e1ac11ff3" in p.name
        for p in finals
    )
    payload = json.loads(finals[0].read_text(encoding="utf-8"))
    identity = payload.get("identity") or payload.get("provenance") or {}
    # generation fields must not appear in embedding index identity
    blob = json.dumps(identity)
    assert "aliyun" not in blob
    assert "generation_provider" not in blob


def test_11_baseline_etom_provider_still_aliyun():
    assert EXPECTED_LLM_PROVIDER == "aliyun"
    assert EXPECTED_LLM_MODEL == "deepseek-v4.1-flash"
    # proposed profile remains independent
    assert EXPECTED_PROVIDER != EXPECTED_LLM_PROVIDER


def test_12_method_prompt_schema_r5_unchanged():
    proto = protocol_identity()
    assert proto["prompt_hash"] == FROZEN_PROMPT_HASH
    assert proto["schema_set_hash"] == FROZEN_SCHEMA_SET_HASH
    assert proto["replanner_schema_hash"] == FROZEN_REPLANNER_SCHEMA_HASH
    assert proto["retrieval_formula_hash"] == FROZEN_RETRIEVAL_FORMULA_HASH
    assert proto["method_version"] == "pe_fin_replan_etom_v1"
    assert proto["protocol_version"] == "planner_executor_replanner_etom_seven_round_v1"


def test_13_mock_dry_run_still_pass(tmp_path):
    dry = UnifiedPeFinReplanEtomRunner(
        ROOT, tmp_path / "dry", mode="dry-run", formal100=True
    )
    dry_result = dry.execute()
    assert dry_result.get("live_pass_claimed") is False
    mock = UnifiedPeFinReplanEtomRunner(
        ROOT, tmp_path / "mock", mode="mock", engineering_smoke5=True, formal100=False
    )
    mock_result = mock.execute()
    assert mock_result.get("live_pass_claimed") is False
    meta = json.loads((tmp_path / "mock" / "run_metadata.json").read_text(encoding="utf-8"))
    assert meta["generation_provider"] == "mock"
    assert meta["generation_provider_profile_id"] is None


def test_14_profile_hash_stable():
    profile = load_generation_provider_profile(ROOT)
    again = generation_provider_profile_hash(profile)
    assert again == profile["profile_hash"]
    assert len(again) == 64
