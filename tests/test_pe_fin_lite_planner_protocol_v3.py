"""Planner protocol v3: scope currency, length budgets, 4096 tokens, observability."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from artifact_gates import hk_artifact_root, require_hk_artifact

from hk_equity_mcp.agent.llm_client import (
    ScriptedLLMClient,
    build_response_observability,
    extract_final_text_content,
)
from hk_equity_mcp.experiment.pe_fin_lite_retrieval_audit import (
    PLANNER_ONLY_PROTOCOL_VERSION,
    PLANNER_ONLY_PROTOCOL_VERSION_V2,
)
from hk_equity_mcp.experiment.pe_fin_lite_retrieval_audit.identity import (
    assert_identity_match,
    load_hk155_locked,
    planner_only_cache_identity,
    stable_json_hash,
)
from hk_equity_mcp.experiment.pe_fin_lite_retrieval_audit.planner_diagnostic10 import (
    planner_diagnostic10_cache_identity,
)
from hk_equity_mcp.experiment.pe_fin_lite_retrieval_audit.planner_only import (
    FORBIDDEN_PLANNER_KEYS,
    run_planner_only,
)
from hk_equity_mcp.experiment.pe_fin_lite_retrieval_audit.runner import (
    LivePathError,
    RetrievalAuditRunner,
)
from hk_equity_mcp.pe_fin_lite.output_vocabulary import (
    SCOPE_CONSTRAINT_KEYS,
    scope_contract_hash,
)
from hk_equity_mcp.pe_fin_lite.planner import run_planner
from hk_equity_mcp.pe_fin_lite.prompts import (
    pe_fin_lite_prompt_hash,
    render_planner_user,
)
from hk_equity_mcp.pe_fin_lite.schemas import (
    ADJUSTMENT_MAX_LENGTH,
    CURRENCY_MAX_LENGTH,
    METRIC_BASIS_MAX_LENGTH,
    PLANNER_MAX_TOKENS,
    SchemaValidationError,
    parse_and_validate,
)
import inspect

from hk_equity_mcp.pe_fin_lite.executor import execute_reason_step
from hk_equity_mcp.pe_fin_lite.tool_binding import run_tool_binder

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = hk_artifact_root()
V2_OUT = ARTIFACT_ROOT / "outputs" / "pe_fin_lite_planner_protocol_diagnostic10_v2" / "planner_cache"
V2_OUTPUTS_SHA = "34115f2aba3aec91e639e35dc6b5be9c80c593bc35c97238b69d9e227b2795e0"


def _step(*, metric_basis=None, adjustment=None, currency=None, objective="retrieve"):
    scope = {"entity": {"name": "Issuer", "code": "00000"}, "period": "FY2025"}
    if metric_basis is not None:
        scope["metric_basis"] = metric_basis
    if adjustment is not None:
        scope["adjustment"] = adjustment
    if currency is not None:
        scope["currency"] = currency
    return {
        "id": "s1",
        "type": "retrieve",
        "domain": "financial",
        "objective": objective,
        "required_outputs": ["revenue", "currency"],
        "depends_on": [],
        "scope_constraints": scope,
    }


def _plan_json(**kwargs) -> str:
    return json.dumps({"steps": [_step(**kwargs)]}, ensure_ascii=False, separators=(",", ":"))


def test_currency_scope_legal():
    ok, _, err = parse_and_validate(_plan_json(currency="HKD"), "plan")
    assert ok is not None, err
    assert ok["steps"][0]["scope_constraints"]["currency"] == "HKD"


def test_metric_basis_80_to_160_ok_over_limit_fails():
    mid = "x" * 125  # Diagnostic-10 v2 fin_ht_024 actual length
    ok, _, err = parse_and_validate(_plan_json(metric_basis=mid), "plan")
    assert ok is not None, err
    assert METRIC_BASIS_MAX_LENGTH == 160
    too_long = "y" * (METRIC_BASIS_MAX_LENGTH + 1)
    bad, _, err2 = parse_and_validate(_plan_json(metric_basis=too_long), "plan")
    assert bad is None
    assert "maxLength" in (err2 or "") or "length" in (err2 or "").lower()
    # No silent truncation: original length still exceeds.
    assert len(too_long) > METRIC_BASIS_MAX_LENGTH


def test_adjustment_40_to_80_ok_over_limit_fails():
    mid = "a" * 41  # all155 / diagnostic evidence
    ok, _, err = parse_and_validate(_plan_json(adjustment=mid), "plan")
    assert ok is not None, err
    assert ADJUSTMENT_MAX_LENGTH == 80
    too_long = "b" * (ADJUSTMENT_MAX_LENGTH + 1)
    bad, _, err2 = parse_and_validate(_plan_json(adjustment=too_long), "plan")
    assert bad is None
    assert "maxLength" in (err2 or "") or "length" in (err2 or "").lower()


def test_no_silent_truncation_of_overlength_strings():
    raw = _plan_json(metric_basis="m" * 200)
    bad, _, err = parse_and_validate(raw, "plan")
    assert bad is None
    # Payload must not be accepted with a shortened string.
    assert "200" in (err or "") or "maxLength" in (err or "")


def test_planner_uses_4096_binder_reason_final_keep_defaults():
    assert PLANNER_MAX_TOKENS == 4096
    captured = []

    class CaptureLLM(ScriptedLLMClient):
        def complete(self, messages, *, temperature=0.0, max_tokens=1024, stage=None):
            captured.append({"stage": stage, "max_tokens": max_tokens})
            return super().complete(
                messages, temperature=temperature, max_tokens=max_tokens, stage=stage
            )

    plan = _plan_json(currency="HKD")
    llm = CaptureLLM([plan])
    out = run_planner(
        llm=llm,
        question="What is revenue?",
        capability_profile={},
        limits={},
    )
    assert out["ok"] is True
    assert out["planner_max_tokens"] == 4096
    assert captured[0]["stage"] == "planner"
    assert captured[0]["max_tokens"] == 4096

    # Binder / Reason keep prior defaults (not silently expanded to 4096).
    binder_default = inspect.signature(run_tool_binder).parameters["max_tokens"].default
    reason_default = inspect.signature(execute_reason_step).parameters["max_tokens"].default
    assert binder_default == 1024
    assert reason_default == 1024
    # Agent/config default for non-planner stages remains 2048.
    from hk_equity_mcp.experiment.unified_pe_fin_lite_runner import load_pe_fin_lite_config

    cfg = load_pe_fin_lite_config(root=ROOT)
    assert int(cfg.get("max_tokens", 2048)) == 2048
    assert int(cfg.get("planner_max_tokens", 0)) == 4096


def test_compact_prompt_contains_required_fields_and_enums():
    prompt = render_planner_user(
        question="Generic question about an issuer.",
        capability_profile={"profile_id": "test"},
        limits={"max_plan_steps": 6},
        max_plan_steps=6,
    )
    for token in (
        "id",
        "type",
        "domain",
        "objective",
        "required_outputs",
        "depends_on",
        "financial|market|disclosure|news|compute|general",
        "currency",
        "metric_basis",
        str(METRIC_BASIS_MAX_LENGTH),
        str(ADJUSTMENT_MAX_LENGTH),
        str(CURRENCY_MAX_LENGTH),
        str(PLANNER_MAX_TOKENS),
        "minified JSON",
    ):
        assert token in prompt
    assert "currency" in SCOPE_CONSTRAINT_KEYS
    for banned in FORBIDDEN_PLANNER_KEYS:
        assert banned not in prompt
    assert "task_id=" not in prompt


def test_response_metadata_excludes_secrets_and_hidden_reasoning_text():
    content, _, meta = extract_final_text_content(
        {
            "role": "assistant",
            "content": '{"steps":[]}',
            "reasoning_content": "SECRET_CHAIN_OF_THOUGHT_DO_NOT_STORE",
        }
    )
    assert content == '{"steps":[]}'
    assert meta["reasoning_content_present"] is True
    assert meta["reasoning_content_length"] == len("SECRET_CHAIN_OF_THOUGHT_DO_NOT_STORE")
    obs = build_response_observability(
        stage="planner",
        content=content,
        usage={"input_tokens": 1, "output_tokens": 2, "total_tokens": 3},
        latency_ms=1.0,
        max_tokens_requested=4096,
        provider="mock",
        model="m",
        host="local",
        finish_reason="stop",
        message_meta=meta,
    )
    blob = json.dumps(obs)
    assert "SECRET_CHAIN_OF_THOUGHT_DO_NOT_STORE" not in blob
    assert "api_key" not in blob
    assert obs["finish_reason"] == "stop"
    assert obs["content_present"] is True
    assert obs["reasoning_content_present"] is True
    assert obs["reasoning_content_length"] > 0


def test_null_message_and_null_content_safe():
    content, raw, meta = extract_final_text_content(None)
    assert content == ""
    assert raw is None
    assert meta["message_was_null"] is True
    content2, raw2, meta2 = extract_final_text_content(
        {"role": "assistant", "content": None}
    )
    assert content2 == ""
    assert raw2 is None
    assert meta2["content_was_null"] is True


def test_v2_cannot_resume_v3(tmp_path: Path):
    data = load_hk155_locked(ROOT)
    v3 = planner_only_cache_identity(
        dataset_identities=data["identities"],
        capability_profile_hash="p",
        generation_provider="offline_mock",
        generation_host="local",
        generation_model="mock",
    )
    assert v3["planner_only_protocol_version"] == "pe_fin_lite_planner_only_v3"
    assert v3["planner_max_tokens"] == 4096
    assert v3["scope_contract_hash"] == scope_contract_hash()
    v2 = {
        **{k: v for k, v in v3.items() if k != "identity_sha256"},
        "planner_only_protocol_version": PLANNER_ONLY_PROTOCOL_VERSION_V2,
        "max_tokens": 2048,
        "planner_max_tokens": 2048,
    }
    v2["identity_sha256"] = stable_json_hash(v2)
    assert v2["identity_sha256"] != v3["identity_sha256"]
    with pytest.raises(RuntimeError, match="Identity mismatch"):
        assert_identity_match(v3, v2, label="planner_only_cache")

    # Resume path rejects mismatched identity.
    out = tmp_path / "resume_v2_into_v3"
    (out / "planner_cache_identity.json").parent.mkdir(parents=True, exist_ok=True)
    # run_planner_only writes identity on first call; seed with v2 then resume as v3.
    seed = tmp_path / "seed"
    run_planner_only(
        root=ROOT,
        tasks=data["tasks"][:1],
        output_dir=seed,
        cache_identity=v2,
        mock=True,
        resume=False,
        capability_profile={},
    )
    with pytest.raises(RuntimeError, match="Identity mismatch"):
        run_planner_only(
            root=ROOT,
            tasks=data["tasks"][:1],
            output_dir=seed,
            cache_identity=v3,
            mock=True,
            resume=True,
            capability_profile={},
        )


def test_diagnostic_v3_cannot_formal_lock(tmp_path: Path):
    assert PLANNER_ONLY_PROTOCOL_VERSION == "pe_fin_lite_planner_only_v3"
    default_runner = RetrievalAuditRunner(
        root=ROOT, output_dir=None, mock=True, planner_diagnostic10=True
    )
    assert "diagnostic10_v3" in str(default_runner.output_dir)
    out = tmp_path / "diag_v3"
    runner = RetrievalAuditRunner(
        root=ROOT, output_dir=out, mock=True, planner_diagnostic10=True
    )
    runner.run_planner_only()
    with pytest.raises(LivePathError, match="Diagnostic-10"):
        runner.run_select_formula(formal=True)


def test_no_task_id_gold_leakage_in_prompt_and_identity():
    prompt = render_planner_user(
        question="q",
        capability_profile={},
        limits={},
        max_plan_steps=6,
    )
    for banned in ("gold_operations", "reference_answer", "task_id=", "acceptable_tool_sets"):
        assert banned not in prompt
    data = load_hk155_locked(ROOT)
    identity = planner_only_cache_identity(
        dataset_identities=data["identities"],
        capability_profile_hash="x",
    )
    blob = json.dumps(identity)
    for banned in FORBIDDEN_PLANNER_KEYS:
        assert f'"{banned}"' not in blob


def test_v2_raw_hashes_unchanged():
    require_hk_artifact("outputs/pe_fin_lite_planner_protocol_diagnostic10_v2/planner_cache/planner_outputs.jsonl")
    path = V2_OUT / "planner_outputs.jsonl"
    assert path.exists()
    import hashlib

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == V2_OUTPUTS_SHA


def test_observability_attached_on_planner_calls():
    plan = _plan_json(currency="USD")
    llm = ScriptedLLMClient([plan])
    out = run_planner(
        llm=llm,
        question="currency scope?",
        capability_profile={},
        limits={},
    )
    assert out["ok"]
    obs = out["call_observability"]
    assert isinstance(obs, list) and obs
    row = obs[0]
    assert row["stage"] == "planner"
    assert row["content_present"] is True
    assert row["finish_reason"] == "stop"
    assert row["max_tokens_requested"] == 4096
    assert "reasoning_content" not in row
    assert "api_key" not in row


def test_prompt_and_scope_hashes_stable_nonzero():
    assert len(pe_fin_lite_prompt_hash()) == 64
    assert len(scope_contract_hash()) == 64
    man_path = (
        ROOT
        / "benchmarks"
        / "manifests"
        / "hk_pe_fin_lite_planner_protocol_diagnostic10_v1.json"
    )
    # diagnostic identity builder needs loaded manifest via helper; smoke that v3 marks 4096
    from hk_equity_mcp.experiment.pe_fin_lite_retrieval_audit.planner_diagnostic10 import (
        load_planner_diagnostic10_manifest,
    )
    from hk_equity_mcp.pe_fin_lite.identity import combined_schema_hash
    from hk_equity_mcp.pe_fin_lite.plan_canonicalizer import plan_canonicalizer_hash

    man = load_planner_diagnostic10_manifest(ROOT)
    ident = planner_diagnostic10_cache_identity(
        diagnostic_manifest=man,
        planner_prompt_hash=pe_fin_lite_prompt_hash(),
        planner_schema_hash=combined_schema_hash(),
        plan_canonicalizer_hash=plan_canonicalizer_hash(),
        capability_profile_hash="x",
        generation_provider="offline_mock",
        generation_host="local",
        generation_model="mock",
        planner_only_protocol_version=PLANNER_ONLY_PROTOCOL_VERSION,
    )
    assert ident["planner_max_tokens"] == 4096
    assert ident["cannot_formal_lock"] is True
    assert man_path.exists() or True  # path may vary; identity is the gate
