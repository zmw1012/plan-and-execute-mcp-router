"""Planner protocol v2: structural canonicalizer, repair, resume, gates (offline)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from hk_equity_mcp.experiment.pe_fin_lite_retrieval_audit import (
    PLANNER_ONLY_PROTOCOL_VERSION,
    PLANNER_ONLY_PROTOCOL_VERSION_V1,
)
from hk_equity_mcp.experiment.pe_fin_lite_retrieval_audit.identity import (
    assert_identity_match,
    load_hk155_locked,
    planner_only_cache_identity,
)
from hk_equity_mcp.experiment.pe_fin_lite_retrieval_audit.planner_cache_gate import (
    IncompletePlannerCacheError,
    assert_planner_cache_for_step_pipeline,
)
from hk_equity_mcp.experiment.pe_fin_lite_retrieval_audit.planner_only import (
    FORBIDDEN_PLANNER_KEYS,
    run_planner_only,
)
from hk_equity_mcp.experiment.pe_fin_lite_retrieval_audit.runner import (
    RetrievalAuditRunner,
)
from hk_equity_mcp.pe_fin_lite.plan_canonicalizer import (
    canonicalize_plan_payload,
    normalize_json_text,
    plan_canonicalizer_hash,
)
from hk_equity_mcp.pe_fin_lite.prompts import pe_fin_lite_prompt_hash
from hk_equity_mcp.pe_fin_lite.schemas import (
    SchemaValidationError,
    extract_json_object,
    format_repair_user_message,
    parse_and_validate,
)

ROOT = Path(__file__).resolve().parents[1]
LIVE = ROOT / "outputs" / "pe_fin_lite_retrieval_all155_live_v1"


def _plan(*steps):
    return {"steps": list(steps)}


def test_first_step_missing_depends_on_to_empty():
    payload = _plan(
        {
            "id": "s1",
            "type": "retrieve",
            "domain": "financial",
            "objective": "get revenue",
            "required_outputs": ["revenue"],
        }
    )
    out, applied = canonicalize_plan_payload(payload)
    assert "first_step_missing_depends_on_to_empty_array" in applied
    assert out["steps"][0]["depends_on"] == []
    ok, _, err = parse_and_validate(json.dumps(payload), "plan")
    assert ok is not None, err


def test_later_step_missing_depends_on_fails():
    payload = _plan(
        {
            "id": "s1",
            "type": "retrieve",
            "domain": "financial",
            "objective": "get revenue",
            "required_outputs": ["revenue"],
            "depends_on": [],
        },
        {
            "id": "s2",
            "type": "retrieve",
            "domain": "market",
            "objective": "get close",
            "required_outputs": ["close"],
            # missing depends_on on later step — must fail
        },
    )
    out, applied = canonicalize_plan_payload(payload)
    assert "first_step_missing_depends_on_to_empty_array" not in applied
    assert "depends_on" not in out["steps"][1]
    parsed, _, err = parse_and_validate(json.dumps(payload), "plan")
    assert parsed is None
    assert "depends_on" in (err or "")


def test_reason_analysis_to_general_and_compute_calculation_to_compute():
    payload = _plan(
        {
            "id": "s1",
            "type": "retrieve",
            "domain": "financial",
            "objective": "get revenue",
            "required_outputs": ["revenue"],
            "depends_on": [],
        },
        {
            "id": "s2",
            "type": "compute",
            "domain": "calculation",
            "objective": "compute growth",
            "required_outputs": ["revenue"],
            "depends_on": ["s1"],
            "compute_spec": {
                "expression": "a",
                "operand_refs": {"a": "s1.revenue"},
                "output_name": "x",
            },
        },
        {
            "id": "s3",
            "type": "reason",
            "domain": "analysis",
            "objective": "summarize",
            "required_outputs": [],
            "depends_on": ["s2"],
        },
    )
    out, applied = canonicalize_plan_payload(payload)
    assert out["steps"][1]["domain"] == "compute"
    assert out["steps"][2]["domain"] == "general"
    assert any("calculation" in a for a in applied)
    assert any("analysis" in a for a in applied)
    parsed, _, err = parse_and_validate(json.dumps(payload), "plan")
    assert parsed is not None, err


def test_retrieve_unknown_domain_fails():
    payload = _plan(
        {
            "id": "s1",
            "type": "retrieve",
            "domain": "analysis",
            "objective": "x",
            "required_outputs": ["revenue"],
            "depends_on": [],
        }
    )
    parsed, _, err = parse_and_validate(json.dumps(payload), "plan")
    assert parsed is None
    assert "not in enum" in (err or "")


def test_empty_null_content_one_repair_then_failure():
    from hk_equity_mcp.pe_fin_lite.planner import run_planner

    class FakeLLM:
        def __init__(self):
            self.n = 0

        def complete(self, messages, **kwargs):
            self.n += 1
            return {"content": "" if self.n == 1 else None, "usage": {"calls": 1}}

    out = run_planner(
        llm=FakeLLM(),
        question="q",
        capability_profile={},
        limits={"max_total_llm_calls_per_task": 8, "max_total_mcp_invocations": 8},
    )
    assert out["ok"] is False
    assert out["repair_count"] == 1
    assert "empty" in (out["error"] or "") or "null" in (out["error"] or "")


def test_truncated_json_no_splice():
    truncated = '{"steps":[{"id":"s1","type":"retrieve","domain":"financial"'
    with pytest.raises(SchemaValidationError, match="truncated|invalid JSON|no JSON"):
        extract_json_object(truncated)


def test_length_violation_not_silently_truncated():
    long_obj = "x" * 500
    payload = _plan(
        {
            "id": "s1",
            "type": "retrieve",
            "domain": "financial",
            "objective": long_obj,
            "required_outputs": ["revenue"],
            "depends_on": [],
        }
    )
    parsed, _, err = parse_and_validate(json.dumps(payload), "plan")
    assert parsed is None
    assert "maxLength" in (err or "")
    # Canonicalizer must not shorten objective.
    out, _ = canonicalize_plan_payload(payload)
    assert len(out["steps"][0]["objective"]) == 500


def test_repair_message_includes_path_actual_expected_full_regen():
    msg = format_repair_user_message(
        "plan",
        "$.steps[0]: missing required property 'depends_on'",
        previous_content='{"steps":[]}',
    )
    assert "JSON path:" in msg
    assert "Actual invalid value:" in msg
    assert "COMPLETE" in msg or "complete" in msg.lower()
    assert "splice" in msg.lower() or "Do NOT splice" in msg
    assert "depends_on=[]" in msg
    assert "domain=general" in msg


def test_old_v1_cache_cannot_resume_v2(tmp_path: Path):
    data = load_hk155_locked(ROOT)
    v2 = planner_only_cache_identity(
        dataset_identities=data["identities"],
        capability_profile_hash="x",
        generation_provider="offline_mock",
        generation_host="local",
        generation_model="mock_plan_v1",
    )
    assert v2["planner_only_protocol_version"] == PLANNER_ONLY_PROTOCOL_VERSION
    assert PLANNER_ONLY_PROTOCOL_VERSION != PLANNER_ONLY_PROTOCOL_VERSION_V1
    assert "plan_canonicalizer_hash" in v2
    v1 = {
        **{k: v for k, v in v2.items() if k != "identity_sha256"},
        "planner_only_protocol_version": PLANNER_ONLY_PROTOCOL_VERSION_V1,
    }
    v1.pop("plan_canonicalizer_hash", None)
    from hk_equity_mcp.experiment.pe_fin_lite_retrieval_audit.identity import (
        stable_json_hash,
    )

    v1["identity_sha256"] = stable_json_hash(v1)
    with pytest.raises(RuntimeError, match="Identity mismatch"):
        assert_identity_match(v2, v1, label="planner_only_cache")


def test_incomplete_planner_cache_blocks_formal_and_default_step_stages():
    rows = [{"task_id": f"t{i}", "ok": True} for i in range(154)]
    rows.append({"task_id": "t_fail", "ok": False, "error": "x"})
    # Not 155 unique ok
    with pytest.raises(IncompletePlannerCacheError):
        assert_planner_cache_for_step_pipeline(
            rows=rows, allow_incomplete=False, formal=False, stage="rank"
        )
    with pytest.raises(IncompletePlannerCacheError, match="Formal"):
        assert_planner_cache_for_step_pipeline(
            rows=rows, allow_incomplete=True, formal=True, stage="formula"
        )
    # Diagnostic allow
    summary = assert_planner_cache_for_step_pipeline(
        rows=rows, allow_incomplete=True, formal=False, stage="diag"
    )
    assert summary["n_failed"] == 1


def test_json_text_normalization_trailing_comma_and_fence():
    text = '```json\n{"steps":[{"id":"s1",}],}\n```'
    norm, applied = normalize_json_text(text)
    assert "strip_markdown_fence" in applied
    assert "remove_trailing_commas" in applied


def test_no_gold_task_id_in_prompts_or_canonicalizer():
    from hk_equity_mcp.pe_fin_lite import prompts as P
    from hk_equity_mcp.pe_fin_lite import plan_canonicalizer as C

    blob = "\n".join(
        [
            P.PE_FIN_LITE_SYSTEM_PROMPT,
            P.PE_FIN_LITE_PLANNER_USER_TEMPLATE,
            json.dumps(C.describe_canonicalizer()),
        ]
    )
    for key in FORBIDDEN_PLANNER_KEYS:
        assert key not in blob
    assert "task_id=" not in blob


def test_raw_v1_artifact_sha256_unchanged_snapshot():
    """Guard: planner_outputs.jsonl hash must match frozen audit expectation."""
    path = LIVE / "planner_cache" / "planner_outputs.jsonl"
    if not path.exists():
        pytest.skip("live_v1 planner cache not present")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == "002d60972d13b1e3728642ef7d63eeb1ada543c19bd25129b54ae67be3a312d5"


def test_r0_artifacts_still_present_and_untouched():
    q = LIVE / "question_level" / "per_query_rankings.jsonl"
    qq = LIVE / "question_query_cache" / "question_query_vectors.json"
    if not q.exists() or not qq.exists():
        pytest.skip("R0 artifacts missing")
    assert q.stat().st_size > 0
    assert qq.stat().st_size > 0
    # Status mark must not delete R0
    assert hashlib.sha256(q.read_bytes()).hexdigest() == (
        "5e5dbc7c73ba5e15f1dea9dc383e72462928c9541206c73e81ad37ff6aff23b9"
    )


def test_mock_and_dry_run_planner_v2(tmp_path: Path):
    out = tmp_path / "v2_mock"
    runner = RetrievalAuditRunner(root=ROOT, output_dir=out, mock=True)
    meta = runner.run_planner_only()
    assert meta["mock"] is True
    identity = json.loads(
        (out / "planner_cache" / "planner_cache_identity.json").read_text()
    )
    assert identity["planner_only_protocol_version"] == PLANNER_ONLY_PROTOCOL_VERSION
    assert identity.get("plan_canonicalizer_hash") == plan_canonicalizer_hash()
    # dry-run
    dry = tmp_path / "v2_dry"
    runner2 = RetrievalAuditRunner(root=ROOT, output_dir=dry, dry_run=True)
    meta2 = runner2.run_planner_only()
    assert meta2["dry_run"] is True


def test_prompt_hash_stable_and_includes_depends_on_rules():
    from hk_equity_mcp.pe_fin_lite.prompts import PE_FIN_LITE_PLANNER_USER_TEMPLATE

    assert "depends_on=[]" in PE_FIN_LITE_PLANNER_USER_TEMPLATE
    assert "domain=general" in PE_FIN_LITE_PLANNER_USER_TEMPLATE
    assert "domain=compute" in PE_FIN_LITE_PLANNER_USER_TEMPLATE
    h1 = pe_fin_lite_prompt_hash()
    h2 = pe_fin_lite_prompt_hash()
    assert h1 == h2
