"""Planner protocol v2: answerability_reason, compactness, repair, artifacts (offline)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from hk_equity_mcp.agent.llm_client import ScriptedLLMClient
from hk_equity_mcp.apv_fin.identity import (
    PLANNER_PROTOCOL_CHANGE,
    combined_schema_hash,
    schema_hashes,
)
from hk_equity_mcp.apv_fin.planner import run_initial_planner
from hk_equity_mcp.apv_fin.prompts import (
    APV_FIN_PLANNER_USER_TEMPLATE,
    APV_FIN_SYSTEM_PROMPT,
    apv_fin_prompt_hash,
    assert_prompts_clean,
)
from hk_equity_mcp.apv_fin.schemas import (
    ANSWERABILITY_HYPOTHESIS_ENUM,
    APV_FIN_PLAN_V1,
    PLAN_COMPACT_LIMITS,
    SchemaValidationError,
    format_repair_user_message,
    parse_and_validate,
    validate_against_schema,
)
from hk_equity_mcp.experiment.unified_apv_fin_metrics import aggregate_apv_fin_metrics
from hk_equity_mcp.experiment.unified_apv_fin_runner import (
    UnifiedHkApvFinRunner,
    assert_output_dir_safe,
)

ROOT = Path(__file__).resolve().parents[1]
MAX_TOKENS = 2048
# Conservative chars→tokens estimate (≈4 chars/token) with headroom.
CHARS_PER_TOKEN_EST = 4


def _compact_plan(**overrides: Any) -> Dict[str, Any]:
    plan: Dict[str, Any] = {
        "schema_version": "apv_fin_plan_v1",
        "task_spec": {
            "goal": "Obtain the requested HK equity metric from supported sources.",
            "entities": ["hk_listed_issuer"],
            "periods": ["FY2025"],
            "metrics": ["requested_metric"],
            "constraints": ["cutoff_2026-08-12"],
            "ambiguities": [],
            "answerability_hypothesis": "answerable",
            "answerability_reason": "Required evidence is available from supported sources.",
            "cutoff_date": "2026-08-12",
            "currency_unit_hints": [],
            "metric_basis_hints": [],
        },
        "initial_plan": {
            "steps": [
                {
                    "step_id": "s1_retrieve",
                    "step_type": "retrieve",
                    "objective": "Retrieve supporting evidence for the metric.",
                    "depends_on": [],
                    "mandatory": True,
                    "success_criteria": "Evidence rows obtained",
                    "required_capability": "financial.reported_metric",
                    "required_outputs": ["value"],
                    "argument_slots": {"ticker": "00700"},
                    "retrieval_query_hint": "hong kong equity metric",
                    "notes": None,
                },
                {
                    "step_id": "s2_reason",
                    "step_type": "reason",
                    "objective": "Summarize evidence for the answer.",
                    "depends_on": ["s1_retrieve"],
                    "mandatory": True,
                    "success_criteria": "Concise summary ready",
                    "retrieval_query_hint": None,
                    "notes": None,
                },
                {
                    "step_id": "s3_answer",
                    "step_type": "answer",
                    "objective": "Emit final answer.",
                    "depends_on": ["s2_reason"],
                    "mandatory": True,
                    "success_criteria": "Final JSON ready",
                    "retrieval_query_hint": None,
                    "notes": None,
                },
            ],
            "planning_notes": None,
        },
    }
    for key, value in overrides.items():
        if key == "task_spec":
            plan["task_spec"].update(value)
        elif key == "steps":
            plan["initial_plan"]["steps"] = value
        else:
            plan[key] = value
    return plan


def _cross_domain_compact_plan() -> Dict[str, Any]:
    """Complex but still compact cross-domain fixture under output budget."""
    return _compact_plan(
        task_spec={
            "goal": "Link FY disclosure metric to same-day market close under cutoff.",
            "entities": ["hk_listed_issuer", "hk_index"],
            "periods": ["FY2025", "2026-07-31"],
            "metrics": ["eps", "close", "yoy_change"],
            "constraints": [
                "cutoff_2026-08-12",
                "prefer_official_disclosure",
                "hkd_if_stated",
            ],
            "ambiguities": ["basis_ifrs_vs_adjusted"],
            "answerability_hypothesis": "answerable",
            "answerability_reason": "Disclosure and curated market series are both supported.",
            "currency_unit_hints": ["HKD"],
            "metric_basis_hints": ["reported_ifrs"],
        },
        steps=[
            {
                "step_id": "s1_fin",
                "step_type": "retrieve",
                "objective": "Retrieve FY disclosure metric.",
                "depends_on": [],
                "mandatory": True,
                "success_criteria": "Disclosure value found",
                    "required_capability": "financial.reported_metric",
                    "required_outputs": ["value"],
                    "argument_slots": {"ticker": "00700"},
                "retrieval_query_hint": "hong kong issuer fy eps",
                "notes": None,
            },
            {
                "step_id": "s2_mkt",
                "step_type": "retrieve",
                "objective": "Retrieve same-day market close.",
                "depends_on": [],
                "mandatory": True,
                "success_criteria": "Close price found",
                    "required_capability": "financial.reported_metric",
                    "required_outputs": ["value"],
                    "argument_slots": {"ticker": "00700"},
                "retrieval_query_hint": "stock hk hist close",
                "notes": None,
            },
            {
                "step_id": "s3_compute",
                "step_type": "compute",
                "objective": "Compute simple YoY change if both ends present.",
                "depends_on": ["s1_fin"],
                "mandatory": False,
                "success_criteria": "Numeric change ready",
                "retrieval_query_hint": None,
                "notes": None,
            },
            {
                "step_id": "s4_reason",
                "step_type": "reason",
                "objective": "Join disclosure and market evidence.",
                "depends_on": ["s1_fin", "s2_mkt"],
                "mandatory": True,
                "success_criteria": "Joined summary ready",
                "retrieval_query_hint": None,
                "notes": None,
            },
            {
                "step_id": "s5_answer",
                "step_type": "answer",
                "objective": "Emit final answer with units.",
                "depends_on": ["s4_reason"],
                "mandatory": True,
                "success_criteria": "Final JSON ready",
                "retrieval_query_hint": None,
                "notes": None,
            },
        ],
    )


def test_hypothesis_enum_plus_reason_valid():
    plan = _compact_plan()
    validate_against_schema(plan, APV_FIN_PLAN_V1)
    parsed, repair, err = parse_and_validate(json.dumps(plan), "apv_fin_plan_v1")
    assert err is None and repair == 0
    assert parsed["task_spec"]["answerability_hypothesis"] == "answerable"
    assert "available" in parsed["task_spec"]["answerability_reason"].lower()


def test_prose_in_hypothesis_rejected():
    plan = _compact_plan(
        task_spec={
            "answerability_hypothesis": (
                "Likely answerable if the issuer's official FY2025 report discloses EPS."
            )
        }
    )
    with pytest.raises(SchemaValidationError, match="enum mismatch"):
        validate_against_schema(plan, APV_FIN_PLAN_V1)


def test_repair_splits_prose_into_enum_and_reason():
    bad = _compact_plan(
        task_spec={
            "answerability_hypothesis": "Likely answerable from supported disclosure sources."
        }
    )
    try:
        validate_against_schema(bad, APV_FIN_PLAN_V1)
        raise AssertionError("expected enum mismatch")
    except SchemaValidationError as exc:
        msg = format_repair_user_message(
            "apv_fin_plan_v1", f"SchemaValidationError: {exc}"
        )
    assert "$.task_spec.answerability_hypothesis" in msg
    assert "Do not place explanation prose" in msg
    assert "answerability_reason" in msg
    for token in ANSWERABILITY_HYPOTHESIS_ENUM:
        assert token in msg

    fixed = _compact_plan(
        task_spec={
            "answerability_hypothesis": "answerable",
            "answerability_reason": "Required evidence is available from supported sources.",
        }
    )
    payload, repair, err = parse_and_validate(
        json.dumps(bad), "apv_fin_plan_v1", repair_content=json.dumps(fixed)
    )
    assert err is None and repair == 1
    assert payload["task_spec"]["answerability_hypothesis"] == "answerable"


def test_compact_plan_fixture_valid():
    validate_against_schema(_compact_plan(), APV_FIN_PLAN_V1)
    validate_against_schema(_cross_domain_compact_plan(), APV_FIN_PLAN_V1)


def test_max_items_rejected():
    plan = _compact_plan(
        task_spec={
            "entities": ["a", "b", "c", "d", "e"],  # max 4
        }
    )
    with pytest.raises(SchemaValidationError, match="maxItems"):
        validate_against_schema(plan, APV_FIN_PLAN_V1)


def test_max_length_rejected():
    plan = _compact_plan(
        task_spec={"goal": "x" * (PLAN_COMPACT_LIMITS["goal_max"] + 1)}
    )
    with pytest.raises(SchemaValidationError, match="maxLength"):
        validate_against_schema(plan, APV_FIN_PLAN_V1)


def test_overlong_ambiguity_rejected():
    plan = _compact_plan(
        task_spec={
            "ambiguities": ["y" * (PLAN_COMPACT_LIMITS["ambiguity_item_max"] + 1)]
        }
    )
    with pytest.raises(SchemaValidationError, match="maxLength"):
        validate_against_schema(plan, APV_FIN_PLAN_V1)


def test_overlong_step_objective_rejected():
    plan = _compact_plan()
    plan["initial_plan"]["steps"][0]["objective"] = "z" * (
        PLAN_COMPACT_LIMITS["objective_max"] + 1
    )
    with pytest.raises(SchemaValidationError, match="maxLength"):
        validate_against_schema(plan, APV_FIN_PLAN_V1)


def test_truncated_json_full_regeneration_not_splice():
    truncated = '{"schema_version":"apv_fin_plan_v1","task_spec":{"goal":"x"'
    msg = format_repair_user_message(
        "apv_fin_plan_v1",
        "JSONDecodeError: Expecting ',' delimiter: line 1 column 5919 (char 5918)",
    )
    assert "truncated_or_invalid_json" in msg
    assert "do NOT patch" in msg or "Do NOT patch" in msg
    assert "Regenerate the full object" in msg

    fixed = json.dumps(_compact_plan())
    payload, repair, err = parse_and_validate(
        truncated, "apv_fin_plan_v1", repair_content=fixed
    )
    assert err is None and repair == 1
    assert payload["schema_version"] == "apv_fin_plan_v1"


def test_second_repair_failure_fail_closed():
    llm = ScriptedLLMClient(
        [
            "not-json-at-all",
            "still-broken-{",
            # A third turn must never be consumed by planner (one repair only).
            json.dumps(_compact_plan()),
        ]
    )
    out = run_initial_planner(llm, "What is the FY metric?", max_tokens=MAX_TOKENS)
    assert out["ok"] is False
    assert out["fail_closed"] is True
    assert int(out["repair_count"]) == 1
    # Scripted client queue should still have the unused third payload.
    assert len(llm._queue) == 1


def test_no_task_id_special_cases_in_schema_or_prompt():
    blob = json.dumps(APV_FIN_PLAN_V1, ensure_ascii=False) + APV_FIN_PLANNER_USER_TEMPLATE
    for needle in (
        "fin_cf_003",
        "fin_cf_023",
        "xd_dev_002",
        "mkt_dev_014",
        "Tencent",
        "HSBC",
        "Alibaba",
        "Pop Mart",
        "Xiaomi",
        "gold_operations",
    ):
        assert needle not in blob


def test_all_failed_run_still_writes_all_artifacts(tmp_path: Path):
    """Simulate all-failed predictions and assert full artifact set is written."""
    out = tmp_path / "all_failed"
    out.mkdir()
    failed_rows: List[Dict[str, Any]] = []
    for tid in ("t1", "t2", "t3", "t4"):
        failed_rows.append(
            {
                "task_id": tid,
                "id": tid,
                "status": "failed",
                "result_status": "failed",
                "stopping_reason": "planner_fail_closed",
                "llm_calls": 2,
                "llm_calls_used": 2,
                "calls_by_stage": {"planner": 1, "planner_repair": 1},
                "tokens": {
                    "input_tokens": 100,
                    "output_tokens": 50,
                    "total_tokens": 150,
                },
                "latency_ms": 12.5,
                "real_mcp_execution_count": 0,
                "embedding_calls": 0,
                "auto_grade_status": "manual_required",
                "answer_correct": None,
                "domain": "financial",
                "domain_combination": "financial",
                "project_difficulty": "easy",
                "exact_etom_level": "",
                "exact_etom_alignment": "",
                "answerability_class": "answerable",
            }
        )
    # Use runner side-artifact writer via a lightweight stub.
    runner = UnifiedHkApvFinRunner(
        output_dir=out,
        split="development",
        smoke=True,
        dry_run=True,
        allow_dirty=True,
        root=ROOT,
    )
    items = [
        {
            "task_id": r["task_id"],
            "domain": r["domain"],
            "domain_combination": r["domain_combination"],
            "project_difficulty": r["project_difficulty"],
            "exact_etom_level": r["exact_etom_level"],
            "exact_etom_alignment": r["exact_etom_alignment"],
            "answerability_class": r["answerability_class"],
            "question": "q",
            "gold_status": "",
            "gold_operations": [],
            "reference_answer": "",
            "company": "",
            "source_split": "development",
        }
        for r in failed_rows
    ]
    preds = {r["task_id"]: r for r in failed_rows}
    (out / "predictions.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in failed_rows) + "\n",
        encoding="utf-8",
    )
    runner._write_side_artifacts(items=items, predictions_by_id=preds, run_meta={})

    required = [
        "predictions.jsonl",
        "per_task_metrics.csv",
        "manual_grading.csv",
        "cost_summary.json",
        "error_summary.json",
        "grouped_aggregates.json",
        "grading_summary.json",
        "apv_fin_metrics.json",
    ]
    for name in required:
        assert (out / name).exists(), name

    cost = json.loads((out / "cost_summary.json").read_text(encoding="utf-8"))
    assert cost["n_tasks"] == 4
    assert cost["failed_tasks"] == 4
    assert cost["planner_calls"] == 4
    assert cost["planner_repair_calls"] == 4
    assert cost["total_real_mcp_executions"] == 0
    assert cost["total_embedding_query_calls"] == 0
    assert cost["estimated_cost_usd"] is None
    assert cost["total_tokens"] == 600

    metrics = aggregate_apv_fin_metrics(failed_rows)
    assert metrics["n_tasks"] == 4
    assert metrics["cost_summary"]["planner_calls"] == 4


def test_planner_prompt_has_example_no_company_whitelist_or_gold():
    assert_prompts_clean()
    assert '"answerability_hypothesis": "answerable"' in APV_FIN_PLANNER_USER_TEMPLATE
    assert (
        '"answerability_reason": "Required evidence is available from supported sources."'
        in APV_FIN_PLANNER_USER_TEMPLATE
    )
    assert "Output ONE compact JSON object only" in APV_FIN_PLANNER_USER_TEMPLATE
    assert "Do not write prose in enum fields" in APV_FIN_PLANNER_USER_TEMPLATE
    for needle in (
        "Tencent",
        "HSBC",
        "Alibaba",
        "Pop Mart",
        "Xiaomi",
        "gold_operations",
        "reference_answer",
        "fin_cf_",
        "mkt_dev_",
    ):
        assert needle not in APV_FIN_PLANNER_USER_TEMPLATE
        assert needle not in APV_FIN_SYSTEM_PROMPT


def test_schema_prompt_identity_mismatch_prevents_resume(tmp_path: Path):
    out = tmp_path / "resume_guard"
    out.mkdir()
    meta = {
        "condition": "apv_fin",
        "method_version": "apv_fin_v1",
        "execution_backend": "real_mcp",
        "generation_provider": "openai_compatible",
        "generation_base_url_host": "api.zhizengzeng.com",
        "generation_model": "deepseek-v4.1-flash",
        "embedding_provider": "openai_compatible",
        "embedding_base_url_host": "api.zhizengzeng.com",
        "embedding_model": "text-embedding-3-small",
        "system_prompt_hash": "deadbeef" * 8,
        "schema_hash": "cafebabe" * 8,
        "operation_metadata_hash": "0" * 64,
        "catalog_fingerprint": "0" * 64,
        "dataset_split": "development",
        "subset_kind": "smoke",
        "subset_manifest_hash": None,
        "subset_ordered_id_hash": None,
        "diagnostic_manifest_hash": None,
        "smoke4_manifest_hash": None,
        "temperature": 0.0,
        "max_tokens": 2048,
        "json_mode": True,
        "limits": {},
        "max_total_llm_calls_per_task": 14,
        "mock": True,
        "dry_run": False,
    }
    (out / "run_metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out / "predictions.jsonl").write_text("", encoding="utf-8")
    current = dict(meta)
    current["system_prompt_hash"] = apv_fin_prompt_hash()
    current["schema_hash"] = combined_schema_hash()
    with pytest.raises(RuntimeError, match="identity mismatch"):
        assert_output_dir_safe(
            output_dir=out,
            dry_run=False,
            mock=True,
            resume=True,
            run_meta=current,
        )


def test_compact_plan_under_output_token_budget():
    """Offline fixture: typical + cross-domain plans stay under max_tokens=2048."""
    for label, plan in (
        ("typical", _compact_plan()),
        ("cross_domain", _cross_domain_compact_plan()),
    ):
        blob = json.dumps(plan, ensure_ascii=False, separators=(",", ":"))
        char_count = len(blob)
        token_est = (char_count + CHARS_PER_TOKEN_EST - 1) // CHARS_PER_TOKEN_EST
        # Leave substantial headroom vs 2048 — verbosity must not require raising budget.
        assert token_est < 900, (
            f"{label}: chars={char_count} token_est={token_est} exceeds compact budget"
        )
        assert token_est < MAX_TOKENS


def test_planner_protocol_change_label():
    assert PLANNER_PROTOCOL_CHANGE == (
        "planner_protocol_v2_before_valid_method_evaluation"
    )
    assert apv_fin_prompt_hash()
    assert schema_hashes()["apv_fin_plan_v1"]
    assert combined_schema_hash()
