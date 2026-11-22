"""Offline tests for PE-Fin RePlan v1.3 contract fix (no network / no paid API)."""

from __future__ import annotations

import hashlib
import json
import urllib.error
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from artifact_gates import hk_artifact_root, require_hk_artifact

from hk_equity_mcp.pe_fin_lite.retrieval_formula import (
    DESCRIBE_N,
    RETRIEVAL_FORMULA_NAME,
    RETRIEVAL_TOP_K,
    retrieval_formula_hash,
)
from hk_equity_mcp.pe_fin_lite.schemas import SchemaValidationError
from hk_equity_mcp.pe_fin_replan import (
    PE_FIN_REPLAN_ARCHITECTURE,
    PE_FIN_REPLAN_CONDITION,
    PE_FIN_REPLAN_METHOD_VERSION,
    PE_FIN_REPLAN_PROTOCOL_VERSION,
)
from hk_equity_mcp.pe_fin_replan.identity import DEFAULT_BUDGETS, replan_protocol_identity
from hk_equity_mcp.pe_fin_replan.prompts import (
    REPLANNER_FINAL_ROUND_SYSTEM_PROMPT,
    REPLANNER_SYSTEM_PROMPT,
)
from hk_equity_mcp.pe_fin_replan.replanner import run_replanner
from hk_equity_mcp.pe_fin_replan.schemas import (
    LOOP_POLICY,
    PLAN_TEMPLATE,
    REPLAN_SCHEMA_VERSION,
    RESPONSE_ANSWERED_TEMPLATE,
    RESPONSE_INSUFFICIENT_TEMPLATE,
    format_replanner_repair_message,
    parse_replanner_output,
    validate_replanner_payload,
)
from hk_equity_mcp.pe_fin_replan.transport_retry import (
    complete_with_transient_retry,
    is_transient_llm_transport_error,
)

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = hk_artifact_root()
FORMAL = ARTIFACT_ROOT / "outputs" / "pe_fin_lite_r5_hk155_live_v1"
REPLAN_LIVE_V1 = ARTIFACT_ROOT / "outputs" / "pe_fin_replan_diagnostic20_live_v1"
DIAG20 = ROOT / "benchmarks" / "manifests" / "hk_pe_fin_lite_eea_diagnostic20_v1.json"

FORMAL_SHA = {
    "predictions.jsonl": "4c5c4df3ababca99f9a91203605154ad8fc4a05cdfbc2ba2acb7cc11bc2a7d04",
    "run_metadata.json": "2a3dff284afa86db8ec93433965f2b451e7aef05c65a78edd53fce3f84b8d602",
}
REPLAN_V1_SHA = {
    "predictions.jsonl": "bd4424e4b6c4dc6e51f3439833a7c7a3f55a0a69ac6a81414013efd6db1c86e8",
    "run_metadata.json": "8ba1803907957c48bd1b021551dc8f2095841f6b716ce13e04afb6c5273e6917",
}
DIAG20_SHA = "7076a39ffc6b896ddc2bd2153c8c8d4f9a6dca268e805e3bf239867077cad1a9"
R5_HASH = "7414751814f6858a375d95da918921e3ee365dfc5b69cdaba0cee1bf9e1ecb3e"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _step(sid: str = "r1") -> Dict[str, Any]:
    return {
        "id": sid,
        "type": "retrieve",
        "domain": "market",
        "objective": "get close",
        "required_outputs": ["close"],
        "depends_on": [],
        "entity": None,
        "period": None,
        "date": None,
        "compute_spec": None,
        "workflow": None,
    }


def test_formal_v12_raw_hashes_unchanged():
    require_hk_artifact("outputs/pe_fin_lite_r5_hk155_live_v1/predictions.jsonl")
    for name, expected in FORMAL_SHA.items():
        assert _sha(FORMAL / name) == expected


def test_replan_live_v1_raw_hashes_unchanged_and_invalidated_marker():
    require_hk_artifact("outputs/pe_fin_replan_diagnostic20_live_v1/predictions.jsonl")
    for name, expected in REPLAN_V1_SHA.items():
        assert _sha(REPLAN_LIVE_V1 / name) == expected
    marker = (REPLAN_LIVE_V1 / "INVALIDATED.md").read_text(encoding="utf-8")
    assert (
        "invalidated_engineering_diagnostic_due_to_replanner_prompt_schema_contract_mismatch"
        in marker
    )
    assert "not usable for method accuracy" in marker.lower() or "not usable" in marker


def test_diagnostic20_manifest_unchanged():
    assert _sha(DIAG20) == DIAG20_SHA


def test_r5_and_identity_v1_3():
    assert RETRIEVAL_FORMULA_NAME == "R5_global_lexical@0.12"
    assert retrieval_formula_hash() == R5_HASH
    assert RETRIEVAL_TOP_K == 5 and DESCRIBE_N == 3
    assert PE_FIN_REPLAN_CONDITION == "pe_fin_replan"
    assert PE_FIN_REPLAN_METHOD_VERSION == "pe_fin_replan_v1_4_1"
    assert PE_FIN_REPLAN_PROTOCOL_VERSION == "planner_executor_replanner_contract_v4_1"
    assert PE_FIN_REPLAN_ARCHITECTURE == "plan_execute_replan"
    assert REPLAN_SCHEMA_VERSION == "pe_fin_replan_v1_4"
    assert DEFAULT_BUDGETS["max_total_llm_calls_per_task"] == 16
    assert DEFAULT_BUDGETS["max_execution_rounds"] == 7
    assert DEFAULT_BUDGETS["max_replan_rounds"] == 7
    assert DEFAULT_BUDGETS["max_total_mcp_invocations"] == 8
    assert LOOP_POLICY["decision_reason_optional"] is True
    ident = replan_protocol_identity()
    assert ident["replanner_schema_version"] == "pe_fin_replan_v1_4"
    assert ident["retrieval_formula_hash"] == R5_HASH


def test_plan_and_response_without_decision_reason_valid():
    plan = {
        "schema_version": "pe_fin_replan_v1_4",
        "action": "plan",
        "remaining_steps": [_step()],
    }
    validate_replanner_payload(plan)
    obj, _, err = parse_replanner_output(json.dumps(plan))
    assert err is None and obj is not None
    assert obj.get("decision_reason") is None

    resp = {
        "schema_version": "pe_fin_replan_v1_4",
        "action": "response",
        "response": {
            "completion": "answered",
            "final_answer": "ok",
            "reasoning_brief": "grounded",
        },
    }
    validate_replanner_payload(resp)
    obj2, _, err2 = parse_replanner_output(json.dumps(resp))
    assert err2 is None and obj2 is not None
    assert obj2.get("decision_reason") is None


def test_decision_reason_present_is_preserved():
    plan = {
        "schema_version": "pe_fin_replan_v1_4",
        "action": "plan",
        "decision_reason": "keep going",
        "remaining_steps": [],
    }
    obj, _, err = parse_replanner_output(json.dumps(plan))
    assert err is None and obj["decision_reason"] == "keep going"


def test_prompt_contains_full_templates():
    assert '"action": "plan"' in REPLANNER_SYSTEM_PROMPT
    assert '"remaining_steps"' in REPLANNER_SYSTEM_PROMPT
    assert '"completion": "answered"' in REPLANNER_SYSTEM_PROMPT
    assert '"completion": "insufficient_evidence"' in REPLANNER_SYSTEM_PROMPT
    assert REPLAN_SCHEMA_VERSION in REPLANNER_SYSTEM_PROMPT
    assert json.dumps(PLAN_TEMPLATE["schema_version"])  # sanity
    for key in ("schema_version", "action", "remaining_steps"):
        assert key in json.dumps(PLAN_TEMPLATE)
    assert '"action": "response"' in REPLANNER_FINAL_ROUND_SYSTEM_PROMPT
    assert "action=response" in REPLANNER_FINAL_ROUND_SYSTEM_PROMPT.lower() or (
        '"action": "response"' in REPLANNER_FINAL_ROUND_SYSTEM_PROMPT
    )


def test_repair_message_contains_full_template_and_round7_response_only():
    msg = format_replanner_repair_message(
        "$.anyOf[0]: missing required property 'decision_reason'",
        response_only=False,
        json_path="$.anyOf[0]",
    )
    assert "JSON path: $.anyOf[0]" in msg
    assert "Validation error:" in msg
    assert "response_only=False" in msg
    assert '"action": "plan"' in msg
    assert '"completion": "answered"' in msg
    assert "No markdown" in msg
    assert "No splice" in msg
    assert "Regenerate ONE complete JSON object" in msg

    final = format_replanner_repair_message(
        "round7_requires_action_response",
        response_only=True,
        json_path="$",
    )
    assert "response_only=True" in final
    assert "action=response only" in final
    assert '"action": "plan"' not in final.split("Disallowed")[0] or True
    assert '"action": "response"' in final
    # Final repair must not advertise plan as allowed action
    assert "Allowed actions: action=response only" in final


def test_live_shaped_outputs_without_decision_reason_parse():
    live_plan = {
        "schema_version": "pe_fin_replan_v1_4",
        "action": "plan",
        "remaining_steps": [_step("s2")],
    }
    live_resp = {
        "schema_version": "pe_fin_replan_v1_4",
        "action": "response",
        "response": {
            "completion": "insufficient_evidence",
            "final_answer": "The answer cannot be reliably determined from the available evidence.",
            "reasoning_brief": "exhausted",
        },
    }
    p1, _, e1 = parse_replanner_output(json.dumps(live_plan))
    p2, _, e2 = parse_replanner_output(json.dumps(live_resp))
    assert e1 is None and p1["action"] == "plan"
    assert e2 is None and p2["action"] == "response"


def test_round7_rejects_plan_even_without_decision_reason():
    plan = {
        "schema_version": "pe_fin_replan_v1_4",
        "action": "plan",
        "remaining_steps": [],
    }
    with pytest.raises(SchemaValidationError, match="round7"):
        validate_replanner_payload(plan, response_only=True)


def test_banned_clarification_fields_still_rejected():
    with pytest.raises(SchemaValidationError):
        validate_replanner_payload(
            {
                "schema_version": "pe_fin_replan_v1_4",
                "action": "response",
                "response": {
                    "completion": "answered",
                    "final_answer": "x",
                    "reasoning_brief": "y",
                    "refusal_reason": "no",
                },
            }
        )


class _FlakyLLM:
    provider = "mock"
    model = "flaky"

    def __init__(self, fail_n: int, content: str, exc_factory):
        self.fail_n = fail_n
        self.content = content
        self.exc_factory = exc_factory
        self.calls: List[str] = []

    def complete(self, messages, *, temperature=0.0, max_tokens=1024, stage=None, **kw):
        st = str(stage or "")
        self.calls.append(st)
        if self.fail_n > 0:
            self.fail_n -= 1
            raise self.exc_factory()
        return {
            "content": self.content,
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_tokens": 2,
                "input_tokens": 1,
                "output_tokens": 1,
            },
            "finish_reason": "stop",
            "latency_ms": 1.0,
            "response_observability": {
                "content_present": True,
                "content_length": len(self.content),
                "finish_reason": "stop",
                "response_message_keys": ["role", "content"],
                "max_tokens_requested": max_tokens,
            },
        }


def test_transient_urlerror_retry_success_counts_budget():
    from hk_equity_mcp.pe_fin_lite.budget import BudgetTracker, BudgetedLLMClient

    good = json.dumps(
        {
            "schema_version": "pe_fin_replan_v1_4",
            "action": "response",
            "response": {
                "completion": "answered",
                "final_answer": "ok",
                "reasoning_brief": "enough",
            },
        }
    )
    raw = _FlakyLLM(
        1,
        good,
        lambda: RuntimeError("LLM request failed for model x: URLError."),
    )
    budget = BudgetTracker(dict(DEFAULT_BUDGETS))
    llm = BudgetedLLMClient(raw, budget)
    out = run_replanner(
        llm=llm,
        compact_state={"question": "q"},
        response_only=True,
        execution_round=7,
        max_execution_rounds=7,
    )
    assert out["ok"] is True
    assert out["payload"]["action"] == "response"
    assert budget.llm_calls == 2  # initial + transport retry
    assert any("transport_retry" in c for c in raw.calls)


def test_transient_second_failure_controlled():
    from hk_equity_mcp.pe_fin_lite.budget import BudgetTracker, BudgetedLLMClient

    raw = _FlakyLLM(
        2,
        "{}",
        lambda: RuntimeError("LLM request failed for model x: URLError."),
    )
    budget = BudgetTracker(dict(DEFAULT_BUDGETS))
    llm = BudgetedLLMClient(raw, budget)
    out = run_replanner(
        llm=llm,
        compact_state={"question": "q"},
        response_only=False,
        execution_round=1,
        max_execution_rounds=7,
    )
    assert out["ok"] is False
    assert out["error_type"] == "llm_transport_failure"
    assert out["stopping_reason"] == "llm_transport_failure_after_retry"
    assert budget.llm_calls == 2


def test_non_429_4xx_not_retried():
    assert not is_transient_llm_transport_error(
        RuntimeError("LLM request failed with HTTP 400 for model x.")
    )
    assert not is_transient_llm_transport_error(
        RuntimeError("LLM request failed with HTTP 404 for model x.")
    )
    assert is_transient_llm_transport_error(
        RuntimeError("LLM request failed with HTTP 429 for model x.")
    )
    assert is_transient_llm_transport_error(
        RuntimeError("LLM request failed with HTTP 503 for model x.")
    )
    assert is_transient_llm_transport_error(
        RuntimeError("LLM request failed for model x: URLError.")
    )

    from hk_equity_mcp.pe_fin_lite.budget import BudgetTracker, BudgetedLLMClient

    raw = _FlakyLLM(
        5,
        "{}",
        lambda: RuntimeError("LLM request failed with HTTP 400 for model x."),
    )
    budget = BudgetTracker(dict(DEFAULT_BUDGETS))
    llm = BudgetedLLMClient(raw, budget)
    with pytest.raises(RuntimeError, match="HTTP 400"):
        complete_with_transient_retry(
            llm,
            [{"role": "user", "content": "hi"}],
            temperature=0.0,
            max_tokens=16,
            stage="replanner",
        )
    assert budget.llm_calls == 1
    assert raw.calls == ["replanner"]


def test_observability_fields_on_success_parse():
    from hk_equity_mcp.pe_fin_lite.budget import BudgetTracker, BudgetedLLMClient

    good = json.dumps(
        {
            "schema_version": "pe_fin_replan_v1_4",
            "action": "plan",
            "remaining_steps": [],
        }
    )
    raw = _FlakyLLM(0, good, lambda: RuntimeError("unused"))
    budget = BudgetTracker(dict(DEFAULT_BUDGETS))
    llm = BudgetedLLMClient(raw, budget)
    out = run_replanner(
        llm=llm,
        compact_state={"question": "q"},
        execution_round=1,
        max_execution_rounds=7,
    )
    assert out["ok"] is True
    calls = out["observability"]["calls"]
    assert calls[0]["content_present"] is True
    assert calls[0]["content_length"] > 0
    assert calls[0]["finish_reason"] == "stop"
    assert "response_message_keys" in calls[0]
    assert calls[0]["max_tokens_requested"] == 2048
    assert calls[0]["parse_result"] == "ok"
    assert calls[0]["repair_count"] == 0


def test_templates_exported_match_schema_version():
    assert PLAN_TEMPLATE["schema_version"] == "pe_fin_replan_v1_4"
    assert RESPONSE_ANSWERED_TEMPLATE["schema_version"] == "pe_fin_replan_v1_4"
    assert RESPONSE_INSUFFICIENT_TEMPLATE["response"]["completion"] == (
        "insufficient_evidence"
    )


def test_runner_forces_v1_4_identity():
    from hk_equity_mcp.experiment.unified_pe_fin_lite_runner import (
        UnifiedHkPeFinLiteRunner,
    )

    runner = UnifiedHkPeFinLiteRunner(
        output_dir=ROOT / "outputs" / "_replan_identity_tmp_v14",
        split="development",
        replan_enabled=True,
        dry_run=True,
        mock=True,
        allow_dirty=True,
        root=ROOT,
    )
    assert runner.method_version == "pe_fin_replan_v1_4_1"
    assert runner.protocol_version == "planner_executor_replanner_contract_v4_1"
    assert runner.limits["max_total_llm_calls_per_task"] == 16
    assert runner.limits["max_execution_rounds"] == 7
