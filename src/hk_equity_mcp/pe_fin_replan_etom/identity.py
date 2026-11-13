"""Identity for PE-Fin RePlan-ETOM v1: budgets, protocol hashes, frozen ETOM-100 constants.

Frozen dataset identity is HARD-CODED here and asserted against the lock file.
Nothing in this module reads gold labels.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from hk_equity_mcp.pe_fin_replan_etom import (
    EMBEDDING_CACHE_NAMESPACE,
    ETOM_FLAGS,
    PE_FIN_REPLAN_ETOM_ARCHITECTURE,
    PE_FIN_REPLAN_ETOM_BASE_METHOD,
    PE_FIN_REPLAN_ETOM_CONDITION,
    PE_FIN_REPLAN_ETOM_METHOD_VERSION,
    PE_FIN_REPLAN_ETOM_PROTOCOL_VERSION,
    PE_FIN_REPLAN_ETOM_TRANSFER_SOURCE,
)

# --------------------------------------------------------------------------
# Frozen ETOM-100 identity (must match benchmarks/etom/*.lock.json + pin)
# --------------------------------------------------------------------------
FROZEN_SAMPLER = "etom_stratified_external_100_v1"
FROZEN_SEED = 20261106
FROZEN_SAMPLE_SHA256 = "99cb671ea3528c1c69ab6fc2a3d622c4d6f6bb07058728071aae5eae8720cbb1"
FROZEN_ORDERED_TASK_IDS_SHA256 = "3be67c36e4eec58920864a401bbaa73db55136930b0f406e2da56c4988734479"
FROZEN_LOCK_FILE_SHA256 = "2bda1a2f16dd1b36b5a73b4057b7305b4901ed83eba0b537db2f187a56ed309b"
FROZEN_SOURCE_PIN_SHA256 = "5f1ae09e7a8fbb7b554ed1302433feacaffb8645ccc560c7bb87ff64893f2692"
FROZEN_SOURCE_COMMIT = "790f28727f567b88918420d163c524d46644a694"
FROZEN_REGISTRY_SHA256 = "3c362877ec5733cf0fa1d3451b2f353fb97674e66075da0d7a6f41c0f98cc608"
FROZEN_CATALOG_TOTAL_TOOLS = 2375
FROZEN_TASK_COUNT = 100
FROZEN_PER_LEVEL = 20
FROZEN_LOCK_REL = "benchmarks/etom/formal_100_v1.lock.json"
FROZEN_PIN_REL = "benchmarks/etom/etom_source_pin_v1.json"

METHOD_IDENTITY = {
    "condition": PE_FIN_REPLAN_ETOM_CONDITION,
    "method_version": PE_FIN_REPLAN_ETOM_METHOD_VERSION,
    "protocol_version": PE_FIN_REPLAN_ETOM_PROTOCOL_VERSION,
    "architecture": PE_FIN_REPLAN_ETOM_ARCHITECTURE,
    "base_method": PE_FIN_REPLAN_ETOM_BASE_METHOD,
    "transfer_source": PE_FIN_REPLAN_ETOM_TRANSFER_SOURCE,
    "posthoc_exploratory": True,
    "formal_primary_result": False,
    "cross_benchmark_transfer_evaluation": True,
}

DEFAULT_BUDGETS: Dict[str, Any] = {
    "max_total_llm_calls_per_task": 16,
    "max_total_mcp_invocations": 8,
    "max_execution_rounds": 7,
    "max_executed_steps": 7,
    "max_replan_rounds": 7,
    "max_new_steps_per_replan": 2,
    "max_plan_steps": 6,
    "max_local_fallback_per_retrieve": 1,
    "retrieval_top_k": 5,
    "describe_top_n": 3,
}

LLM_BUDGET_RATIONALE = {
    "initial_planner": 1,
    "max_executor_internal_llm": 7,
    "max_replanner": 7,
    "repair_slack": 1,
    "hard_ceiling": 16,
    "note": "transport retries and repairs also count toward the hard ceiling",
}


class FrozenIdentityError(RuntimeError):
    pass


def _sha(obj: Any) -> str:
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_budgets(budgets: Dict[str, Any]) -> None:
    for key in (
        "max_total_llm_calls_per_task",
        "max_total_mcp_invocations",
        "max_execution_rounds",
        "max_executed_steps",
        "max_replan_rounds",
        "retrieval_top_k",
        "describe_top_n",
    ):
        if int(budgets.get(key)) != int(DEFAULT_BUDGETS[key]):
            raise FrozenIdentityError(
                f"budget {key}={budgets.get(key)!r} != frozen {DEFAULT_BUDGETS[key]!r}"
            )


def assert_frozen_lock_payload(lock: Dict[str, Any]) -> None:
    """Assert a parsed lock JSON matches the hard-coded frozen identity."""
    checks = [
        ("sampler_version", lock.get("sampler_version"), FROZEN_SAMPLER),
        ("seed", lock.get("seed"), FROZEN_SEED),
        ("sample_sha256", lock.get("sample_sha256"), FROZEN_SAMPLE_SHA256),
        ("ordered_task_ids_sha256", lock.get("ordered_task_ids_sha256"), FROZEN_ORDERED_TASK_IDS_SHA256),
        ("source_commit", lock.get("source_commit"), FROZEN_SOURCE_COMMIT),
        ("source_pin_sha256", lock.get("source_pin_sha256"), FROZEN_SOURCE_PIN_SHA256),
        ("sample_size", lock.get("sample_size"), FROZEN_TASK_COUNT),
    ]
    for name, got, want in checks:
        if got != want:
            raise FrozenIdentityError(f"lock {name}={got!r} != frozen {want!r}")
    ids = list(lock.get("ordered_task_ids") or [])
    if len(ids) != FROZEN_TASK_COUNT or len(set(ids)) != FROZEN_TASK_COUNT:
        raise FrozenIdentityError("lock must contain exactly 100 unique ordered task ids")
    blob = "\n".join(ids).encode("utf-8")
    if hashlib.sha256(blob).hexdigest() != FROZEN_ORDERED_TASK_IDS_SHA256:
        raise FrozenIdentityError("ordered_task_ids do not hash to the frozen sha256")
    per_level = {level: 0 for level in range(1, 6)}
    for tid in ids:
        per_level[int(str(tid).split("-")[0][1:])] += 1
    if any(per_level[level] != FROZEN_PER_LEVEL for level in per_level):
        raise FrozenIdentityError(f"lock must have 20 tasks per level, got {per_level}")
    quotas = lock.get("quotas") or {}
    if {k: int(v) for k, v in quotas.items()} != {f"L{n}": FROZEN_PER_LEVEL for n in range(1, 6)}:
        raise FrozenIdentityError(f"lock quotas drifted: {quotas}")
    overlap = set(lock.get("excluded_task_ids") or []) & set(ids)
    if overlap:
        raise FrozenIdentityError(f"lock overlaps cost-pilot ids: {sorted(overlap)}")


def assert_frozen_identity(root: Path) -> Dict[str, Any]:
    """Read the lock + pin from disk and assert every frozen hash. Returns the lock."""
    root = Path(root)
    lock_path = root / FROZEN_LOCK_REL
    pin_path = root / FROZEN_PIN_REL
    if file_sha256(lock_path) != FROZEN_LOCK_FILE_SHA256:
        raise FrozenIdentityError("formal_100_v1.lock.json file sha256 drifted")
    if file_sha256(pin_path) != FROZEN_SOURCE_PIN_SHA256:
        raise FrozenIdentityError("etom_source_pin_v1.json file sha256 drifted")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    assert_frozen_lock_payload(lock)
    pin = json.loads(pin_path.read_text(encoding="utf-8"))
    if pin.get("commit_sha") != FROZEN_SOURCE_COMMIT:
        raise FrozenIdentityError("pin commit_sha drifted")
    if (pin.get("dataset_files") or {}).get("data/mcp_registry.json", {}).get("sha256") != FROZEN_REGISTRY_SHA256:
        raise FrozenIdentityError("pin registry sha256 drifted")
    if int((pin.get("official_counts") or {}).get("tools") or 0) != FROZEN_CATALOG_TOTAL_TOOLS:
        raise FrozenIdentityError("pin tool count drifted")
    return lock


def frozen_dataset_identity() -> Dict[str, Any]:
    return {
        "sampler": FROZEN_SAMPLER,
        "seed": FROZEN_SEED,
        "sample_sha256": FROZEN_SAMPLE_SHA256,
        "ordered_task_ids_sha256": FROZEN_ORDERED_TASK_IDS_SHA256,
        "lock_file_sha256": FROZEN_LOCK_FILE_SHA256,
        "source_pin_sha256": FROZEN_SOURCE_PIN_SHA256,
        "source_commit": FROZEN_SOURCE_COMMIT,
        "registry_sha256": FROZEN_REGISTRY_SHA256,
        "catalog_total_tools": FROZEN_CATALOG_TOTAL_TOOLS,
        "tasks": FROZEN_TASK_COUNT,
        "per_level": FROZEN_PER_LEVEL,
        "evaluator": "hk_equity_mcp.etom.evaluator",
        "evaluation": "routing_only_graph_em_precision_recall_f1",
    }


def budget_identity_hash(budgets: Optional[Dict[str, Any]] = None) -> str:
    return _sha(budgets if budgets is not None else DEFAULT_BUDGETS)


def protocol_identity() -> Dict[str, Any]:
    """Full method/protocol identity including prompt/schema/grounding hashes."""
    from hk_equity_mcp.pe_fin_lite.retrieval_formula import (
        RETRIEVAL_FORMULA_NAME,
        retrieval_formula_hash,
    )
    from hk_equity_mcp.pe_fin_replan.failed_invocation import (
        FAILED_INVOCATION_POLICY_VERSION,
        failed_invocation_policy_hash,
    )
    from hk_equity_mcp.pe_fin_replan_etom import schemas as S
    from hk_equity_mcp.pe_fin_replan_etom.envelope import envelope_hash
    from hk_equity_mcp.pe_fin_replan_etom.grounding import (
        GROUNDING_CONTRACT_VERSION,
        grounding_contract_hash,
    )
    from hk_equity_mcp.pe_fin_replan_etom.prompts import (
        pe_fin_replan_etom_prompt_hash,
        prompt_hashes,
        replanner_prompt_hash,
    )
    from hk_equity_mcp.pe_fin_replan_etom.tool_binding import binder_protocol_hash

    return {
        **METHOD_IDENTITY,
        "flags": dict(ETOM_FLAGS),
        "budgets": dict(DEFAULT_BUDGETS),
        "budget_identity_hash": budget_identity_hash(),
        "llm_budget_rationale": dict(LLM_BUDGET_RATIONALE),
        "embedding_cache_namespace": EMBEDDING_CACHE_NAMESPACE,
        "retrieval_formula_name": RETRIEVAL_FORMULA_NAME,
        "retrieval_formula_hash": retrieval_formula_hash(),
        "retrieval": {"top_k": 5, "describe_n": 3, "domain_hard_filter": False, "filter": "safety_only"},
        "prompt_hash": pe_fin_replan_etom_prompt_hash(),
        "prompt_hashes": prompt_hashes(),
        "replanner_prompt_hash": replanner_prompt_hash(),
        "schema_set_hash": S.schema_set_hash(),
        "replanner_schema_version": S.REPLAN_SCHEMA_VERSION,
        "replanner_schema_hash": S.replanner_schema_hash(),
        "loop_policy_hash": S.loop_policy_hash(),
        "semantic_plan_repair_policy_hash": S.semantic_plan_repair_policy_hash(),
        "step_canonicalizer_hash": S.step_canonicalizer_hash(),
        "grounding_contract_version": GROUNDING_CONTRACT_VERSION,
        "grounding_contract_hash": grounding_contract_hash(),
        "failed_invocation_policy_version": FAILED_INVOCATION_POLICY_VERSION,
        "failed_invocation_policy_hash": failed_invocation_policy_hash(),
        "binder_protocol_hash": binder_protocol_hash(),
        "envelope_hash": envelope_hash(),
        "frozen_dataset": frozen_dataset_identity(),
    }


RUN_ROLE_FORMAL100 = "formal100"
RUN_ROLE_ENGINEERING_SMOKE5 = "engineering_live_path_smoke_only"
RUN_ROLE_MOCK = "mock"
RUN_ROLE_DRY_RUN = "dry-run"
RUN_ROLE_INDEX = "index"

# Frozen-order first task per level (L1–L5). Derived ONLY from lock task_id
# prefixes + ordered_task_ids; never from gold / accepted tools / correctness.
ENGINEERING_SMOKE5_EXPECTED_IDS = (
    "L1-q556",
    "L2-q766",
    "L3-q108",
    "L4-q57",
    "L5-q68",
)


def engineering_smoke5_task_ids(ordered_task_ids: List[str]) -> List[str]:
    """Select the first frozen-order task of each level L1–L5.

    Uses only ``task_id`` strings (level prefix). Does not read questions,
    gold, accepted tools, or baseline scores.
    """
    chosen: Dict[str, str] = {}
    for tid in ordered_task_ids:
        level = str(tid).split("-", 1)[0]
        if level in {f"L{n}" for n in range(1, 6)} and level not in chosen:
            chosen[level] = str(tid)
        if len(chosen) == 5:
            break
    ids = [chosen[f"L{n}"] for n in range(1, 6) if f"L{n}" in chosen]
    if len(ids) != 5:
        raise FrozenIdentityError(
            f"engineering Smoke-5 requires one task per L1–L5; got {ids!r}"
        )
    if tuple(ids) != ENGINEERING_SMOKE5_EXPECTED_IDS:
        raise FrozenIdentityError(
            "engineering Smoke-5 ids drifted from the frozen first-per-level pin: "
            f"got {ids!r} want {list(ENGINEERING_SMOKE5_EXPECTED_IDS)!r}"
        )
    return ids


RESUME_IDENTITY_KEYS = (
    "condition",
    "method_version",
    "protocol_version",
    "dataset_sample_sha256",
    "dataset_manifest_sha256",
    "ordered_task_ids_sha256",
    "capability_profile_hash",
    "prompt_hash",
    "schema_set_hash",
    "replanner_schema_hash",
    "grounding_contract_hash",
    "cli_output_contract_version",
    "catalog_registry_sha256",
    "operation_documents_sha256",
    "operation_index_identity",
    "embedding_cache_namespace",
    "generation_provider_profile_id",
    "generation_provider_profile_hash",
    "generation_provider",
    "generation_model",
    "generation_host",
    "embedding_model",
    "embedding_host",
    "budgets",
    "evaluator_identity",
    "run_mode",
    "run_role",
    "git_commit",
    "git_dirty",
    "fixture_run",
)


def resume_identity_hash(metadata: Dict[str, Any]) -> str:
    """Hash of every field a resumed run must keep identical.

    ``max_tasks`` is intentionally excluded for formal100 prefix extension.
    ``run_role`` IS included so Smoke-5 can never resume into formal100 (or
    vice versa), even if an operator reuses an output directory by mistake.
    """
    payload = {key: metadata.get(key) for key in RESUME_IDENTITY_KEYS}
    return _sha(payload)


def evaluator_identity() -> Dict[str, Any]:
    return {
        "module": "hk_equity_mcp.etom.evaluator.evaluate_prediction",
        "metrics": ["exact_match", "precision", "recall", "f1"],
        "scope": "routing_only",
        "formal_metrics": "hk_equity_mcp.etom.formal_metrics",
    }
