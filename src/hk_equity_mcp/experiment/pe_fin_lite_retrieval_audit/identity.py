"""Dataset / cache / index identities for Retrieval Audit (fail-closed)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from hk_equity_mcp.experiment.pe_fin_lite_retrieval_audit import (
    AUDIT_VERSION,
    DESCRIBE_N,
    DOMAIN_FILTER_VERSION,
    OPERATION_DOCUMENT_VERSION,
    PLANNER_ONLY_PROTOCOL_VERSION,
    QUERY_CONSTRUCTION_VERSION,
    RANKING_PROTOCOL_VERSION,
    RESULT_ROLE,
    TOP_K,
)
from hk_equity_mcp.experiment.runner import file_sha256
from hk_equity_mcp.experiment.unified_exploratory import (
    EXPLORATORY_MANIFEST,
    EXPLORATORY_RUN_ORDER,
    load_exploratory_manifest,
    load_exploratory_run_order,
    load_exploratory_tasks_in_locked_order,
)
from hk_equity_mcp.experiment.unified_runner import catalog_fingerprint
from hk_equity_mcp.pe_fin_lite.identity import combined_schema_hash
from hk_equity_mcp.pe_fin_lite.prompts import pe_fin_lite_prompt_hash
from hk_equity_mcp.real_mcp.unified_catalog import build_unified_catalog


def stable_json_hash(obj: Any) -> str:
    blob = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def question_identity_hash(tasks: Sequence[Dict[str, Any]]) -> str:
    rows = []
    for t in tasks:
        rows.append(
            {
                "task_id": t.get("task_id") or t.get("id"),
                "question": t.get("question"),
                "domain": t.get("domain"),
                "source_split": t.get("source_split"),
                "answerability_class": t.get("answerability_class"),
            }
        )
    return stable_json_hash(rows)


def ordered_task_ids_hash(task_ids: Sequence[str]) -> str:
    return stable_json_hash(list(task_ids))


def split_task_ids(tasks: Sequence[Dict[str, Any]]) -> Dict[str, List[str]]:
    dev = [
        str(t.get("task_id") or t.get("id"))
        for t in tasks
        if t.get("source_split") == "development" or t.get("previously_reserved") is False
    ]
    reserved = [
        str(t.get("task_id") or t.get("id"))
        for t in tasks
        if t.get("source_split") == "previously_reserved"
        or t.get("previously_reserved") is True
    ]
    all_ids = [str(t.get("task_id") or t.get("id")) for t in tasks]
    return {
        "development70": dev,
        "previously_reserved85": reserved,
        "all155": all_ids,
    }


def load_hk155_locked(root: Path) -> Dict[str, Any]:
    """Load frozen HK-155 exploratory tasks + identity hashes (read-only)."""
    root = Path(root)
    tasks = load_exploratory_tasks_in_locked_order(root)
    manifest = load_exploratory_manifest(root)
    run_order = load_exploratory_run_order(root)
    task_ids = [str(t.get("task_id") or t.get("id")) for t in tasks]
    splits = split_task_ids(tasks)
    if len(tasks) != 155:
        raise RuntimeError(f"HK-155 expected 155 tasks, got {len(tasks)}")
    if len(splits["development70"]) != 70:
        raise RuntimeError(
            f"development-origin expected 70, got {len(splits['development70'])}"
        )
    if len(splits["previously_reserved85"]) != 85:
        raise RuntimeError(
            f"previously-reserved-origin expected 85, got {len(splits['previously_reserved85'])}"
        )
    identities = {
        "audit_version": AUDIT_VERSION,
        "result_role": RESULT_ROLE,
        "benchmark_id": manifest.get("benchmark_id"),
        "manifest_path": EXPLORATORY_MANIFEST,
        "run_order_path": EXPLORATORY_RUN_ORDER,
        "manifest_sha256": file_sha256(root / EXPLORATORY_MANIFEST),
        "run_order_sha256": file_sha256(root / EXPLORATORY_RUN_ORDER),
        "run_order_hash": run_order.get("run_order_hash"),
        "ordered_task_ids_hash": ordered_task_ids_hash(task_ids),
        "question_identity_hash": question_identity_hash(tasks),
        "n_tasks": 155,
        "development70_count": 70,
        "previously_reserved85_count": 85,
        "domain_counts": manifest.get("domain_counts"),
        "source_split_counts": manifest.get("source_split_counts"),
        "not_for_unseen_final_claims": True,
        "no_final_test_claim": True,
        "notes": [
            "exploratory/descriptive only",
            "previously-reserved 85 are evaluation-only after formula lock",
            "do not claim unseen held-out",
        ],
    }
    return {
        "tasks": tasks,
        "manifest": manifest,
        "run_order": run_order,
        "task_ids": task_ids,
        "splits": splits,
        "identities": identities,
    }


def build_catalog_context(root: Path) -> Dict[str, Any]:
    bundle = build_unified_catalog(root=root)
    cat_fp = catalog_fingerprint(bundle.stats, bundle.source_catalogs)
    runtime_ids = list(bundle.stats.runtime_operation_ids)
    formal = [
        oid
        for oid, rec in bundle.records.items()
        if rec.formal_gold_eligible
    ]
    distractors = [
        oid
        for oid, rec in bundle.records.items()
        if rec.is_distractor and rec.runtime_exposed
    ]
    excluded = list(bundle.stats.blocked_operation_ids)
    return {
        "bundle": bundle,
        "catalog_fingerprint": cat_fp,
        "runtime_operation_ids": runtime_ids,
        "formal_gold_eligible_ids": sorted(formal),
        "distractor_ids": sorted(distractors),
        "excluded_ids": sorted(excluded),
        "stats": {
            "unique_operation_ids": bundle.stats.unique_operation_ids,
            "runtime_exposed_unique": bundle.stats.runtime_exposed_unique,
            "formal_gold_eligible_unique": bundle.stats.formal_gold_eligible_unique,
            "distractor_runtime_unique": bundle.stats.distractor_runtime_unique,
            "blocked_or_rejected_unique": bundle.stats.blocked_or_rejected_unique,
        },
    }


def question_level_cache_identity(
    *,
    dataset_identities: Dict[str, Any],
    operation_documents_sha256: str,
    catalog_fingerprint: str,
    embedding_model: str,
    embedding_provider_host: str,
    query_construction_version: str = QUERY_CONSTRUCTION_VERSION,
) -> Dict[str, Any]:
    payload = {
        "kind": "question_level_retrieval_cache",
        "manifest_sha256": dataset_identities.get("manifest_sha256"),
        "ordered_task_ids_hash": dataset_identities.get("ordered_task_ids_hash"),
        "question_identity_hash": dataset_identities.get("question_identity_hash"),
        "operation_documents_sha256": operation_documents_sha256,
        "catalog_fingerprint": catalog_fingerprint,
        "embedding_model": embedding_model,
        "embedding_provider_host": embedding_provider_host,
        "query_construction_version": query_construction_version,
        "top_k": TOP_K,
        "describe_n": DESCRIBE_N,
        "ranking_protocol_version": RANKING_PROTOCOL_VERSION,
    }
    return {**payload, "identity_sha256": stable_json_hash(payload)}


def planner_only_cache_identity(
    *,
    dataset_identities: Dict[str, Any],
    planner_prompt_hash: Optional[str] = None,
    planner_schema_hash: Optional[str] = None,
    capability_profile_hash: Optional[str] = None,
    plan_canonicalizer_hash: Optional[str] = None,
    scope_contract_hash: Optional[str] = None,
    generation_provider: str = "offline_mock",
    generation_host: str = "local",
    generation_model: str = "mock",
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    from hk_equity_mcp.pe_fin_lite.output_vocabulary import (
        scope_contract_hash as _scope_hash,
    )
    from hk_equity_mcp.pe_fin_lite.plan_canonicalizer import (
        plan_canonicalizer_hash as _canon_hash,
    )
    from hk_equity_mcp.pe_fin_lite.schemas import PLANNER_MAX_TOKENS

    planner_max_tokens = (
        int(max_tokens) if max_tokens is not None else int(PLANNER_MAX_TOKENS)
    )
    payload = {
        "kind": "planner_only_cache",
        "manifest_sha256": dataset_identities.get("manifest_sha256"),
        "ordered_task_ids_hash": dataset_identities.get("ordered_task_ids_hash"),
        "question_identity_hash": dataset_identities.get("question_identity_hash"),
        "planner_prompt_hash": planner_prompt_hash or pe_fin_lite_prompt_hash(),
        "planner_schema_hash": planner_schema_hash or combined_schema_hash(),
        "capability_profile_hash": capability_profile_hash,
        "plan_canonicalizer_hash": plan_canonicalizer_hash or _canon_hash(),
        "scope_contract_hash": scope_contract_hash or _scope_hash(),
        "generation_provider": generation_provider,
        "generation_host": generation_host,
        "generation_model": generation_model,
        "temperature": temperature,
        "max_tokens": planner_max_tokens,
        "planner_max_tokens": planner_max_tokens,
        "planner_only_protocol_version": PLANNER_ONLY_PROTOCOL_VERSION,
        "no_binder": True,
        "no_mcp": True,
        "no_embedding_in_planner_stage": True,
        "no_gold_in_planner": True,
    }
    return {**payload, "identity_sha256": stable_json_hash(payload)}


def step_retrieval_cache_identity(
    *,
    planner_cache_sha256: str,
    operation_documents_sha256: str,
    catalog_fingerprint: str,
    ranking_variant: str,
    alpha: Optional[float],
    domain_filter_version: str = DOMAIN_FILTER_VERSION,
    top_k: int = TOP_K,
    describe_n: int = DESCRIBE_N,
    embedding_model: str = "mock-hash-embedding-v1",
) -> Dict[str, Any]:
    payload = {
        "kind": "step_level_retrieval_cache",
        "planner_cache_sha256": planner_cache_sha256,
        "operation_documents_sha256": operation_documents_sha256,
        "catalog_fingerprint": catalog_fingerprint,
        "operation_document_version": OPERATION_DOCUMENT_VERSION,
        "ranking_variant": ranking_variant,
        "alpha": alpha,
        "domain_filter_version": domain_filter_version,
        "top_k": top_k,
        "describe_n": describe_n,
        "embedding_model": embedding_model,
        "ranking_protocol_version": RANKING_PROTOCOL_VERSION,
        "query_construction_version": QUERY_CONSTRUCTION_VERSION,
    }
    return {**payload, "identity_sha256": stable_json_hash(payload)}


def assert_identity_match(expected: Dict[str, Any], observed: Dict[str, Any], *, label: str) -> None:
    exp = expected.get("identity_sha256")
    obs = observed.get("identity_sha256")
    if exp != obs:
        raise RuntimeError(
            f"Identity mismatch for {label}: expected={exp} observed={obs}"
        )
