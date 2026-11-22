"""Formula selection on development-70 only; frozen evaluation on reserved-85."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from hk_equity_mcp.experiment.pe_fin_lite_retrieval_audit import (
    DESCRIBE_N,
    DOMAIN_FILTER_VERSION,
    HYBRID_ALPHAS,
    OPERATION_DOCUMENT_VERSION,
    OUTPUT_COVERAGE_VERSION,
    PLAN_FIRST_SELECTABLE_VARIANTS,
    QUERY_CONSTRUCTION_VERSION,
    RUNTIME_LEXICAL_WEIGHT,
    TOP_K,
)
from hk_equity_mcp.experiment.pe_fin_lite_retrieval_audit.identity import stable_json_hash
from hk_equity_mcp.experiment.pe_fin_lite_retrieval_audit.metrics import aggregate_metric_rows


VARIANT_COMPLEXITY = {
    "R0": 0,  # baseline only — never selected for Plan-first
    "R1": 1,
    "R_runtime": 2,
    "R2": 3,
    "R3": 4,
}

REAL_EMBEDDING_BACKEND = "real_text_embedding_3_small"


def assert_formal_formula_lock_allowed(
    *,
    embedding_backend: str,
    mock: bool,
) -> None:
    """Refuse formal lock from mock/hash rankings."""
    if mock:
        raise RuntimeError(
            "mock results cannot generate formal formula lock (fail-closed)"
        )
    if embedding_backend != REAL_EMBEDDING_BACKEND:
        raise RuntimeError(
            "Formal formula lock requires embedding_backend="
            f"{REAL_EMBEDDING_BACKEND!r} real cache; got {embedding_backend!r}"
        )


def parse_formula_key(key: str) -> tuple:
    if "@" in key:
        variant, alpha_s = key.split("@", 1)
        return variant, float(alpha_s)
    return key, None


def filter_plan_first_selection_inputs(
    development_metrics_by_formula: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Drop R0 (question-level baseline) from Plan-first selection pool."""
    out: Dict[str, Dict[str, Any]] = {}
    for key, agg in development_metrics_by_formula.items():
        variant, _alpha = parse_formula_key(key)
        if variant == "R0":
            continue
        if variant not in PLAN_FIRST_SELECTABLE_VARIANTS:
            continue
        out[key] = agg
    return out


def _selection_key(agg: Dict[str, Any], *, variant: str, alpha: Optional[float]) -> tuple:
    """Priority: complete-route@3, macro Recall@3, min-domain Recall@3, MRR,
    then simpler variant and larger alpha (deterministic).
    """
    return (
        float(agg.get("complete_route@3") or 0.0),
        float(agg.get("recall@3") or 0.0),
        float(agg.get("min_domain_recall@3") or 0.0),
        float(agg.get("mrr") or 0.0),
        -VARIANT_COMPLEXITY.get(variant, 99),
        float(alpha if alpha is not None else 1.0),
    )


def compare_formulas(
    *,
    development_metrics_by_formula: Dict[str, Dict[str, Any]],
    exclude_r0: bool = True,
) -> Dict[str, Any]:
    """``development_metrics_by_formula`` keys like ``R2``, ``R3@0.8``, ``R_runtime``."""
    pool = (
        filter_plan_first_selection_inputs(development_metrics_by_formula)
        if exclude_r0
        else dict(development_metrics_by_formula)
    )
    ranked = []
    for key, agg in pool.items():
        variant, alpha = parse_formula_key(key)
        ranked.append(
            {
                "formula_key": key,
                "variant": variant,
                "alpha": alpha,
                "metrics": agg,
                "selection_tuple": list(
                    _selection_key(agg, variant=variant, alpha=alpha)
                ),
            }
        )
    ranked.sort(key=lambda r: tuple(r["selection_tuple"]), reverse=True)
    selected = ranked[0] if ranked else None
    runtime_sync_required = None
    runtime_sync_spec = None
    if selected:
        v = selected["variant"]
        if v == "R_runtime":
            runtime_sync_required = False
            runtime_sync_spec = {
                "note": "Selected formula already mirrors PE-Fin Lite runtime retrieval.",
                "lexical_weight": RUNTIME_LEXICAL_WEIGHT,
                "filter_timing": "after_global_topk",
            }
        else:
            runtime_sync_required = True
            runtime_sync_spec = {
                "note": (
                    "Selected Plan-first formula differs from current runtime "
                    "(cosine+0.12*lexical, Top-K then hard filter). Do not modify "
                    "runtime in this experiment round; sync separately after confirmation."
                ),
                "selected_variant": v,
                "alpha": selected.get("alpha"),
                "minimal_sync": {
                    "query_construction": QUERY_CONSTRUCTION_VERSION,
                    "replace_runtime_score_with": (
                        "cosine only"
                        if v in {"R1", "R2"}
                        else f"alpha*cosine+(1-alpha)*coverage alpha={selected.get('alpha')}"
                    ),
                    "domain_filter_timing": (
                        "before_scoring" if v in {"R2", "R3"} else "none"
                    ),
                    "drop_lexical_term": True,
                    "keep_top_k": TOP_K,
                    "keep_describe_n": DESCRIBE_N,
                },
            }
    return {
        "ranking": ranked,
        "selected": selected,
        "selection_policy": [
            "complete_route@3",
            "macro_recall@3",
            "minimum_per_domain_recall@3",
            "mrr",
            "simpler_formula",
            "larger_alpha",
        ],
        "development_only": True,
        "r0_excluded_from_plan_first_selection": True,
        "selectable_variants": list(PLAN_FIRST_SELECTABLE_VARIANTS),
        "excluded_from_inputs": sorted(
            set(development_metrics_by_formula) - set(pool)
        ),
        "runtime_sync_required": runtime_sync_required,
        "runtime_sync_specification": runtime_sync_spec,
    }


def build_formula_lock(
    *,
    selected: Dict[str, Any],
    development_metrics: Dict[str, Any],
    config: Dict[str, Any],
    runtime_sync_required: bool,
    runtime_sync_specification: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    variant = selected.get("variant")
    lexical_weight = (
        RUNTIME_LEXICAL_WEIGHT if variant == "R_runtime" else None
    )
    lock = {
        "lock_version": "retrieval_formula_lock_v2",
        "selected_variant": variant,
        "alpha": selected.get("alpha"),
        "formula_key": selected.get("formula_key"),
        "lexical_weight": lexical_weight,
        "query_construction_version": QUERY_CONSTRUCTION_VERSION,
        "operation_document_version": OPERATION_DOCUMENT_VERSION,
        "operation_documents_sha256": config.get("operation_documents_sha256"),
        "domain_filter_version": DOMAIN_FILTER_VERSION,
        "filter_version": DOMAIN_FILTER_VERSION,
        "output_coverage_version": OUTPUT_COVERAGE_VERSION,
        "embedding_provider_host": config.get("embedding_provider_host"),
        "embedding_model": config.get("embedding_model"),
        "embedding_dimension": config.get("embedding_dimension"),
        "catalog_fingerprint": config.get("catalog_fingerprint"),
        "operation_index_identity": config.get("operation_index_identity"),
        "planner_cache_sha256": config.get("planner_cache_sha256"),
        "step_query_cache_sha256": config.get("step_query_cache_sha256"),
        "top_k": TOP_K,
        "describe_n": DESCRIBE_N,
        "development_metrics": development_metrics,
        "selection_policy": [
            "complete_route@3",
            "macro_recall@3",
            "minimum_per_domain_recall@3",
            "mrr",
            "simpler_formula",
            "larger_alpha",
        ],
        "runtime_sync_required": bool(runtime_sync_required),
        "runtime_sync_specification": runtime_sync_specification,
        "config": config,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "exploratory_descriptive": True,
        "result_role": "exploratory_descriptive",
        "no_retune_on_reserved85": True,
        "no_unseen_heldout_claim": True,
        "r0_excluded_from_selection": True,
    }
    lock["lock_sha256"] = stable_json_hash(
        {k: v for k, v in lock.items() if k != "lock_sha256"}
    )
    return lock


def assert_frozen_evaluation_allowed(
    *,
    lock: Dict[str, Any],
    requested_variant: str,
    requested_alpha: Optional[float],
) -> None:
    if lock.get("selected_variant") != requested_variant:
        raise RuntimeError(
            "Frozen formula evaluation cannot retune variant on reserved-85 "
            f"(locked={lock.get('selected_variant')} requested={requested_variant})"
        )
    locked_alpha = lock.get("alpha")
    if requested_variant == "R3":
        if locked_alpha is None or abs(float(locked_alpha) - float(requested_alpha or -1)) > 1e-12:
            raise RuntimeError(
                "Frozen formula evaluation cannot retune alpha on reserved-85 "
                f"(locked={locked_alpha} requested={requested_alpha})"
            )
    elif requested_alpha not in (None, 1.0):
        raise RuntimeError("Non-R3 frozen evaluation must not pass a tuning alpha")


def write_formula_lock(path: Path, lock: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(lock, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_formula_lock(path: Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def attach_min_domain_recall(
    agg: Dict[str, Any],
    *,
    per_domain_recall_at_3: Dict[str, float],
) -> Dict[str, Any]:
    out = dict(agg)
    if per_domain_recall_at_3:
        out["min_domain_recall@3"] = min(per_domain_recall_at_3.values())
        out["per_domain_recall@3"] = dict(per_domain_recall_at_3)
    else:
        out["min_domain_recall@3"] = 0.0
        out["per_domain_recall@3"] = {}
    return out
