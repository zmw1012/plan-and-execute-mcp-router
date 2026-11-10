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
    QUERY_CONSTRUCTION_VERSION,
    TOP_K,
)
from hk_equity_mcp.experiment.pe_fin_lite_retrieval_audit.identity import stable_json_hash
from hk_equity_mcp.experiment.pe_fin_lite_retrieval_audit.metrics import aggregate_metric_rows


VARIANT_COMPLEXITY = {
    "R0": 0,
    "R1": 1,
    "R2": 2,
    "R3": 3,
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


def _selection_key(agg: Dict[str, Any], *, variant: str, alpha: Optional[float]) -> tuple:
    """Priority: complete-route@3, macro Recall@3, min-domain Recall@3, MRR,
    then simpler variant and larger alpha.
    """
    return (
        float(agg.get("complete_route@3") or 0.0),
        float(agg.get("recall@3") or 0.0),
        float(agg.get("min_domain_recall@3") or 0.0),
        float(agg.get("mrr") or 0.0),
        # Prefer simpler formula / larger alpha when approximately tied:
        # invert complexity and use alpha directly in sort via secondary tuple
        # handled by caller with reverse on first four only.
        -VARIANT_COMPLEXITY.get(variant, 99),
        float(alpha if alpha is not None else 1.0),
    )


def compare_formulas(
    *,
    development_metrics_by_formula: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """``development_metrics_by_formula`` keys like ``R2`` or ``R3@0.8``."""
    ranked = []
    for key, agg in development_metrics_by_formula.items():
        if "@" in key:
            variant, alpha_s = key.split("@", 1)
            alpha = float(alpha_s)
        else:
            variant, alpha = key, None
        ranked.append(
            {
                "formula_key": key,
                "variant": variant,
                "alpha": alpha,
                "metrics": agg,
                "selection_tuple": list(_selection_key(agg, variant=variant, alpha=alpha)),
            }
        )
    ranked.sort(key=lambda r: tuple(r["selection_tuple"]), reverse=True)
    return {
        "ranking": ranked,
        "selected": ranked[0] if ranked else None,
        "selection_policy": [
            "complete_route@3",
            "macro_recall@3",
            "minimum_per_domain_recall@3",
            "mrr",
            "simpler_formula",
            "larger_alpha",
        ],
        "development_only": True,
    }


def build_formula_lock(
    *,
    selected: Dict[str, Any],
    development_metrics: Dict[str, Any],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    lock = {
        "lock_version": "retrieval_formula_lock_v1",
        "selected_variant": selected.get("variant"),
        "alpha": selected.get("alpha"),
        "formula_key": selected.get("formula_key"),
        "query_construction_version": QUERY_CONSTRUCTION_VERSION,
        "operation_document_version": OPERATION_DOCUMENT_VERSION,
        "filter_version": DOMAIN_FILTER_VERSION,
        "top_k": TOP_K,
        "describe_n": DESCRIBE_N,
        "development_metrics": development_metrics,
        "config": config,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "exploratory_descriptive": True,
        "no_retune_on_reserved85": True,
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
