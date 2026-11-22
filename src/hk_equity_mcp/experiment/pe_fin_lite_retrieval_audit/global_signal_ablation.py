"""Zero-network R4_global_coverage / R5_global_lexical ablations on frozen caches."""

from __future__ import annotations

import hashlib
import json
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from hk_equity_mcp.experiment.pe_fin_lite_retrieval_audit import (
    DESCRIBE_N,
    HYBRID_ALPHAS,
    OPERATION_DOCUMENT_VERSION,
    QUERY_CONSTRUCTION_VERSION,
    RANKING_PROTOCOL_VERSION,
    RUNTIME_LEXICAL_WEIGHT,
    TOP_K,
)
from hk_equity_mcp.experiment.pe_fin_lite_retrieval_audit.embedding_identity_gate import (
    assert_query_catalog_alignment,
    assert_real_backend_identity,
)
from hk_equity_mcp.experiment.pe_fin_lite_retrieval_audit.formula_selection import (
    REAL_EMBEDDING_BACKEND,
    attach_min_domain_recall,
)
from hk_equity_mcp.experiment.pe_fin_lite_retrieval_audit.gold_reference import (
    build_evaluation_reference,
)
from hk_equity_mcp.experiment.pe_fin_lite_retrieval_audit.identity import (
    build_catalog_context,
    load_hk155_locked,
    stable_json_hash,
)
from hk_equity_mcp.experiment.pe_fin_lite_retrieval_audit.live_path import LivePathError
from hk_equity_mcp.experiment.pe_fin_lite_retrieval_audit.metrics import (
    aggregate_metric_rows,
    score_multi_step_route,
)
from hk_equity_mcp.experiment.pe_fin_lite_retrieval_audit.operation_documents import (
    build_all_operation_documents,
)
from hk_equity_mcp.experiment.pe_fin_lite_retrieval_audit.planner_cache_gate import (
    assert_planner_cache_for_step_pipeline,
)
from hk_equity_mcp.experiment.pe_fin_lite_retrieval_audit.query_cache import (
    load_complete_step_query_cache,
)
from hk_equity_mcp.experiment.pe_fin_lite_retrieval_audit.ranking import (
    OfflineEmbeddingIndex,
    build_step_query,
    rank_operations,
)
from hk_equity_mcp.experiment.pe_fin_lite_retrieval_audit.catalog_audit import (
    analyze_catalog_coverage,
)
from hk_equity_mcp.operation_resolver.retrieval import LEXICAL_WEIGHT
from hk_equity_mcp.pe_fin_lite.identity import EXPECTED_EMBEDDING_MODEL
from hk_equity_mcp.pe_fin_lite.operation_embedding_index import (
    identity_digest,
    load_complete_index,
    build_index_identity,
)

ABLATION_VERSION = "pe_fin_lite_global_signal_ablation_v1"
EXPECTED_PLANNER_SHA = (
    "e16eed7e1c1e832ed2fa3991cea05b8bd424af37489ab6915353f8fe7b599f84"
)
EXPECTED_STEP_IDENTITY = (
    "cf915cfc2755672a8d0ccd5681f5ea4332ed2e57cf51fdaeb7d020c4b7b87da7"
)
EXPECTED_CATALOG_IDENTITY = (
    "ac4c109bed0edb21fb2b837dfdbd745680b24f33efa62e13ad6bca53f174b0df"
)
EXPECTED_OP_DOCS = (
    "5989196c1c2765cfe5ec6c094c80189cb9b3b18ed9cc2072ab057b29d3f0dd27"
)
EXPECTED_RUNTIME_OPS = 39
EXPECTED_RETRIEVE_STEPS = 240

ABLATION_FORMULAS = (
    "R1",
    "R4_global_coverage@1.0",
    "R4_global_coverage@0.9",
    "R4_global_coverage@0.8",
    "R4_global_coverage@0.7",
    "R5_global_lexical@0.12",
)

ABLATION_COMPLEXITY = {
    "R1": 1,
    "R5_global_lexical": 2,
    "R4_global_coverage": 3,
}


def _file_sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _metadata_by_id(bundle: Any) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for oid, rec in bundle.records.items():
        raw = rec.source_raw or {}
        out[oid] = {
            "domain": rec.preferred_domain
            or (rec.domains[0] if rec.domains else "general"),
            "provides": list(raw.get("provides") or []),
            "capabilities": list(raw.get("capabilities") or []),
            "availability": "blocked"
            if rec.is_blocked
            else ("runtime" if rec.runtime_exposed else "hidden"),
            "runtime_role": "blocked"
            if rec.is_blocked
            else ("distractor" if rec.is_distractor else "runtime"),
            "blocked": rec.is_blocked,
            "rejected": rec.is_blocked,
            "formal_gold_eligible": rec.formal_gold_eligible,
            "is_distractor": rec.is_distractor,
        }
    return out


def _parse_ablation_key(key: str) -> Tuple[str, Optional[float], Optional[float]]:
    if key == "R1":
        return "R1", None, None
    if key.startswith("R4_global_coverage@"):
        return "R4", float(key.split("@", 1)[1]), None
    if key.startswith("R5_global_lexical@"):
        return "R5", None, float(key.split("@", 1)[1])
    raise ValueError(f"Unknown ablation key: {key}")


def _selection_key(agg: Dict[str, Any], *, formula_key: str) -> tuple:
    variant, alpha, _lw = _parse_ablation_key(formula_key)
    family = (
        "R4_global_coverage"
        if variant == "R4"
        else ("R5_global_lexical" if variant == "R5" else "R1")
    )
    return (
        float(agg.get("complete_route@3") or 0.0),
        float(agg.get("recall@3") or 0.0),
        float(agg.get("min_domain_recall@3") or 0.0),
        float(agg.get("mrr") or 0.0),
        -ABLATION_COMPLEXITY.get(family, 99),
        float(alpha if alpha is not None else 1.0),
    )


def compare_ablation_formulas(
    development_metrics_by_formula: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    ranked = []
    for key, agg in development_metrics_by_formula.items():
        if key not in ABLATION_FORMULAS:
            continue
        ranked.append(
            {
                "formula_key": key,
                "metrics": agg,
                "selection_tuple": list(_selection_key(agg, formula_key=key)),
            }
        )
    ranked.sort(key=lambda r: tuple(r["selection_tuple"]), reverse=True)
    selected = ranked[0] if ranked else None
    exact = None
    if selected:
        v, a, lw = _parse_ablation_key(selected["formula_key"])
        if v == "R1":
            exact = "cosine_similarity"
        elif v == "R4":
            exact = f"{a}*cosine_similarity + {1-a}*required_output_coverage"
        else:
            exact = f"cosine_similarity + {lw}*lexical_score"
    return {
        "ranking": ranked,
        "selected": selected,
        "selection_policy": [
            "complete_route@3",
            "macro_recall@3",
            "minimum_per_domain_recall@3",
            "mrr",
            "simpler_formula",
            "larger_alpha_for_R4",
        ],
        "development_only": True,
        "exact_formula": exact,
    }


def _top_set(ids: Sequence[str], k: int) -> Set[str]:
    return {str(x) for x in list(ids)[:k]}


def _analyze_reorder(
    *,
    baseline_full: Sequence[Dict[str, Any]],
    other_full: Sequence[Dict[str, Any]],
    gold: Sequence[str],
) -> Dict[str, Any]:
    b_ids = [r["operation_id"] for r in baseline_full]
    o_ids = [r["operation_id"] for r in other_full]
    b1 = b_ids[:1]
    o1 = o_ids[:1]
    b3 = _top_set(b_ids, 3)
    o3 = _top_set(o_ids, 3)
    gold_set = {str(x) for x in gold}
    cov_vals = [
        float(r.get("required_output_coverage"))
        for r in other_full
        if r.get("required_output_coverage") is not None
    ]
    constant_cov = bool(cov_vals) and max(cov_vals) - min(cov_vals) < 1e-12
    return {
        "changed_top1": b1 != o1,
        "changed_top3_set": b3 != o3,
        "rescued_gold_into_top3": bool(gold_set & o3) and not bool(gold_set & b3),
        "displaced_gold_from_top3": bool(gold_set & b3) and not bool(gold_set & o3),
        "unchanged_due_to_constant_coverage": constant_cov and b_ids == o_ids,
        "missing_or_empty_provides": sum(
            1 for r in other_full if r.get("provides_empty")
        ),
    }


def assert_r4_alpha1_matches_r1(
    *,
    r1_full: Sequence[Dict[str, Any]],
    r4_full: Sequence[Dict[str, Any]],
    context: str,
) -> None:
    r1_ids = [r["operation_id"] for r in r1_full]
    r4_ids = [r["operation_id"] for r in r4_full]
    if r1_ids != r4_ids:
        raise LivePathError(
            f"R4@1.0 full order mismatch vs R1 at {context}: "
            f"r1_top5={r1_ids[:5]} r4_top5={r4_ids[:5]}"
        )
    for a, b in zip(r1_full, r4_full):
        if abs(float(a["cosine_similarity"]) - float(b["cosine_similarity"])) > 1e-12:
            raise LivePathError(f"R4@1.0 cosine mismatch at {context}/{a['operation_id']}")
        if abs(float(a["score"]) - float(b["score"])) > 1e-9:
            raise LivePathError(f"R4@1.0 score mismatch at {context}/{a['operation_id']}")


def run_global_signal_ablation(
    *,
    root: Path,
    output_dir: Path,
    embedding_cache_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run R1/R4/R5 zero-network ablations into ``global_signal_ablation_v1/``."""
    root = Path(root)
    out_root = Path(output_dir)
    abl_dir = out_root / "global_signal_ablation_v1"
    abl_dir.mkdir(parents=True, exist_ok=True)

    # ---- freeze input hashes (before) ----
    watched = [
        "planner_cache/planner_outputs.jsonl",
        "planner_cache/planner_only_summary.json",
        "planner_cache_audit/planner_cache_audit.json",
        "step_query_cache/step_query_vectors.json",
        "step_query_cache/step_query_cache_identity.json",
        "catalog_index/index_status.json",
        "question_level/retrieval_metrics.json",
        "question_level/per_query_rankings.jsonl",
        "ranking_variants/per_step_rankings.jsonl",
        "ranking_variants/formula_comparison.json",
        "ranking_variants/retrieval_metrics.json",
        "formula_selection/formula_lock.json",
    ]
    raw_before = {}
    for rel in watched:
        p = out_root / rel
        if p.exists():
            raw_before[rel] = {"sha256": _file_sha(p), "bytes": p.stat().st_size}

    planner_path = out_root / "planner_cache" / "planner_outputs.jsonl"
    planner_sha = _file_sha(planner_path)
    if planner_sha != EXPECTED_PLANNER_SHA:
        raise LivePathError(
            f"Planner cache SHA mismatch: expected={EXPECTED_PLANNER_SHA} got={planner_sha}"
        )

    planner_rows = []
    for line in planner_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            planner_rows.append(json.loads(line))
    assert_planner_cache_for_step_pipeline(
        rows=planner_rows, allow_incomplete=False, formal=True, stage="global_signal_ablation"
    )
    n_retrieve = sum(len(r.get("retrieve_steps") or []) for r in planner_rows)
    if n_retrieve != EXPECTED_RETRIEVE_STEPS:
        raise LivePathError(
            f"retrieve steps={n_retrieve} != expected {EXPECTED_RETRIEVE_STEPS}"
        )

    step_ident_path = out_root / "step_query_cache" / "step_query_cache_identity.json"
    step_ident = json.loads(step_ident_path.read_text(encoding="utf-8"))
    if step_ident.get("identity_sha256") != EXPECTED_STEP_IDENTITY:
        raise LivePathError(
            f"step-query identity mismatch: expected={EXPECTED_STEP_IDENTITY} "
            f"got={step_ident.get('identity_sha256')}"
        )
    if step_ident.get("planner_cache_sha256") != EXPECTED_PLANNER_SHA:
        raise LivePathError("step-query identity planner_cache_sha256 mismatch")
    if step_ident.get("embedding_model") != EXPECTED_EMBEDDING_MODEL:
        raise LivePathError("step-query embedding model mismatch")
    if step_ident.get("embedding_provider_host") != "api.zhizengzeng.com":
        raise LivePathError("step-query embedding host mismatch")
    if int(step_ident.get("embedding_dim") or 0) != 1536:
        raise LivePathError("step-query embedding dim mismatch")

    scache = load_complete_step_query_cache(
        out_root / "step_query_cache" / "step_query_vectors.json"
    )
    s_ident = dict(scache.get("identity") or {})
    # Stale stored digest inside vectors.json is tolerated iff core fields match
    # and recomputed digest equals the authoritative identity file.
    core = {k: v for k, v in s_ident.items() if k != "identity_sha256"}
    recomputed = stable_json_hash(core)
    if recomputed != EXPECTED_STEP_IDENTITY:
        raise LivePathError(
            f"step vectors identity recompute mismatch: {recomputed}"
        )
    stale_vector_digest = s_ident.get("identity_sha256")
    if stale_vector_digest not in {EXPECTED_STEP_IDENTITY, None}:
        # Record only; fail if core fields diverge from identity file.
        for k in (
            "embedding_model",
            "embedding_provider_host",
            "embedding_dim",
            "planner_cache_sha256",
            "query_construction_version",
        ):
            if s_ident.get(k) != step_ident.get(k):
                raise LivePathError(
                    f"step vectors vs identity file field mismatch on {k}"
                )

    catalog_status = json.loads(
        (out_root / "catalog_index" / "index_status.json").read_text(encoding="utf-8")
    )
    cat_digest = (
        (catalog_status.get("selected") or {}).get("identity_digest")
        or catalog_status.get("identity_digest")
    )
    if cat_digest != EXPECTED_CATALOG_IDENTITY:
        raise LivePathError(
            f"catalog identity mismatch: expected={EXPECTED_CATALOG_IDENTITY} got={cat_digest}"
        )
    if int(catalog_status.get("vector_dimension") or 0) != 1536:
        raise LivePathError("catalog vector_dimension != 1536")

    dataset = load_hk155_locked(root)
    catalog = build_catalog_context(root)
    docs = build_all_operation_documents(catalog["bundle"])
    if docs["operation_documents_sha256"] != EXPECTED_OP_DOCS:
        raise LivePathError(
            f"operation_documents_sha256 mismatch: {docs['operation_documents_sha256']}"
        )
    runtime_ids = list(catalog["runtime_operation_ids"])
    if len(runtime_ids) != EXPECTED_RUNTIME_OPS:
        raise LivePathError(f"runtime ops={len(runtime_ids)} != {EXPECTED_RUNTIME_OPS}")

    cache_dir = Path(
        embedding_cache_dir
        or root / ".cache" / "pe_fin_lite_operation_embeddings_v1"
    )
    identity = build_index_identity(
        registry=catalog["bundle"].registry,
        catalog_fingerprint=catalog["catalog_fingerprint"],
        embedding_model=EXPECTED_EMBEDDING_MODEL,
        embedding_provider_host="api.zhizengzeng.com",
        expected_vector_dimension=1536,
    )
    if identity_digest(identity) != EXPECTED_CATALOG_IDENTITY:
        raise LivePathError(
            f"rebuilt catalog identity {identity_digest(identity)} != expected"
        )
    vectors = load_complete_index(
        registry=catalog["bundle"].registry,
        identity=identity,
        cache_dir=cache_dir,
    )
    if len(vectors) != EXPECTED_RUNTIME_OPS:
        # load_complete_index returns registry-complete vectors; intersect runtime.
        pass
    meta_by_id = _metadata_by_id(catalog["bundle"])
    index = OfflineEmbeddingIndex.from_preloaded_vectors(
        documents=docs["documents"],
        metadata_by_id=meta_by_id,
        runtime_operation_ids=runtime_ids,
        vectors=vectors,
        embedding_model=EXPECTED_EMBEDDING_MODEL,
        registry=catalog["bundle"].registry,
    )
    index.embedding_provider_host = "api.zhizengzeng.com"
    index.catalog_source = "real"
    index.catalog_identity_sha256 = EXPECTED_CATALOG_IDENTITY
    assert_real_backend_identity(
        embedding_backend=REAL_EMBEDDING_BACKEND,
        embedding_provider_host="api.zhizengzeng.com",
        vector_dimension=index.embedding_dim,
        source="real",
    )
    assert_query_catalog_alignment(
        query_dim=1536,
        catalog_dim=int(index.embedding_dim or 0),
        query_host="api.zhizengzeng.com",
        catalog_host="api.zhizengzeng.com",
        query_model=EXPECTED_EMBEDDING_MODEL,
        catalog_model=EXPECTED_EMBEDDING_MODEL,
    )
    if abs(float(LEXICAL_WEIGHT) - float(RUNTIME_LEXICAL_WEIGHT)) > 1e-12:
        raise LivePathError("LEXICAL_WEIGHT drifted from 0.12")

    tasks_by_id = {
        str(t.get("task_id") or t.get("id")): t for t in dataset["tasks"]
    }
    step_vectors = dict(scache.get("vectors_by_task_step") or {})
    if len(step_vectors) != EXPECTED_RETRIEVE_STEPS:
        raise LivePathError(
            f"step vectors n={len(step_vectors)} != {EXPECTED_RETRIEVE_STEPS}"
        )

    # formula_key -> per-task metric rows (tool-eligible)
    formula_task_rows: Dict[str, List[Dict[str, Any]]] = {k: [] for k in ABLATION_FORMULAS}
    formula_disc_rows: Dict[str, List[Dict[str, Any]]] = {k: [] for k in ABLATION_FORMULAS}
    per_step_out: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    # Reorder accumulators vs R1
    reorder_stats: Dict[str, Dict[str, Any]] = {
        k: {
            "changed_top1_count": 0,
            "changed_top3_set_count": 0,
            "rescued_gold_into_top3_count": 0,
            "displaced_gold_from_top3_count": 0,
            "unchanged_due_to_constant_coverage_count": 0,
            "missing_or_empty_provides_count": 0,
            "n_steps": 0,
            "coverage_values": [],
            "lexical_values": [],
        }
        for k in ABLATION_FORMULAS
        if k != "R1"
    }

    from hk_equity_mcp.experiment.pe_fin_lite_retrieval_audit.query_cache import (
        step_cache_key,
    )

    for prow in planner_rows:
        tid = str(prow.get("task_id"))
        task = tasks_by_id.get(tid) or {}
        ref = build_evaluation_reference(task)
        steps = prow.get("retrieve_steps") or []
        # Collect per-formula step rankings for this task.
        step_ranked_by_formula: Dict[str, List[List[str]]] = {
            k: [] for k in ABLATION_FORMULAS
        }
        for step in steps:
            if not isinstance(step, dict):
                continue
            sid = str(step.get("id") or step.get("step_id") or "")
            q = str(step.get("search_query") or build_step_query(step))
            entry = step_vectors.get(f"{tid}::{sid}")
            if entry is None:
                key = step_cache_key(tid, sid, q)
                entry = next(
                    (
                        v
                        for v in step_vectors.values()
                        if isinstance(v, dict) and v.get("cache_key") == key
                    ),
                    None,
                )
            if entry is None:
                raise LivePathError(f"Missing step vector for {tid}/{sid}")
            qvec = entry.get("vector") if isinstance(entry, dict) else entry
            req_out = list(step.get("required_outputs") or [])
            gold = list(ref["acceptable_ops_union"])

            # R1 baseline
            try:
                r1 = rank_operations(
                    index=index,
                    query=q,
                    variant="R1",
                    step_domain=str(step.get("domain") or task.get("domain") or "general"),
                    required_outputs=req_out,
                    top_k=TOP_K,
                    gold_ops_for_filter_diag=gold,
                    query_vector=qvec,
                )
            except Exception as exc:  # noqa: BLE001
                errors.append({"task_id": tid, "step_id": sid, "error": f"{type(exc).__name__}:{exc}"})
                continue

            if r1["filter"].get("domain_filter_applied"):
                raise LivePathError("R1 unexpectedly applied domain filter")
            if int(r1["n_candidates_scored"]) != EXPECTED_RUNTIME_OPS:
                raise LivePathError(
                    f"R1 pool size {r1['n_candidates_scored']} != {EXPECTED_RUNTIME_OPS}"
                )
            r1_full = list(r1.get("ranked_full") or r1["ranked"])
            step_ranked_by_formula["R1"].append(list(r1["ranked_ids"]))

            # Emit R1 op rows (top-k + full for analysis compactness: store full)
            for row in r1_full:
                per_step_out.append(
                    {
                        "formula_key": "R1",
                        "variant": "R1",
                        "alpha": None,
                        "lexical_weight": None,
                        "task_id": tid,
                        "step_id": sid,
                        "operation_id": row["operation_id"],
                        "cosine_similarity": row["cosine_similarity"],
                        "required_output_coverage": row.get("required_output_coverage"),
                        "lexical_score": row.get("lexical_score"),
                        "final_score": row.get("final_score", row["score"]),
                        "final_rank": row.get("final_rank"),
                        "domain_filter_applied": False,
                    }
                )

            # R4 alphas
            for alpha in HYBRID_ALPHAS:
                key = f"R4_global_coverage@{alpha}"
                r4 = rank_operations(
                    index=index,
                    query=q,
                    variant="R4",
                    alpha=float(alpha),
                    step_domain=str(step.get("domain") or task.get("domain") or "general"),
                    required_outputs=req_out,
                    top_k=TOP_K,
                    gold_ops_for_filter_diag=gold,
                    query_vector=qvec,
                )
                if r4["filter"].get("domain_filter_applied"):
                    raise LivePathError("R4 must not apply domain filter")
                if int(r4["n_candidates_scored"]) != EXPECTED_RUNTIME_OPS:
                    raise LivePathError(
                        f"R4 pool size {r4['n_candidates_scored']} != {EXPECTED_RUNTIME_OPS}"
                    )
                r4_full = list(r4.get("ranked_full") or [])
                if abs(float(alpha) - 1.0) < 1e-12:
                    assert_r4_alpha1_matches_r1(
                        r1_full=r1_full, r4_full=r4_full, context=f"{tid}/{sid}"
                    )
                step_ranked_by_formula[key].append(list(r4["ranked_ids"]))
                for row in r4_full:
                    per_step_out.append(
                        {
                            "formula_key": key,
                            "variant": "R4",
                            "alpha": float(alpha),
                            "lexical_weight": None,
                            "task_id": tid,
                            "step_id": sid,
                            "operation_id": row["operation_id"],
                            "cosine_similarity": row["cosine_similarity"],
                            "required_output_coverage": row.get(
                                "required_output_coverage"
                            ),
                            "lexical_score": None,
                            "final_score": row.get("final_score", row["score"]),
                            "final_rank": row.get("final_rank"),
                            "domain_filter_applied": False,
                        }
                    )
                an = _analyze_reorder(
                    baseline_full=r1_full, other_full=r4_full, gold=gold
                )
                st = reorder_stats[key]
                st["n_steps"] += 1
                st["changed_top1_count"] += int(an["changed_top1"])
                st["changed_top3_set_count"] += int(an["changed_top3_set"])
                st["rescued_gold_into_top3_count"] += int(an["rescued_gold_into_top3"])
                st["displaced_gold_from_top3_count"] += int(
                    an["displaced_gold_from_top3"]
                )
                st["unchanged_due_to_constant_coverage_count"] += int(
                    an["unchanged_due_to_constant_coverage"]
                )
                st["missing_or_empty_provides_count"] += int(
                    an["missing_or_empty_provides"]
                )
                st["coverage_values"].extend(
                    [
                        float(r["required_output_coverage"])
                        for r in r4_full
                        if r.get("required_output_coverage") is not None
                    ]
                )

            # R5
            key5 = "R5_global_lexical@0.12"
            r5 = rank_operations(
                index=index,
                query=q,
                variant="R5",
                step_domain=str(step.get("domain") or task.get("domain") or "general"),
                required_outputs=req_out,
                top_k=TOP_K,
                gold_ops_for_filter_diag=gold,
                query_vector=qvec,
            )
            if r5["filter"].get("domain_filter_applied"):
                raise LivePathError("R5 must not apply domain filter")
            if abs(float(r5.get("lexical_weight") or -1) - 0.12) > 1e-12:
                raise LivePathError("R5 lexical_weight != 0.12")
            if int(r5["n_candidates_scored"]) != EXPECTED_RUNTIME_OPS:
                raise LivePathError(
                    f"R5 pool size {r5['n_candidates_scored']} != {EXPECTED_RUNTIME_OPS}"
                )
            r5_full = list(r5.get("ranked_full") or [])
            step_ranked_by_formula[key5].append(list(r5["ranked_ids"]))
            for row in r5_full:
                per_step_out.append(
                    {
                        "formula_key": key5,
                        "variant": "R5",
                        "alpha": None,
                        "lexical_weight": 0.12,
                        "task_id": tid,
                        "step_id": sid,
                        "operation_id": row["operation_id"],
                        "cosine_similarity": row["cosine_similarity"],
                        "required_output_coverage": None,
                        "lexical_score": row.get("lexical_score"),
                        "final_score": row.get("final_score", row["score"]),
                        "final_rank": row.get("final_rank"),
                        "domain_filter_applied": False,
                    }
                )
            an5 = _analyze_reorder(
                baseline_full=r1_full, other_full=r5_full, gold=gold
            )
            st5 = reorder_stats[key5]
            st5["n_steps"] += 1
            st5["changed_top1_count"] += int(an5["changed_top1"])
            st5["changed_top3_set_count"] += int(an5["changed_top3_set"])
            st5["rescued_gold_into_top3_count"] += int(an5["rescued_gold_into_top3"])
            st5["displaced_gold_from_top3_count"] += int(
                an5["displaced_gold_from_top3"]
            )
            st5["missing_or_empty_provides_count"] += int(
                an5["missing_or_empty_provides"]
            )
            st5["lexical_values"].extend(
                [
                    float(r["lexical_score"])
                    for r in r5_full
                    if r.get("lexical_score") is not None
                ]
            )

        # Aggregate multi-step metrics per formula for this task.
        for fkey, step_lists in step_ranked_by_formula.items():
            if not step_lists:
                metrics = score_multi_step_route(
                    step_ranked_ids=[],
                    equivalents_by_node=[set(x) for x in ref["equivalents_by_node"]],
                    acceptable_union=set(ref["acceptable_ops_union"]),
                )
            else:
                metrics = score_multi_step_route(
                    step_ranked_ids=step_lists,
                    equivalents_by_node=[set(x) for x in ref["equivalents_by_node"]],
                    acceptable_union=set(ref["acceptable_ops_union"]),
                )
            row = {
                "task_id": tid,
                "formula_key": fkey,
                "metrics": metrics,
                "domain": ref["domain"],
                "source_split": ref["source_split"],
                "is_no_tool_task": ref["is_no_tool_task"],
                "is_disclosure_composite": ref["is_disclosure_composite"],
                "include_in_tool_recall_denominator": ref[
                    "include_in_tool_recall_denominator"
                ],
            }
            if ref["include_in_tool_recall_denominator"]:
                formula_task_rows[fkey].append(row)
            elif ref["is_disclosure_composite"]:
                formula_disc_rows[fkey].append(row)

    def _agg_split(rows: List[Dict[str, Any]], split_ids: Set[str]) -> Dict[str, Any]:
        subset = [r for r in rows if r.get("task_id") in split_ids]
        per_domain: Dict[str, List[float]] = defaultdict(list)
        for r in subset:
            per_domain[str(r.get("domain"))].append(
                float((r.get("metrics") or {}).get("recall", {}).get("@3") or 0.0)
            )
        domain_means = {d: statistics.mean(vs) if vs else 0.0 for d, vs in per_domain.items()}
        return attach_min_domain_recall(
            aggregate_metric_rows([r["metrics"] for r in subset]),
            per_domain_recall_at_3=domain_means,
        )

    splits = dataset["splits"]
    formula_metrics: Dict[str, Any] = {}
    development_inputs: Dict[str, Any] = {}
    for fkey in ABLATION_FORMULAS:
        rows = formula_task_rows[fkey]
        by_split = {
            "development70": _agg_split(rows, set(splits["development70"])),
            "previously_reserved85": _agg_split(
                rows, set(splits["previously_reserved85"])
            ),
            "all155": _agg_split(rows, set(splits["all155"])),
        }
        per_domain_all: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for r in rows:
            per_domain_all[str(r.get("domain"))].append(r)
        per_domain_metrics = {
            d: aggregate_metric_rows([x["metrics"] for x in rs])
            for d, rs in sorted(per_domain_all.items())
        }
        disc = aggregate_metric_rows(
            [r["metrics"] for r in formula_disc_rows[fkey]]
        )
        # rebuild ordered topk per step for catalog coverage
        step_topk: Dict[Tuple[str, str], List[Tuple[int, str]]] = defaultdict(list)
        for r in per_step_out:
            if r["formula_key"] != fkey:
                continue
            if int(r.get("final_rank") or 99) > TOP_K:
                continue
            step_topk[(r["task_id"], r["step_id"])].append(
                (int(r["final_rank"]), r["operation_id"])
            )
        cov_rankings = []
        for (tid, sid), pairs in step_topk.items():
            pairs.sort()
            cov_rankings.append(
                {
                    "task_id": tid,
                    "step_id": sid,
                    "ranked_ids": [oid for _, oid in pairs],
                }
            )
        cat_cov = analyze_catalog_coverage(
            formal_gold_ids=catalog["formal_gold_eligible_ids"],
            distractor_ids=catalog["distractor_ids"],
            excluded_ids=catalog["excluded_ids"],
            per_query_rankings=cov_rankings,
        )
        formula_metrics[fkey] = {
            "by_split": by_split,
            "per_domain": per_domain_metrics,
            "disclosure_separate": disc,
            "catalog_coverage": {
                "distractors": cat_cov.get("distractors"),
                "excluded": cat_cov.get("excluded"),
            },
            "n_tool_tasks": len(rows),
            "n_disclosure_tasks": len(formula_disc_rows[fkey]),
        }
        development_inputs[fkey] = by_split["development70"]

    # R4@1.0 aggregate must match R1 aggregate
    r1_dev = development_inputs["R1"]
    r4_1_dev = development_inputs["R4_global_coverage@1.0"]
    for metric in ("complete_route@3", "recall@3", "mrr", "hit@3"):
        a = float(r1_dev.get(metric) or 0.0)
        b = float(r4_1_dev.get(metric) or 0.0)
        if abs(a - b) > 1e-12:
            raise LivePathError(
                f"R4@1.0 development aggregate {metric}={b} != R1 {a}"
            )

    comparison = compare_ablation_formulas(development_inputs)
    selected = comparison.get("selected") or {}
    sel_key = selected.get("formula_key")
    sel_variant, sel_alpha, sel_lw = (
        _parse_ablation_key(sel_key) if sel_key else (None, None, None)
    )

    # Signal reorder summary (drop raw value lists)
    signal_analysis = {}
    for k, st in reorder_stats.items():
        cov_vals = st.get("coverage_values") or []
        lex_vals = st.get("lexical_values") or []
        signal_analysis[k] = {
            "changed_top1_count": st["changed_top1_count"],
            "changed_top3_set_count": st["changed_top3_set_count"],
            "rescued_gold_into_top3_count": st["rescued_gold_into_top3_count"],
            "displaced_gold_from_top3_count": st["displaced_gold_from_top3_count"],
            "unchanged_due_to_constant_coverage_count": st[
                "unchanged_due_to_constant_coverage_count"
            ],
            "missing_or_empty_provides_count": st["missing_or_empty_provides_count"],
            "n_steps": st["n_steps"],
            "coverage_score_distribution": _dist(cov_vals),
            "lexical_score_distribution": _dist(lex_vals),
        }

    # Old formula lock audit (do not modify)
    old_lock_path = out_root / "formula_selection" / "formula_lock.json"
    old_lock_audit = {"exists": old_lock_path.exists(), "provisional_superseded": False}
    if old_lock_path.exists():
        old = json.loads(old_lock_path.read_text(encoding="utf-8"))
        cfg = old.get("config") or {}
        host = old.get("embedding_provider_host", cfg.get("embedding_provider_host"))
        dim = old.get("embedding_dimension", cfg.get("embedding_dimension"))
        idx = old.get("operation_index_identity", cfg.get("operation_index_identity"))
        psha = old.get("planner_cache_sha256", cfg.get("planner_cache_sha256"))
        incomplete = host is None or dim is None or idx is None
        stale_planner = psha != EXPECTED_PLANNER_SHA
        old_lock_audit.update(
            {
                "selected_variant": old.get("selected_variant"),
                "embedding_provider_host": host,
                "embedding_dimension": dim,
                "operation_index_identity": idx,
                "planner_cache_sha256": psha,
                "identity_incomplete": incomplete,
                "planner_sha_mismatch_vs_final": stale_planner,
                "provisional_superseded": bool(incomplete or stale_planner),
                "note": (
                    "Existing formula_lock.json left untouched; treated as "
                    "provisional/superseded for identity incompleteness or stale "
                    "planner SHA."
                ),
            }
        )

    documents_hash = identity.get("documents_hash")
    registry_fp = identity.get("registry_fingerprint")
    candidate_lock = {
        "lock_version": "retrieval_formula_lock_candidate_v3",
        "result_role": "exploratory_descriptive",
        "selected_variant": (
            "R4_global_coverage"
            if sel_variant == "R4"
            else ("R5_global_lexical" if sel_variant == "R5" else sel_variant)
        ),
        "formula_key": sel_key,
        "alpha": sel_alpha,
        "lexical_weight": sel_lw if sel_variant == "R5" else None,
        "exact_formula": comparison.get("exact_formula"),
        "embedding_provider_host": "api.zhizengzeng.com",
        "embedding_model": EXPECTED_EMBEDDING_MODEL,
        "embedding_dimension": 1536,
        "operation_index_identity": EXPECTED_CATALOG_IDENTITY,
        "operation_documents_sha256": EXPECTED_OP_DOCS,
        "documents_hash": documents_hash,
        "registry_fingerprint": registry_fp,
        "catalog_fingerprint": catalog["catalog_fingerprint"],
        "planner_cache_sha256": EXPECTED_PLANNER_SHA,
        "step_query_cache_identity": EXPECTED_STEP_IDENTITY,
        "query_construction_version": QUERY_CONSTRUCTION_VERSION,
        "ranking_protocol_version": RANKING_PROTOCOL_VERSION,
        "operation_document_version": OPERATION_DOCUMENT_VERSION,
        "ablation_version": ABLATION_VERSION,
        "top_k": TOP_K,
        "describe_n": DESCRIBE_N,
        "development_metrics": selected.get("metrics"),
        "selection_policy": comparison.get("selection_policy"),
        "runtime_sync_required": True,
        "runtime_synced": False,
        "no_retune_on_reserved85": True,
        "no_unseen_heldout_claim": True,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    for req in (
        "embedding_provider_host",
        "embedding_model",
        "embedding_dimension",
        "operation_index_identity",
        "operation_documents_sha256",
        "planner_cache_sha256",
        "step_query_cache_identity",
    ):
        if candidate_lock.get(req) in (None, ""):
            raise LivePathError(f"candidate lock missing required field {req}")
    candidate_lock["candidate_lock_sha256"] = stable_json_hash(
        {k: v for k, v in candidate_lock.items() if k != "candidate_lock_sha256"}
    )

    # Write artifacts
    _write_jsonl(abl_dir / "per_step_rankings.jsonl", per_step_out)
    _write_json(
        abl_dir / "formula_comparison.json",
        {
            "formulas": formula_metrics,
            "development70_selection_inputs": development_inputs,
            "comparison": comparison,
            "embedding_backend": REAL_EMBEDDING_BACKEND,
            "zero_network": True,
        },
    )
    _write_json(
        abl_dir / "retrieval_metrics.json",
        {
            fkey: {
                "development70": formula_metrics[fkey]["by_split"]["development70"],
                "previously_reserved85": formula_metrics[fkey]["by_split"][
                    "previously_reserved85"
                ],
                "all155": formula_metrics[fkey]["by_split"]["all155"],
            }
            for fkey in ABLATION_FORMULAS
        },
    )
    _write_json(
        abl_dir / "per_domain_metrics.json",
        {fkey: formula_metrics[fkey]["per_domain"] for fkey in ABLATION_FORMULAS},
    )
    _write_json(
        abl_dir / "disclosure_metrics.json",
        {
            fkey: formula_metrics[fkey]["disclosure_separate"]
            for fkey in ABLATION_FORMULAS
        },
    )
    _write_json(abl_dir / "signal_reordering_analysis.json", signal_analysis)
    _write_json(abl_dir / "error_summary.json", {"errors": errors})
    _write_json(abl_dir / "formula_lock_candidate_v3.json", candidate_lock)
    _write_json(
        abl_dir / "old_formula_lock_audit.json",
        old_lock_audit,
    )

    identity_payload = {
        "ablation_version": ABLATION_VERSION,
        "planner_cache_sha256": EXPECTED_PLANNER_SHA,
        "step_query_cache_identity": EXPECTED_STEP_IDENTITY,
        "step_vectors_stored_identity_sha256_stale": stale_vector_digest,
        "step_vectors_recomputed_identity_sha256": recomputed,
        "operation_index_identity": EXPECTED_CATALOG_IDENTITY,
        "operation_documents_sha256": EXPECTED_OP_DOCS,
        "documents_hash": documents_hash,
        "registry_fingerprint": registry_fp,
        "catalog_fingerprint": catalog["catalog_fingerprint"],
        "embedding_provider_host": "api.zhizengzeng.com",
        "embedding_model": EXPECTED_EMBEDDING_MODEL,
        "embedding_dimension": 1536,
        "runtime_operations": EXPECTED_RUNTIME_OPS,
        "retrieve_steps": EXPECTED_RETRIEVE_STEPS,
        "top_k": TOP_K,
        "describe_n": DESCRIBE_N,
        "query_construction_version": QUERY_CONSTRUCTION_VERSION,
        "ranking_protocol_version": RANKING_PROTOCOL_VERSION,
        "lexical_weight": 0.12,
        "hybrid_alphas": list(HYBRID_ALPHAS),
    }
    _write_json(abl_dir / "identity.json", identity_payload)

    zero_net = {
        "zero_network": True,
        "generation_llm_calls": 0,
        "embedding_api_calls": 0,
        "mcp_calls": 0,
        "build_index": False,
        "agent_run": False,
        "confirmation": "offline ranking from complete frozen caches only",
    }
    _write_json(abl_dir / "zero_network_confirmation.json", zero_net)

    # After hashes
    raw_after = {}
    for rel in watched:
        p = out_root / rel
        if p.exists():
            raw_after[rel] = {"sha256": _file_sha(p), "bytes": p.stat().st_size}
    unchanged = raw_before == raw_after
    if not unchanged:
        raise LivePathError(
            "Raw input artifacts changed during ablation (fail-closed)"
        )
    _write_json(
        abl_dir / "raw_input_hashes.json",
        {"before": raw_before, "after": raw_after, "unchanged": True},
    )

    run_meta = {
        "ablation_version": ABLATION_VERSION,
        "result_role": "exploratory_descriptive",
        "zero_network": True,
        "embedding_backend": REAL_EMBEDDING_BACKEND,
        "selected_candidate": {
            "formula_key": sel_key,
            "candidate_lock_sha256": candidate_lock["candidate_lock_sha256"],
        },
        "old_formula_lock_audit": old_lock_audit,
        "n_per_step_rows": len(per_step_out),
        "n_errors": len(errors),
        "runtime_synced": False,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(abl_dir / "run_metadata.json", run_meta)

    return {
        "ablation_dir": str(abl_dir),
        "comparison": comparison,
        "formula_metrics": formula_metrics,
        "signal_analysis": signal_analysis,
        "candidate_lock": candidate_lock,
        "old_formula_lock_audit": old_lock_audit,
        "raw_hashes_unchanged": True,
        "zero_network": True,
    }


def _dist(vals: Sequence[float]) -> Dict[str, Any]:
    xs = [float(x) for x in vals]
    if not xs:
        return {"n": 0}
    return {
        "n": len(xs),
        "mean": float(statistics.mean(xs)),
        "median": float(statistics.median(xs)),
        "min": float(min(xs)),
        "max": float(max(xs)),
        "p25": float(sorted(xs)[max(0, int(0.25 * (len(xs) - 1)))]),
        "p75": float(sorted(xs)[max(0, int(0.75 * (len(xs) - 1)))]),
    }
