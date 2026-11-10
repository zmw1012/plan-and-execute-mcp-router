"""Orchestrate Retrieval Audit modes (offline mock/dry-run safe)."""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

from hk_equity_mcp.experiment.pe_fin_lite_retrieval_audit import (
    AUDIT_VERSION,
    DEFAULT_OUTPUT_DIR,
    DESCRIBE_N,
    HYBRID_ALPHAS,
    RESULT_ROLE,
    TOP_K,
)
from hk_equity_mcp.experiment.pe_fin_lite_retrieval_audit.catalog_audit import (
    analyze_catalog_coverage,
)
from hk_equity_mcp.experiment.pe_fin_lite_retrieval_audit.formula_selection import (
    assert_frozen_evaluation_allowed,
    attach_min_domain_recall,
    build_formula_lock,
    compare_formulas,
    write_formula_lock,
)
from hk_equity_mcp.experiment.pe_fin_lite_retrieval_audit.gold_reference import (
    build_evaluation_reference,
)
from hk_equity_mcp.experiment.pe_fin_lite_retrieval_audit.identity import (
    build_catalog_context,
    load_hk155_locked,
    planner_only_cache_identity,
    question_level_cache_identity,
    sha256_text,
    step_retrieval_cache_identity,
)
from hk_equity_mcp.experiment.pe_fin_lite_retrieval_audit.metrics import (
    aggregate_metric_rows,
    cross_domain_route_coverage,
    score_multi_step_route,
    score_ranked_list,
)
from hk_equity_mcp.experiment.pe_fin_lite_retrieval_audit.operation_documents import (
    build_all_operation_documents,
)
from hk_equity_mcp.experiment.pe_fin_lite_retrieval_audit.planner_only import (
    run_planner_only,
)
from hk_equity_mcp.experiment.pe_fin_lite_retrieval_audit.ranking import (
    OfflineEmbeddingIndex,
    build_question_query,
    rank_operations,
)
from hk_equity_mcp.pe_fin_lite.capability_profile import (
    capability_profile_hash,
    load_capability_profile,
)


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _metadata_by_id(bundle) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for oid, rec in bundle.records.items():
        raw = rec.source_raw or {}
        out[oid] = {
            "domain": rec.preferred_domain or (rec.domains[0] if rec.domains else "general"),
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


def _task_map(tasks: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {str(t.get("task_id") or t.get("id")): t for t in tasks}


def _split_filter(
    rows: Sequence[Dict[str, Any]],
    *,
    split_ids: Sequence[str],
) -> List[Dict[str, Any]]:
    allow = set(split_ids)
    return [r for r in rows if str(r.get("task_id")) in allow]


def _aggregate_by_domain(
    scored_rows: Sequence[Dict[str, Any]],
    tasks_by_id: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in scored_rows:
        tid = str(r.get("task_id"))
        dom = str((tasks_by_id.get(tid) or {}).get("domain") or "unknown")
        buckets[dom].append(r["metrics"])
    return {dom: aggregate_metric_rows(rows) for dom, rows in sorted(buckets.items())}


def _aggregate_by_difficulty(
    scored_rows: Sequence[Dict[str, Any]],
    tasks_by_id: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in scored_rows:
        tid = str(r.get("task_id"))
        diff = str(
            (tasks_by_id.get(tid) or {}).get("project_difficulty_level")
            or (tasks_by_id.get(tid) or {}).get("difficulty")
            or "unknown"
        )
        buckets[diff].append(r["metrics"])
    return {d: aggregate_metric_rows(rows) for d, rows in sorted(buckets.items())}


class RetrievalAuditRunner:
    def __init__(
        self,
        *,
        root: Path,
        output_dir: Optional[Path] = None,
        mock: bool = True,
        dry_run: bool = False,
        allow_network: bool = False,
    ):
        self.root = Path(root)
        self.output_dir = Path(output_dir or (self.root / DEFAULT_OUTPUT_DIR))
        self.mock = bool(mock)
        self.dry_run = bool(dry_run)
        self.allow_network = bool(allow_network)
        if self.allow_network:
            raise RuntimeError(
                "Retrieval Audit runner refuses allow_network=True in this offline round"
            )
        self.dataset = load_hk155_locked(self.root)
        self.catalog = build_catalog_context(self.root)
        self.docs = build_all_operation_documents(self.catalog["bundle"])
        self.meta_by_id = _metadata_by_id(self.catalog["bundle"])
        self.tasks_by_id = _task_map(self.dataset["tasks"])
        self.index = OfflineEmbeddingIndex(
            documents=self.docs["documents"],
            metadata_by_id=self.meta_by_id,
            runtime_operation_ids=self.catalog["runtime_operation_ids"],
            allow_network=False,
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _base_run_metadata(self, mode: str) -> Dict[str, Any]:
        return {
            "audit_version": AUDIT_VERSION,
            "result_role": RESULT_ROLE,
            "mode": mode,
            "mock": self.mock,
            "dry_run": self.dry_run,
            "allow_network": False,
            "dataset": self.dataset["identities"],
            "catalog_stats": self.catalog["stats"],
            "catalog_fingerprint": self.catalog["catalog_fingerprint"],
            "operation_documents_sha256": self.docs["operation_documents_sha256"],
            "operation_document_integrity": self.docs["integrity"],
            "embedding_model": self.index.embedding_model,
            "embedding_dim": self.index.embedding_dim,
            "top_k": TOP_K,
            "describe_n": DESCRIBE_N,
            "output_dir": str(self.output_dir),
            "no_unseen_heldout_claim": True,
        }

    def run_question_level(self) -> Dict[str, Any]:
        out_dir = self.output_dir / "question_level"
        out_dir.mkdir(parents=True, exist_ok=True)
        q_identity = question_level_cache_identity(
            dataset_identities=self.dataset["identities"],
            operation_documents_sha256=self.docs["operation_documents_sha256"],
            catalog_fingerprint=self.catalog["catalog_fingerprint"],
            embedding_model=self.index.embedding_model,
            embedding_provider_host="local_hash",
        )
        _write_json(out_dir / "cache_identity.json", q_identity)

        rankings: List[Dict[str, Any]] = []
        scored_tool: List[Dict[str, Any]] = []
        scored_disclosure: List[Dict[str, Any]] = []
        no_tool_rows: List[Dict[str, Any]] = []
        errors: List[Dict[str, Any]] = []

        if self.dry_run:
            meta = self._base_run_metadata("question_level")
            meta["dry_run_note"] = "No rankings computed"
            _write_json(out_dir / "run_metadata.json", meta)
            _write_json(out_dir / "error_summary.json", {"errors": []})
            return meta

        for task in self.dataset["tasks"]:
            tid = str(task.get("task_id") or task.get("id"))
            ref = build_evaluation_reference(task)
            query = build_question_query(str(task.get("question") or ""))
            try:
                ranked = rank_operations(
                    index=self.index,
                    query=query,
                    variant="R0",
                    step_domain=str(task.get("domain") or "general"),
                    required_outputs=[],
                    top_k=TOP_K,
                    gold_ops_for_filter_diag=ref["acceptable_ops_union"],
                )
            except Exception as exc:  # noqa: BLE001
                errors.append({"task_id": tid, "error": f"{type(exc).__name__}:{exc}"})
                continue

            eq_sets = [set(x) for x in ref["equivalents_by_node"]]
            acc = set(ref["acceptable_ops_union"])
            metrics = score_ranked_list(
                ranked_ids=ranked["ranked_ids"],
                equivalents_by_node=eq_sets,
                acceptable_union=acc,
            )
            xd = cross_domain_route_coverage(
                ranked_ids=ranked["ranked_ids"],
                domain_branches=ref["domain_branches"],
                k=TOP_K,
            )
            row = {
                "task_id": tid,
                "variant": "R0",
                "source_split": ref["source_split"],
                "domain": ref["domain"],
                "query": query,
                "ranked_ids": ranked["ranked_ids"],
                "ranked": ranked["ranked"],
                "filter": ranked["filter"],
                "blocked_leakage": ranked["blocked_leakage"],
                "metrics": metrics,
                "cross_domain": xd,
                "is_no_tool_task": ref["is_no_tool_task"],
                "is_disclosure_composite": ref["is_disclosure_composite"],
                "include_in_tool_recall_denominator": ref[
                    "include_in_tool_recall_denominator"
                ],
                "evaluation_reference_task_id_only": tid,
            }
            rankings.append(row)
            if ref["is_no_tool_task"]:
                no_tool_rows.append(
                    {
                        "task_id": tid,
                        "unnecessary_retrieval": 1.0 if ranked["ranked_ids"] else 0.0,
                        "retrieved_any": bool(ranked["ranked_ids"]),
                        "answerability_class": ref["answerability_class"],
                    }
                )
            elif ref["is_disclosure_composite"]:
                scored_disclosure.append(row)
            else:
                scored_tool.append(row)

        tool_metrics = aggregate_metric_rows([r["metrics"] for r in scored_tool])
        disc_metrics = aggregate_metric_rows([r["metrics"] for r in scored_disclosure])
        by_split = {}
        for name, ids in self.dataset["splits"].items():
            subset = _split_filter(scored_tool, split_ids=ids)
            by_split[name] = aggregate_metric_rows([r["metrics"] for r in subset])

        per_domain = _aggregate_by_domain(scored_tool, self.tasks_by_id)
        per_diff = _aggregate_by_difficulty(scored_tool, self.tasks_by_id)
        catalog_cov = analyze_catalog_coverage(
            formal_gold_ids=self.catalog["formal_gold_eligible_ids"],
            distractor_ids=self.catalog["distractor_ids"],
            excluded_ids=self.catalog["excluded_ids"],
            per_query_rankings=rankings,
        )

        filter_miss = sum(
            1 for r in rankings if (r.get("filter") or {}).get("filter_miss")
        )
        embedding_miss = sum(
            1 for r in rankings if (r.get("filter") or {}).get("embedding_miss")
        )

        retrieval_metrics = {
            "variant": "R0",
            "tool_recall_tasks": tool_metrics,
            "disclosure_composite_separate": disc_metrics,
            "no_tool_decision": {
                "n": len(no_tool_rows),
                "unnecessary_retrieval_rate": (
                    statistics.mean(r["unnecessary_retrieval"] for r in no_tool_rows)
                    if no_tool_rows
                    else None
                ),
                "rows": no_tool_rows,
            },
            "by_split": by_split,
            "per_domain": per_domain,
            "per_difficulty": per_diff,
            "filter_miss_count": filter_miss,
            "embedding_miss_count": embedding_miss,
        }
        route_coverage = {
            "complete_route@3_tool": tool_metrics.get("complete_route@3"),
            "complete_route@5_tool": tool_metrics.get("complete_route@5"),
            "cross_domain_complete_rate": statistics.mean(
                [
                    float((r.get("cross_domain") or {}).get("complete_cross_domain_route") or 0.0)
                    for r in scored_tool
                    if str(r.get("domain")) == "cross_domain"
                ]
                or [0.0]
            ),
        }

        _write_jsonl(out_dir / "per_query_rankings.jsonl", rankings)
        _write_json(out_dir / "retrieval_metrics.json", retrieval_metrics)
        _write_json(out_dir / "route_coverage.json", route_coverage)
        _write_json(out_dir / "per_domain_metrics.json", per_domain)
        _write_json(out_dir / "distractor_analysis.json", catalog_cov["distractors"])
        _write_json(out_dir / "operation_coverage.json", catalog_cov)
        _write_json(out_dir / "filter_miss_analysis.json", {
            "filter_miss_count": filter_miss,
            "embedding_miss_count": embedding_miss,
        })
        meta = self._base_run_metadata("question_level")
        meta["n_rankings"] = len(rankings)
        meta["cache_identity"] = q_identity
        _write_json(out_dir / "run_metadata.json", meta)
        _write_json(out_dir / "error_summary.json", {"errors": errors})
        return meta

    def run_planner_only(self) -> Dict[str, Any]:
        out_dir = self.output_dir / "planner_cache"
        profile = load_capability_profile(self.root)
        identity = planner_only_cache_identity(
            dataset_identities=self.dataset["identities"],
            capability_profile_hash=capability_profile_hash(profile),
            generation_provider="offline_mock" if self.mock or self.dry_run else "live",
            generation_host="local",
            generation_model="mock_plan_v1" if self.mock or self.dry_run else "unset",
        )
        summary = run_planner_only(
            root=self.root,
            tasks=self.dataset["tasks"],
            output_dir=out_dir,
            cache_identity=identity,
            mock=self.mock or self.dry_run,
            dry_run=self.dry_run,
            resume=True,
            capability_profile=profile,
        )
        meta = self._base_run_metadata("planner_only")
        meta["planner_only_summary"] = summary
        _write_json(out_dir / "run_metadata.json", meta)
        _write_json(out_dir / "error_summary.json", {"errors": summary.get("errors") or []})
        return meta

    def _load_planner_cache(self) -> Dict[str, Any]:
        path = self.output_dir / "planner_cache" / "planner_outputs.jsonl"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing planner cache: {path}. Run --planner-only first."
            )
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        cache_sha = sha256_text(path.read_text(encoding="utf-8"))
        return {"rows": rows, "planner_cache_sha256": cache_sha, "path": str(path)}

    def run_rank_from_planner_cache(
        self,
        *,
        variants: Optional[Sequence[str]] = None,
        alphas: Optional[Sequence[float]] = None,
    ) -> Dict[str, Any]:
        variants = list(variants or ("R1", "R2", "R3"))
        alphas = list(alphas or HYBRID_ALPHAS)
        out_dir = self.output_dir / "ranking_variants"
        out_dir.mkdir(parents=True, exist_ok=True)
        cache = self._load_planner_cache()
        if self.dry_run:
            meta = self._base_run_metadata("rank_from_planner_cache")
            meta["dry_run_note"] = "Skipped ranking"
            _write_json(out_dir / "run_metadata.json", meta)
            _write_json(
                out_dir / "formula_comparison.json",
                {
                    "formulas": {},
                    "development70_selection_inputs": {},
                    "dry_run": True,
                },
            )
            _write_json(out_dir / "retrieval_metrics.json", {"dry_run": True})
            _write_json(out_dir / "route_coverage.json", {"dry_run": True})
            _write_json(out_dir / "per_domain_metrics.json", {"dry_run": True})
            _write_jsonl(out_dir / "per_step_rankings.jsonl", [])
            _write_json(out_dir / "error_summary.json", {"errors": []})
            return meta

        all_step_rankings: List[Dict[str, Any]] = []
        formula_metrics: Dict[str, Dict[str, Any]] = {}
        comparison_rows: Dict[str, Any] = {}

        for variant in variants:
            alpha_list: List[Optional[float]]
            if variant == "R3":
                alpha_list = list(alphas)
            else:
                alpha_list = [None]

            for alpha in alpha_list:
                formula_key = f"{variant}@{alpha}" if alpha is not None else variant
                identity = step_retrieval_cache_identity(
                    planner_cache_sha256=cache["planner_cache_sha256"],
                    operation_documents_sha256=self.docs["operation_documents_sha256"],
                    catalog_fingerprint=self.catalog["catalog_fingerprint"],
                    ranking_variant=variant,
                    alpha=alpha,
                    embedding_model=self.index.embedding_model,
                )
                scored_tool: List[Dict[str, Any]] = []
                scored_disc: List[Dict[str, Any]] = []
                per_domain_recall3: Dict[str, List[float]] = defaultdict(list)

                for prow in cache["rows"]:
                    tid = str(prow.get("task_id"))
                    task = self.tasks_by_id.get(tid) or {}
                    ref = build_evaluation_reference(task)
                    steps = prow.get("retrieve_steps") or []
                    step_ranked: List[List[str]] = []
                    for step in steps:
                        if not isinstance(step, dict):
                            continue
                        q = str(step.get("search_query") or build_question_query(""))
                        ranked = rank_operations(
                            index=self.index,
                            query=q,
                            variant=variant,
                            alpha=float(alpha) if alpha is not None else 1.0,
                            step_domain=str(step.get("domain") or task.get("domain") or "general"),
                            required_outputs=list(step.get("required_outputs") or []),
                            top_k=TOP_K,
                            gold_ops_for_filter_diag=ref["acceptable_ops_union"],
                        )
                        step_ranked.append(ranked["ranked_ids"])
                        all_step_rankings.append(
                            {
                                "task_id": tid,
                                "step_id": step.get("id"),
                                "variant": variant,
                                "alpha": alpha,
                                "query": q,
                                "ranked_ids": ranked["ranked_ids"],
                                "filter": ranked["filter"],
                                "blocked_leakage": ranked["blocked_leakage"],
                                "identity_sha256": identity["identity_sha256"],
                            }
                        )
                    if not step_ranked:
                        # Planner failure / no retrieve steps.
                        metrics = score_ranked_list(
                            ranked_ids=[],
                            equivalents_by_node=[set(x) for x in ref["equivalents_by_node"]],
                            acceptable_union=set(ref["acceptable_ops_union"]),
                        )
                    else:
                        metrics = score_multi_step_route(
                            step_ranked_ids=step_ranked,
                            equivalents_by_node=[set(x) for x in ref["equivalents_by_node"]],
                            acceptable_union=set(ref["acceptable_ops_union"]),
                        )
                    row = {
                        "task_id": tid,
                        "variant": variant,
                        "alpha": alpha,
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
                        scored_tool.append(row)
                        per_domain_recall3[str(ref["domain"])].append(
                            float(metrics.get("recall", {}).get("@3") or 0.0)
                        )
                    elif ref["is_disclosure_composite"]:
                        scored_disc.append(row)

                domain_means = {
                    d: statistics.mean(vs) if vs else 0.0
                    for d, vs in per_domain_recall3.items()
                }
                agg = attach_min_domain_recall(
                    aggregate_metric_rows([r["metrics"] for r in scored_tool]),
                    per_domain_recall_at_3=domain_means,
                )
                # Split metrics for later selection / reserved eval.
                by_split = {}
                for name, ids in self.dataset["splits"].items():
                    subset = _split_filter(scored_tool, split_ids=ids)
                    by_split[name] = attach_min_domain_recall(
                        aggregate_metric_rows([r["metrics"] for r in subset]),
                        per_domain_recall_at_3={
                            d: statistics.mean(
                                float(r["metrics"]["recall"]["@3"])
                                for r in subset
                                if r.get("domain") == d
                            )
                            if any(r.get("domain") == d for r in subset)
                            else 0.0
                            for d in sorted({r.get("domain") for r in subset})
                        },
                    )
                formula_metrics[formula_key] = {
                    "aggregate_all_tool": agg,
                    "by_split": by_split,
                    "disclosure_separate": aggregate_metric_rows(
                        [r["metrics"] for r in scored_disc]
                    ),
                    "identity": identity,
                }
                comparison_rows[formula_key] = by_split.get("development70") or agg

        _write_jsonl(out_dir / "per_step_rankings.jsonl", all_step_rankings)
        _write_json(out_dir / "formula_comparison.json", {
            "formulas": formula_metrics,
            "development70_selection_inputs": comparison_rows,
        })
        catalog_cov = analyze_catalog_coverage(
            formal_gold_ids=self.catalog["formal_gold_eligible_ids"],
            distractor_ids=self.catalog["distractor_ids"],
            excluded_ids=self.catalog["excluded_ids"],
            per_query_rankings=all_step_rankings,
        )
        _write_json(out_dir / "distractor_analysis.json", catalog_cov["distractors"])
        _write_json(out_dir / "operation_coverage.json", catalog_cov)
        meta = self._base_run_metadata("rank_from_planner_cache")
        meta["n_step_rankings"] = len(all_step_rankings)
        meta["formulas"] = list(formula_metrics.keys())
        _write_json(out_dir / "run_metadata.json", meta)
        _write_json(out_dir / "error_summary.json", {"errors": []})
        _write_json(out_dir / "retrieval_metrics.json", {
            "formulas": {k: v["aggregate_all_tool"] for k, v in formula_metrics.items()}
        })
        _write_json(out_dir / "route_coverage.json", {
            k: {
                "complete_route@3": (v["aggregate_all_tool"] or {}).get("complete_route@3"),
                "complete_route@5": (v["aggregate_all_tool"] or {}).get("complete_route@5"),
            }
            for k, v in formula_metrics.items()
        })
        _write_json(out_dir / "per_domain_metrics.json", {
            k: (v["by_split"].get("all155") or v["aggregate_all_tool"])
            for k, v in formula_metrics.items()
        })
        return meta

    def run_select_formula(self, *, formal: bool = False) -> Dict[str, Any]:
        """Select formula using development-70 metrics only.

        Formal ``formula_lock.json`` is written only when ``formal=True`` and
        metrics are from a real (non-mock) ranking run. Offline/mock rounds
        validate the selection machinery and emit schema-only artifacts.
        """
        out_dir = self.output_dir / "formula_selection"
        out_dir.mkdir(parents=True, exist_ok=True)
        cmp_path = self.output_dir / "ranking_variants" / "formula_comparison.json"
        if not cmp_path.exists():
            if self.dry_run:
                payload = {"formulas": {}, "development70_selection_inputs": {}}
            else:
                raise FileNotFoundError(
                    f"Missing {cmp_path}; run --rank-from-planner-cache first"
                )
        else:
            payload = json.loads(cmp_path.read_text(encoding="utf-8"))
        dev_inputs = payload.get("development70_selection_inputs") or {}
        comparison = compare_formulas(development_metrics_by_formula=dev_inputs)
        _write_json(out_dir / "formula_comparison.json", comparison)

        schema_only = {
            "lock_version": "retrieval_formula_lock_v1",
            "status": "schema_only_no_formal_selection",
            "note": (
                "Formal formula lock requires real ranking metrics on "
                "development-origin 70 after a non-mock embedding run. "
                "This offline/mock round only validates selection machinery."
            ),
            "selection_policy": [
                "complete_route@3",
                "macro_recall@3",
                "minimum_per_domain_recall@3",
                "mrr",
                "simpler_formula",
                "larger_alpha",
            ],
            "no_retune_on_reserved85": True,
            "mock_selection_preview": comparison.get("selected"),
        }
        _write_json(out_dir / "formula_lock_schema.json", schema_only)

        write_formal = bool(formal) and (not self.mock) and (not self.dry_run) and bool(
            comparison.get("selected")
        )
        meta = self._base_run_metadata("select_formula")
        meta["formal_lock_written"] = False
        if write_formal:
            selected = comparison["selected"]
            lock = build_formula_lock(
                selected=selected,
                development_metrics=selected.get("metrics") or {},
                config={
                    "catalog_fingerprint": self.catalog["catalog_fingerprint"],
                    "operation_documents_sha256": self.docs["operation_documents_sha256"],
                    "embedding_model": self.index.embedding_model,
                    "dataset": self.dataset["identities"],
                },
            )
            write_formula_lock(out_dir / "formula_lock.json", lock)
            meta["formal_lock_written"] = True
            meta["selected"] = {
                "variant": lock.get("selected_variant"),
                "alpha": lock.get("alpha"),
                "lock_sha256": lock.get("lock_sha256"),
            }
        _write_json(out_dir / "run_metadata.json", meta)
        return meta

    def run_evaluate_frozen_formula(self) -> Dict[str, Any]:
        out_dir = self.output_dir / "reserved_evaluation"
        out_dir.mkdir(parents=True, exist_ok=True)
        lock_path = self.output_dir / "formula_selection" / "formula_lock.json"
        cmp_path = self.output_dir / "ranking_variants" / "formula_comparison.json"
        if not lock_path.exists():
            # Schema-only path: refuse retune demonstration without claiming results.
            meta = self._base_run_metadata("evaluate_frozen_formula")
            meta["status"] = "skipped_no_formal_lock"
            meta["note"] = (
                "No formal formula_lock.json yet (expected in offline round without "
                "real embeddings). Reserved-85 retune guard is unit-tested."
            )
            _write_json(out_dir / "run_metadata.json", meta)
            return meta
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        payload = json.loads(cmp_path.read_text(encoding="utf-8"))
        formulas = payload.get("formulas") or {}
        key = lock.get("formula_key")
        if key not in formulas:
            raise RuntimeError(f"Locked formula {key} missing from ranking_variants")
        variant = lock.get("selected_variant")
        alpha = lock.get("alpha")
        assert_frozen_evaluation_allowed(
            lock=lock, requested_variant=variant, requested_alpha=alpha
        )
        by_split = (formulas[key].get("by_split") or {})
        report = {
            "locked_formula": {
                "variant": variant,
                "alpha": alpha,
                "lock_sha256": lock.get("lock_sha256"),
            },
            "development70": by_split.get("development70"),
            "previously_reserved85": by_split.get("previously_reserved85"),
            "all155": by_split.get("all155"),
            "retune_forbidden": True,
        }
        _write_json(out_dir / "frozen_formula_evaluation.json", report)
        meta = self._base_run_metadata("evaluate_frozen_formula")
        meta["report"] = {
            "variant": variant,
            "alpha": alpha,
            "has_reserved85": by_split.get("previously_reserved85") is not None,
        }
        _write_json(out_dir / "run_metadata.json", meta)
        return meta

    def run_catalog_coverage(self) -> Dict[str, Any]:
        out_dir = self.output_dir / "catalog_coverage"
        out_dir.mkdir(parents=True, exist_ok=True)
        # Prefer question-level rankings if present; else step rankings.
        rows: List[Dict[str, Any]] = []
        qpath = self.output_dir / "question_level" / "per_query_rankings.jsonl"
        spath = self.output_dir / "ranking_variants" / "per_step_rankings.jsonl"
        for path in (qpath, spath):
            if path.exists():
                for line in path.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        rows.append(json.loads(line))
        cov = analyze_catalog_coverage(
            formal_gold_ids=self.catalog["formal_gold_eligible_ids"],
            distractor_ids=self.catalog["distractor_ids"],
            excluded_ids=self.catalog["excluded_ids"],
            per_query_rankings=rows,
        )
        _write_json(out_dir / "operation_coverage.json", cov)
        _write_json(out_dir / "distractor_analysis.json", cov["distractors"])
        _write_json(
            out_dir / "operation_documents.json",
            {
                "operation_documents_sha256": self.docs["operation_documents_sha256"],
                "integrity": self.docs["integrity"],
                "n_documents": self.docs["n_documents"],
                "catalog_fingerprint": self.catalog["catalog_fingerprint"],
                "embedding_model": self.index.embedding_model,
                "embedding_dim": self.index.embedding_dim,
                "stats": self.catalog["stats"],
            },
        )
        meta = self._base_run_metadata("catalog_coverage")
        _write_json(out_dir / "run_metadata.json", meta)
        _write_json(out_dir / "error_summary.json", {"errors": []})
        return meta

    def run_report_all(self) -> Dict[str, Any]:
        out_dir = self.output_dir / "all155_report"
        out_dir.mkdir(parents=True, exist_ok=True)
        report: Dict[str, Any] = {
            "audit_version": AUDIT_VERSION,
            "result_role": RESULT_ROLE,
            "dataset": self.dataset["identities"],
            "splits": {
                "development70": len(self.dataset["splits"]["development70"]),
                "previously_reserved85": len(
                    self.dataset["splits"]["previously_reserved85"]
                ),
                "all155": len(self.dataset["splits"]["all155"]),
            },
            "catalog_stats": self.catalog["stats"],
            "artifacts_present": {
                "question_level": (self.output_dir / "question_level").exists(),
                "planner_cache": (self.output_dir / "planner_cache").exists(),
                "ranking_variants": (self.output_dir / "ranking_variants").exists(),
                "formula_selection": (self.output_dir / "formula_selection").exists(),
                "reserved_evaluation": (self.output_dir / "reserved_evaluation").exists(),
                "catalog_coverage": (self.output_dir / "catalog_coverage").exists(),
            },
            "notes": [
                "exploratory/descriptive only",
                "no unseen held-out claim",
                "gold/accepted ops used only in offline scoring",
            ],
        }
        # Attach available metrics snapshots.
        for rel in (
            "question_level/retrieval_metrics.json",
            "ranking_variants/retrieval_metrics.json",
            "formula_selection/formula_comparison.json",
            "reserved_evaluation/frozen_formula_evaluation.json",
            "catalog_coverage/operation_coverage.json",
        ):
            path = self.output_dir / rel
            if path.exists():
                report[rel.replace("/", "__")] = json.loads(path.read_text(encoding="utf-8"))
        _write_json(out_dir / "all155_report.json", report)
        meta = self._base_run_metadata("report_all")
        _write_json(out_dir / "run_metadata.json", meta)
        _write_json(self.output_dir / "run_metadata.json", meta)
        _write_json(self.output_dir / "error_summary.json", {"errors": []})
        return meta
