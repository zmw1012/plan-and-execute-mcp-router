"""Orchestrate Retrieval Audit modes (mock / dry-run / live / offline-rank)."""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

from hk_equity_mcp.experiment.pe_fin_lite_retrieval_audit import (
    AUDIT_VERSION,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_STEP_RANK_VARIANTS,
    DESCRIBE_N,
    HYBRID_ALPHAS,
    RESULT_ROLE,
    TOP_K,
)
from hk_equity_mcp.experiment.pe_fin_lite_retrieval_audit.catalog_audit import (
    analyze_catalog_coverage,
)
from hk_equity_mcp.experiment.pe_fin_lite_retrieval_audit.embedding_identity_gate import (
    REAL_TEXT_EMBEDDING_DIM,
    assert_query_catalog_alignment,
    assert_real_backend_identity,
    classify_catalog_source,
    list_catalog_cache_inventory,
    resolve_embedding_meta_for_offline,
    select_matching_catalog_cache,
)
from hk_equity_mcp.experiment.pe_fin_lite_retrieval_audit.formula_selection import (
    REAL_EMBEDDING_BACKEND,
    assert_formal_formula_lock_allowed,
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
from hk_equity_mcp.experiment.pe_fin_lite_retrieval_audit.live_path import (
    AuditMode,
    LivePathError,
    ResolvedMode,
    build_embedding_backend,
    build_generation_llm,
    load_and_assert_embedding,
    load_and_assert_generation,
    public_meta_without_secrets,
    require_live_env_files,
    resolve_cache_dir,
    resolve_mode,
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
from hk_equity_mcp.experiment.pe_fin_lite_retrieval_audit.planner_cache_gate import (
    assert_planner_cache_for_step_pipeline,
    audit_planner_cache_zero_network,
)
from hk_equity_mcp.experiment.pe_fin_lite_retrieval_audit.planner_only import (
    run_planner_only,
)
from hk_equity_mcp.experiment.pe_fin_lite_retrieval_audit.query_cache import (
    QueryCacheError,
    embed_planner_step_queries,
    embed_question_queries,
    load_complete_question_query_cache,
    load_complete_step_query_cache,
    question_query_cache_identity,
    step_cache_key,
    step_query_cache_identity,
)
from hk_equity_mcp.experiment.pe_fin_lite_retrieval_audit.ranking import (
    OfflineEmbeddingIndex,
    build_question_query,
    build_step_query,
    rank_operations,
)
from hk_equity_mcp.pe_fin_lite.capability_profile import (
    capability_profile_hash,
    load_capability_profile,
)
from hk_equity_mcp.pe_fin_lite.identity import (
    EXPECTED_EMBEDDING_MODEL,
    EXPECTED_GENERATION_HOST,
    EXPECTED_GENERATION_MODEL,
    EXPECTED_GENERATION_PROVIDER,
    host_from_base_url,
)
from hk_equity_mcp.pe_fin_lite.operation_embedding_index import (
    IncompleteOperationIndexError,
    build_index_identity,
    build_or_resume_index,
    identity_digest,
    index_status,
    load_complete_index,
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
        mode: Optional[ResolvedMode] = None,
        mock: bool = False,
        dry_run: bool = False,
        live: bool = False,
        no_mock: bool = False,
        offline_rank: bool = False,
        env_file: Optional[str] = None,
        embedding_env_file: Optional[str] = None,
        embedding_cache_dir: Optional[str | Path] = None,
        require_complete_index: bool = False,
        allow_incomplete_planner_cache: bool = False,
        planner_diagnostic10: bool = False,
        allow_network: bool = False,
    ):
        if allow_network and not live:
            raise LivePathError(
                "allow_network is removed; use explicit --live (fail-closed)"
            )
        self.root = Path(root)
        self.planner_diagnostic10 = bool(planner_diagnostic10)
        default_out = DEFAULT_OUTPUT_DIR
        if self.planner_diagnostic10:
            from hk_equity_mcp.experiment.pe_fin_lite_retrieval_audit import (
                DEFAULT_PLANNER_DIAGNOSTIC10_OUTPUT_DIR,
            )

            default_out = DEFAULT_PLANNER_DIAGNOSTIC10_OUTPUT_DIR
        self.output_dir = Path(output_dir or (self.root / default_out))
        self.resolved = mode or resolve_mode(
            mock=mock,
            dry_run=dry_run,
            live=live or allow_network,
            no_mock=no_mock,
            offline_rank=offline_rank,
        )
        self.mock = self.resolved.mock
        self.dry_run = self.resolved.dry_run
        self.live = self.resolved.live
        self.allow_network = self.resolved.allow_network
        self.env_file = env_file
        self.embedding_env_file = embedding_env_file
        self.embedding_cache_dir = resolve_cache_dir(self.root, embedding_cache_dir)
        self.require_complete_index = bool(require_complete_index)
        self.allow_incomplete_planner_cache = bool(allow_incomplete_planner_cache)
        self.gen_meta: Optional[Dict[str, Any]] = None
        self.emb_meta: Optional[Dict[str, Any]] = None
        self._index_identity: Optional[Dict[str, Any]] = None
        self._catalog_vectors: Optional[Dict[str, List[float]]] = None
        self._selected_catalog_cache: Optional[Dict[str, Any]] = None
        self._llm = None

        self.dataset = load_hk155_locked(self.root)
        self.catalog = build_catalog_context(self.root)
        self.docs = build_all_operation_documents(self.catalog["bundle"])
        self.meta_by_id = _metadata_by_id(self.catalog["bundle"])
        self.tasks_by_id = _task_map(self.dataset["tasks"])
        self.registry = self.catalog["bundle"].registry

        # Mock hash index only for mock mode ranking that embeds on the fly.
        # Live/offline ranking uses preloaded catalog + query caches.
        if self.mock:
            self.index = OfflineEmbeddingIndex(
                documents=self.docs["documents"],
                metadata_by_id=self.meta_by_id,
                runtime_operation_ids=self.catalog["runtime_operation_ids"],
                registry=self.registry,
            )
            self.embedding_backend_kind = "mock_hash"
        else:
            self.index = None  # constructed from caches when ranking
            self.embedding_backend_kind = (
                REAL_EMBEDDING_BACKEND
                if self.live or self.resolved.mode == AuditMode.OFFLINE_RANK
                else "none"
            )
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _base_run_metadata(self, mode: str) -> Dict[str, Any]:
        emb_model = (
            self.index.embedding_model
            if self.index is not None
            else (
                (self.emb_meta or {}).get("model")
                if self.emb_meta
                else (
                    EXPECTED_EMBEDDING_MODEL
                    if not self.mock
                    else "mock-hash-embedding-v1"
                )
            )
        )
        emb_dim = self.index.embedding_dim if self.index is not None else None
        meta = {
            "audit_version": AUDIT_VERSION,
            "result_role": RESULT_ROLE,
            "mode": mode,
            "audit_mode": self.resolved.mode.value,
            "mock": self.mock,
            "dry_run": self.dry_run,
            "live": self.live,
            "allow_network": bool(self.allow_network),
            "embedding_backend": self.embedding_backend_kind,
            "dataset": self.dataset["identities"],
            "catalog_stats": self.catalog["stats"],
            "catalog_fingerprint": self.catalog["catalog_fingerprint"],
            "operation_documents_sha256": self.docs["operation_documents_sha256"],
            "operation_document_integrity": self.docs["integrity"],
            "embedding_model": emb_model,
            "embedding_dim": emb_dim,
            "embedding_cache_dir": str(self.embedding_cache_dir),
            "top_k": TOP_K,
            "describe_n": DESCRIBE_N,
            "output_dir": str(self.output_dir),
            "no_unseen_heldout_claim": True,
        }
        if self.gen_meta:
            meta["generation"] = public_meta_without_secrets(self.gen_meta)
        if self.emb_meta:
            meta["embedding"] = public_meta_without_secrets(self.emb_meta)
        return meta

    def _catalog_index_identity(self) -> Dict[str, Any]:
        if self.mock:
            emb_model = "mock-hash-embedding-v1"
            emb_host = "local_hash"
        else:
            if self.emb_meta is None:
                raise LivePathError(
                    "emb_meta with non-null host required before catalog index "
                    "identity (pass --embedding-env-file; refuse host=null)"
                )
            emb_model = str(
                (self.emb_meta or {}).get("model") or EXPECTED_EMBEDDING_MODEL
            )
            emb_host = (
                (self.emb_meta or {}).get("host")
                or host_from_base_url((self.emb_meta or {}).get("base_url"))
            )
            if not emb_host:
                raise LivePathError(
                    "catalog index identity requires non-null embedding_provider_host "
                    "(fail-closed; no mock/hash selection)"
                )
            if emb_model != EXPECTED_EMBEDDING_MODEL and not self.dry_run:
                raise LivePathError(
                    f"embedding model locked to {EXPECTED_EMBEDDING_MODEL}"
                )
        expected_dim = None
        if self.emb_meta and self.emb_meta.get("expected_vector_dimension") is not None:
            expected_dim = int(self.emb_meta["expected_vector_dimension"])
        elif not self.mock:
            expected_dim = REAL_TEXT_EMBEDDING_DIM
        identity = build_index_identity(
            registry=self.registry,
            catalog_fingerprint=self.catalog["catalog_fingerprint"],
            embedding_model=emb_model,
            embedding_provider_host=emb_host,
            expected_vector_dimension=expected_dim,
        )
        self._index_identity = identity
        return identity

    def _ensure_real_embedding_meta_offline(
        self, *, query_cache_identity: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Load embedding identity for real offline stages (zero network)."""
        if self.mock:
            self.emb_meta = {
                "model": "mock-hash-embedding-v1",
                "base_url": None,
                "host": "local_hash",
                "provider": "mock_hash",
                "identity_source": "mock_mode",
                "expected_vector_dimension": 64,
            }
            return self.emb_meta
        meta = resolve_embedding_meta_for_offline(
            root=self.root,
            embedding_env_file=self.embedding_env_file,
            query_cache_identity=query_cache_identity,
        )
        self.emb_meta = meta
        return meta

    def get_index_status(self) -> Dict[str, Any]:
        """Zero-network catalog index status / inventory.

        Without ``--embedding-env-file``: inventory-only (does not claim a formal
        selected identity). With env file: select exact real identity and report
        completeness for that digest only.
        """
        out_dir = self.output_dir / "catalog_index"
        out_dir.mkdir(parents=True, exist_ok=True)
        inventory = list_catalog_cache_inventory(self.embedding_cache_dir)
        _write_json(out_dir / "cache_inventory.json", {"caches": inventory})

        selected: Optional[Dict[str, Any]] = None
        status_payload: Dict[str, Any]
        if self.mock:
            self._ensure_real_embedding_meta_offline()
            identity = self._catalog_index_identity()
            status = index_status(
                registry=self.registry,
                identity=identity,
                cache_dir=self.embedding_cache_dir,
            ).to_dict()
            status_payload = {
                **status,
                "final_complete": bool(status.get("complete")),
                "expected": status.get("total"),
                "identity_digest": identity_digest(identity),
                "source": "mock_hash",
                "selection_mode": "mock",
                "zero_network": True,
                "cache_inventory": inventory,
            }
        elif self.embedding_env_file:
            self._ensure_real_embedding_meta_offline()
            identity = self._catalog_index_identity()
            digest = identity_digest(identity)
            status = index_status(
                registry=self.registry,
                identity=identity,
                cache_dir=self.embedding_cache_dir,
            ).to_dict()
            dim = status.get("vector_dimension")
            host = identity.get("embedding_provider_host")
            source = classify_catalog_source(
                embedding_provider_host=host,
                vector_dimension=dim,
                embedding_model=identity.get("embedding_model"),
            )
            assert_real_backend_identity(
                embedding_backend=REAL_EMBEDDING_BACKEND,
                embedding_provider_host=host,
                vector_dimension=dim,
                source=source,
            )
            if self.require_complete_index and not status.get("complete"):
                raise LivePathError(
                    f"--require-complete-index: real catalog identity {digest} incomplete"
                )
            if self.require_complete_index and source != "real":
                raise LivePathError(
                    f"--require-complete-index requires real catalog source; got {source}"
                )
            selected = {
                "identity_digest": digest,
                "embedding_provider_host": host,
                "embedding_model": identity.get("embedding_model"),
                "vector_dimension": dim,
                "documents_hash": identity.get("documents_hash"),
                "registry_fingerprint": identity.get("registry_fingerprint"),
                "complete": bool(status.get("complete")),
                "source": source,
            }
            status_payload = {
                **status,
                "final_complete": bool(status.get("complete")),
                "expected": status.get("total"),
                "identity_digest": digest,
                "embedding_provider_host": host,
                "embedding_model": identity.get("embedding_model"),
                "vector_dimension": dim,
                "documents_hash": identity.get("documents_hash"),
                "source": source,
                "selection_mode": "embedding_env_file",
                "selected": selected,
                "zero_network": True,
                "cache_inventory": inventory,
            }
        else:
            # Inventory only — do not silently select host=null mock cache.
            status_payload = {
                "namespace": "pe_fin_lite_operation_embeddings_v1",
                "cache_dir": str(self.embedding_cache_dir),
                "selection_mode": "inventory_only",
                "selected": None,
                "note": (
                    "No --embedding-env-file: listed all catalog caches only. "
                    "Refusing to claim host=null/64-d mock cache as formal index."
                ),
                "cache_inventory": inventory,
                "zero_network": True,
            }
            if self.require_complete_index:
                raise LivePathError(
                    "--require-complete-index requires --embedding-env-file to select "
                    "the real catalog identity (fail-closed)"
                )

        _write_json(out_dir / "index_status.json", status_payload)
        meta = self._base_run_metadata("index_status")
        meta["index_status"] = {
            "selection_mode": status_payload.get("selection_mode"),
            "selected": selected,
            "n_inventory": len(inventory),
            "identity_sha256": (selected or {}).get("identity_digest"),
            "zero_network": True,
        }
        if self.emb_meta:
            meta["embedding"] = public_meta_without_secrets(self.emb_meta)
        _write_json(out_dir / "run_metadata.json", meta)
        return status_payload

    def _load_complete_catalog_index(
        self, *, query_cache_identity: Optional[Dict[str, Any]] = None
    ) -> OfflineEmbeddingIndex:
        if self.mock:
            assert self.index is not None
            return self.index
        self._ensure_real_embedding_meta_offline(
            query_cache_identity=query_cache_identity
        )
        identity = self._catalog_index_identity()
        digest = identity_digest(identity)
        inventory = list_catalog_cache_inventory(self.embedding_cache_dir)
        expected_dim = int(
            (self.emb_meta or {}).get("expected_vector_dimension")
            or (
                (query_cache_identity or {}).get("embedding_dim")
                if query_cache_identity
                else REAL_TEXT_EMBEDDING_DIM
            )
            or REAL_TEXT_EMBEDDING_DIM
        )
        chosen = select_matching_catalog_cache(
            inventory=inventory,
            expected_identity=identity,
            expected_digest=digest,
            require_real=True,
            require_dim=expected_dim,
        )
        assert_real_backend_identity(
            embedding_backend=REAL_EMBEDDING_BACKEND,
            embedding_provider_host=chosen.get("embedding_provider_host"),
            vector_dimension=chosen.get("vector_dimension"),
            source=chosen.get("source"),
        )
        try:
            vectors = load_complete_index(
                registry=self.registry,
                identity=identity,
                cache_dir=self.embedding_cache_dir,
            )
        except IncompleteOperationIndexError as exc:
            raise LivePathError(
                f"Incomplete catalog embedding index (fail-closed): {exc}"
            ) from exc
        self._catalog_vectors = vectors
        dim = len(next(iter(vectors.values()))) if vectors else None
        if dim != expected_dim:
            raise LivePathError(
                f"Loaded catalog dim={dim} != expected_dim={expected_dim} "
                f"(identity={digest}; fail-closed)"
            )
        if query_cache_identity:
            assert_query_catalog_alignment(
                query_dim=int(query_cache_identity.get("embedding_dim") or expected_dim),
                catalog_dim=int(dim or 0),
                query_host=query_cache_identity.get("embedding_provider_host"),
                catalog_host=chosen.get("embedding_provider_host"),
                query_model=query_cache_identity.get("embedding_model"),
                catalog_model=chosen.get("embedding_model"),
            )
        self.embedding_backend_kind = REAL_EMBEDDING_BACKEND
        index = OfflineEmbeddingIndex.from_preloaded_vectors(
            documents=self.docs["documents"],
            metadata_by_id=self.meta_by_id,
            runtime_operation_ids=self.catalog["runtime_operation_ids"],
            vectors=vectors,
            embedding_model=str(
                (self.emb_meta or {}).get("model") or EXPECTED_EMBEDDING_MODEL
            ),
            registry=self.registry,
        )
        index.embedding_dim = dim
        index.embedding_provider_host = chosen.get("embedding_provider_host")
        index.catalog_identity_sha256 = digest
        index.catalog_source = chosen.get("source")
        self.index = index
        self._index_identity = identity
        self._selected_catalog_cache = chosen
        return index

    def build_index_only(self) -> Dict[str, Any]:
        """Live embedding only: build/resume catalog index. No generation/planner."""
        if self.dry_run:
            meta = self._base_run_metadata("build_index_only")
            meta["dry_run_note"] = "Skipped embedding calls"
            _write_json(self.output_dir / "catalog_index" / "run_metadata.json", meta)
            return meta
        if not self.live and not self.mock:
            raise LivePathError(
                "build-index-only requires --live (or --mock for hash index)"
            )
        if self.live:
            require_live_env_files(
                env_file=self.env_file,
                embedding_env_file=self.embedding_env_file,
                need_generation=False,
                need_embedding=True,
            )
            self.emb_meta = load_and_assert_embedding(
                self.root, self.embedding_env_file  # type: ignore[arg-type]
            )
            self.embedding_backend_kind = REAL_EMBEDDING_BACKEND
        identity = self._catalog_index_identity()
        backend = build_embedding_backend(
            mode=self.resolved, emb_meta=self.emb_meta, for_index_build=True
        )
        result = build_or_resume_index(
            registry=self.registry,
            identity=identity,
            cache_dir=self.embedding_cache_dir,
            embed_fn=backend.embed,
            batch_size=16,
            ov_cache_dir=None,
        )
        out_dir = self.output_dir / "catalog_index"
        out_dir.mkdir(parents=True, exist_ok=True)
        _write_json(out_dir / "build_index_result.json", public_meta_without_secrets(result))
        meta = self._base_run_metadata("build_index_only")
        meta["build_index_result"] = {
            "completed": result.get("completed"),
            "total": result.get("total"),
            "complete": result.get("complete"),
            "action": result.get("action"),
            "identity_sha256": result.get("identity_sha256"),
        }
        _write_json(out_dir / "run_metadata.json", meta)
        return meta

    def embed_question_queries_stage(self) -> Dict[str, Any]:
        """Live: embed 155 question queries. No planner/MCP/ranking."""
        out_dir = self.output_dir / "question_query_cache"
        if self.dry_run:
            meta = self._base_run_metadata("embed_question_queries")
            meta["dry_run_note"] = "Skipped embedding"
            _write_json(out_dir / "run_metadata.json", meta)
            return meta
        if not self.live and not self.mock:
            raise LivePathError("embed-question-queries requires --live or --mock")
        if self.live:
            require_live_env_files(
                env_file=self.env_file,
                embedding_env_file=self.embedding_env_file,
                need_generation=False,
                need_embedding=True,
            )
            self.emb_meta = load_and_assert_embedding(
                self.root, self.embedding_env_file  # type: ignore[arg-type]
            )
            self.embedding_backend_kind = REAL_EMBEDDING_BACKEND
        backend = build_embedding_backend(
            mode=self.resolved, emb_meta=self.emb_meta, for_index_build=True
        )
        identity = question_query_cache_identity(
            dataset_identities=self.dataset["identities"],
            embedding_model=str(
                (self.emb_meta or {}).get("model")
                or getattr(backend, "model_name", EXPECTED_EMBEDDING_MODEL)
            ),
            embedding_provider_host=(
                host_from_base_url((self.emb_meta or {}).get("base_url"))
                if self.emb_meta
                else "local_hash"
            ),
            embedding_dim=None,
        )
        summary = embed_question_queries(
            tasks=self.dataset["tasks"],
            output_dir=out_dir,
            identity=identity,
            embed_fn=backend.embed,
            batch_size=16,
            resume=True,
        )
        meta = self._base_run_metadata("embed_question_queries")
        meta["question_query_cache"] = summary
        _write_json(out_dir / "run_metadata.json", meta)
        return meta

    def embed_planner_step_queries_stage(self) -> Dict[str, Any]:
        """Live: embed retrieve-step queries from frozen planner cache."""
        out_dir = self.output_dir / "step_query_cache"
        if self.dry_run:
            meta = self._base_run_metadata("embed_planner_step_queries")
            meta["dry_run_note"] = "Skipped embedding"
            _write_json(out_dir / "run_metadata.json", meta)
            return meta
        if not self.live and not self.mock:
            raise LivePathError("embed-planner-step-queries requires --live or --mock")
        cache = self._require_planner_for_step_stage(
            stage="embed_planner_step_queries", formal=False
        )
        if self.live:
            require_live_env_files(
                env_file=self.env_file,
                embedding_env_file=self.embedding_env_file,
                need_generation=False,
                need_embedding=True,
            )
            self.emb_meta = load_and_assert_embedding(
                self.root, self.embedding_env_file  # type: ignore[arg-type]
            )
            self.embedding_backend_kind = REAL_EMBEDDING_BACKEND
        backend = build_embedding_backend(
            mode=self.resolved, emb_meta=self.emb_meta, for_index_build=True
        )
        identity = step_query_cache_identity(
            dataset_identities=self.dataset["identities"],
            planner_cache_sha256=cache["planner_cache_sha256"],
            embedding_model=str(
                (self.emb_meta or {}).get("model")
                or getattr(backend, "model_name", EXPECTED_EMBEDDING_MODEL)
            ),
            embedding_provider_host=(
                host_from_base_url((self.emb_meta or {}).get("base_url"))
                if self.emb_meta
                else "local_hash"
            ),
            embedding_dim=None,
        )
        summary = embed_planner_step_queries(
            planner_rows=cache["rows"],
            output_dir=out_dir,
            identity=identity,
            embed_fn=backend.embed,
            batch_size=16,
            resume=True,
        )
        meta = self._base_run_metadata("embed_planner_step_queries")
        meta["step_query_cache"] = summary
        meta["planner_cache_sha256"] = cache["planner_cache_sha256"]
        _write_json(out_dir / "run_metadata.json", meta)
        return meta

    def run_question_level(self) -> Dict[str, Any]:
        out_dir = self.output_dir / "question_level"
        out_dir.mkdir(parents=True, exist_ok=True)

        if self.dry_run:
            meta = self._base_run_metadata("question_level")
            meta["dry_run_note"] = "No rankings computed"
            _write_json(out_dir / "run_metadata.json", meta)
            _write_json(out_dir / "error_summary.json", {"errors": []})
            return meta

        if self.live:
            raise LivePathError(
                "question-level ranking is zero-network offline; "
                "use --embed-question-queries --live first, then --question-level "
                "without --live"
            )

        query_vectors: Optional[Dict[str, List[float]]] = None
        query_dim: Optional[int] = None
        catalog_dim: Optional[int] = None
        alignment: Dict[str, bool] = {}
        if self.mock:
            index = self.index
            assert index is not None
            emb_model = index.embedding_model
            emb_host = "local_hash"
            self.embedding_backend_kind = "mock_hash"
            catalog_dim = index.embedding_dim
            query_dim = catalog_dim
            alignment = {
                "identity_match": True,
                "dimension_match": True,
                "host_match": True,
                "model_match": True,
            }
        else:
            # Offline ranking from complete caches — catalog must match query identity.
            qpath = out_dir.parent / "question_query_cache" / "question_query_vectors.json"
            try:
                qcache = load_complete_question_query_cache(qpath)
            except QueryCacheError as exc:
                raise LivePathError(
                    f"Incomplete question-query cache (fail-closed): {exc}"
                ) from exc
            q_ident_raw = dict(qcache.get("identity") or {})
            query_vectors = {
                str(k): list(v)
                for k, v in (qcache.get("vectors_by_task_id") or {}).items()
            }
            if query_vectors:
                query_dim = len(next(iter(query_vectors.values())))
                q_ident_raw["embedding_dim"] = query_dim
            emb_model = str(
                q_ident_raw.get("embedding_model") or EXPECTED_EMBEDDING_MODEL
            )
            emb_host = str(q_ident_raw.get("embedding_provider_host") or "").strip()
            if not emb_host:
                raise LivePathError(
                    "question query cache missing embedding_provider_host "
                    "(fail-closed; refuse host=null catalog)"
                )
            index = self._load_complete_catalog_index(
                query_cache_identity=q_ident_raw
            )
            catalog_dim = index.embedding_dim
            alignment = assert_query_catalog_alignment(
                query_dim=int(query_dim or 0),
                catalog_dim=int(catalog_dim or 0),
                query_host=emb_host,
                catalog_host=getattr(index, "embedding_provider_host", None),
                query_model=emb_model,
                catalog_model=index.embedding_model,
            )
            assert_real_backend_identity(
                embedding_backend=REAL_EMBEDDING_BACKEND,
                embedding_provider_host=getattr(
                    index, "embedding_provider_host", emb_host
                ),
                vector_dimension=catalog_dim,
                source=getattr(index, "catalog_source", None),
            )
            self.embedding_backend_kind = REAL_EMBEDDING_BACKEND

        q_identity = question_level_cache_identity(
            dataset_identities=self.dataset["identities"],
            operation_documents_sha256=self.docs["operation_documents_sha256"],
            catalog_fingerprint=self.catalog["catalog_fingerprint"],
            embedding_model=emb_model,
            embedding_provider_host=emb_host,
        )
        _write_json(out_dir / "cache_identity.json", q_identity)

        rankings: List[Dict[str, Any]] = []
        scored_tool: List[Dict[str, Any]] = []
        scored_disclosure: List[Dict[str, Any]] = []
        no_tool_rows: List[Dict[str, Any]] = []
        errors: List[Dict[str, Any]] = []

        for task in self.dataset["tasks"]:
            tid = str(task.get("task_id") or task.get("id"))
            ref = build_evaluation_reference(task)
            query = build_question_query(str(task.get("question") or ""))
            try:
                qvec = None if query_vectors is None else query_vectors.get(tid)
                if query_vectors is not None and qvec is None:
                    raise LivePathError(f"Missing question query vector for {tid}")
                ranked = rank_operations(
                    index=index,
                    query=query,
                    variant="R0",
                    step_domain=str(task.get("domain") or "general"),
                    required_outputs=[],
                    top_k=TOP_K,
                    gold_ops_for_filter_diag=ref["acceptable_ops_union"],
                    query_vector=qvec,
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
                "embedding_backend": self.embedding_backend_kind,
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
            "embedding_backend": self.embedding_backend_kind,
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
        meta["zero_network"] = True
        meta["mock"] = bool(self.mock)
        meta["query_identity"] = {
            "embedding_model": emb_model,
            "embedding_provider_host": emb_host,
            "embedding_dim": query_dim,
            "identity_sha256": (q_identity or {}).get("identity_sha256"),
        }
        meta["catalog_identity"] = {
            "identity_sha256": getattr(index, "catalog_identity_sha256", None)
            or (
                identity_digest(self._index_identity)
                if self._index_identity
                else None
            ),
            "embedding_provider_host": getattr(
                index, "embedding_provider_host", emb_host
            ),
            "embedding_model": getattr(index, "embedding_model", emb_model),
            "vector_dimension": catalog_dim,
            "source": getattr(index, "catalog_source", None)
            or ("mock_hash" if self.mock else "real"),
            "selected_cache": self._selected_catalog_cache,
        }
        meta["query_dimension"] = query_dim
        meta["catalog_dimension"] = catalog_dim
        meta["embedding_dim"] = catalog_dim
        meta["identity_match"] = bool(alignment.get("identity_match"))
        meta["dimension_match"] = bool(alignment.get("dimension_match"))
        if self.emb_meta:
            meta["embedding"] = public_meta_without_secrets(self.emb_meta)
        _write_json(out_dir / "run_metadata.json", meta)
        _write_json(out_dir / "error_summary.json", {"errors": errors})
        return meta

    def run_planner_only(self) -> Dict[str, Any]:
        out_dir = self.output_dir / "planner_cache"
        profile = load_capability_profile(self.root)

        if self.live:
            require_live_env_files(
                env_file=self.env_file,
                embedding_env_file=self.embedding_env_file,
                need_generation=True,
                need_embedding=False,
            )
            self.gen_meta = load_and_assert_generation(
                self.root, self.env_file  # type: ignore[arg-type]
            )
            self._llm = build_generation_llm(json_mode=True)
            gen_provider = EXPECTED_GENERATION_PROVIDER
            gen_host = EXPECTED_GENERATION_HOST
            gen_model = EXPECTED_GENERATION_MODEL
        elif self.mock or self.dry_run:
            gen_provider = "offline_mock"
            gen_host = "local"
            gen_model = "mock_plan_v1"
            self._llm = None
        else:
            raise LivePathError(
                "planner-only without --mock/--dry-run requires --live "
                "(fail-closed; no silent mock)"
            )

        from hk_equity_mcp.experiment.pe_fin_lite_retrieval_audit import (
            PLANNER_ONLY_PROTOCOL_VERSION,
        )
        from hk_equity_mcp.experiment.pe_fin_lite_retrieval_audit.planner_diagnostic10 import (
            build_planner_protocol_diagnostic,
            planner_diagnostic10_cache_identity,
            select_diagnostic10_tasks,
        )
        from hk_equity_mcp.pe_fin_lite.identity import combined_schema_hash
        from hk_equity_mcp.pe_fin_lite.output_vocabulary import scope_contract_hash
        from hk_equity_mcp.pe_fin_lite.plan_canonicalizer import plan_canonicalizer_hash
        from hk_equity_mcp.pe_fin_lite.prompts import pe_fin_lite_prompt_hash
        from hk_equity_mcp.pe_fin_lite.schemas import PLANNER_MAX_TOKENS

        if self.planner_diagnostic10:
            selected = select_diagnostic10_tasks(
                root=self.root, all_tasks=self.dataset["tasks"]
            )
            tasks = selected["tasks"]
            identity = planner_diagnostic10_cache_identity(
                diagnostic_manifest=selected["manifest"],
                planner_prompt_hash=pe_fin_lite_prompt_hash(),
                planner_schema_hash=combined_schema_hash(),
                plan_canonicalizer_hash=plan_canonicalizer_hash(),
                capability_profile_hash=capability_profile_hash(profile),
                generation_provider=gen_provider,
                generation_host=gen_host,
                generation_model=gen_model,
                planner_only_protocol_version=PLANNER_ONLY_PROTOCOL_VERSION,
                max_tokens=int(PLANNER_MAX_TOKENS),
                scope_contract_hash=scope_contract_hash(),
            )
            mode_name = "planner_only_diagnostic10"
        else:
            tasks = self.dataset["tasks"]
            identity = planner_only_cache_identity(
                dataset_identities=self.dataset["identities"],
                capability_profile_hash=capability_profile_hash(profile),
                generation_provider=gen_provider,
                generation_host=gen_host,
                generation_model=gen_model,
                max_tokens=int(PLANNER_MAX_TOKENS),
                scope_contract_hash=scope_contract_hash(),
            )
            selected = None
            mode_name = "planner_only"

        summary = run_planner_only(
            root=self.root,
            tasks=tasks,
            output_dir=out_dir,
            cache_identity=identity,
            llm=self._llm,
            mock=self.mock,
            dry_run=self.dry_run,
            live=self.live,
            resume=True,
            capability_profile=profile,
        )
        meta = self._base_run_metadata(mode_name)
        meta["planner_diagnostic10"] = bool(self.planner_diagnostic10)
        meta["explicitly_not_formal_accuracy"] = bool(self.planner_diagnostic10)
        meta["planner_only_summary"] = {
            k: v
            for k, v in summary.items()
            if k != "identity" or isinstance(v, dict)
        }
        # Ensure secrets never land in metadata.
        meta = public_meta_without_secrets(meta)
        _write_json(out_dir / "run_metadata.json", meta)
        _write_json(out_dir / "error_summary.json", {"errors": summary.get("errors") or []})

        if self.planner_diagnostic10 and selected is not None:
            # Reload rows for diagnostic metrics (includes resumed prior rows).
            rows = []
            out_path = out_dir / "planner_outputs.jsonl"
            if out_path.exists():
                for line in out_path.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        rows.append(json.loads(line))
            diagnostic = build_planner_protocol_diagnostic(
                rows=rows,
                manifest=selected["manifest"],
                manifest_task_meta_by_id=selected["manifest_task_meta_by_id"],
                cache_identity=identity,
            )
            if self.dry_run or self.mock:
                # Do not fabricate a live protocol pass under mock/dry-run.
                diagnostic["dry_run"] = bool(self.dry_run)
                diagnostic["mock"] = bool(self.mock)
                diagnostic["engineering_pass"] = False
                diagnostic["engineering_pass_criteria"] = {
                    **(diagnostic.get("engineering_pass_criteria") or {}),
                    "live_backend": False,
                }
                diagnostic["structural_complete_under_current_mode"] = diagnostic.get(
                    "complete_for_diagnostic10"
                )
                diagnostic["note"] = (
                    "mock/dry-run cannot satisfy live engineering_pass; "
                    "structural completeness is recorded separately. "
                    + str(diagnostic.get("note") or "")
                )
            _write_json(out_dir / "planner_protocol_diagnostic.json", diagnostic)
            meta["planner_protocol_diagnostic"] = {
                "complete_for_diagnostic10": diagnostic.get("complete_for_diagnostic10"),
                "ok": diagnostic.get("ok"),
                "failed": diagnostic.get("failed"),
                "repair_rate": diagnostic.get("repair_rate"),
                "engineering_pass": diagnostic.get("engineering_pass"),
                "explicitly_not_formal_accuracy": True,
            }
            _write_json(out_dir / "run_metadata.json", public_meta_without_secrets(meta))
        return meta

    def _load_planner_cache(self) -> Dict[str, Any]:
        from hk_equity_mcp.experiment.pe_fin_lite_retrieval_audit.planner_cache_gate import (
            summarize_planner_rows,
        )

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
        summary = summarize_planner_rows(rows)
        return {
            "rows": rows,
            "planner_cache_sha256": cache_sha,
            "path": str(path),
            "completeness": summary,
        }

    def _require_planner_for_step_stage(self, *, stage: str, formal: bool = False) -> Dict[str, Any]:
        identity_path = (
            self.output_dir / "planner_cache" / "planner_cache_identity.json"
        )
        if identity_path.exists():
            ident = json.loads(identity_path.read_text(encoding="utf-8"))
            if ident.get("kind") == "planner_only_diagnostic10_cache":
                raise LivePathError(
                    f"{stage}: Diagnostic-10 planner cache cannot drive step "
                    "embedding / ranking / formal formula selection"
                )
        cache = self._load_planner_cache()
        # Mock offline pipeline may have 155 ok mock plans — still gate on failures.
        allow = bool(self.allow_incomplete_planner_cache) and not formal
        if self.mock and not formal:
            # Mock offline paths may still proceed; formal never.
            allow = True
        summary = assert_planner_cache_for_step_pipeline(
            rows=cache["rows"],
            allow_incomplete=allow,
            formal=formal,
            stage=stage,
        )
        cache["completeness"] = summary
        return cache

    def run_rank_from_planner_cache(
        self,
        *,
        variants: Optional[Sequence[str]] = None,
        alphas: Optional[Sequence[float]] = None,
    ) -> Dict[str, Any]:
        variants = list(variants or DEFAULT_STEP_RANK_VARIANTS)
        alphas = list(alphas or HYBRID_ALPHAS)
        out_dir = self.output_dir / "ranking_variants"
        out_dir.mkdir(parents=True, exist_ok=True)

        if self.live:
            raise LivePathError(
                "--rank-from-planner-cache is zero-network; do not pass --live. "
                "Use --embed-planner-step-queries --live first."
            )

        if self.dry_run:
            cache = {"rows": [], "planner_cache_sha256": "", "completeness": {}}
        else:
            cache = self._require_planner_for_step_stage(
                stage="rank_from_planner_cache", formal=False
            )
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
                    "embedding_backend": self.embedding_backend_kind,
                },
            )
            _write_json(out_dir / "retrieval_metrics.json", {"dry_run": True})
            _write_json(out_dir / "route_coverage.json", {"dry_run": True})
            _write_json(out_dir / "per_domain_metrics.json", {"dry_run": True})
            _write_jsonl(out_dir / "per_step_rankings.jsonl", [])
            _write_json(out_dir / "error_summary.json", {"errors": []})
            return meta

        step_vectors: Optional[Dict[str, Dict[str, Any]]] = None
        query_dim: Optional[int] = None
        catalog_dim: Optional[int] = None
        alignment: Dict[str, bool] = {}
        emb_host: Optional[str] = None
        if self.mock:
            index = self.index
            assert index is not None
            emb_model = index.embedding_model
            emb_host = "local_hash"
            self.embedding_backend_kind = "mock_hash"
            catalog_dim = index.embedding_dim
            query_dim = catalog_dim
            alignment = {
                "identity_match": True,
                "dimension_match": True,
                "host_match": True,
                "model_match": True,
            }
        else:
            spath = (
                self.output_dir / "step_query_cache" / "step_query_vectors.json"
            )
            try:
                scache = load_complete_step_query_cache(spath)
            except QueryCacheError as exc:
                raise LivePathError(
                    f"Incomplete step-query cache (fail-closed): {exc}"
                ) from exc
            step_vectors = dict(scache.get("vectors_by_task_step") or {})
            s_ident = dict(scache.get("identity") or {})
            # Infer dim from first step vector when present.
            if step_vectors and s_ident.get("embedding_dim") is None:
                first = next(iter(step_vectors.values()))
                vec = first.get("vector") if isinstance(first, dict) else first
                if isinstance(vec, list):
                    s_ident["embedding_dim"] = len(vec)
            query_dim = (
                int(s_ident["embedding_dim"])
                if s_ident.get("embedding_dim") is not None
                else None
            )
            emb_model = str(
                s_ident.get("embedding_model") or EXPECTED_EMBEDDING_MODEL
            )
            emb_host = str(s_ident.get("embedding_provider_host") or "").strip() or None
            if not emb_host:
                raise LivePathError(
                    "step query cache missing embedding_provider_host "
                    "(fail-closed; refuse host=null catalog)"
                )
            index = self._load_complete_catalog_index(query_cache_identity=s_ident)
            catalog_dim = index.embedding_dim
            alignment = assert_query_catalog_alignment(
                query_dim=int(query_dim or 0),
                catalog_dim=int(catalog_dim or 0),
                query_host=emb_host,
                catalog_host=getattr(index, "embedding_provider_host", None),
                query_model=emb_model,
                catalog_model=index.embedding_model,
            )
            assert_real_backend_identity(
                embedding_backend=REAL_EMBEDDING_BACKEND,
                embedding_provider_host=getattr(
                    index, "embedding_provider_host", emb_host
                ),
                vector_dimension=catalog_dim,
                source=getattr(index, "catalog_source", None),
            )
            self.embedding_backend_kind = REAL_EMBEDDING_BACKEND

        all_step_rankings: List[Dict[str, Any]] = []
        formula_metrics: Dict[str, Dict[str, Any]] = {}
        comparison_rows: Dict[str, Any] = {}

        for variant in variants:
            alpha_list: List[Optional[float]]
            if variant == "R3":
                alpha_list = list(alphas)
            else:
                alpha_list = [None]  # R1/R2/R_runtime: no alpha grid

            for alpha in alpha_list:
                formula_key = f"{variant}@{alpha}" if alpha is not None else variant
                identity = step_retrieval_cache_identity(
                    planner_cache_sha256=cache["planner_cache_sha256"],
                    operation_documents_sha256=self.docs["operation_documents_sha256"],
                    catalog_fingerprint=self.catalog["catalog_fingerprint"],
                    ranking_variant=variant,
                    alpha=alpha,
                    embedding_model=emb_model,
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
                        q = str(step.get("search_query") or build_step_query(step))
                        sid = str(step.get("id") or step.get("step_id") or "")
                        qvec = None
                        if step_vectors is not None:
                            entry = step_vectors.get(f"{tid}::{sid}")
                            if entry is None:
                                # Fallback key lookup via cache_key.
                                key = step_cache_key(tid, sid, q)
                                entry = next(
                                    (
                                        v
                                        for v in step_vectors.values()
                                        if v.get("cache_key") == key
                                    ),
                                    None,
                                )
                            if entry is None:
                                raise LivePathError(
                                    f"Missing step query vector for {tid}/{sid}"
                                )
                            qvec = entry.get("vector")
                        ranked = rank_operations(
                            index=index,
                            query=q,
                            variant=variant,
                            alpha=float(alpha) if alpha is not None else 1.0,
                            step_domain=str(
                                step.get("domain") or task.get("domain") or "general"
                            ),
                            required_outputs=list(step.get("required_outputs") or []),
                            top_k=TOP_K,
                            gold_ops_for_filter_diag=ref["acceptable_ops_union"],
                            query_vector=qvec,
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
                                "embedding_backend": self.embedding_backend_kind,
                            }
                        )
                    if not step_ranked:
                        metrics = score_ranked_list(
                            ranked_ids=[],
                            equivalents_by_node=[
                                set(x) for x in ref["equivalents_by_node"]
                            ],
                            acceptable_union=set(ref["acceptable_ops_union"]),
                        )
                    else:
                        metrics = score_multi_step_route(
                            step_ranked_ids=step_ranked,
                            equivalents_by_node=[
                                set(x) for x in ref["equivalents_by_node"]
                            ],
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
                    "embedding_backend": self.embedding_backend_kind,
                }
                comparison_rows[formula_key] = by_split.get("development70") or agg

        _write_jsonl(out_dir / "per_step_rankings.jsonl", all_step_rankings)
        _write_json(out_dir / "formula_comparison.json", {
            "formulas": formula_metrics,
            "development70_selection_inputs": comparison_rows,
            "embedding_backend": self.embedding_backend_kind,
            "zero_network": True,
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
        meta["zero_network"] = True
        meta["mock"] = bool(self.mock)
        meta["query_identity"] = {
            "embedding_model": emb_model,
            "embedding_provider_host": emb_host,
            "embedding_dim": query_dim,
        }
        meta["catalog_identity"] = {
            "identity_sha256": getattr(index, "catalog_identity_sha256", None)
            or (
                identity_digest(self._index_identity)
                if self._index_identity
                else None
            ),
            "embedding_provider_host": getattr(
                index, "embedding_provider_host", emb_host
            ),
            "embedding_model": getattr(index, "embedding_model", emb_model),
            "vector_dimension": catalog_dim,
            "source": getattr(index, "catalog_source", None)
            or ("mock_hash" if self.mock else "real"),
            "selected_cache": self._selected_catalog_cache,
        }
        meta["query_dimension"] = query_dim
        meta["catalog_dimension"] = catalog_dim
        meta["embedding_dim"] = catalog_dim
        meta["identity_match"] = bool(alignment.get("identity_match"))
        meta["dimension_match"] = bool(alignment.get("dimension_match"))
        if self.emb_meta:
            meta["embedding"] = public_meta_without_secrets(self.emb_meta)
        _write_json(out_dir / "run_metadata.json", meta)
        _write_json(out_dir / "error_summary.json", {"errors": []})
        _write_json(out_dir / "retrieval_metrics.json", {
            "formulas": {k: v["aggregate_all_tool"] for k, v in formula_metrics.items()},
            "embedding_backend": self.embedding_backend_kind,
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
        """Select formula using development-70 metrics only (zero network)."""
        if self.live:
            raise LivePathError("--select-formula is zero-network; do not pass --live")
        # Formal lock always requires a complete Planner cache in the same output tree.
        if formal and not self.dry_run:
            if self.planner_diagnostic10:
                raise LivePathError(
                    "Diagnostic-10 cannot generate formal formula lock "
                    "(engineering_protocol_diagnostic_only)"
                )
            identity_path = (
                self.output_dir / "planner_cache" / "planner_cache_identity.json"
            )
            if identity_path.exists():
                ident = json.loads(identity_path.read_text(encoding="utf-8"))
                if ident.get("kind") == "planner_only_diagnostic10_cache":
                    raise LivePathError(
                        "Diagnostic-10 planner cache cannot generate formal formula lock"
                    )
            planner_path = self.output_dir / "planner_cache" / "planner_outputs.jsonl"
            if planner_path.exists():
                self._require_planner_for_step_stage(
                    stage="select_formula_formal", formal=True
                )
            else:
                raise LivePathError(
                    "Formal formula selection requires complete planner_cache "
                    "(155 ok rows) under the same output directory"
                )
        out_dir = self.output_dir / "formula_selection"
        out_dir.mkdir(parents=True, exist_ok=True)
        cmp_path = self.output_dir / "ranking_variants" / "formula_comparison.json"
        if not cmp_path.exists():
            if self.dry_run:
                payload = {
                    "formulas": {},
                    "development70_selection_inputs": {},
                    "embedding_backend": self.embedding_backend_kind,
                }
            else:
                raise FileNotFoundError(
                    f"Missing {cmp_path}; run --rank-from-planner-cache first"
                )
        else:
            payload = json.loads(cmp_path.read_text(encoding="utf-8"))
        embedding_backend = str(
            payload.get("embedding_backend") or self.embedding_backend_kind
        )
        dev_inputs = payload.get("development70_selection_inputs") or {}
        comparison = compare_formulas(
            development_metrics_by_formula=dev_inputs, exclude_r0=True
        )
        comparison["embedding_backend"] = embedding_backend
        _write_json(out_dir / "formula_comparison.json", comparison)

        schema_only = {
            "lock_version": "retrieval_formula_lock_v2",
            "status": "schema_only_no_formal_selection",
            "note": (
                "Formal formula lock requires real text-embedding-3-small ranking "
                "caches on development-origin 70. Mock/hash results cannot formal-lock. "
                "R0 is excluded from Plan-first selection."
            ),
            "selection_policy": [
                "complete_route@3",
                "macro_recall@3",
                "minimum_per_domain_recall@3",
                "mrr",
                "simpler_formula",
                "larger_alpha",
            ],
            "selectable_variants": list(comparison.get("selectable_variants") or []),
            "r0_excluded_from_plan_first_selection": True,
            "no_retune_on_reserved85": True,
            "embedding_backend": embedding_backend,
            "mock_selection_preview": comparison.get("selected"),
            "runtime_sync_required": comparison.get("runtime_sync_required"),
        }
        _write_json(out_dir / "formula_lock_schema.json", schema_only)

        meta = self._base_run_metadata("select_formula")
        meta["formal_lock_written"] = False
        meta["embedding_backend"] = embedding_backend
        meta["zero_network"] = True
        meta["runtime_sync_required"] = comparison.get("runtime_sync_required")

        write_formal = bool(formal) and bool(comparison.get("selected"))
        if write_formal:
            try:
                assert_formal_formula_lock_allowed(
                    embedding_backend=embedding_backend,
                    mock=self.mock,
                )
            except RuntimeError as exc:
                meta["formal_lock_refused"] = str(exc)
                write_formal = False

        if write_formal:
            selected = comparison["selected"]
            planner_sha = None
            step_sha = None
            p_ident = self.output_dir / "planner_cache" / "planner_cache_identity.json"
            if p_ident.exists():
                planner_sha = json.loads(p_ident.read_text(encoding="utf-8")).get(
                    "identity_sha256"
                )
            s_ident = (
                self.output_dir / "step_query_cache" / "step_query_cache_identity.json"
            )
            if s_ident.exists():
                step_sha = json.loads(s_ident.read_text(encoding="utf-8")).get(
                    "identity_sha256"
                )
            emb_host = None
            emb_dim = None
            index_id = None
            if self._selected_catalog_cache:
                emb_host = self._selected_catalog_cache.get("embedding_provider_host")
                emb_dim = self._selected_catalog_cache.get("vector_dimension")
                index_id = self._selected_catalog_cache.get("identity_digest")
            elif self.index is not None:
                emb_host = getattr(self.index, "embedding_provider_host", None)
                emb_dim = getattr(self.index, "embedding_dim", None)
                index_id = getattr(self.index, "catalog_identity_sha256", None)
            lock = build_formula_lock(
                selected=selected,
                development_metrics=selected.get("metrics") or {},
                runtime_sync_required=bool(comparison.get("runtime_sync_required")),
                runtime_sync_specification=comparison.get(
                    "runtime_sync_specification"
                ),
                config={
                    "catalog_fingerprint": self.catalog["catalog_fingerprint"],
                    "operation_documents_sha256": self.docs[
                        "operation_documents_sha256"
                    ],
                    "embedding_model": EXPECTED_EMBEDDING_MODEL,
                    "embedding_provider_host": emb_host,
                    "embedding_dimension": emb_dim,
                    "embedding_backend": embedding_backend,
                    "operation_index_identity": index_id,
                    "planner_cache_sha256": planner_sha,
                    "step_query_cache_sha256": step_sha,
                    "dataset": self.dataset["identities"],
                },
            )
            write_formula_lock(out_dir / "formula_lock.json", lock)
            meta["formal_lock_written"] = True
            meta["selected"] = {
                "variant": lock.get("selected_variant"),
                "alpha": lock.get("alpha"),
                "lock_sha256": lock.get("lock_sha256"),
                "runtime_sync_required": lock.get("runtime_sync_required"),
            }
        _write_json(out_dir / "run_metadata.json", meta)
        return meta

    def run_planner_cache_audit(self) -> Dict[str, Any]:
        """Zero-network Planner-only gate after future live Planner-155."""
        if self.live:
            raise LivePathError(
                "--planner-cache-audit is zero-network; do not pass --live"
            )
        audit = audit_planner_cache_zero_network(
            output_dir=self.output_dir,
            expected_task_ids=self.dataset["task_ids"],
        )
        meta = self._base_run_metadata("planner_cache_audit")
        meta["zero_network"] = True
        meta["planner_cache_audit"] = {
            "complete_155_ok": audit.get("complete_155_ok"),
            "formal_step_pipeline_allowed": audit.get("formal_step_pipeline_allowed"),
            "ok": audit.get("ok"),
            "failed": audit.get("failed"),
            "gate_recommendation": audit.get("gate_recommendation"),
            "planner_outputs_sha256": audit.get("planner_outputs_sha256"),
        }
        _write_json(
            self.output_dir / "planner_cache_audit" / "run_metadata.json", meta
        )
        return audit

    def run_retry_failed_planner_cache(self) -> Dict[str, Any]:
        """Retry only ok=false Planner rows; freeze attempt1; atomic promote if complete.

        Live CLI path only. Does not call Embedding/MCP. Does not hardcode task IDs.
        """
        from hk_equity_mcp.experiment.pe_fin_lite_retrieval_audit.planner_cache_gate import (
            IncompletePlannerCacheError,
        )
        from hk_equity_mcp.experiment.pe_fin_lite_retrieval_audit.planner_cache_retry import (
            PlannerCacheRetryError,
            retry_failed_planner_cache,
        )
        from hk_equity_mcp.pe_fin_lite.identity import combined_schema_hash
        from hk_equity_mcp.pe_fin_lite.output_vocabulary import scope_contract_hash
        from hk_equity_mcp.pe_fin_lite.plan_canonicalizer import plan_canonicalizer_hash
        from hk_equity_mcp.pe_fin_lite.prompts import pe_fin_lite_prompt_hash
        from hk_equity_mcp.pe_fin_lite.schemas import PLANNER_MAX_TOKENS

        if self.planner_diagnostic10:
            raise LivePathError(
                "--retry-failed-planner-cache cannot combine with --planner-diagnostic10"
            )
        profile = load_capability_profile(self.root)

        if self.live:
            require_live_env_files(
                env_file=self.env_file,
                embedding_env_file=self.embedding_env_file,
                need_generation=True,
                need_embedding=False,
            )
            self.gen_meta = load_and_assert_generation(
                self.root, self.env_file  # type: ignore[arg-type]
            )
            self._llm = build_generation_llm(json_mode=True)
            gen_provider = EXPECTED_GENERATION_PROVIDER
            gen_host = EXPECTED_GENERATION_HOST
            gen_model = EXPECTED_GENERATION_MODEL
        elif self.mock or self.dry_run:
            self._llm = None
            # Expected identity must match the on-disk planner_cache_identity.
            # For dry-run against a live cache, rebuild from the frozen live identity
            # fields below after load; mock tests supply matching offline identity.
            gen_provider = EXPECTED_GENERATION_PROVIDER
            gen_host = EXPECTED_GENERATION_HOST
            gen_model = EXPECTED_GENERATION_MODEL
        else:
            raise LivePathError(
                "--retry-failed-planner-cache requires --live "
                "(or --mock/--dry-run for offline tests only)"
            )

        # Prefer on-disk identity as source of truth for generation fields when present,
        # then assert it matches current prompt/schema/canonicalizer hashes.
        identity_path = (
            self.output_dir / "planner_cache" / "planner_cache_identity.json"
        )
        if identity_path.exists():
            on_disk = json.loads(identity_path.read_text(encoding="utf-8"))
            gen_provider = str(
                on_disk.get("generation_provider") or gen_provider
            )
            gen_host = str(on_disk.get("generation_host") or gen_host)
            gen_model = str(on_disk.get("generation_model") or gen_model)

        expected = planner_only_cache_identity(
            dataset_identities=self.dataset["identities"],
            planner_prompt_hash=pe_fin_lite_prompt_hash(),
            planner_schema_hash=combined_schema_hash(),
            capability_profile_hash=capability_profile_hash(profile),
            plan_canonicalizer_hash=plan_canonicalizer_hash(),
            generation_provider=gen_provider,
            generation_host=gen_host,
            generation_model=gen_model,
            max_tokens=int(PLANNER_MAX_TOKENS),
            scope_contract_hash=scope_contract_hash(),
        )
        try:
            report = retry_failed_planner_cache(
                output_dir=self.output_dir,
                tasks_by_id=self.tasks_by_id,
                ordered_task_ids=self.dataset["task_ids"],
                expected_identity=expected,
                llm=self._llm,
                capability_profile=profile,
                mock=bool(self.mock),
                dry_run=bool(self.dry_run),
            )
        except (PlannerCacheRetryError, IncompletePlannerCacheError) as exc:
            raise LivePathError(str(exc)) from exc

        meta = self._base_run_metadata("retry_failed_planner_cache")
        meta["retry"] = {
            "promoted": report.get("promoted"),
            "failed_ids": report.get("failed_ids_derived_from_source_cache"),
            "merged_sha256": report.get("merged_planner_outputs_sha256"),
            "source_sha256": report.get("source_planner_cache_sha256"),
            "combined_costs": report.get("combined_costs"),
            "formal_step_pipeline_allowed": report.get("formal_step_pipeline_allowed"),
            "dry_run": bool(self.dry_run),
        }
        if self.gen_meta:
            meta["generation"] = public_meta_without_secrets(self.gen_meta)
        _write_json(
            self.output_dir / "planner_cache" / "retry_run_metadata.json",
            public_meta_without_secrets(meta),
        )
        return report

    def run_evaluate_frozen_formula(self) -> Dict[str, Any]:
        if self.live:
            raise LivePathError(
                "--evaluate-frozen-formula is zero-network; do not pass --live"
            )
        out_dir = self.output_dir / "reserved_evaluation"
        out_dir.mkdir(parents=True, exist_ok=True)
        lock_path = self.output_dir / "formula_selection" / "formula_lock.json"
        cmp_path = self.output_dir / "ranking_variants" / "formula_comparison.json"
        if not lock_path.exists():
            meta = self._base_run_metadata("evaluate_frozen_formula")
            meta["status"] = "skipped_no_formal_lock"
            meta["note"] = (
                "No formal formula_lock.json yet. Reserved-85 retune guard is "
                "unit-tested. Mock/hash cannot produce formal lock."
            )
            meta["zero_network"] = True
            _write_json(out_dir / "run_metadata.json", meta)
            return meta
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        backend = str(
            ((lock.get("config") or {}).get("embedding_backend"))
            or REAL_EMBEDDING_BACKEND
        )
        assert_formal_formula_lock_allowed(
            embedding_backend=backend, mock=False
        )
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
                "embedding_backend": backend,
            },
            "development70": by_split.get("development70"),
            "previously_reserved85": by_split.get("previously_reserved85"),
            "all155": by_split.get("all155"),
            "retune_forbidden": True,
            "zero_network": True,
        }
        _write_json(out_dir / "frozen_formula_evaluation.json", report)
        meta = self._base_run_metadata("evaluate_frozen_formula")
        meta["report"] = {
            "variant": variant,
            "alpha": alpha,
            "has_reserved85": by_split.get("previously_reserved85") is not None,
            "embedding_backend": backend,
        }
        meta["zero_network"] = True
        _write_json(out_dir / "run_metadata.json", meta)
        return meta

    def run_catalog_coverage(self) -> Dict[str, Any]:
        out_dir = self.output_dir / "catalog_coverage"
        out_dir.mkdir(parents=True, exist_ok=True)
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
        emb_model = (
            self.index.embedding_model
            if self.index is not None
            else ("mock-hash-embedding-v1" if self.mock else EXPECTED_EMBEDDING_MODEL)
        )
        emb_dim = self.index.embedding_dim if self.index is not None else None
        _write_json(out_dir / "operation_coverage.json", cov)
        _write_json(out_dir / "distractor_analysis.json", cov["distractors"])
        _write_json(
            out_dir / "operation_documents.json",
            {
                "operation_documents_sha256": self.docs["operation_documents_sha256"],
                "integrity": self.docs["integrity"],
                "n_documents": self.docs["n_documents"],
                "catalog_fingerprint": self.catalog["catalog_fingerprint"],
                "embedding_model": emb_model,
                "embedding_dim": emb_dim,
                "stats": self.catalog["stats"],
            },
        )
        meta = self._base_run_metadata("catalog_coverage")
        meta["zero_network"] = True
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
                "question_query_cache": (self.output_dir / "question_query_cache").exists(),
                "step_query_cache": (self.output_dir / "step_query_cache").exists(),
                "catalog_index": (self.output_dir / "catalog_index").exists(),
            },
            "notes": [
                "exploratory/descriptive only",
                "no unseen held-out claim",
                "gold/accepted ops used only in offline scoring",
            ],
        }
        for rel in (
            "question_level/retrieval_metrics.json",
            "ranking_variants/retrieval_metrics.json",
            "formula_selection/formula_comparison.json",
            "reserved_evaluation/frozen_formula_evaluation.json",
            "catalog_coverage/operation_coverage.json",
            "catalog_index/index_status.json",
        ):
            path = self.output_dir / rel
            if path.exists():
                report[rel.replace("/", "__")] = json.loads(path.read_text(encoding="utf-8"))
        _write_json(out_dir / "all155_report.json", report)
        meta = self._base_run_metadata("report_all")
        meta["zero_network"] = True
        _write_json(out_dir / "run_metadata.json", meta)
        _write_json(self.output_dir / "run_metadata.json", meta)
        _write_json(self.output_dir / "error_summary.json", {"errors": []})
        return meta
