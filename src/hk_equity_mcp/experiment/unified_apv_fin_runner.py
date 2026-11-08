"""APV-Fin unified experiment runner (proposed method; separate from OV baseline).

Reuses: UnifiedRealMcpGateway, frozen catalog, dataset loader, evaluator v1.2,
resume/checkpoint helpers, provider metadata / fail-closed env loading.
Does not modify Operation-Visible baseline prompts, configs, or raw outputs.
"""

from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import yaml

from hk_equity_mcp.agent.llm_client import OpenAICompatibleChatClient, ScriptedLLMClient
from hk_equity_mcp.apv_fin import APV_FIN_CONDITION, APV_FIN_METHOD_VERSION
from hk_equity_mcp.apv_fin.agent import ApvFinAgent
from hk_equity_mcp.apv_fin.metadata import load_and_validate_metadata, validate_metadata_bundle
from hk_equity_mcp.apv_fin.prompts import APV_FIN_SYSTEM_PROMPT, apv_fin_prompt_hash
from hk_equity_mcp.common.env_loader import (
    EMBEDDING_REQUIRED,
    GENERATION_REQUIRED,
    load_embedding_profile,
    load_generation_profile,
    missing_keys,
)
from hk_equity_mcp.experiment.runner import (
    already_done_ids,
    file_sha256,
    git_branch,
    git_commit,
    git_dirty,
    redact_secrets,
    url_host,
)
from hk_equity_mcp.experiment.unified_apv_fin_metrics import aggregate_apv_fin_metrics
from hk_equity_mcp.experiment.unified_dataset import (
    assert_heldout_unlocked,
    load_smoke_tasks,
    load_split_tasks,
    select_by_ids,
)
from hk_equity_mcp.experiment.unified_evaluator import (
    evaluate_task,
    summarize_predictions,
)
from hk_equity_mcp.experiment.unified_exploratory import (
    RETIREMENT_POLICY_VERSION,
    assert_exploratory_unlocked,
    build_retirement_metadata,
    load_exploratory_manifest,
    load_exploratory_run_order,
)
from hk_equity_mcp.experiment.unified_runner import (
    RESUME_IDENTITY_KEYS,
    append_jsonl,
    assert_resume_compatible_unified,
    atomic_write_json,
    catalog_fingerprint,
    fsync_file,
    repo_root,
)
from hk_equity_mcp.experiment.unified_schema import (
    BANNED_PROMPT_SUBSTRINGS,
    MANUAL_GRADING_FIELDS,
    PER_TASK_METRIC_FIELDS,
    PREDICTION_FIELDS,
)
from hk_equity_mcp.operation_resolver.retrieval import (
    DeterministicHashEmbeddingBackend,
    OpenAICompatibleEmbeddingBackend,
)
from hk_equity_mcp.operation_resolver.server import OperationResolverServer
from hk_equity_mcp.real_mcp.unified_catalog import build_unified_catalog
from hk_equity_mcp.real_mcp.unified_gateway import (
    ScriptedMockTransport,
    UnifiedRealMcpGateway,
)

DIAGNOSTIC12_MANIFEST = "benchmarks/manifests/hk_apv_fin_diagnostic12_v1.json"

APV_RESUME_IDENTITY_KEYS = RESUME_IDENTITY_KEYS + (
    "method_version",
    "operation_metadata_hash",
    "diagnostic_manifest_hash",
)


def load_apv_fin_config(
    path: Optional[Path] = None, root: Optional[Path] = None
) -> Dict[str, Any]:
    root = root or repo_root()
    cfg_path = path or (root / "configs" / "hk_unified_apv_fin_v1.yaml")
    with cfg_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_diagnostic12_manifest(root: Optional[Path] = None) -> Dict[str, Any]:
    root = root or repo_root()
    path = root / DIAGNOSTIC12_MANIFEST
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("manifest_id") != "hk_apv_fin_diagnostic12_v1":
        raise RuntimeError("Unexpected diagnostic12 manifest_id")
    if len(data.get("ordered_task_ids") or []) != 12:
        raise RuntimeError("diagnostic12 must contain exactly 12 ordered_task_ids")
    return data


def load_diagnostic12_tasks(root: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Pointer-only: load exploratory_all tasks, then select Diagnostic-12 IDs."""
    root = root or repo_root()
    man = load_diagnostic12_manifest(root)
    all_tasks = load_split_tasks(root, "exploratory_all")
    return select_by_ids(all_tasks, man["ordered_task_ids"], preserve_id_order=True)


class _CyclingScriptedLLM(ScriptedLLMClient):
    def __init__(self, scripted_contents: list[str], model: str = "mock-llm"):
        super().__init__(scripted_contents, model=model)
        self._script = list(scripted_contents)

    def reset_script(self) -> None:
        """Restart the offline trajectory at each task boundary."""
        self._queue = list(self._script)

    def complete(self, messages, *, temperature: float = 0.0, max_tokens: int = 1024):
        if not self._queue:
            self._queue = list(self._script)
        return super().complete(
            messages, temperature=temperature, max_tokens=max_tokens
        )


def _mock_apv_scripted_llm() -> ScriptedLLMClient:
    """Offline trajectory covering planner → monitor continue → retrieve → monitor answer."""
    plan = {
        "schema_version": "apv_fin_plan_v1",
        "task_spec": {
            "goal": "answer the HK equity question from available evidence",
            "entities": [],
            "periods": [],
            "metrics": ["requested_metric"],
            "constraints": ["cutoff_2026-08-12"],
            "ambiguities": [],
            "answerability_hypothesis": "answerable",
            "cutoff_date": "2026-08-12",
            "currency_unit_hints": [],
            "metric_basis_hints": [],
        },
        "initial_plan": {
            "steps": [
                {
                    "step_id": "s1_retrieve",
                    "step_type": "retrieve",
                    "objective": "retrieve supporting HK equity evidence",
                    "depends_on": [],
                    "mandatory": True,
                    "success_criteria": "mcp evidence obtained",
                    "retrieval_query_hint": "hong kong equity financial indicator",
                    "notes": None,
                },
                {
                    "step_id": "s2_reason",
                    "step_type": "reason",
                    "objective": "summarize evidence for final answer",
                    "depends_on": ["s1_retrieve"],
                    "mandatory": True,
                    "success_criteria": "concise summary ready",
                    "retrieval_query_hint": None,
                    "notes": None,
                },
                {
                    "step_id": "s3_answer",
                    "step_type": "answer",
                    "objective": "emit final answer",
                    "depends_on": ["s2_reason"],
                    "mandatory": True,
                    "success_criteria": "final JSON ready",
                    "retrieval_query_hint": None,
                    "notes": None,
                },
            ],
            "planning_notes": "offline mock plan",
        },
    }
    monitor_continue = {
        "schema_version": "apv_fin_monitor_v1",
        "decision": "continue",
        "decision_reason": "execute next ready step",
        "ready_step_ids": ["s1_retrieve"],
        "patch": {
            "completed_steps": [],
            "updated_steps": [],
            "new_steps": [],
            "removed_steps": [],
        },
        "clarify_question": None,
        "refuse_reason": None,
        "fail_reason": None,
    }
    monitor_continue2 = {
        **monitor_continue,
        "ready_step_ids": ["s2_reason"],
        "decision_reason": "continue to reason",
    }
    monitor_continue3 = {
        **monitor_continue,
        "ready_step_ids": ["s3_answer"],
        "decision_reason": "continue to answer node",
    }
    monitor_answer = {
        "schema_version": "apv_fin_monitor_v1",
        "decision": "answer",
        "decision_reason": "enough evidence for mock answer",
        "ready_step_ids": [],
        "patch": {
            "completed_steps": ["s3_answer"],
            "updated_steps": [],
            "new_steps": [],
            "removed_steps": [],
        },
        "clarify_question": None,
        "refuse_reason": None,
        "fail_reason": None,
    }
    reason = {
        "summary": "mock evidence supports a concise answer",
        "decision": "answer",
        "resolved": True,
        "claims": ["mock claim"],
    }
    final = {
        "schema_version": "apv_fin_final_v1",
        "status": "answered",
        "final_answer": "mock answer",
        "reasoning_brief": "offline APV-Fin mock trajectory",
        "evidence_ids": [],
        "clarification_question": None,
        "refusal_reason": None,
        "currency": None,
        "unit": None,
        "metric_basis": None,
        "period": None,
        "entity": None,
    }
    # Sequence consumed across one task; cycling restarts for next task.
    turns = [
        plan,
        monitor_continue,
        monitor_continue2,
        reason,
        monitor_continue3,
        monitor_answer,
        final,
    ]
    return _CyclingScriptedLLM([json.dumps(t, ensure_ascii=False) for t in turns])


class UnifiedHkApvFinRunner:
    def __init__(
        self,
        *,
        output_dir: Path,
        split: str = "development",
        smoke: bool = False,
        diagnostic12: bool = False,
        ids: Optional[Sequence[str]] = None,
        max_tasks: Optional[int] = None,
        resume: bool = False,
        dry_run: bool = False,
        mock: bool = False,
        allow_dirty: bool = False,
        confirm_heldout_final: bool = False,
        confirm_retire_heldout: bool = False,
        env_file: Optional[str] = None,
        embedding_env_file: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        root: Optional[Path] = None,
        allow_hkex_download: bool = False,
        json_mode: bool = True,
    ):
        self.root = root or repo_root()
        self.output_dir = Path(output_dir)
        self.split = split
        self.smoke = bool(smoke)
        self.diagnostic12 = bool(diagnostic12)
        self.ids = list(ids) if ids else None
        self.max_tasks = max_tasks
        self.resume = resume
        self.dry_run = dry_run
        self.mock = mock
        self.allow_dirty = allow_dirty
        self.confirm_heldout_final = confirm_heldout_final
        self.confirm_retire_heldout = confirm_retire_heldout
        self.env_file = env_file
        self.embedding_env_file = embedding_env_file
        self.config = config or load_apv_fin_config(root=self.root)
        self.allow_hkex_download = allow_hkex_download
        self.json_mode = json_mode
        self.temperature = float(self.config.get("temperature", 0.0))
        self.max_tokens = int(self.config.get("max_tokens", 2048))
        self.limits = dict(self.config.get("limits") or {})
        self.retrieval = dict(self.config.get("retrieval") or {})
        self.condition = APV_FIN_CONDITION
        self.method_version = APV_FIN_METHOD_VERSION
        self.execution_backend = "real_mcp"
        self.system_prompt = APV_FIN_SYSTEM_PROMPT
        self.gen_meta: Dict[str, str] = {}
        self.emb_meta: Optional[Dict[str, str]] = None
        self._gateway: Optional[UnifiedRealMcpGateway] = None
        self.bundle = build_unified_catalog(root=self.root)
        meta_path = self.root / str(
            self.config.get("operation_metadata")
            or "configs/operations/hk_unified_operation_metadata_v1.yaml"
        )
        self.metadata_path = meta_path
        self.metadata_by_id = load_and_validate_metadata(
            meta_path, catalog=self.bundle, root=self.root
        )
        self._exploratory_manifest: Optional[Dict[str, Any]] = None
        self._exploratory_run_order: Optional[Dict[str, Any]] = None
        self._diagnostic_manifest: Optional[Dict[str, Any]] = None

    def _load_tasks(self) -> List[Dict[str, Any]]:
        effective_split = self.split
        confirm_retire = self.confirm_retire_heldout
        if self.diagnostic12:
            # Diagnostic-12 is always a pointer subset of exploratory_all.
            effective_split = "exploratory_all"
            if self.dry_run and not confirm_retire:
                # Offline dry-run may unlock for metadata validation only.
                confirm_retire = True
            if not self.dry_run and not confirm_retire:
                raise RuntimeError(
                    "diagnostic12 requires --split exploratory_all "
                    "--confirm-retire-heldout (except dry-run)."
                )

        heldout_status = assert_heldout_unlocked(
            split=effective_split, confirm_heldout_final=self.confirm_heldout_final
        )
        self._heldout_status = heldout_status
        exploratory_status = assert_exploratory_unlocked(
            split=effective_split, confirm_retire_heldout=confirm_retire
        )
        self._exploratory_status = exploratory_status

        if self.diagnostic12:
            # Diagnostic-12 is a pointer subset of exploratory_all; no gold mutation.
            self._diagnostic_manifest = load_diagnostic12_manifest(self.root)
            self.split = "exploratory_all"
            tasks = load_diagnostic12_tasks(self.root)
            self._exploratory_manifest = load_exploratory_manifest(self.root)
            self._exploratory_run_order = load_exploratory_run_order(self.root)
        elif self.smoke:
            if self.split != "development":
                raise RuntimeError("--smoke is only valid with --split development")
            tasks = load_smoke_tasks(self.root)
        else:
            if self.split == "exploratory_all":
                self._exploratory_manifest = load_exploratory_manifest(self.root)
                self._exploratory_run_order = load_exploratory_run_order(self.root)
            tasks = load_split_tasks(self.root, self.split)

        if self.ids is not None:
            tasks = select_by_ids(
                tasks,
                self.ids,
                preserve_id_order=True,
            )
        if self.max_tasks is not None:
            tasks = tasks[: max(0, int(self.max_tasks))]
        return tasks

    def prepare(self) -> Dict[str, Any]:
        dirty = git_dirty(self.root)
        if not self.dry_run and not self.mock and dirty and not self.allow_dirty:
            raise RuntimeError(
                "Paid/formal run refused: git working tree is dirty. "
                "Commit changes first, or pass --allow-dirty explicitly."
            )

        for banned in BANNED_PROMPT_SUBSTRINGS:
            if banned in self.system_prompt:
                raise RuntimeError(f"APV-Fin prompt unexpectedly contains {banned}")

        prompt_h = apv_fin_prompt_hash()
        tasks = self._load_tasks()
        stats = self.bundle.stats
        meta_hash = file_sha256(self.metadata_path)

        if self.mock or self.dry_run:
            try:
                self.gen_meta = load_generation_profile(self.root, self.env_file)
            except FileNotFoundError:
                self.gen_meta = {
                    "provider": "aliyun",
                    "model": "deepseek-v4.1-flash",
                    "profile_file": None,
                    "base_url": None,
                }
            try:
                self.emb_meta = load_embedding_profile(
                    self.root, self.embedding_env_file
                )
            except FileNotFoundError:
                self.emb_meta = {
                    "model": "text-embedding-3-small",
                    "profile_file": None,
                    "base_url": None,
                }
            if self.dry_run and not self.mock:
                self._dry_run_missing = {
                    "generation": missing_keys(GENERATION_REQUIRED),
                    "embedding": missing_keys(EMBEDDING_REQUIRED),
                }
        else:
            self.gen_meta = load_generation_profile(self.root, self.env_file)
            self.emb_meta = load_embedding_profile(self.root, self.embedding_env_file)

        expected_model = str(
            self.config.get("generation_model") or "deepseek-v4.1-flash"
        )
        expected_provider = str(self.config.get("generation_provider") or "aliyun")
        expected_emb = str(
            self.config.get("embedding_model") or "text-embedding-3-small"
        )
        if not self.mock:
            actual_model = self.gen_meta.get("model") or os.environ.get("LLM_MODEL")
            actual_provider = self.gen_meta.get("provider") or os.environ.get(
                "LLM_PROVIDER"
            )
            actual_emb = (self.emb_meta or {}).get("model") or os.environ.get(
                "EMBEDDING_MODEL"
            )
            if actual_model and actual_model != expected_model:
                raise RuntimeError(
                    f"generation model mismatch: expected {expected_model}, got {actual_model}"
                )
            if actual_provider and actual_provider != expected_provider:
                raise RuntimeError(
                    f"generation provider mismatch: expected {expected_provider}, "
                    f"got {actual_provider}"
                )
            if actual_emb and actual_emb != expected_emb:
                raise RuntimeError(
                    f"embedding model mismatch: expected {expected_emb}, got {actual_emb}"
                )

        self.output_dir.mkdir(parents=True, exist_ok=True)
        fingerprint = catalog_fingerprint(stats, self.bundle.source_catalogs)

        domain_counts: Dict[str, int] = {}
        for t in tasks:
            domain_counts[str(t.get("domain"))] = domain_counts.get(
                str(t.get("domain")), 0
            ) + 1

        diagnostic_manifest_hash = None
        if self._diagnostic_manifest is not None:
            diagnostic_manifest_hash = self._diagnostic_manifest.get("manifest_hash")

        exploratory_manifest_hash = None
        run_order_hash = None
        retirement_policy_version = None
        retirement_meta = None
        if self.split == "exploratory_all" or self.diagnostic12:
            man = self._exploratory_manifest or load_exploratory_manifest(self.root)
            order = self._exploratory_run_order or load_exploratory_run_order(self.root)
            exploratory_manifest_hash = man.get("manifest_hash")
            run_order_hash = order.get("run_order_hash")
            retirement_policy_version = RETIREMENT_POLICY_VERSION
            existing_first = None
            meta_path = self.output_dir / "run_metadata.json"
            if self.resume and meta_path.exists():
                try:
                    existing = json.loads(meta_path.read_text(encoding="utf-8"))
                    existing_first = (existing.get("retirement") or {}).get(
                        "first_real_exploratory_run"
                    )
                except Exception:  # noqa: BLE001
                    existing_first = None
            retirement_meta = build_retirement_metadata(
                dry_run=self.dry_run,
                mock=self.mock,
                confirm_retire_heldout=self.confirm_retire_heldout
                or bool(self.diagnostic12 and self.dry_run),
                git_commit=git_commit(self.root),
                prompt_hash=prompt_h,
                catalog_fingerprint=fingerprint,
                manifest_hash=str(exploratory_manifest_hash),
                run_order_hash=str(run_order_hash),
                timestamp=datetime.now(timezone.utc).isoformat(),
                existing_first_real=existing_first,
            )

        meta = {
            "condition": self.condition,
            "method_version": self.method_version,
            "execution_backend": self.execution_backend,
            "real_mcp": True,
            "mock": self.mock,
            "dry_run": self.dry_run,
            "dataset_split": self.split,
            "smoke": self.smoke,
            "diagnostic12": self.diagnostic12,
            "diagnostic_manifest_hash": diagnostic_manifest_hash,
            "heldout_lock": getattr(self, "_heldout_status", {}),
            "exploratory_lock": getattr(self, "_exploratory_status", {}),
            "exploratory_manifest_hash": exploratory_manifest_hash,
            "run_order_hash": run_order_hash,
            "retirement_policy_version": retirement_policy_version,
            "retirement": retirement_meta,
            "question_ids": [t["task_id"] for t in tasks],
            "question_count": len(tasks),
            "domain_counts": domain_counts,
            "generation_provider": self.gen_meta.get("provider") or expected_provider,
            "generation_model": self.gen_meta.get("model") or expected_model,
            "generation_profile_file": self.gen_meta.get("profile_file"),
            "generation_base_url_host": url_host(
                self.gen_meta.get("base_url") or os.environ.get("LLM_BASE_URL")
            ),
            "embedding_model": (self.emb_meta or {}).get("model") or expected_emb,
            "embedding_profile_file": (self.emb_meta or {}).get("profile_file"),
            "embedding_base_url_host": url_host(
                (self.emb_meta or {}).get("base_url")
                or os.environ.get("EMBEDDING_BASE_URL")
            ),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "json_mode": self.json_mode,
            "retrieval_top_k": int(self.retrieval.get("top_k", 20)),
            "lexical_weight": float(self.retrieval.get("lexical_weight", 0.12)),
            "limits": self.limits,
            "system_prompt_hash": prompt_h,
            "system_prompt": self.system_prompt,
            "catalog_fingerprint": fingerprint,
            "operation_metadata_path": str(self.metadata_path),
            "operation_metadata_hash": meta_hash,
            "operation_metadata_coverage": len(self.metadata_by_id),
            "catalog_stats": {
                "namespaced_entries": stats.namespaced_entries,
                "unique_operation_ids": stats.unique_operation_ids,
                "runtime_exposed_unique": stats.runtime_exposed_unique,
                "formal_gold_eligible_unique": stats.formal_gold_eligible_unique,
                "distractor_runtime_unique": stats.distractor_runtime_unique,
                "blocked_or_rejected_unique": stats.blocked_or_rejected_unique,
                "blocked_operation_ids": stats.blocked_operation_ids,
                "runtime_operation_ids": stats.runtime_operation_ids,
                "overlap_operation_ids": stats.overlap_operation_ids,
            },
            "source_catalogs": {
                k: str(v) for k, v in self.bundle.source_catalogs.items()
            },
            "branch": git_branch(self.root),
            "git_commit": git_commit(self.root),
            "git_dirty": dirty,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "resume": self.resume,
            "allow_dirty": self.allow_dirty,
            "max_tasks": self.max_tasks,
            "ids": self.ids,
            "run_status": "dry_run"
            if self.dry_run
            else ("mock" if self.mock else "running"),
            "fidelity_note": (
                "APV-Fin proposed method v1; separated from Operation-Visible baseline"
            ),
            "dry_run_missing_env": getattr(self, "_dry_run_missing", None),
            "baseline_separation": {
                "does_not_modify_operation_visible_baseline": True,
                "does_not_modify_baseline_raw_outputs": True,
                "reuses_unified_gateway_and_catalog": True,
            },
        }
        return {"items": tasks, "run_metadata": redact_secrets(meta)}

    def build_agent(self) -> ApvFinAgent:
        if self.dry_run:
            raise RuntimeError("build_agent must not be called during dry-run.")

        if self.mock:
            llm: Any = _mock_apv_scripted_llm()
            embedding: Any = DeterministicHashEmbeddingBackend()
            gateway = UnifiedRealMcpGateway(
                catalog_bundle=self.bundle,
                mock_transport=ScriptedMockTransport(),
            )
        else:
            llm = OpenAICompatibleChatClient(
                base_url=os.environ["LLM_BASE_URL"],
                api_key=os.environ["LLM_API_KEY"],
                model=os.environ["LLM_MODEL"],
                provider=os.environ.get("LLM_PROVIDER") or "openai_compatible",
                json_mode=self.json_mode,
            )
            embedding = OpenAICompatibleEmbeddingBackend(
                base_url=os.environ["EMBEDDING_BASE_URL"],
                api_key=os.environ["EMBEDDING_API_KEY"],
                model=os.environ["EMBEDDING_MODEL"],
            )
            gateway = UnifiedRealMcpGateway(
                catalog_bundle=self.bundle,
                allow_hkex_download=self.allow_hkex_download,
                persistent=True,
            )

        gateway.open()
        self._gateway = gateway
        cache_dir = None
        if self.mock:
            cache_dir = self.output_dir / ".cache" / "operation_embeddings"
            cache_dir.mkdir(parents=True, exist_ok=True)
        resolver = OperationResolverServer(
            embedding_backend=embedding,
            registry=self.bundle.registry,
            real_mcp_gateway=gateway,
            cache_dir=cache_dir,
        )
        if resolver.financial is not None:
            gateway.close()
            raise RuntimeError(
                "APV-Fin fail-closed: OperationResolverServer unexpectedly retained "
                "FinancialServer."
            )
        cfg = {
            "llm": {"temperature": self.temperature, "max_tokens": self.max_tokens},
            "limits": dict(self.limits),
            "retrieval": dict(self.retrieval),
            # Symbol-only args match the forced financial-indicator mock target.
            "mock_retrieve_arguments": {"*": {"symbol": "0700"}},
            # Deterministic offline invoke target with symbol-only required args.
            "mock_select_operation_id": (
                "aktools.stock_financial_hk_analysis_indicator_em"
                if self.mock
                else None
            ),
        }
        return ApvFinAgent(
            llm=llm,
            resolver=resolver,
            config=cfg,
            metadata_by_id=self.metadata_by_id,
            runtime_operation_ids=list(self.bundle.stats.runtime_operation_ids),
        )

    def close(self) -> None:
        if self._gateway is not None:
            self._gateway.close()
            self._gateway = None

    def _build_prediction_row(
        self, item: Dict[str, Any], result: Dict[str, Any]
    ) -> Dict[str, Any]:
        from hk_equity_mcp.experiment.unified_grading import grade_task

        metrics = evaluate_task(item=item, result=result)
        grade = grade_task(item, {**result, **metrics, "task_id": item["task_id"]})
        tokens = result.get("tokens") or {}
        row = {
            "task_id": item["task_id"],
            "id": item["task_id"],
            "domain": item.get("domain"),
            "domain_combination": item.get("domain_combination"),
            "project_difficulty": item.get("project_difficulty"),
            "exact_etom_level": item.get("exact_etom_level"),
            "exact_etom_alignment": item.get("exact_etom_alignment"),
            "etom_level": item.get("etom_level"),
            "etom_alignment": item.get("etom_alignment"),
            "question": item["question"],
            "status": result.get("status"),
            "error": result.get("error"),
            "final_answer": result.get("final_answer"),
            "predicted_status": result.get("predicted_status"),
            "answerability_class": item.get("answerability_class"),
            "source_split": item.get("source_split"),
            "company": item.get("company"),
            "gold_status": item.get("gold_status"),
            "gold_values": item.get("gold_values"),
            "llm_calls": result.get("llm_calls"),
            "tokens": tokens,
            "embedding_calls": result.get("embedding_calls"),
            "embedding_tokens": result.get("embedding_tokens"),
            "latency_ms": result.get("latency_ms"),
            "retry_count": result.get("retry_count", 0),
            "timeout": result.get("error") == "timeout",
            "parse_repair_count": result.get("parse_repair_count"),
            "trace": result.get("trace") or [],
            "resolver_calls": result.get("resolver_calls")
            or {"find_operations": 0, "describe_operation": 0},
            "underlying_operation_calls": result.get("underlying_operation_calls", 0),
            "provider_metadata": result.get("provider_metadata") or {},
            "condition": self.condition,
            "method_version": self.method_version,
            "task_spec": result.get("task_spec"),
            "initial_plan": result.get("initial_plan"),
            "plan_state": result.get("plan_state"),
            "plan_revisions": result.get("plan_revisions"),
            "evidence_ledger": result.get("evidence_ledger"),
            "compute_traces": result.get("compute_traces"),
            "validator": result.get("validator"),
            "stopping_reason": result.get("stopping_reason"),
            "budget": result.get("budget"),
            "auto_grade_status": grade.get("auto_grade_status"),
            "answer_correct": grade.get("answer_correct"),
            "auto_grade_reason": grade.get("reason"),
            "manual_answer_correctness": "",
            "manual_evidence_correctness": "",
            "manual_unsupported_claim": "",
            "manual_causal_overclaim": "",
            "manual_overall_task_success": "",
        }
        row.update(metrics)
        for key in PREDICTION_FIELDS:
            row.setdefault(key, None)
        return redact_secrets(row)

    def _write_side_artifacts(
        self,
        *,
        items: List[Dict[str, Any]],
        predictions_by_id: Dict[str, Dict[str, Any]],
        run_meta: Dict[str, Any],
    ) -> None:
        metrics_path = self.output_dir / "per_task_metrics.csv"
        with metrics_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(PER_TASK_METRIC_FIELDS))
            writer.writeheader()
            for item in items:
                pred = predictions_by_id.get(item["task_id"], {})
                tokens = pred.get("tokens") or {}
                writer.writerow(
                    {
                        "task_id": item["task_id"],
                        "domain": item.get("domain"),
                        "domain_combination": item.get("domain_combination"),
                        "project_difficulty": item.get("project_difficulty"),
                        "exact_etom_level": item.get("exact_etom_level"),
                        "exact_etom_alignment": item.get("exact_etom_alignment"),
                        "answerability_class": item.get("answerability_class"),
                        "status": pred.get("status"),
                        "operation_precision": pred.get("operation_precision"),
                        "operation_recall": pred.get("operation_recall"),
                        "operation_f1": pred.get("operation_f1"),
                        "exact_tool_set": pred.get("exact_tool_set"),
                        "graph_dependency_accuracy": pred.get(
                            "graph_dependency_accuracy"
                        ),
                        "tool_execution_success": pred.get("tool_execution_success"),
                        "all_tool_invocations_successful": pred.get(
                            "all_tool_invocations_successful"
                        ),
                        "successful_invocation_count": pred.get(
                            "successful_invocation_count"
                        ),
                        "failed_invocation_count": pred.get(
                            "failed_invocation_count"
                        ),
                        "recovered_after_tool_error": pred.get(
                            "recovered_after_tool_error"
                        ),
                        "protocol_success_rate": pred.get("protocol_success_rate"),
                        "wrong_tool_rate": pred.get("wrong_tool_rate"),
                        "refusal_correctness": pred.get("refusal_correctness"),
                        "llm_calls": pred.get("llm_calls"),
                        "input_tokens": tokens.get("input_tokens"),
                        "output_tokens": tokens.get("output_tokens"),
                        "total_tokens": tokens.get("total_tokens"),
                        "embedding_calls": pred.get("embedding_calls"),
                        "embedding_tokens": pred.get("embedding_tokens"),
                        "latency_ms": pred.get("latency_ms"),
                        "real_mcp_execution_count": pred.get(
                            "real_mcp_execution_count"
                        ),
                        "retry_count": pred.get("retry_count"),
                        "timeout": pred.get("timeout"),
                        "parse_repair_count": pred.get("parse_repair_count"),
                    }
                )
        fsync_file(metrics_path)

        rows = [
            predictions_by_id[i["task_id"]]
            for i in items
            if i["task_id"] in predictions_by_id
        ]
        summary = summarize_predictions(rows)
        apv_metrics = aggregate_apv_fin_metrics(rows)
        atomic_write_json(self.output_dir / "apv_fin_metrics.json", apv_metrics)
        atomic_write_json(
            self.output_dir / "error_summary.json",
            {
                "aggregate": summary,
                "apv_fin_metrics": apv_metrics,
            },
        )

        grading_path = self.output_dir / "manual_grading.csv"
        with grading_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(MANUAL_GRADING_FIELDS))
            writer.writeheader()
            for item in items:
                pred = predictions_by_id.get(item["task_id"], {})
                writer.writerow(
                    {
                        "task_id": item["task_id"],
                        "domain": item.get("domain"),
                        "domain_combination": item.get("domain_combination"),
                        "source_split": item.get("source_split", ""),
                        "company": item.get("company", ""),
                        "project_difficulty": item.get("project_difficulty", ""),
                        "exact_etom_level": item.get("exact_etom_level", ""),
                        "exact_etom_alignment": item.get("exact_etom_alignment", ""),
                        "question": item["question"],
                        "gold_status": item.get("gold_status", ""),
                        "answerability_class": item.get("answerability_class", ""),
                        "gold_operations": json.dumps(
                            item.get("gold_operations") or [], ensure_ascii=False
                        ),
                        "reference_answer": item.get("reference_answer", ""),
                        "predicted_status": pred.get("predicted_status", ""),
                        "predicted_answer": pred.get("final_answer", ""),
                        "predicted_operations": json.dumps(
                            pred.get("predicted_operations") or [], ensure_ascii=False
                        ),
                        "evidence": json.dumps(
                            pred.get("evidence_ledger") or pred.get("evidence") or [],
                            ensure_ascii=False,
                        ),
                        "auto_grade_status": pred.get("auto_grade_status", ""),
                        "answer_correct": pred.get("answer_correct", ""),
                        "answer_correct_0_or_1": "",
                        "evidence_correct_0_or_1": "",
                        "unsupported_claim_0_or_1": "",
                        "causal_overclaim_0_or_1": "",
                        "overall_task_success_0_or_1": "",
                        "grader_notes": "",
                        "manual_answer_correctness": "",
                        "manual_evidence_correctness": "",
                        "manual_unsupported_claim": "",
                        "manual_causal_overclaim": "",
                        "manual_overall_task_success": "",
                        "notes": "",
                    }
                )
        fsync_file(grading_path)

    def run(self) -> Dict[str, Any]:
        prepared = self.prepare()
        items = prepared["items"]
        run_meta = prepared["run_metadata"]
        meta_path = self.output_dir / "run_metadata.json"
        predictions_path = self.output_dir / "predictions.jsonl"

        if self.dry_run:
            atomic_write_json(meta_path, run_meta)
            print("APV-FIN DRY-RUN OK")
            print(f"condition={run_meta['condition']}")
            print(f"method_version={run_meta['method_version']}")
            print(f"split={self.split}")
            print(f"diagnostic12={self.diagnostic12}")
            print(f"question_count={run_meta['question_count']}")
            print(f"domain_counts={run_meta['domain_counts']}")
            print(f"system_prompt_hash={run_meta['system_prompt_hash']}")
            print(f"operation_metadata_coverage={run_meta['operation_metadata_coverage']}")
            print(f"operation_metadata_hash={run_meta['operation_metadata_hash']}")
            print(f"diagnostic_manifest_hash={run_meta.get('diagnostic_manifest_hash')}")
            print(f"catalog_fingerprint={run_meta['catalog_fingerprint']}")
            print(f"output_dir={self.output_dir}")
            return run_meta

        if self.resume and meta_path.exists():
            existing = json.loads(meta_path.read_text(encoding="utf-8"))
            mismatches = []
            for key in APV_RESUME_IDENTITY_KEYS:
                if existing.get(key) != run_meta.get(key):
                    mismatches.append(
                        f"{key}: existing={existing.get(key)!r} current={run_meta.get(key)!r}"
                    )
            if mismatches:
                raise RuntimeError(
                    "Resume refused: APV-Fin run_metadata identity mismatch. "
                    + "; ".join(mismatches)
                )
        elif self.resume and not meta_path.exists():
            raise RuntimeError(
                "Resume refused: run_metadata.json is missing. "
                "Cannot verify experiment identity."
            )

        run_meta["run_status"] = "running" if not self.mock else "mock_running"
        atomic_write_json(meta_path, run_meta)

        try:
            agent = self.build_agent()
            done = already_done_ids(predictions_path) if self.resume else set()
            if self.resume and predictions_path.exists():
                done |= {
                    row.get("task_id") or row.get("id")
                    for row in (
                        json.loads(line)
                        for line in predictions_path.read_text(
                            encoding="utf-8"
                        ).splitlines()
                        if line.strip()
                    )
                    if (row.get("task_id") or row.get("id"))
                }
            if not self.resume and predictions_path.exists():
                predictions_path.unlink()

            predictions_by_id: Dict[str, Dict[str, Any]] = {}
            if self.resume and predictions_path.exists():
                for line in predictions_path.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    predictions_by_id[row.get("task_id") or row["id"]] = row

            for index, item in enumerate(items, start=1):
                qid = item["task_id"]
                if qid in done:
                    print(f"[{index:02d}/{len(items)}] skip {qid} (resume)", flush=True)
                    continue
                print(f"[{index:02d}/{len(items)}] {qid}", flush=True)
                # Model receives ONLY the question string (no gold leakage).
                if hasattr(agent.llm, "reset_script"):
                    agent.llm.reset_script()
                result = agent.run(item["question"])
                # Scrub gold keys from any accidental prompt surfaces in trace text.
                row = self._build_prediction_row(item, result)
                append_jsonl(predictions_path, row)
                predictions_by_id[qid] = row
                self._write_side_artifacts(
                    items=items,
                    predictions_by_id=predictions_by_id,
                    run_meta=run_meta,
                )

            run_meta["run_status"] = "completed"
            run_meta["completed_at"] = datetime.now(timezone.utc).isoformat()
            if self._gateway is not None:
                run_meta["gateway_invoke_count"] = self._gateway.invoke_count
            atomic_write_json(meta_path, run_meta)
            print(f"Predictions: {predictions_path}")
            print(f"Run metadata: {meta_path}")
            return run_meta
        finally:
            self.close()
