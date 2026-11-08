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
from hk_equity_mcp.apv_fin.identity import (
    EXPECTED_EMBEDDING_MODEL,
    EXPECTED_GENERATION_HOST,
    EXPECTED_GENERATION_MODEL,
    EXPECTED_GENERATION_PROVIDER,
    assert_live_provider_identity,
    combined_schema_hash,
    host_from_base_url,
    schema_hashes,
    worst_case_budget_estimate,
)
from hk_equity_mcp.apv_fin.metadata import load_and_validate_metadata
from hk_equity_mcp.apv_fin.operation_embedding_index import (
    build_index_identity,
    build_or_resume_index,
    default_shared_cache_dir,
    index_status,
    load_complete_index,
    resolve_embedding_cache_dir,
    sibling_ov_cache_dir,
)
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
SMOKE4_MANIFEST = "benchmarks/manifests/hk_apv_fin_real_smoke4_v1.json"

APV_RESUME_IDENTITY_KEYS = (
    "condition",
    "method_version",
    "execution_backend",
    "generation_provider",
    "generation_base_url_host",
    "generation_model",
    "embedding_provider",
    "embedding_base_url_host",
    "embedding_model",
    "system_prompt_hash",
    "schema_hash",
    "operation_metadata_hash",
    "catalog_fingerprint",
    "dataset_split",
    "subset_kind",
    "subset_manifest_hash",
    "subset_ordered_id_hash",
    "diagnostic_manifest_hash",
    "smoke4_manifest_hash",
    "temperature",
    "max_tokens",
    "json_mode",
    "limits",
    "max_total_llm_calls_per_task",
    "mock",
    "dry_run",
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


def load_smoke4_manifest(root: Optional[Path] = None) -> Dict[str, Any]:
    root = root or repo_root()
    path = root / SMOKE4_MANIFEST
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("manifest_id") != "hk_apv_fin_real_smoke4_v1":
        raise RuntimeError("Unexpected smoke4 manifest_id")
    if len(data.get("ordered_task_ids") or []) != 4:
        raise RuntimeError("smoke4 must contain exactly 4 ordered_task_ids")
    return data


def load_smoke4_tasks(root: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Pointer-only Smoke-4 subset of Diagnostic-12 / exploratory_all."""
    root = root or repo_root()
    man = load_smoke4_manifest(root)
    all_tasks = load_split_tasks(root, "exploratory_all")
    return select_by_ids(all_tasks, man["ordered_task_ids"], preserve_id_order=True)


def assert_output_dir_safe(
    *,
    output_dir: Path,
    dry_run: bool,
    mock: bool,
    resume: bool,
    run_meta: Dict[str, Any],
) -> None:
    """Refuse unsafe overwrite / dry-mock→live resume / subset mixing."""
    meta_path = output_dir / "run_metadata.json"
    predictions_path = output_dir / "predictions.jsonl"
    if predictions_path.exists() and not resume and not dry_run:
        raise RuntimeError(
            f"Output directory already has predictions.jsonl at {predictions_path}. "
            "Pass --resume explicitly; refusing overwrite."
        )
    if not meta_path.exists():
        return
    existing = json.loads(meta_path.read_text(encoding="utf-8"))
    if resume:
        if bool(existing.get("dry_run")) and not dry_run:
            raise RuntimeError(
                "Resume refused: cannot resume a dry-run directory as a live/mock run."
            )
        if bool(existing.get("mock")) and not mock and not dry_run:
            raise RuntimeError(
                "Resume refused: cannot resume a mock directory as a live run."
            )
        if bool(existing.get("dry_run")) != bool(dry_run):
            raise RuntimeError("Resume refused: dry_run flag mismatch.")
        if bool(existing.get("mock")) != bool(mock):
            raise RuntimeError("Resume refused: mock flag mismatch.")
        # Subset mixing guard
        if existing.get("subset_kind") != run_meta.get("subset_kind"):
            raise RuntimeError(
                "Resume refused: subset_kind mismatch "
                f"(existing={existing.get('subset_kind')!r} "
                f"current={run_meta.get('subset_kind')!r})."
            )
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
    elif not dry_run and meta_path.exists() and predictions_path.exists():
        raise RuntimeError(
            "Output directory already contains a previous run. "
            "Use a fresh directory or pass --resume."
        )


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
    """Offline trajectory for hardened order: plan→execute→monitor→…→final."""
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
        "decision_reason": "continue after observation",
        "ready_step_ids": [],
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
    # Hardened order: planner → (retrieve) → monitor → reason → monitor → (answer) → final
    turns = [plan, monitor_continue, reason, monitor_continue, final]
    return _CyclingScriptedLLM([json.dumps(t, ensure_ascii=False) for t in turns])


class UnifiedHkApvFinRunner:
    def __init__(
        self,
        *,
        output_dir: Path,
        split: str = "development",
        smoke: bool = False,
        diagnostic12: bool = False,
        smoke4: bool = False,
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
        require_complete_index: bool = False,
        embedding_cache_dir: Optional[Path] = None,
        allow_auto_build_index: bool = False,
    ):
        self.root = root or repo_root()
        self.output_dir = Path(output_dir)
        self.split = split
        self.smoke = bool(smoke)
        self.diagnostic12 = bool(diagnostic12)
        self.smoke4 = bool(smoke4)
        if self.diagnostic12 and self.smoke4:
            raise RuntimeError("Cannot combine --diagnostic12 and --smoke4")
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
        self.require_complete_index = bool(require_complete_index)
        self.allow_auto_build_index = bool(allow_auto_build_index)
        self.embedding_cache_dir = resolve_embedding_cache_dir(
            self.root, embedding_cache_dir
        )
        self.temperature = float(self.config.get("temperature", 0.0))
        self.max_tokens = int(self.config.get("max_tokens", 2048))
        self.limits = dict(self.config.get("limits") or {})
        self.retrieval = dict(self.config.get("retrieval") or {})
        self.embedding_batch_size = int(self.config.get("embedding_batch_size", 16))
        self.embedding_request_timeout_sec = float(
            self.config.get("embedding_request_timeout_sec", 120)
        )
        self.embedding_max_retries = int(self.config.get("embedding_max_retries", 3))
        self.embedding_retry_backoff_seconds = [
            float(x)
            for x in (
                self.config.get("embedding_retry_backoff_seconds") or [1, 2, 4]
            )
        ]
        self.condition = APV_FIN_CONDITION
        self.method_version = APV_FIN_METHOD_VERSION
        self.execution_backend = "real_mcp"
        self.system_prompt = APV_FIN_SYSTEM_PROMPT
        self.gen_meta: Dict[str, str] = {}
        self.emb_meta: Optional[Dict[str, str]] = None
        self._gateway: Optional[UnifiedRealMcpGateway] = None
        self._index_identity: Optional[Dict[str, Any]] = None
        self._index_status: Optional[Dict[str, Any]] = None
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
        self._smoke4_manifest: Optional[Dict[str, Any]] = None

    def _load_tasks(self) -> List[Dict[str, Any]]:
        effective_split = self.split
        confirm_retire = self.confirm_retire_heldout
        if self.diagnostic12 or self.smoke4:
            # Pointer subsets of exploratory_all.
            effective_split = "exploratory_all"
            if self.dry_run and not confirm_retire:
                # Offline dry-run may unlock for metadata validation only.
                confirm_retire = True
            if not self.dry_run and not confirm_retire:
                raise RuntimeError(
                    "diagnostic12/smoke4 requires --split exploratory_all "
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
        elif self.smoke4:
            self._smoke4_manifest = load_smoke4_manifest(self.root)
            self._diagnostic_manifest = load_diagnostic12_manifest(self.root)
            self.split = "exploratory_all"
            tasks = load_smoke4_tasks(self.root)
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

        expected_model = str(
            self.config.get("generation_model") or EXPECTED_GENERATION_MODEL
        )
        expected_provider = str(
            self.config.get("generation_provider") or EXPECTED_GENERATION_PROVIDER
        )
        expected_host = str(
            self.config.get("generation_host") or EXPECTED_GENERATION_HOST
        )
        expected_emb = str(
            self.config.get("embedding_model") or EXPECTED_EMBEDDING_MODEL
        )
        expected_emb_provider = str(
            self.config.get("embedding_provider") or "openai_compatible"
        )

        if self.mock or self.dry_run:
            try:
                self.gen_meta = load_generation_profile(self.root, self.env_file)
            except FileNotFoundError:
                self.gen_meta = {
                    "provider": expected_provider,
                    "model": expected_model,
                    "profile_file": None,
                    "base_url": f"https://{expected_host}/v1",
                }
            try:
                self.emb_meta = load_embedding_profile(
                    self.root, self.embedding_env_file
                )
            except FileNotFoundError:
                self.emb_meta = {
                    "provider": expected_emb_provider,
                    "model": expected_emb,
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

        gen_host = host_from_base_url(
            self.gen_meta.get("base_url") or os.environ.get("LLM_BASE_URL")
        ) or url_host(
            self.gen_meta.get("base_url") or os.environ.get("LLM_BASE_URL")
        )
        emb_host = host_from_base_url(
            (self.emb_meta or {}).get("base_url")
            or os.environ.get("EMBEDDING_BASE_URL")
        ) or url_host(
            (self.emb_meta or {}).get("base_url")
            or os.environ.get("EMBEDDING_BASE_URL")
        )
        actual_model = self.gen_meta.get("model") or os.environ.get("LLM_MODEL")
        actual_provider = self.gen_meta.get("provider") or os.environ.get(
            "LLM_PROVIDER"
        )
        actual_emb = (self.emb_meta or {}).get("model") or os.environ.get(
            "EMBEDDING_MODEL"
        )
        actual_emb_provider = (self.emb_meta or {}).get("provider") or os.environ.get(
            "EMBEDDING_PROVIDER"
        )

        # Live (non-mock, non-dry-run): fail-closed identity — no Aliyun/mock/lexical fallback.
        if not self.mock and not self.dry_run:
            assert_live_provider_identity(
                generation_provider=str(actual_provider or ""),
                generation_host=gen_host,
                generation_model=str(actual_model or ""),
                embedding_model=str(actual_emb or ""),
                embedding_host=emb_host,
                embedding_provider=str(actual_emb_provider)
                if actual_emb_provider
                else None,
            )
            if expected_provider != EXPECTED_GENERATION_PROVIDER:
                raise RuntimeError(
                    f"config generation_provider must be {EXPECTED_GENERATION_PROVIDER}"
                )
            if expected_host != EXPECTED_GENERATION_HOST:
                raise RuntimeError(
                    f"config generation_host must be {EXPECTED_GENERATION_HOST}"
                )
            if expected_model != EXPECTED_GENERATION_MODEL:
                raise RuntimeError(
                    f"config generation_model must be {EXPECTED_GENERATION_MODEL}"
                )
        elif not self.mock:
            # Dry-run: still validate when env values are present.
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
            if gen_host and gen_host != expected_host:
                raise RuntimeError(
                    f"generation host mismatch: expected {expected_host}, got {gen_host}"
                )

        self.output_dir.mkdir(parents=True, exist_ok=True)
        fingerprint = catalog_fingerprint(stats, self.bundle.source_catalogs)

        domain_counts: Dict[str, int] = {}
        for t in tasks:
            domain_counts[str(t.get("domain"))] = domain_counts.get(
                str(t.get("domain")), 0
            ) + 1

        diagnostic_manifest_hash = None
        diagnostic_ordered_id_hash = None
        if self._diagnostic_manifest is not None:
            diagnostic_manifest_hash = self._diagnostic_manifest.get("manifest_hash")
            diagnostic_ordered_id_hash = self._diagnostic_manifest.get(
                "ordered_id_hash"
            )

        smoke4_manifest_hash = None
        smoke4_ordered_id_hash = None
        if self._smoke4_manifest is not None:
            smoke4_manifest_hash = self._smoke4_manifest.get("manifest_hash")
            smoke4_ordered_id_hash = self._smoke4_manifest.get("ordered_id_hash")

        if self.smoke4:
            subset_kind = "smoke4"
            subset_manifest_hash = smoke4_manifest_hash
            subset_ordered_id_hash = smoke4_ordered_id_hash
        elif self.diagnostic12:
            subset_kind = "diagnostic12"
            subset_manifest_hash = diagnostic_manifest_hash
            subset_ordered_id_hash = diagnostic_ordered_id_hash
        elif self.smoke:
            subset_kind = "smoke8"
            subset_manifest_hash = None
            subset_ordered_id_hash = None
        else:
            subset_kind = "split"
            subset_manifest_hash = None
            subset_ordered_id_hash = None

        exploratory_manifest_hash = None
        run_order_hash = None
        retirement_policy_version = None
        retirement_meta = None
        if self.split == "exploratory_all" or self.diagnostic12 or self.smoke4:
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
                or bool((self.diagnostic12 or self.smoke4) and self.dry_run),
                git_commit=git_commit(self.root),
                prompt_hash=prompt_h,
                catalog_fingerprint=fingerprint,
                manifest_hash=str(exploratory_manifest_hash),
                run_order_hash=str(run_order_hash),
                timestamp=datetime.now(timezone.utc).isoformat(),
                existing_first_real=existing_first,
            )

        schema_h = combined_schema_hash()
        worst_case = worst_case_budget_estimate(self.limits)

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
            "smoke4": self.smoke4,
            "subset_kind": subset_kind,
            "subset_manifest_hash": subset_manifest_hash,
            "subset_ordered_id_hash": subset_ordered_id_hash,
            "diagnostic_manifest_hash": diagnostic_manifest_hash,
            "diagnostic_ordered_id_hash": diagnostic_ordered_id_hash,
            "smoke4_manifest_hash": smoke4_manifest_hash,
            "smoke4_ordered_id_hash": smoke4_ordered_id_hash,
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
            "generation_base_url_host": gen_host
            or url_host(
                self.gen_meta.get("base_url") or os.environ.get("LLM_BASE_URL")
            ),
            "embedding_provider": (self.emb_meta or {}).get("provider")
            or actual_emb_provider
            or expected_emb_provider,
            "embedding_model": (self.emb_meta or {}).get("model") or expected_emb,
            "embedding_profile_file": (self.emb_meta or {}).get("profile_file"),
            "embedding_base_url_host": emb_host
            or url_host(
                (self.emb_meta or {}).get("base_url")
                or os.environ.get("EMBEDDING_BASE_URL")
            ),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "json_mode": self.json_mode,
            "retrieval_top_k": int(self.retrieval.get("top_k", 20)),
            "lexical_weight": float(self.retrieval.get("lexical_weight", 0.12)),
            "limits": self.limits,
            "max_total_llm_calls_per_task": int(
                self.limits.get("max_total_llm_calls_per_task", 14)
            ),
            "worst_case_budget_estimate": worst_case,
            "system_prompt_hash": prompt_h,
            "system_prompt": self.system_prompt,
            "schema_hash": schema_h,
            "schema_hashes": schema_hashes(),
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
            "not_formal_accuracy_result": bool(self.smoke4),
            "dry_run_missing_env": getattr(self, "_dry_run_missing", None),
            "baseline_separation": {
                "does_not_modify_operation_visible_baseline": True,
                "does_not_modify_baseline_raw_outputs": True,
                "reuses_unified_gateway_and_catalog": True,
            },
            "formal_live_identity": {
                "generation_provider": EXPECTED_GENERATION_PROVIDER,
                "generation_host": EXPECTED_GENERATION_HOST,
                "generation_model": EXPECTED_GENERATION_MODEL,
                "embedding_model": EXPECTED_EMBEDDING_MODEL,
            },
            "no_mcp_zero_atr": True,
            "embedding_cache_dir": str(self.embedding_cache_dir),
            "embedding_batch_size": self.embedding_batch_size,
            "embedding_request_timeout_sec": self.embedding_request_timeout_sec,
            "embedding_max_retries": self.embedding_max_retries,
            "embedding_retry_backoff_seconds": list(
                self.embedding_retry_backoff_seconds
            ),
            "require_complete_index": self.require_complete_index,
            "allow_auto_build_index": self.allow_auto_build_index,
            "embedding_index_status": self.get_index_status(),
        }
        return {"items": tasks, "run_metadata": redact_secrets(meta)}

    def _catalog_fingerprint(self) -> str:
        return catalog_fingerprint(self.bundle.stats, self.bundle.source_catalogs)

    def _embedding_index_identity(self) -> Dict[str, Any]:
        emb_model = (
            (self.emb_meta or {}).get("model")
            or os.environ.get("EMBEDDING_MODEL")
            or self.config.get("embedding_model")
            or EXPECTED_EMBEDDING_MODEL
        )
        emb_host = host_from_base_url(
            (self.emb_meta or {}).get("base_url")
            or os.environ.get("EMBEDDING_BASE_URL")
        ) or url_host(
            (self.emb_meta or {}).get("base_url")
            or os.environ.get("EMBEDDING_BASE_URL")
        )
        if self.mock and not emb_host:
            emb_host = "mock-local"
        identity = build_index_identity(
            registry=self.bundle.registry,
            catalog_fingerprint=self._catalog_fingerprint(),
            embedding_model=str(emb_model),
            embedding_provider_host=emb_host,
        )
        self._index_identity = identity
        return identity

    def _make_embedding_backend(self, *, for_index_build: bool) -> Any:
        if self.mock:
            return DeterministicHashEmbeddingBackend()
        # Live / build-index: load embedding profile if not yet loaded.
        if self.emb_meta is None:
            self.emb_meta = load_embedding_profile(self.root, self.embedding_env_file)
        kwargs: Dict[str, Any] = {
            "base_url": os.environ.get("EMBEDDING_BASE_URL")
            or (self.emb_meta or {}).get("base_url"),
            "api_key": os.environ.get("EMBEDDING_API_KEY"),
            "model": os.environ.get("EMBEDDING_MODEL")
            or (self.emb_meta or {}).get("model"),
            "timeout_sec": self.embedding_request_timeout_sec,
        }
        if for_index_build:
            kwargs["max_retries"] = self.embedding_max_retries
            kwargs["retry_backoff_seconds"] = self.embedding_retry_backoff_seconds

            def _on_attempt(event: Dict[str, Any]) -> None:
                print(
                    "embedding_transport "
                    f"phase={event.get('phase')} attempt={event.get('attempt')}/"
                    f"{event.get('max_attempts')} batch_size={event.get('batch_size')} "
                    f"error_type={event.get('error_type')}",
                    flush=True,
                )

            kwargs["on_attempt"] = _on_attempt
        else:
            # Query-time embeddings: short retries only (index must already be complete).
            kwargs["max_retries"] = min(1, self.embedding_max_retries)
            kwargs["retry_backoff_seconds"] = self.embedding_retry_backoff_seconds[:1]
        return OpenAICompatibleEmbeddingBackend(**kwargs)

    def get_index_status(self) -> Dict[str, Any]:
        """Pure local index status — zero network calls."""
        if self.emb_meta is None and not self.mock:
            try:
                self.emb_meta = load_embedding_profile(
                    self.root, self.embedding_env_file
                )
            except FileNotFoundError:
                self.emb_meta = {
                    "model": self.config.get("embedding_model")
                    or EXPECTED_EMBEDDING_MODEL,
                    "base_url": None,
                }
        identity = self._embedding_index_identity()
        status = index_status(
            registry=self.bundle.registry,
            identity=identity,
            cache_dir=self.embedding_cache_dir,
        ).to_dict()
        self._index_status = status
        return status

    def build_index_only(self) -> Dict[str, Any]:
        """Build/resume shared catalog embedding index. No generation LLM / MCP / agent."""
        if self.emb_meta is None and not self.mock:
            self.emb_meta = load_embedding_profile(self.root, self.embedding_env_file)
        identity = self._embedding_index_identity()
        backend = self._make_embedding_backend(for_index_build=True)
        ov_dir = sibling_ov_cache_dir(self.root) if not self.mock else None
        result = build_or_resume_index(
            registry=self.bundle.registry,
            identity=identity,
            cache_dir=self.embedding_cache_dir,
            embed_fn=backend.embed,
            batch_size=self.embedding_batch_size,
            ov_cache_dir=ov_dir,
        )
        self._index_status = result
        return result

    def build_agent(self) -> ApvFinAgent:
        if self.dry_run:
            raise RuntimeError("build_agent must not be called during dry-run.")

        identity = self._embedding_index_identity()
        status = index_status(
            registry=self.bundle.registry,
            identity=identity,
            cache_dir=self.embedding_cache_dir,
        )
        self._index_status = status.to_dict()

        if self.require_complete_index and not status.complete:
            raise RuntimeError(
                "APV-Fin require-complete-index: shared operation embedding index "
                f"incomplete ({status.completed}/{status.total}) at "
                f"{self.embedding_cache_dir}. Run --build-index-only first. "
                "Live runs must not auto-rebuild the catalog index."
            )
        if (
            not self.mock
            and not status.complete
            and not self.allow_auto_build_index
        ):
            raise RuntimeError(
                "APV-Fin fail-closed: catalog embedding index incomplete and "
                "auto-build disabled for live runs. Use --build-index-only then "
                "--require-complete-index."
            )
        if self.mock and not status.complete:
            # Offline mock may build the deterministic index into the shared cache.
            self.build_index_only()
            status = index_status(
                registry=self.bundle.registry,
                identity=identity,
                cache_dir=self.embedding_cache_dir,
            )
            self._index_status = status.to_dict()
        if not status.complete:
            raise RuntimeError(
                "APV-Fin fail-closed: refusing to start agent without a complete "
                "operation embedding index."
            )

        preloaded = load_complete_index(
            registry=self.bundle.registry,
            identity=identity,
            cache_dir=self.embedding_cache_dir,
        )

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
            embedding = self._make_embedding_backend(for_index_build=False)
            gateway = UnifiedRealMcpGateway(
                catalog_bundle=self.bundle,
                allow_hkex_download=self.allow_hkex_download,
                persistent=True,
            )

        gateway.open()
        self._gateway = gateway
        resolver = OperationResolverServer(
            embedding_backend=embedding,
            registry=self.bundle.registry,
            real_mcp_gateway=gateway,
            preloaded_doc_vectors=preloaded,
            ensure_docs=False,
        )
        if resolver.financial is not None:
            gateway.close()
            raise RuntimeError(
                "APV-Fin fail-closed: OperationResolverServer unexpectedly retained "
                "FinancialServer."
            )
        live_mode = not self.mock
        cfg: Dict[str, Any] = {
            "llm": {"temperature": self.temperature, "max_tokens": self.max_tokens},
            "limits": dict(self.limits),
            "retrieval": dict(self.retrieval),
            "allow_mock_hooks": bool(self.mock),
        }
        if self.mock:
            # Mock-only hooks — unreachable in live_mode.
            cfg["mock_retrieve_arguments"] = {"*": {"symbol": "0700"}}
            cfg["mock_select_operation_id"] = (
                "aktools.stock_financial_hk_analysis_indicator_em"
            )
        return ApvFinAgent(
            llm=llm,
            resolver=resolver,
            config=cfg,
            metadata_by_id=self.metadata_by_id,
            runtime_operation_ids=list(self.bundle.stats.runtime_operation_ids),
            live_mode=live_mode,
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
            "plan_revision_log": result.get("plan_revision_log"),
            "scheduler_decisions": result.get("scheduler_decisions"),
            "monitor_decisions": result.get("monitor_decisions"),
            "retrieved_candidates": result.get("retrieved_candidates"),
            "selected_operations": result.get("selected_operations"),
            "invoked_operations": result.get("invoked_operations"),
            "successful_invoked_operations": result.get(
                "successful_invoked_operations"
            ),
            "describe_calls": result.get("describe_calls"),
            "invoke_calls": result.get("invoke_calls"),
            "evidence_ledger": result.get("evidence_ledger"),
            "compute_traces": result.get("compute_traces"),
            "validator": result.get("validator"),
            "stopping_reason": result.get("stopping_reason"),
            "budget": result.get("budget"),
            "error_type": result.get("error_type"),
            "llm_call_budget": result.get("llm_call_budget"),
            "llm_calls_used": result.get("llm_calls_used"),
            "llm_calls_remaining": result.get("llm_calls_remaining"),
            "llm_budget_exhausted": result.get("llm_budget_exhausted"),
            "calls_by_stage": result.get("calls_by_stage"),
            "execution_order": result.get("execution_order"),
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

        assert_output_dir_safe(
            output_dir=self.output_dir,
            dry_run=self.dry_run,
            mock=self.mock,
            resume=self.resume,
            run_meta=run_meta,
        )

        if self.dry_run:
            atomic_write_json(meta_path, run_meta)
            print("APV-FIN DRY-RUN OK")
            print(f"condition={run_meta['condition']}")
            print(f"method_version={run_meta['method_version']}")
            print(f"split={self.split}")
            print(f"subset_kind={run_meta.get('subset_kind')}")
            print(f"diagnostic12={self.diagnostic12}")
            print(f"smoke4={self.smoke4}")
            print(f"question_count={run_meta['question_count']}")
            print(f"domain_counts={run_meta['domain_counts']}")
            print(f"generation_provider={run_meta.get('generation_provider')}")
            print(f"generation_base_url_host={run_meta.get('generation_base_url_host')}")
            print(f"generation_model={run_meta.get('generation_model')}")
            print(f"embedding_model={run_meta.get('embedding_model')}")
            print(f"embedding_base_url_host={run_meta.get('embedding_base_url_host')}")
            print(f"system_prompt_hash={run_meta['system_prompt_hash']}")
            print(f"schema_hash={run_meta.get('schema_hash')}")
            print(f"operation_metadata_coverage={run_meta['operation_metadata_coverage']}")
            print(f"operation_metadata_hash={run_meta['operation_metadata_hash']}")
            print(f"diagnostic_manifest_hash={run_meta.get('diagnostic_manifest_hash')}")
            print(f"smoke4_manifest_hash={run_meta.get('smoke4_manifest_hash')}")
            print(f"catalog_fingerprint={run_meta['catalog_fingerprint']}")
            print(f"output_dir={self.output_dir}")
            return run_meta

        if self.resume and meta_path.exists():
            # assert_output_dir_safe already checked identity; re-raise clearly if needed
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
