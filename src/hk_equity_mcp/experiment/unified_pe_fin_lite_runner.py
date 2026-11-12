"""PE-Fin Lite unified experiment runner.

Reuses: UnifiedRealMcpGateway, frozen catalog, dataset loader, evaluator v1.2,
resume/checkpoint helpers, provider metadata / fail-closed env loading.
Does not copy APV-Fin Monitor / Capability Compiler / Replan controllers.
"""

from __future__ import annotations

import csv
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import yaml

from hk_equity_mcp.agent.llm_client import OpenAICompatibleChatClient, ScriptedLLMClient
from hk_equity_mcp.apv_fin.llm_budget import LlmCallBudgetExhausted
from hk_equity_mcp.apv_fin.metadata import load_and_validate_metadata
from hk_equity_mcp.apv_fin.runtime_safety import summarize_prediction_outcomes
from hk_equity_mcp.common.env_loader import (
    load_embedding_profile,
    load_generation_profile,
)
from hk_equity_mcp.experiment.runner import (
    already_done_ids,
    file_sha256,
    git_branch,
    git_commit,
    git_dirty,
    redact_secrets,
)
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
    assert_exploratory_unlocked,
    build_retirement_metadata,
    load_exploratory_manifest,
    load_exploratory_run_order,
)
from hk_equity_mcp.experiment.unified_pe_fin_lite_metrics import (
    aggregate_pe_fin_lite_metrics,
)
from hk_equity_mcp.experiment.unified_runner import (
    append_jsonl,
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
from hk_equity_mcp.pe_fin_lite import (
    PE_FIN_LITE_ARCHITECTURE,
    PE_FIN_LITE_CONDITION,
    PE_FIN_LITE_METHOD_VERSION,
)
from hk_equity_mcp.pe_fin_lite.eea_identity import (
    EEA_FLAGS,
    PE_FIN_LITE_EEA_CONDITION,
    PE_FIN_LITE_EEA_METHOD_VERSION,
    PE_FIN_LITE_EEA_PROTOCOL_VERSION,
)
from hk_equity_mcp.pe_fin_lite.agent import PeFinLiteAgent
from hk_equity_mcp.pe_fin_lite.budget import McpBudgetExhausted, structured_budget_failure
from hk_equity_mcp.pe_fin_replan import (
    PE_FIN_REPLAN_ARCHITECTURE,
    PE_FIN_REPLAN_CONDITION,
    PE_FIN_REPLAN_METHOD_VERSION,
    PE_FIN_REPLAN_PROTOCOL_VERSION,
    REPLAN_FLAGS,
)
from hk_equity_mcp.pe_fin_replan.agent import PeFinReplanAgent
from hk_equity_mcp.pe_fin_replan.identity import DEFAULT_BUDGETS as REPLAN_DEFAULT_BUDGETS
from hk_equity_mcp.pe_fin_replan.identity import replan_protocol_identity
from hk_equity_mcp.pe_fin_lite.checkpoint_observability import (
    CHECKPOINT_TRACE_CONTRACT_VERSION,
    checkpoint_observability_hash,
)
from hk_equity_mcp.pe_fin_lite.capability_profile import (
    capability_profile_hash,
    load_capability_profile,
)
from hk_equity_mcp.pe_fin_lite.disclosure_workflow import disclosure_workflow_hash
from hk_equity_mcp.pe_fin_lite.identity import (
    EXPECTED_EMBEDDING_MODEL,
    EXPECTED_GENERATION_HOST,
    EXPECTED_GENERATION_MODEL,
    EXPECTED_GENERATION_PROVIDER,
    METRICS_VERSION,
    assert_live_provider_identity,
    combined_schema_hash,
    host_from_base_url,
    protocol_identity_hashes,
    schema_hashes,
    worst_case_budget_estimate,
)
from hk_equity_mcp.pe_fin_lite.operation_embedding_index import (
    build_index_identity,
    build_or_resume_index,
    default_shared_cache_dir,
    identity_digest,
    index_status,
    load_complete_index,
    resolve_embedding_cache_dir,
)
from hk_equity_mcp.pe_fin_lite.prompts import PE_FIN_LITE_SYSTEM_PROMPT, pe_fin_lite_prompt_hash
from hk_equity_mcp.pe_fin_lite.retrieval_formula import (
    DESCRIBE_N,
    DOMAIN_HARD_FILTER,
    EXPECTED_EMBEDDING_DIMENSION,
    R5_CANDIDATE_LOCK_SHA256,
    RETRIEVAL_FORMULA_EXACT,
    RETRIEVAL_FORMULA_NAME,
    RETRIEVAL_FORMULA_VERSION,
    RETRIEVAL_TOP_K,
    runtime_retrieval_metadata,
)
from hk_equity_mcp.real_mcp.unified_catalog import build_unified_catalog
from hk_equity_mcp.real_mcp.unified_gateway import (
    ScriptedMockTransport,
    UnifiedRealMcpGateway,
)

DIAGNOSTIC12_MANIFEST = "benchmarks/manifests/hk_pe_fin_lite_diagnostic12_v1.json"
DIAGNOSTIC20_EEA_MANIFEST = "benchmarks/manifests/hk_pe_fin_lite_eea_diagnostic20_v1.json"
SMOKE5_MANIFEST = "benchmarks/manifests/hk_pe_fin_lite_smoke5_v1.json"

PE_RESUME_IDENTITY_KEYS = (
    "condition",
    "method_version",
    "architecture",
    "execution_backend",
    "generation_provider",
    "generation_base_url_host",
    "generation_model",
    "embedding_provider",
    "embedding_base_url_host",
    "embedding_model",
    "embedding_dimension",
    "system_prompt_hash",
    "schema_hash",
    "capability_profile_hash",
    "operation_metadata_hash",
    "catalog_fingerprint",
    "embedding_index_identity_sha256",
    "operation_documents_sha256",
    "retrieval_formula_name",
    "retrieval_formula_version",
    "retrieval_formula_hash",
    "lexical_weight",
    "domain_hard_filter",
    "candidate_lock_sha256",
    "dataset_split",
    "subset_kind",
    "subset_manifest_hash",
    "subset_ordered_id_hash",
    "diagnostic_manifest_hash",
    "smoke5_manifest_hash",
    "temperature",
    "max_tokens",
    "planner_max_tokens",
    "json_mode",
    "limits",
    "max_total_llm_calls_per_task",
    "mock",
    "dry_run",
    "checkpoint_trace_contract_version",
)


def load_pe_fin_lite_config(
    path: Optional[Path] = None, root: Optional[Path] = None
) -> Dict[str, Any]:
    root = root or repo_root()
    cfg_path = path or (root / "configs" / "hk_unified_pe_fin_lite_v1.yaml")
    with cfg_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_diagnostic12_manifest(root: Optional[Path] = None) -> Dict[str, Any]:
    root = root or repo_root()
    path = root / DIAGNOSTIC12_MANIFEST
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("manifest_id") != "hk_pe_fin_lite_diagnostic12_v1":
        raise RuntimeError("Unexpected diagnostic12 manifest_id")
    if len(data.get("ordered_task_ids") or []) != 12:
        raise RuntimeError("diagnostic12 must contain exactly 12 ordered_task_ids")
    return data


def load_diagnostic12_tasks(root: Optional[Path] = None) -> List[Dict[str, Any]]:
    root = root or repo_root()
    man = load_diagnostic12_manifest(root)
    all_tasks = load_split_tasks(root, "exploratory_all")
    return select_by_ids(all_tasks, man["ordered_task_ids"], preserve_id_order=True)


def load_smoke5_manifest(root: Optional[Path] = None) -> Dict[str, Any]:
    root = root or repo_root()
    path = root / SMOKE5_MANIFEST
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("manifest_id") != "hk_pe_fin_lite_smoke5_v1":
        raise RuntimeError("Unexpected smoke5 manifest_id")
    if len(data.get("ordered_task_ids") or []) != 5:
        raise RuntimeError("smoke5 must contain exactly 5 ordered_task_ids")
    return data


def load_smoke5_tasks(root: Optional[Path] = None) -> List[Dict[str, Any]]:
    root = root or repo_root()
    man = load_smoke5_manifest(root)
    all_tasks = load_split_tasks(root, "exploratory_all")
    return select_by_ids(all_tasks, man["ordered_task_ids"], preserve_id_order=True)


def load_diagnostic20_eea_manifest(root: Optional[Path] = None) -> Dict[str, Any]:
    root = root or repo_root()
    path = root / DIAGNOSTIC20_EEA_MANIFEST
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("manifest_id") != "hk_pe_fin_lite_eea_diagnostic20_v1":
        raise RuntimeError("Unexpected diagnostic20 EEA manifest_id")
    if len(data.get("ordered_task_ids") or []) != 20:
        raise RuntimeError("diagnostic20 EEA must contain exactly 20 ordered_task_ids")
    if data.get("explicitly_not_formal_accuracy") is not True:
        raise RuntimeError("diagnostic20 EEA must declare explicitly_not_formal_accuracy")
    return data


def load_diagnostic20_eea_tasks(root: Optional[Path] = None) -> List[Dict[str, Any]]:
    root = root or repo_root()
    man = load_diagnostic20_eea_manifest(root)
    all_tasks = load_split_tasks(root, "exploratory_all")
    tasks = select_by_ids(all_tasks, man["ordered_task_ids"], preserve_id_order=True)
    for t in tasks:
        if t.get("source_split") != "development":
            raise RuntimeError(
                f"diagnostic20 EEA task {t.get('task_id')} is not development-origin"
            )
    return tasks


def assert_output_dir_safe(
    *,
    output_dir: Path,
    dry_run: bool,
    mock: bool,
    resume: bool,
    run_meta: Dict[str, Any],
) -> None:
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
        if existing.get("subset_kind") != run_meta.get("subset_kind"):
            raise RuntimeError(
                "Resume refused: subset_kind mismatch "
                f"(existing={existing.get('subset_kind')!r} "
                f"current={run_meta.get('subset_kind')!r})."
            )
        mismatches = []
        for key in PE_RESUME_IDENTITY_KEYS:
            if existing.get(key) != run_meta.get(key):
                mismatches.append(
                    f"{key}: existing={existing.get(key)!r} current={run_meta.get(key)!r}"
                )
        if mismatches:
            raise RuntimeError(
                "Resume refused: PE-Fin Lite run_metadata identity mismatch. "
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
        self._queue = list(self._script)

    def complete(
        self,
        messages,
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        **kwargs,
    ):
        if not self._queue:
            self._queue = list(self._script)
        return super().complete(
            messages, temperature=temperature, max_tokens=max_tokens, **kwargs
        )


def _mock_pe_fin_lite_scripted_llm() -> ScriptedLLMClient:
    plan = {
        "schema_version": "pe_fin_lite_plan_v1",
        "steps": [
            {
                "id": "s1",
                "type": "retrieve",
                "domain": "market",
                "objective": "retrieve supporting HK equity market evidence",
                "required_outputs": ["close", "trade_date"],
                "depends_on": [],
                "entity": None,
                "period": None,
                "date": None,
                "compute_spec": None,
                "workflow": None,
            },
            {
                "id": "s2",
                "type": "reason",
                "domain": "general",
                "objective": "summarize verified evidence for final answer",
                "required_outputs": [],
                "depends_on": ["s1"],
                "entity": None,
                "period": None,
                "date": None,
                "compute_spec": None,
                "workflow": None,
            },
        ],
    }
    reason = {
        "summary": "mock evidence supports a concise answer with stated limitations",
        "claims": ["mock claim"],
        "gaps": [],
    }
    final = {
        "schema_version": "pe_fin_lite_final_v1",
        "status": "answered",
        "final_answer": "mock answer",
        "reasoning_brief": "offline PE-Fin Lite mock trajectory",
        "evidence_ids": [],
        "clarification_question": None,
        "refusal_reason": None,
        "currency": None,
        "unit": None,
        "metric_basis": None,
        "period": None,
        "entity": None,
    }
    # planner → binder → reason → final  (binder may fail schema then mock_binder used)
    # With use_mock_binder=True, binder LLM turn is skipped.
    turns = [plan, reason, final]
    return _CyclingScriptedLLM([json.dumps(t, ensure_ascii=False) for t in turns])


class _StageAwareReplanMockLLM(ScriptedLLMClient):
    """Offline mock: Planner once, then Replanner Plan/Response by stage."""

    def __init__(self, model: str = "mock-llm-replan"):
        plan = {
            "schema_version": "pe_fin_lite_plan_v1",
            "steps": [
                {
                    "id": "s1",
                    "type": "retrieve",
                    "domain": "market",
                    "objective": "retrieve supporting HK equity market evidence",
                    "required_outputs": ["close", "trade_date"],
                    "depends_on": [],
                    "entity": None,
                    "period": None,
                    "date": None,
                    "compute_spec": None,
                    "workflow": None,
                },
                {
                    "id": "s2",
                    "type": "reason",
                    "domain": "general",
                    "objective": "summarize verified evidence for final answer",
                    "required_outputs": [],
                    "depends_on": ["s1"],
                    "entity": None,
                    "period": None,
                    "date": None,
                    "compute_spec": None,
                    "workflow": None,
                },
            ],
        }
        keep_reason = {
            "schema_version": "pe_fin_replan_v1_1",
            "action": "plan",
            "decision_reason": "Keep remaining reason step after retrieve",
            "remaining_steps": [
                {
                    "id": "s2",
                    "type": "reason",
                    "domain": "general",
                    "objective": "summarize verified evidence for final answer",
                    "required_outputs": [],
                    "depends_on": [],
                    "entity": None,
                    "period": None,
                    "date": None,
                    "compute_spec": None,
                    "workflow": None,
                }
            ],
        }
        reason = {
            "summary": "mock evidence supports a concise answer with stated limitations",
            "claims": ["mock claim"],
            "gaps": [],
        }
        final_resp = {
            "schema_version": "pe_fin_replan_v1_1",
            "action": "response",
            "decision_reason": "Goal achieved from verified mock trajectory",
            "response": {
                "completion": "answered",
                "final_answer": "mock answer from replanner",
                "reasoning_brief": "offline PE-Fin RePlan mock trajectory",
            },
        }
        # Queue unused — complete() is stage-routed.
        plan_s = json.dumps(plan, ensure_ascii=False)
        super().__init__([plan_s], model=model)
        self._script = [plan_s]
        self._plan = plan_s
        self._keep_reason = json.dumps(keep_reason, ensure_ascii=False)
        self._reason = json.dumps(reason, ensure_ascii=False)
        self._final = json.dumps(final_resp, ensure_ascii=False)
        self._replan_calls = 0

    def reset_script(self) -> None:
        self._queue = list(self._script)
        self._replan_calls = 0

    def complete(
        self,
        messages,
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        stage: Optional[str] = None,
        **kwargs,
    ):
        st = str(stage or "")
        if st.startswith("planner"):
            content = self._plan
        elif st.startswith("reason"):
            content = self._reason
        elif st.startswith("replanner"):
            self._replan_calls += 1
            # First replan after retrieve → keep reason; later → response.
            content = self._keep_reason if self._replan_calls == 1 else self._final
        else:
            content = self._final
        return {
            "content": content,
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 10,
                "total_tokens": 20,
                "input_tokens": 10,
                "output_tokens": 10,
            },
            "latency_ms": 1.0,
            "model": self.model,
            "provider": "mock",
        }


def _mock_pe_fin_replan_scripted_llm() -> ScriptedLLMClient:
    return _StageAwareReplanMockLLM()


class UnifiedHkPeFinLiteRunner:
    def __init__(
        self,
        *,
        output_dir: Path,
        split: str = "development",
        smoke: bool = False,
        diagnostic12: bool = False,
        smoke5: bool = False,
        diagnostic20_eea: bool = False,
        diagnostic20_replan: bool = False,
        eea_enabled: bool = False,
        replan_enabled: bool = False,
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
        self.smoke5 = bool(smoke5)
        self.diagnostic20_eea = bool(diagnostic20_eea)
        self.diagnostic20_replan = bool(diagnostic20_replan)
        self.replan_enabled = bool(replan_enabled) or self.diagnostic20_replan
        # RePlan always includes EEA execution/evidence alignment.
        self.eea_enabled = (
            bool(eea_enabled) or self.diagnostic20_eea or self.replan_enabled
        )
        if self.diagnostic12 and self.smoke5:
            raise RuntimeError("Cannot combine --diagnostic12 and --smoke5")
        if self.diagnostic20_eea and (self.diagnostic12 or self.smoke5):
            raise RuntimeError("Cannot combine --diagnostic20-eea with diagnostic12/smoke5")
        if self.diagnostic20_replan and (self.diagnostic12 or self.smoke5):
            raise RuntimeError(
                "Cannot combine --diagnostic20-replan with diagnostic12/smoke5"
            )
        if self.diagnostic20_eea and self.diagnostic20_replan:
            raise RuntimeError(
                "Cannot combine --diagnostic20-eea with --diagnostic20-replan "
                "(separate output dirs / resume identities)"
            )
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
        self.config = config or load_pe_fin_lite_config(root=self.root)
        self.allow_hkex_download = allow_hkex_download
        self.json_mode = json_mode
        self.require_complete_index = bool(require_complete_index)
        self.allow_auto_build_index = bool(allow_auto_build_index)
        self.embedding_cache_dir = resolve_embedding_cache_dir(
            self.root, embedding_cache_dir
        )
        self.temperature = float(self.config.get("temperature", 0.0))
        self.max_tokens = int(self.config.get("max_tokens", 2048))
        from hk_equity_mcp.pe_fin_lite.schemas import PLANNER_MAX_TOKENS

        self.planner_max_tokens = int(
            self.config.get("planner_max_tokens", PLANNER_MAX_TOKENS)
        )
        self.limits = dict(self.config.get("limits") or {})
        if self.replan_enabled:
            # Force RePlan v1.1 budgets/rounds (do not inherit Lite's LLM=8 / steps=6).
            for k, v in REPLAN_DEFAULT_BUDGETS.items():
                self.limits[k] = v
        self.retrieval = dict(self.config.get("retrieval") or {})
        self.embedding_batch_size = int(self.config.get("embedding_batch_size", 16))
        self.embedding_request_timeout_sec = float(
            self.config.get("embedding_request_timeout_sec", 120)
        )
        self.embedding_max_retries = int(self.config.get("embedding_max_retries", 3))
        self.embedding_retry_backoff_seconds = [
            float(x)
            for x in (self.config.get("embedding_retry_backoff_seconds") or [1, 2, 4])
        ]
        if self.replan_enabled:
            self.condition = PE_FIN_REPLAN_CONDITION
            self.method_version = PE_FIN_REPLAN_METHOD_VERSION
            self.protocol_version = PE_FIN_REPLAN_PROTOCOL_VERSION
            self.architecture = PE_FIN_REPLAN_ARCHITECTURE
        elif self.eea_enabled:
            self.condition = PE_FIN_LITE_EEA_CONDITION
            self.method_version = PE_FIN_LITE_EEA_METHOD_VERSION
            self.protocol_version = PE_FIN_LITE_EEA_PROTOCOL_VERSION
            self.architecture = PE_FIN_LITE_ARCHITECTURE
        else:
            self.condition = PE_FIN_LITE_CONDITION
            self.method_version = PE_FIN_LITE_METHOD_VERSION
            self.protocol_version = None
            self.architecture = PE_FIN_LITE_ARCHITECTURE
        self.execution_backend = "real_mcp"
        self.system_prompt = PE_FIN_LITE_SYSTEM_PROMPT
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
        profile_rel = self.config.get("capability_profile") or (
            "configs/capabilities/hk155_capability_profile_v1.yaml"
        )
        self.capability_profile = load_capability_profile(
            self.root, path=self.root / str(profile_rel)
        )
        self._exploratory_manifest: Optional[Dict[str, Any]] = None
        self._exploratory_run_order: Optional[Dict[str, Any]] = None
        self._diagnostic_manifest: Optional[Dict[str, Any]] = None
        self._smoke5_manifest: Optional[Dict[str, Any]] = None

    def _load_tasks(self) -> List[Dict[str, Any]]:
        effective_split = self.split
        confirm_retire = self.confirm_retire_heldout
        if (
            self.diagnostic12
            or self.smoke5
            or self.diagnostic20_eea
            or self.diagnostic20_replan
        ):
            effective_split = "exploratory_all"
            if self.dry_run and not confirm_retire:
                confirm_retire = True
            if not self.dry_run and not confirm_retire:
                raise RuntimeError(
                    "diagnostic12/smoke5/diagnostic20-eea/diagnostic20-replan requires "
                    "--split exploratory_all --confirm-retire-heldout (except dry-run)."
                )

        self._heldout_status = assert_heldout_unlocked(
            split=effective_split, confirm_heldout_final=self.confirm_heldout_final
        )
        self._exploratory_status = assert_exploratory_unlocked(
            split=effective_split, confirm_retire_heldout=confirm_retire
        )

        if self.diagnostic20_replan:
            # Reuse frozen Diagnostic-20 IDs/order; separate subset_kind for resume.
            self._diagnostic20_eea_manifest = load_diagnostic20_eea_manifest(self.root)
            self.split = "exploratory_all"
            tasks = load_diagnostic20_eea_tasks(self.root)
            self._exploratory_manifest = load_exploratory_manifest(self.root)
            self._exploratory_run_order = load_exploratory_run_order(self.root)
        elif self.diagnostic20_eea:
            self._diagnostic20_eea_manifest = load_diagnostic20_eea_manifest(self.root)
            self.split = "exploratory_all"
            tasks = load_diagnostic20_eea_tasks(self.root)
            self._exploratory_manifest = load_exploratory_manifest(self.root)
            self._exploratory_run_order = load_exploratory_run_order(self.root)
        elif self.diagnostic12:
            self._diagnostic_manifest = load_diagnostic12_manifest(self.root)
            self.split = "exploratory_all"
            tasks = load_diagnostic12_tasks(self.root)
            self._exploratory_manifest = load_exploratory_manifest(self.root)
            self._exploratory_run_order = load_exploratory_run_order(self.root)
        elif self.smoke5:
            self._smoke5_manifest = load_smoke5_manifest(self.root)
            self._diagnostic_manifest = load_diagnostic12_manifest(self.root)
            self.split = "exploratory_all"
            tasks = load_smoke5_tasks(self.root)
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
            tasks = select_by_ids(tasks, self.ids, preserve_id_order=True)
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
                raise RuntimeError(f"PE-Fin Lite prompt unexpectedly contains {banned}")

        prompt_h = pe_fin_lite_prompt_hash()
        tasks = self._load_tasks()
        stats = self.bundle.stats
        meta_hash = file_sha256(self.metadata_path)
        profile_hash = capability_profile_hash(self.capability_profile)
        schema_h = combined_schema_hash()
        schemas = schema_hashes()
        protocol_ids = protocol_identity_hashes()

        expected_model = str(
            self.config.get("generation_model") or EXPECTED_GENERATION_MODEL
        )
        expected_provider = str(
            self.config.get("generation_provider") or EXPECTED_GENERATION_PROVIDER
        )
        expected_host = str(
            self.config.get("generation_host") or EXPECTED_GENERATION_HOST
        )
        expected_emb_model = str(
            self.config.get("embedding_model") or EXPECTED_EMBEDDING_MODEL
        )

        if self.dry_run:
            self.gen_meta = {
                "provider": expected_provider,
                "model": expected_model,
                "base_url": f"https://{expected_host}/v1",
            }
            self.emb_meta = {
                "provider": "openai_compatible",
                "model": expected_emb_model,
                "base_url": f"https://{expected_host}/v1",
            }
        elif self.mock:
            self.gen_meta = {
                "provider": "mock",
                "model": "mock-llm",
                "base_url": None,
            }
            self.emb_meta = {
                "provider": "mock",
                "model": expected_emb_model,
                "base_url": None,
            }
        else:
            self.gen_meta = load_generation_profile(self.root, self.env_file)
            self.emb_meta = load_embedding_profile(self.root, self.embedding_env_file)
            assert_live_provider_identity(
                generation_provider=str(
                    self.gen_meta.get("provider") or expected_provider
                ),
                generation_host=host_from_base_url(self.gen_meta.get("base_url"))
                or expected_host,
                generation_model=str(self.gen_meta.get("model") or expected_model),
                embedding_model=str(self.emb_meta.get("model") or expected_emb_model),
                embedding_host=host_from_base_url(self.emb_meta.get("base_url")),
                embedding_provider=str(self.emb_meta.get("provider") or "openai_compatible"),
            )

        domain_counts = dict(Counter(t.get("domain") for t in tasks))
        subset_kind = "full"
        subset_manifest_hash = None
        subset_ordered_id_hash = None
        diagnostic_manifest_hash = None
        smoke5_manifest_hash = None
        if self.diagnostic12:
            subset_kind = "diagnostic12"
            man = self._diagnostic_manifest or load_diagnostic12_manifest(self.root)
            subset_manifest_hash = man.get("manifest_hash")
            subset_ordered_id_hash = man.get("ordered_id_hash")
            diagnostic_manifest_hash = man.get("manifest_hash")
        elif self.smoke5:
            subset_kind = "smoke5"
            man = self._smoke5_manifest or load_smoke5_manifest(self.root)
            subset_manifest_hash = man.get("manifest_hash")
            subset_ordered_id_hash = man.get("ordered_id_hash")
            smoke5_manifest_hash = man.get("manifest_hash")
            dman = self._diagnostic_manifest or load_diagnostic12_manifest(self.root)
            diagnostic_manifest_hash = dman.get("manifest_hash")
        elif self.diagnostic20_replan:
            subset_kind = "diagnostic20_replan"
            man = getattr(self, "_diagnostic20_eea_manifest", None) or load_diagnostic20_eea_manifest(
                self.root
            )
            subset_manifest_hash = man.get("manifest_sha256")
            subset_ordered_id_hash = man.get("ordered_task_ids_hash")
            diagnostic_manifest_hash = man.get("manifest_sha256")
        elif self.diagnostic20_eea:
            subset_kind = "diagnostic20_eea"
            man = getattr(self, "_diagnostic20_eea_manifest", None) or load_diagnostic20_eea_manifest(
                self.root
            )
            subset_manifest_hash = man.get("manifest_sha256")
            subset_ordered_id_hash = man.get("ordered_task_ids_hash")
            diagnostic_manifest_hash = man.get("manifest_sha256")
        elif self.smoke:
            subset_kind = "smoke"

        cat_fp = catalog_fingerprint(self.bundle.stats, self.bundle.source_catalogs)
        index_identity = self._embedding_index_identity()
        index_identity_sha = identity_digest(index_identity)
        limits = dict(self.limits)
        # R5 formula identity — changing these blocks resume of pre-R5 outputs.
        emb_dim = int(
            index_identity.get("expected_vector_dimension")
            or EXPECTED_EMBEDDING_DIMENSION
        )
        docs_hash = index_identity.get("documents_hash")
        formula_meta = runtime_retrieval_metadata(
            embedding_provider_host=host_from_base_url(
                (self.emb_meta or {}).get("base_url")
            ),
            embedding_model=(self.emb_meta or {}).get("model") or EXPECTED_EMBEDDING_MODEL,
            embedding_dimension=emb_dim,
            operation_documents_sha256=docs_hash,
            operation_index_identity=index_identity_sha,
            catalog_fingerprint=cat_fp,
        )
        run_meta: Dict[str, Any] = {
            "condition": self.condition,
            "method_version": self.method_version,
            "architecture": self.architecture,
            "execution_backend": self.execution_backend,
            "metrics_version": METRICS_VERSION,
            "generation_provider": self.gen_meta.get("provider"),
            "generation_base_url_host": host_from_base_url(self.gen_meta.get("base_url")),
            "generation_model": self.gen_meta.get("model"),
            "embedding_provider": (self.emb_meta or {}).get("provider"),
            "embedding_base_url_host": host_from_base_url(
                (self.emb_meta or {}).get("base_url")
            ),
            "embedding_model": (self.emb_meta or {}).get("model"),
            "embedding_dimension": emb_dim,
            "operation_documents_sha256": docs_hash,
            "retrieval_formula_name": RETRIEVAL_FORMULA_NAME,
            "retrieval_formula_version": RETRIEVAL_FORMULA_VERSION,
            "retrieval_formula_exact": RETRIEVAL_FORMULA_EXACT,
            "retrieval_formula_hash": formula_meta["retrieval_formula_hash"],
            "lexical_weight": 0.12,
            "domain_hard_filter": DOMAIN_HARD_FILTER,
            "candidate_lock_sha256": R5_CANDIDATE_LOCK_SHA256,
            "retrieval_top_k": int(limits.get("retrieval_top_k", RETRIEVAL_TOP_K)),
            "describe_top_n": int(limits.get("describe_top_n", DESCRIBE_N)),
            "retrieval_formula": formula_meta,
            "system_prompt_hash": prompt_h,
            "schema_hash": schema_h,
            "schema_hashes": schemas,
            "protocol_identity": protocol_ids,
            "canonical_output_vocabulary_hash": protocol_ids[
                "canonical_output_vocabulary_hash"
            ],
            "scope_contract_hash": protocol_ids["scope_contract_hash"],
            "argument_grounding_hash": protocol_ids["argument_grounding_hash"],
            "provenance_contract_hash": protocol_ids["provenance_contract_hash"],
            "evidence_acceptance_hash": protocol_ids["evidence_acceptance_hash"],
            "disclosure_workflow_hash": protocol_ids.get("disclosure_workflow_hash")
            or disclosure_workflow_hash(),
            "capability_profile_id": self.capability_profile.get("profile_id"),
            "capability_profile_hash": profile_hash,
            "disclosure_workflow_hash": disclosure_workflow_hash(),
            "operation_metadata_hash": meta_hash,
            "operation_metadata_coverage": {
                "runtime_operation_count": len(stats.runtime_operation_ids),
                "metadata_count": len(self.metadata_by_id),
            },
            "catalog_fingerprint": cat_fp,
            "embedding_index_identity_sha256": index_identity_sha,
            "dataset_split": self.split,
            "subset_kind": subset_kind,
            "subset_manifest_hash": subset_manifest_hash,
            "subset_ordered_id_hash": subset_ordered_id_hash,
            "diagnostic_manifest_hash": diagnostic_manifest_hash,
            "smoke5_manifest_hash": smoke5_manifest_hash,
            "diagnostic12": self.diagnostic12,
            "smoke5": self.smoke5,
            "diagnostic20_eea": self.diagnostic20_eea,
            "diagnostic20_replan": self.diagnostic20_replan,
            "eea_enabled": self.eea_enabled,
            "replan_enabled": self.replan_enabled,
            "protocol_version": self.protocol_version,
            **(
                {
                    "posthoc_exploratory": True,
                    "formal_primary_result": False,
                    "developed_after_hk155_result_inspection": True,
                    "eea_flags": dict(EEA_FLAGS),
                }
                if self.eea_enabled and not self.replan_enabled
                else {}
            ),
            **(
                {
                    "posthoc_exploratory": True,
                    "formal_primary_result": False,
                    "developed_after_hk155_result_inspection": True,
                    "base_method": "pe_fin_lite_v1_2_eea",
                    "replan_flags": dict(REPLAN_FLAGS),
                    "replan_identity": replan_protocol_identity(),
                    "eea_flags": dict(EEA_FLAGS),
                }
                if self.replan_enabled
                else {}
            ),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "planner_max_tokens": self.planner_max_tokens,
            "checkpoint_trace_contract_version": CHECKPOINT_TRACE_CONTRACT_VERSION,
            "checkpoint_observability_hash": checkpoint_observability_hash(),
            "json_mode": self.json_mode,
            "limits": limits,
            "max_total_llm_calls_per_task": int(
                limits.get("max_total_llm_calls_per_task", 8)
            ),
            "worst_case_budget": worst_case_budget_estimate(limits),
            "mock": self.mock,
            "dry_run": self.dry_run,
            "question_count": len(tasks),
            "domain_counts": domain_counts,
            "git_commit": git_commit(self.root),
            "git_branch": git_branch(self.root),
            "git_dirty": dirty,
            "embedding_cache_dir": str(self.embedding_cache_dir),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "monitor_enabled": False,
            "global_replan_enabled": False,
            "capability_compiler_enabled": False,
        }
        if self._exploratory_manifest is not None:
            run_meta["exploratory"] = build_retirement_metadata(
                dry_run=self.dry_run,
                mock=self.mock,
                confirm_retire_heldout=self.confirm_retire_heldout
                or bool(
                    (
                        self.diagnostic12
                        or self.smoke5
                        or self.diagnostic20_eea
                        or self.diagnostic20_replan
                    )
                    and self.dry_run
                ),
                git_commit=git_commit(self.root),
                prompt_hash=prompt_h,
                catalog_fingerprint=cat_fp,
                manifest_hash=str(
                    (self._exploratory_manifest or {}).get("manifest_hash") or ""
                ),
                run_order_hash=str(
                    (self._exploratory_run_order or {}).get("run_order_hash")
                    or (self._exploratory_run_order or {}).get("ordered_id_hash")
                    or ""
                ),
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        return {"items": tasks, "run_metadata": run_meta}

    def _embedding_index_identity(self) -> Dict[str, Any]:
        emb_model = str(
            (self.emb_meta or {}).get("model")
            or self.config.get("embedding_model")
            or EXPECTED_EMBEDDING_MODEL
        )
        emb_host = host_from_base_url((self.emb_meta or {}).get("base_url"))
        identity = build_index_identity(
            registry=self.bundle.registry,
            catalog_fingerprint=catalog_fingerprint(
                self.bundle.stats, self.bundle.source_catalogs
            ),
            embedding_model=emb_model,
            embedding_provider_host=emb_host,
        )
        self._index_identity = identity
        return identity

    def _make_embedding_backend(self, *, for_index_build: bool) -> Any:
        if self.mock:
            return DeterministicHashEmbeddingBackend()
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
        else:
            kwargs["max_retries"] = min(1, self.embedding_max_retries)
            kwargs["retry_backoff_seconds"] = self.embedding_retry_backoff_seconds[:1]
        return OpenAICompatibleEmbeddingBackend(**kwargs)

    def get_index_status(self) -> Dict[str, Any]:
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
        if self.emb_meta is None and not self.mock:
            self.emb_meta = load_embedding_profile(self.root, self.embedding_env_file)
        identity = self._embedding_index_identity()
        backend = self._make_embedding_backend(for_index_build=True)
        result = build_or_resume_index(
            registry=self.bundle.registry,
            identity=identity,
            cache_dir=self.embedding_cache_dir,
            embed_fn=backend.embed,
            batch_size=self.embedding_batch_size,
            ov_cache_dir=None,
        )
        self._index_status = result
        return result

    def build_agent(self):
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
                "PE-Fin Lite require-complete-index: shared operation embedding index "
                f"incomplete ({status.completed}/{status.total}) at "
                f"{self.embedding_cache_dir}. Run --build-index-only first."
            )
        if not self.mock and not status.complete and not self.allow_auto_build_index:
            raise RuntimeError(
                "PE-Fin Lite fail-closed: catalog embedding index incomplete and "
                "auto-build disabled for live runs. Use --build-index-only then "
                "--require-complete-index."
            )
        if self.mock and not status.complete:
            self.build_index_only()
            status = index_status(
                registry=self.bundle.registry,
                identity=identity,
                cache_dir=self.embedding_cache_dir,
            )
            self._index_status = status.to_dict()
        if not status.complete:
            raise RuntimeError(
                "PE-Fin Lite fail-closed: refusing to start agent without a complete "
                "operation embedding index."
            )

        preloaded = load_complete_index(
            registry=self.bundle.registry,
            identity=identity,
            cache_dir=self.embedding_cache_dir,
        )

        if self.mock:
            llm: Any = (
                _mock_pe_fin_replan_scripted_llm()
                if self.replan_enabled
                else _mock_pe_fin_lite_scripted_llm()
            )
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
                "PE-Fin Lite fail-closed: OperationResolverServer unexpectedly retained "
                "FinancialServer."
            )
        cfg: Dict[str, Any] = {
            "llm": {
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "planner_max_tokens": self.planner_max_tokens,
            },
            "planner_max_tokens": self.planner_max_tokens,
            "limits": dict(self.limits),
            "retrieval": dict(self.retrieval),
            "use_mock_binder": bool(self.mock),
            "eea_enabled": bool(self.eea_enabled),
        }
        agent_kwargs = dict(
            llm=llm,
            resolver=resolver,
            capability_profile=self.capability_profile,
            config=cfg,
            metadata_by_id=self.metadata_by_id,
            runtime_operation_ids=list(self.bundle.stats.runtime_operation_ids),
            live_mode=not self.mock,
            allow_hkex_download=self.allow_hkex_download,
        )
        if self.replan_enabled:
            return PeFinReplanAgent(**agent_kwargs)
        return PeFinLiteAgent(**agent_kwargs)

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
            "gold_operations": item.get("gold_operations"),
            "llm_calls": result.get("llm_calls"),
            "tokens": tokens,
            "embedding_calls": result.get("embedding_query_calls")
            or (result.get("budget") or {}).get("embedding_query_calls"),
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
            "architecture": self.architecture,
            "eea_enabled": bool(self.eea_enabled),
            "replan_enabled": bool(self.replan_enabled),
            "protocol_version": self.protocol_version,
            "posthoc_exploratory": bool(self.eea_enabled or self.replan_enabled),
            "formal_primary_result": (
                False if (self.eea_enabled or self.replan_enabled) else None
            ),
            "answerability_state": result.get("answerability_state"),
            "answerability_reason_code": result.get("answerability_reason_code"),
            "core_outputs_satisfied": result.get("core_outputs_satisfied"),
            "optional_outputs_missing": result.get("optional_outputs_missing"),
            "infeasibility_evidence": result.get("infeasibility_evidence"),
            "initial_plan": result.get("initial_plan"),
            "original_plan_hash": result.get("original_plan_hash"),
            "executed_steps": result.get("executed_steps"),
            "remaining_plan_history": result.get("remaining_plan_history"),
            "past_steps": result.get("past_steps"),
            "replan_calls": result.get("replan_calls"),
            "replan_repairs": result.get("replan_repairs"),
            "replan_decisions": result.get("replan_decisions"),
            "plan_diff_history": result.get("plan_diff_history"),
            "steps_added": result.get("steps_added"),
            "steps_removed": result.get("steps_removed"),
            "steps_modified": result.get("steps_modified"),
            "early_answer_count": result.get("early_answer_count"),
            "recovery_attempts": result.get("recovery_attempts"),
            "recovery_successes": result.get("recovery_successes"),
            "failed_signature_blocks": result.get("failed_signature_blocks"),
            "no_progress_rounds": result.get("no_progress_rounds"),
            "final_response_source": result.get("final_response_source"),
            "completion": result.get("completion"),
            "execution_round_count": result.get("execution_round_count"),
            "max_execution_rounds": result.get("max_execution_rounds"),
            "replan_round_count": result.get("replan_round_count"),
            "max_replan_rounds": result.get("max_replan_rounds"),
            "forced_final_round": result.get("forced_final_round"),
            "round7_response_only": result.get("round7_response_only"),
            "premature_insufficient_blocked_count": result.get(
                "premature_insufficient_blocked_count"
            ),
            "llm_calls_by_stage": result.get("llm_calls_by_stage")
            or result.get("calls_by_stage"),
            "resource_limit_failure": result.get("resource_limit_failure"),
            "loop_prevention_trigger": result.get("loop_prevention_trigger"),
            "graph_edges": result.get("graph_edges"),
            "replan_identity": result.get("replan_identity"),
            "replan_flags": result.get("replan_flags"),
            "step_statuses": result.get("step_statuses"),
            "retrieval_funnels": result.get("retrieval_funnels"),
            "predicted_operations": result.get("predicted_operations"),
            "selected_operations": result.get("selected_operations"),
            "invoked_operations": result.get("invoked_operations"),
            "successful_invoked_operations": result.get(
                "successful_invoked_operations"
            ),
            "binding_calls": result.get("binding_calls"),
            "binding_repair_count": result.get("binding_repair_count"),
            "describe_calls": result.get("describe_calls"),
            "invoke_calls": result.get("invoke_calls"),
            "evidence_store": result.get("evidence_store"),
            "evidence_ledger": result.get("evidence_ledger"),
            "compute_traces": result.get("compute_traces"),
            "reason_artifacts": result.get("reason_artifacts"),
            "budget": result.get("budget"),
            "error_type": result.get("error_type"),
            "stopping_reason": result.get("stopping_reason"),
            "monitor_decisions": [],
            "plan_revisions": result.get("plan_revisions")
            if self.replan_enabled
            else [],
            "primary_selected_count": result.get("primary_selected_count"),
            "fallback_used_count": result.get("fallback_used_count"),
            "local_fallback_count": result.get("local_fallback_count"),
            "tool_execution_success": result.get("tool_execution_success"),
            "all_tool_invocations_successful": result.get(
                "all_tool_invocations_successful"
            ),
            "successful_invocation_count": result.get("successful_invocation_count"),
            "failed_invocation_count": result.get("failed_invocation_count"),
            "real_mcp_execution_count": result.get("real_mcp_execution_count"),
            "auto_grade_status": grade.get("auto_grade_status"),
            "answer_correct": grade.get("answer_correct"),
            "auto_grade_reason": grade.get("reason"),
            "manual_answer_correctness": "",
            "manual_evidence_correctness": "",
            "manual_unsupported_claim": "",
            "manual_causal_overclaim": "",
            "manual_overall_task_success": "",
            "checkpoint_observability": result.get("checkpoint_observability"),
            "checkpoint_trace_contract_version": result.get(
                "checkpoint_trace_contract_version"
            ),
            "checkpoint_observability_hash": result.get(
                "checkpoint_observability_hash"
            ),
        }
        agent_tool_ok = result.get("all_tool_invocations_successful")
        agent_succ = result.get("successful_invocation_count")
        agent_fail = result.get("failed_invocation_count")
        row.update(metrics)
        if "all_tool_invocations_successful" in result:
            row["all_tool_invocations_successful"] = agent_tool_ok
            row["tool_execution_success"] = result.get("tool_execution_success")
        if agent_succ is not None:
            row["successful_invocation_count"] = agent_succ
        if agent_fail is not None:
            row["failed_invocation_count"] = agent_fail
        # Prefer agent-recorded gateway invocation accounting over empty
        # trace-derived evaluate_task defaults (PE-Fin Lite funnel traces).
        agent_mcp = result.get("real_mcp_execution_count")
        agent_underlying = result.get("underlying_operation_calls")
        if agent_mcp is not None:
            row["real_mcp_execution_count"] = int(agent_mcp)
        elif agent_underlying is not None:
            row["real_mcp_execution_count"] = int(agent_underlying)
        if agent_underlying is not None:
            row["underlying_operation_calls"] = int(agent_underlying)
        agent_invoke = result.get("invoke_records")
        if agent_invoke is None:
            agent_invoke = result.get("invoke_calls")
        if isinstance(agent_invoke, list) and agent_invoke:
            row["invoke_calls"] = agent_invoke
        elif isinstance(agent_invoke, int):
            # Legacy int count → synthesize records from invoked_operations.
            ops = list(result.get("invoked_operations") or [])
            row["invoke_calls"] = [
                {"operation_id": oid, "source": "gateway"} for oid in ops
            ] or [{"source": "gateway"} for _ in range(int(agent_invoke))]
        if result.get("error_type"):
            row["error_type"] = result.get("error_type")
        if result.get("stopping_reason"):
            row["stopping_reason"] = result.get("stopping_reason")
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
                        "failed_invocation_count": pred.get("failed_invocation_count"),
                        "recovered_after_tool_error": pred.get(
                            "recovered_after_tool_error"
                        ),
                        "protocol_success_rate": pred.get("protocol_success_rate"),
                        "wrong_tool_rate": pred.get("wrong_tool_rate"),
                        "refusal_correctness": pred.get("refusal_correctness"),
                        "llm_calls": pred.get("llm_calls"),
                        "input_tokens": tokens.get("prompt_tokens")
                        or tokens.get("input_tokens"),
                        "output_tokens": tokens.get("completion_tokens")
                        or tokens.get("output_tokens"),
                        "total_tokens": tokens.get("total_tokens"),
                        "embedding_calls": pred.get("embedding_calls"),
                        "embedding_tokens": pred.get("embedding_tokens"),
                        "latency_ms": pred.get("latency_ms"),
                        "real_mcp_execution_count": pred.get("real_mcp_execution_count"),
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
        outcomes = summarize_prediction_outcomes(
            expected_ids=[i["task_id"] for i in items],
            predictions_by_id=predictions_by_id,
        )
        run_meta.update(
            {
                "processed_task_count": outcomes.get("processed_task_count"),
                "full_count": outcomes.get("full_count"),
                "partial_count": outcomes.get("partial_count"),
                "failed_count": outcomes.get("failed_count"),
                "completed_ids": outcomes.get("completed_ids"),
                "completed_task_ids": outcomes.get("completed_task_ids"),
                "full_task_ids": outcomes.get("full_task_ids"),
                "partial_ids": outcomes.get("partial_ids"),
                "partial_task_ids": outcomes.get("partial_task_ids"),
                "failed_ids": outcomes.get("failed_ids"),
                "failed_task_ids": outcomes.get("failed_task_ids"),
                "missing_ids": outcomes.get("missing_ids"),
                "missing_task_ids": outcomes.get("missing_task_ids"),
                "expected_task_count": outcomes.get("expected_task_count"),
                "all_predictions_written": outcomes.get("all_predictions_written"),
            }
        )
        summary = summarize_predictions(rows)
        pe_metrics = aggregate_pe_fin_lite_metrics(rows)
        cost = pe_metrics.get("cost_summary") or {}
        from hk_equity_mcp.experiment.unified_aggregates import build_grouped_aggregates
        from hk_equity_mcp.experiment.unified_grading import summarize_grading

        grading_summary = summarize_grading(rows)
        grouped = build_grouped_aggregates(rows)
        atomic_write_json(self.output_dir / "pe_fin_lite_metrics.json", pe_metrics)
        atomic_write_json(self.output_dir / "cost_summary.json", cost)
        atomic_write_json(self.output_dir / "grouped_aggregates.json", grouped)
        atomic_write_json(self.output_dir / "grading_summary.json", grading_summary)
        atomic_write_json(
            self.output_dir / "retrieval_analysis.json",
            {
                "recall_at_5_mean": pe_metrics.get("retrieval_recall_at_5_mean"),
                "primary_selection_rate": pe_metrics.get("primary_selection_rate"),
                "fallback_use_rate": pe_metrics.get("fallback_use_rate"),
                "retrieve_step_completion_mean": pe_metrics.get(
                    "retrieve_step_completion_mean"
                ),
                "required_output_satisfaction_mean": pe_metrics.get(
                    "required_output_satisfaction_mean"
                ),
                "scope_match_rate_mean": pe_metrics.get("scope_match_rate_mean"),
                "note": "Retrieval success is not answer success.",
            },
        )
        atomic_write_json(
            self.output_dir / "step_funnel_summary.json",
            pe_metrics.get("mechanism_funnel") or {},
        )
        atomic_write_json(
            self.output_dir / "error_summary.json",
            {
                "aggregate": summary,
                "pe_fin_lite_metrics": pe_metrics,
                "grading": grading_summary,
                "grouped_aggregates": grouped,
                "n_tasks": len(rows),
                "n_successful": cost.get("successful_tasks"),
                "n_failed": cost.get("failed_tasks"),
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
                            pred.get("evidence_store") or pred.get("evidence_ledger") or [],
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
            label = "PE-FIN-REPLAN DRY-RUN OK" if self.replan_enabled else "PE-FIN-LITE DRY-RUN OK"
            print(label)
            print(f"condition={run_meta['condition']}")
            print(f"method_version={run_meta['method_version']}")
            print(f"architecture={run_meta['architecture']}")
            print(f"protocol_version={run_meta.get('protocol_version')}")
            print(f"subset_kind={run_meta.get('subset_kind')}")
            print(f"diagnostic12={self.diagnostic12}")
            print(f"smoke5={self.smoke5}")
            print(f"diagnostic20_eea={self.diagnostic20_eea}")
            print(f"diagnostic20_replan={self.diagnostic20_replan}")
            print(f"replan_enabled={self.replan_enabled}")
            print(f"question_count={run_meta['question_count']}")
            print(f"capability_profile_hash={run_meta['capability_profile_hash']}")
            print(f"schema_hash={run_meta.get('schema_hash')}")
            print(f"system_prompt_hash={run_meta['system_prompt_hash']}")
            print(f"output_dir={self.output_dir}")
            return run_meta

        if self.resume and meta_path.exists():
            existing = json.loads(meta_path.read_text(encoding="utf-8"))
            mismatches = []
            for key in PE_RESUME_IDENTITY_KEYS:
                if existing.get(key) != run_meta.get(key):
                    mismatches.append(
                        f"{key}: existing={existing.get(key)!r} current={run_meta.get(key)!r}"
                    )
            if mismatches:
                raise RuntimeError(
                    "Resume refused: PE-Fin Lite run_metadata identity mismatch. "
                    + "; ".join(mismatches)
                )
        elif self.resume and not meta_path.exists():
            raise RuntimeError(
                "Resume refused: run_metadata.json is missing. "
                "Cannot verify experiment identity."
            )

        run_meta["run_status"] = "running" if not self.mock else "mock_running"
        run_meta["expected_task_count"] = len(items)
        atomic_write_json(meta_path, run_meta)

        predictions_by_id: Dict[str, Dict[str, Any]] = {}
        try:
            agent = self.build_agent()
            done = already_done_ids(predictions_path) if self.resume else set()
            if not self.resume and predictions_path.exists():
                predictions_path.unlink()
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
                # Reset scripted LLM per task (mock).
                raw = getattr(agent, "_raw_llm", None)
                if raw is not None and hasattr(raw, "reset_script"):
                    raw.reset_script()
                # Per-task HKEX download budget (max 1); never inherit prior tasks.
                if self._gateway is not None and hasattr(self._gateway, "begin_task"):
                    self._gateway.begin_task()
                try:
                    result = agent.run(
                        item["question"],
                        answerability_class=item.get("answerability_class"),
                    )
                except (LlmCallBudgetExhausted, McpBudgetExhausted) as exc:
                    result = structured_budget_failure(exc)
                except (KeyboardInterrupt, SystemExit):
                    raise
                except Exception as exc:  # noqa: BLE001 — isolate per task
                    result = {
                        "status": "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                        "error_type": "task_exception",
                        "final_answer": "",
                        "predicted_status": "unavailable",
                        "trace": [{"stage": "runner_isolation", "error": str(exc)}],
                        "tokens": {},
                        "llm_calls": 0,
                        "resolver_calls": {"find_operations": 0, "describe_operation": 0},
                        "underlying_operation_calls": 0,
                        "monitor_decisions": [],
                        "plan_revisions": [],
                    }

                row = self._build_prediction_row(item, result)
                append_jsonl(predictions_path, row)
                predictions_by_id[qid] = row
                self._write_side_artifacts(
                    items=items,
                    predictions_by_id=predictions_by_id,
                    run_meta=run_meta,
                )
                atomic_write_json(meta_path, run_meta)

            if not predictions_path.exists():
                predictions_path.write_text("", encoding="utf-8")
                fsync_file(predictions_path)
            else:
                fsync_file(predictions_path)
            self._write_side_artifacts(
                items=items,
                predictions_by_id=predictions_by_id,
                run_meta=run_meta,
            )
            run_meta["run_status"] = "completed"
            run_meta["processed_task_count"] = len(predictions_by_id)
            run_meta["completed_at"] = datetime.now(timezone.utc).isoformat()
            if self._gateway is not None:
                run_meta["gateway_invoke_count"] = self._gateway.invoke_count
            atomic_write_json(meta_path, run_meta)
            print(f"Predictions: {predictions_path}")
            print(f"Run metadata: {meta_path}")
            return run_meta
        except (KeyboardInterrupt, SystemExit) as exc:
            run_meta["run_status"] = "aborted"
            run_meta["abort_reason"] = type(exc).__name__
            run_meta["aborted_at"] = datetime.now(timezone.utc).isoformat()
            run_meta["processed_task_count"] = len(predictions_by_id)
            try:
                if predictions_path.exists():
                    fsync_file(predictions_path)
                self._write_side_artifacts(
                    items=items,
                    predictions_by_id=predictions_by_id,
                    run_meta=run_meta,
                )
                atomic_write_json(meta_path, run_meta)
            finally:
                raise
        except Exception as exc:
            # build_agent / unexpected runner failures must not leave status=running
            if run_meta.get("run_status") in {"running", "mock_running"}:
                run_meta["run_status"] = "aborted"
                run_meta["abort_reason"] = "internal_error"
                run_meta["abort_error_type"] = type(exc).__name__
                run_meta["abort_error"] = str(exc)[:2000]
                run_meta["aborted_at"] = datetime.now(timezone.utc).isoformat()
                run_meta["processed_task_count"] = len(predictions_by_id)
            try:
                if predictions_path.exists():
                    fsync_file(predictions_path)
                self._write_side_artifacts(
                    items=items,
                    predictions_by_id=predictions_by_id,
                    run_meta=run_meta,
                )
                atomic_write_json(meta_path, run_meta)
            finally:
                raise
        finally:
            self.close()
