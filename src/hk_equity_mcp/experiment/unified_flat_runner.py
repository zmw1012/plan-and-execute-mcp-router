"""Unified HK Flat All Tools Real-MCP baseline runner.

Offline-safe modes: --dry-run and --mock. Paid LLM/MCP only when explicitly
requested without those flags. Never mixes dry-run / mock / live output dirs.
"""

from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import yaml

from hk_equity_mcp.agent.flat_all_tools import FlatAllToolsAgent
from hk_equity_mcp.agent.llm_client import OpenAICompatibleChatClient, ScriptedLLMClient
from hk_equity_mcp.common.env_loader import (
    GENERATION_REQUIRED,
    load_generation_profile,
    missing_keys,
)
from hk_equity_mcp.experiment.runner import (
    already_done_ids,
    git_branch,
    git_commit,
    git_dirty,
    redact_secrets,
    url_host,
)
from hk_equity_mcp.experiment.unified_dataset import (
    assert_heldout_unlocked,
    load_smoke_tasks,
    load_split_tasks,
    select_by_ids,
)
from hk_equity_mcp.experiment.unified_exploratory import (
    RETIREMENT_POLICY_VERSION,
    assert_exploratory_unlocked,
    build_retirement_metadata,
    load_exploratory_manifest,
    load_exploratory_run_order,
)
from hk_equity_mcp.experiment.unified_flat_metrics import (
    RETRIEVAL_NA,
    build_flat_grouped_aggregates,
    evaluate_flat_task,
    summarize_flat_predictions,
)
from hk_equity_mcp.experiment.unified_flat_prompt import (
    build_flat_all_tools_system_prompt,
    flat_prompt_rules_hash,
    unified_flat_prompt_hash,
)
from hk_equity_mcp.experiment.unified_flat_schema_bundle import build_flat_schema_bundle
from hk_equity_mcp.experiment.unified_runner import (
    append_jsonl,
    atomic_write_json,
    catalog_fingerprint,
    fsync_file,
)
from hk_equity_mcp.experiment.unified_schema import (
    BANNED_PROMPT_SUBSTRINGS,
    MANUAL_GRADING_FIELDS,
    PER_TASK_METRIC_FIELDS,
    PREDICTION_FIELDS,
)
from hk_equity_mcp.real_mcp.unified_catalog import build_unified_catalog
from hk_equity_mcp.real_mcp.unified_gateway import (
    ScriptedMockTransport,
    UnifiedRealMcpGateway,
)

RESUME_IDENTITY_KEYS = (
    "condition",
    "execution_backend",
    "generation_model",
    "generation_provider",
    "generation_base_url_host",
    "provider_host",
    "dataset_split",
    "system_prompt_hash",
    "schema_bundle_sha256",
    "action_mapping_fingerprint",
    "catalog_fingerprint",
    "temperature",
    "max_tokens",
    "max_llm_turns",
    "exploratory_manifest_hash",
    "run_order_hash",
    "retirement_policy_version",
)

DEFAULT_GENERATION_PROVIDER = "aliyun"
DEFAULT_GENERATION_MODEL = "deepseek-v4.1-flash"


def _run_mode(*, dry_run: bool, mock: bool) -> str:
    if dry_run:
        return "dry_run"
    if mock:
        return "mock"
    return "live"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_flat_config(
    path: Optional[Path] = None, root: Optional[Path] = None
) -> Dict[str, Any]:
    root = root or repo_root()
    cfg_path = path or (root / "configs" / "hk_unified_flat_all_tools_v1.yaml")
    if not cfg_path.exists():
        return {
            "condition": "flat",
            "execution_backend": "real_mcp",
            "temperature": 0.0,
            "max_tokens": 1024,
            "generation_model": DEFAULT_GENERATION_MODEL,
            "generation_provider": DEFAULT_GENERATION_PROVIDER,
            "limits": {
                "max_llm_turns": 8,
                "max_invocations": 6,
                "timeout_sec": 120,
                "bounded_retry": 1,
            },
        }
    with cfg_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def assert_resume_compatible_flat(
    existing: Dict[str, Any], current: Dict[str, Any]
) -> None:
    mismatches = []
    for key in RESUME_IDENTITY_KEYS:
        if existing.get(key) != current.get(key):
            mismatches.append(
                f"{key}: existing={existing.get(key)!r} current={current.get(key)!r}"
            )
    if mismatches:
        raise RuntimeError(
            "Resume refused: run_metadata identity mismatch. "
            + "; ".join(mismatches)
        )


class _CyclingScriptedLLM(ScriptedLLMClient):
    def __init__(self, scripted_contents: list[str], model: str = "mock-llm"):
        super().__init__(scripted_contents, model=model)
        self._script = list(scripted_contents)

    def complete(
        self,
        messages,
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ):
        if not self._queue:
            self._queue = list(self._script)
        return super().complete(
            messages, temperature=temperature, max_tokens=max_tokens
        )


def _mock_flat_llm(action_name: str, operation_id: str) -> ScriptedLLMClient:
    """Offline scripted trajectory: direct operation call → answer."""
    turns = [
        {
            "thought": "call a runtime operation directly (Flat)",
            "tool_calls": [
                {
                    "name": action_name,
                    "arguments": {"symbol": "0700"},
                }
            ],
            "final_answer": None,
        },
        {
            "thought": "answer from evidence",
            "tool_calls": [],
            "final_answer": {
                "status": "answered",
                "final_answer": "mock flat answer",
                "reasoning_brief": "offline Flat All Tools mock trajectory",
                "evidence": [
                    {
                        "operation_id": operation_id,
                        "fields": ["mock"],
                        "note": "mock transport",
                    }
                ],
            },
        },
    ]
    return _CyclingScriptedLLM(
        [json.dumps(t, ensure_ascii=False) for t in turns]
    )


class UnifiedHkFlatAllToolsRunner:
    def __init__(
        self,
        *,
        output_dir: Path,
        split: str = "development",
        smoke: bool = False,
        ids: Optional[Sequence[str]] = None,
        max_tasks: Optional[int] = None,
        resume: bool = False,
        dry_run: bool = False,
        mock: bool = False,
        allow_dirty: bool = False,
        confirm_heldout_final: bool = False,
        confirm_retire_heldout: bool = False,
        env_file: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        root: Optional[Path] = None,
        allow_hkex_download: bool = False,
        json_mode: bool = True,
    ):
        self.root = root or repo_root()
        self.output_dir = Path(output_dir)
        self.split = split
        self.smoke = bool(smoke)
        self.ids = list(ids) if ids else None
        self.max_tasks = max_tasks
        self.resume = resume
        self.dry_run = dry_run
        self.mock = mock
        self.allow_dirty = allow_dirty
        self.confirm_heldout_final = confirm_heldout_final
        self.confirm_retire_heldout = confirm_retire_heldout
        self.env_file = env_file
        self.config = config or load_flat_config(root=self.root)
        self.allow_hkex_download = allow_hkex_download
        self.json_mode = json_mode
        self.temperature = float(self.config.get("temperature", 0.0))
        self.max_tokens = int(self.config.get("max_tokens", 1024))
        self.limits = dict(self.config.get("limits") or {})
        self.condition = "flat"
        self.execution_backend = "real_mcp"
        self.gen_meta: Dict[str, str] = {}
        self.bundle = build_unified_catalog(root=self.root)
        self.schema_bundle = build_flat_schema_bundle(self.bundle)
        self.system_prompt = build_flat_all_tools_system_prompt(self.schema_bundle)
        self._gateway: Optional[UnifiedRealMcpGateway] = None
        self._exploratory_manifest: Optional[Dict[str, Any]] = None
        self._exploratory_run_order: Optional[Dict[str, Any]] = None

    def _load_tasks(self) -> List[Dict[str, Any]]:
        heldout_status = assert_heldout_unlocked(
            split=self.split, confirm_heldout_final=self.confirm_heldout_final
        )
        self._heldout_status = heldout_status
        exploratory_status = assert_exploratory_unlocked(
            split=self.split, confirm_retire_heldout=self.confirm_retire_heldout
        )
        self._exploratory_status = exploratory_status
        if self.smoke:
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
                tasks, self.ids, preserve_id_order=(self.split == "exploratory_all")
            )
        # max_tasks applied in prepare as cumulative prefix; not in resume identity.
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
                raise RuntimeError(
                    f"Flat All Tools system prompt unexpectedly contains {banned}"
                )

        # Catalog identity fail-closed.
        stats = self.bundle.stats
        if stats.unique_operation_ids != 42:
            raise RuntimeError(
                f"Catalog drift: unique_operation_ids={stats.unique_operation_ids} expected 42"
            )
        if stats.runtime_exposed_unique != 39:
            raise RuntimeError(
                f"Catalog drift: runtime_exposed_unique={stats.runtime_exposed_unique} expected 39"
            )
        if stats.blocked_or_rejected_unique != 3:
            raise RuntimeError(
                "Catalog drift: blocked_or_rejected_unique="
                f"{stats.blocked_or_rejected_unique} expected 3"
            )
        if len(self.schema_bundle.entries) != 39:
            raise RuntimeError(
                f"Schema bundle must expose 39 ops; got {len(self.schema_bundle.entries)}"
            )
        for oid in self.schema_bundle.blocked_operation_ids:
            if oid in self.schema_bundle.runtime_operation_ids:
                raise RuntimeError(f"Blocked op leaked into Flat exposure: {oid}")

        prompt_h = unified_flat_prompt_hash(self.system_prompt)
        tasks_full = self._load_tasks()
        tasks = tasks_full
        if self.max_tasks is not None:
            tasks = tasks[: max(0, int(self.max_tasks))]

        expected_model = str(
            self.config.get("generation_model") or DEFAULT_GENERATION_MODEL
        )
        expected_provider = str(
            self.config.get("generation_provider") or DEFAULT_GENERATION_PROVIDER
        )

        if self.mock or self.dry_run:
            try:
                self.gen_meta = load_generation_profile(self.root, self.env_file)
            except FileNotFoundError:
                self.gen_meta = {
                    "provider": expected_provider,
                    "model": expected_model,
                    "profile_file": None,
                    "base_url": None,
                }
            if self.dry_run and not self.mock:
                self._dry_run_missing = {
                    "generation": missing_keys(GENERATION_REQUIRED),
                    "embedding": ["N/A — Flat All Tools does not use embedding"],
                }
        else:
            self.gen_meta = load_generation_profile(self.root, self.env_file)

        if not self.mock:
            actual_model = (
                self.gen_meta.get("model") or os.environ.get("LLM_MODEL") or ""
            ).strip()
            actual_provider = (
                self.gen_meta.get("provider") or os.environ.get("LLM_PROVIDER") or ""
            ).strip()
            if actual_model and actual_model != expected_model:
                raise RuntimeError(
                    f"generation model mismatch: expected {expected_model}, got {actual_model}"
                )
            if actual_provider and actual_provider != expected_provider:
                raise RuntimeError(
                    f"generation provider mismatch: expected {expected_provider}, "
                    f"got {actual_provider}"
                )

        self.output_dir.mkdir(parents=True, exist_ok=True)
        fingerprint = catalog_fingerprint(stats, self.bundle.source_catalogs)
        gen_host = url_host(
            self.gen_meta.get("base_url") or os.environ.get("LLM_BASE_URL")
        )

        domain_counts: Dict[str, int] = {}
        source_split_counts: Dict[str, int] = {}
        for t in tasks:
            domain_counts[str(t.get("domain"))] = domain_counts.get(
                str(t.get("domain")), 0
            ) + 1
            if t.get("source_split") is not None:
                source_split_counts[str(t.get("source_split"))] = (
                    source_split_counts.get(str(t.get("source_split")), 0) + 1
                )

        exploratory_manifest_hash = None
        run_order_hash = None
        retirement_policy_version = None
        retirement_meta = None
        if self.split == "exploratory_all":
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
                confirm_retire_heldout=self.confirm_retire_heldout,
                git_commit=git_commit(self.root),
                prompt_hash=prompt_h,
                catalog_fingerprint=fingerprint,
                manifest_hash=str(exploratory_manifest_hash),
                run_order_hash=str(run_order_hash),
                timestamp=datetime.now(timezone.utc).isoformat(),
                existing_first_real=existing_first,
            )

        max_llm_turns = int(self.limits.get("max_llm_turns", 8))
        meta = {
            "condition": self.condition,
            "execution_backend": self.execution_backend,
            "architecture": "flat_all_tools",
            "real_mcp": True,
            "embedding_enabled": False,
            "retrieval_enabled": False,
            "mock": self.mock,
            "dry_run": self.dry_run,
            "run_mode": _run_mode(dry_run=self.dry_run, mock=self.mock),
            "dataset_split": self.split,
            "smoke": self.smoke,
            "heldout_lock": getattr(self, "_heldout_status", {}),
            "exploratory_lock": getattr(self, "_exploratory_status", {}),
            "exploratory_manifest_hash": exploratory_manifest_hash,
            "run_order_hash": run_order_hash,
            "retirement_policy_version": retirement_policy_version,
            "retirement": retirement_meta,
            "question_ids": [t["task_id"] for t in tasks],
            "question_count": len(tasks),
            "max_tasks_semantics": (
                "cumulative_prefix_of_locked_run_order_or_ids; "
                "max_tasks is NOT part of resume identity"
            ),
            "domain_counts": domain_counts,
            "source_split_counts": source_split_counts or None,
            "generation_provider": self.gen_meta.get("provider") or expected_provider,
            "generation_model": self.gen_meta.get("model") or expected_model,
            "generation_profile_file": self.gen_meta.get("profile_file"),
            "generation_base_url_host": gen_host,
            "provider_host": gen_host,
            "embedding_model": None,
            "embedding_profile_file": None,
            "embedding_base_url_host": None,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "max_llm_turns": max_llm_turns,
            "json_mode": self.json_mode,
            "limits": self.limits,
            "system_prompt_hash": prompt_h,
            "system_prompt_rules_hash": flat_prompt_rules_hash(),
            "system_prompt": self.system_prompt,
            "schema_bundle_sha256": self.schema_bundle.schema_bundle_sha256,
            "serialized_schema_char_count": (
                self.schema_bundle.serialized_schema_char_count
            ),
            "schema_bundle_token_estimate": self.schema_bundle.token_estimate,
            "token_estimate_method": "ceil(char_count/4)",
            "action_mapping_fingerprint": self.schema_bundle.mapping.fingerprint,
            "action_mapping": self.schema_bundle.mapping.to_dict(),
            "catalog_fingerprint": fingerprint,
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
            "retrieval_metrics": RETRIEVAL_NA,
            "retrieval_calls": 0,
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
                "unified HK Flat All Tools Real-MCP baseline; "
                "routing ablation vs Operation-Visible (same catalog/model/contract); "
                "no embedding/retrieval/find/describe"
            ),
            "dataset_note": (
                "Same exploratory_all 155 identity as Operation-Visible. "
                "exploratory_all requires --confirm-retire-heldout."
            ),
            "dry_run_missing_env": getattr(self, "_dry_run_missing", None),
            "tool_protocol": "prompt_json_react",
            "native_function_calling": False,
        }
        return {"items": tasks, "run_metadata": redact_secrets(meta)}

    def build_agent(self) -> FlatAllToolsAgent:
        if self.dry_run:
            raise RuntimeError("build_agent must not be called during dry-run.")

        if self.mock:
            # Prefer a known runtime op that accepts symbol=0700; else first runtime.
            preferred = "aktools.stock_financial_hk_analysis_indicator_em"
            if preferred in self.schema_bundle.runtime_operation_ids:
                oid = preferred
            else:
                oid = self.schema_bundle.runtime_operation_ids[0]
            action = self.schema_bundle.mapping.action_name(oid)
            llm: Any = _mock_flat_llm(action, oid)
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
            gateway = UnifiedRealMcpGateway(
                catalog_bundle=self.bundle,
                allow_hkex_download=self.allow_hkex_download,
                persistent=True,
            )

        gateway.open()
        self._gateway = gateway
        return FlatAllToolsAgent(
            llm=llm,
            gateway=gateway,
            schema_bundle=self.schema_bundle,
            system_prompt=self.system_prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            max_llm_turns=int(self.limits.get("max_llm_turns", 8)),
            max_invocations=int(self.limits.get("max_invocations", 6)),
            timeout_sec=float(self.limits.get("timeout_sec", 120)),
        )

    def close(self) -> None:
        if self._gateway is not None:
            self._gateway.close()
            self._gateway = None

    def _build_prediction_row(
        self, item: Dict[str, Any], result: Dict[str, Any]
    ) -> Dict[str, Any]:
        from hk_equity_mcp.experiment.unified_grading import grade_task

        metrics = evaluate_flat_task(item=item, result=result)
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
            "previously_reserved": item.get("previously_reserved"),
            "benchmark_role": item.get("benchmark_role"),
            "company": item.get("company"),
            "gold_status": item.get("gold_status"),
            "gold_values": item.get("gold_values"),
            "llm_calls": result.get("llm_calls"),
            "tokens": tokens,
            "embedding_calls": 0,
            "embedding_tokens": 0,
            "retrieval_calls": 0,
            "retrieval_metrics": RETRIEVAL_NA,
            "latency_ms": result.get("latency_ms"),
            "retry_count": result.get("retry_count", 0),
            "timeout": result.get("error") == "timeout",
            "trace": result.get("trace") or [],
            "resolver_calls": result.get("resolver_calls")
            or {"find_operations": 0, "describe_operation": 0},
            "underlying_operation_calls": result.get("underlying_operation_calls", 0),
            "provider_metadata": result.get("provider_metadata") or {},
            "condition": self.condition,
            "execution_backend": self.execution_backend,
            "auto_grade_status": grade.get("auto_grade_status"),
            "answer_correct": grade.get("answer_correct"),
            "auto_grade_reason": grade.get("reason"),
            "manual_answer_correctness": "",
            "manual_evidence_correctness": "",
            "manual_unsupported_claim": "",
            "manual_causal_overclaim": "",
            "manual_overall_task_success": "",
            "answer_correct_0_or_1": "",
            "evidence_correct_0_or_1": "",
            "unsupported_claim_0_or_1": "",
            "causal_overclaim_0_or_1": "",
            "overall_task_success_0_or_1": "",
            "grader_notes": "",
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
                        "embedding_calls": 0,
                        "embedding_tokens": 0,
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
                            pred.get("evidence") or [], ensure_ascii=False
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

        rows = [
            predictions_by_id[i["task_id"]]
            for i in items
            if i["task_id"] in predictions_by_id
        ]
        summary = summarize_flat_predictions(rows)
        cost = {
            "n_tasks": len(rows),
            "total_input_tokens": sum(
                int((r.get("tokens") or {}).get("input_tokens") or 0) for r in rows
            ),
            "total_output_tokens": sum(
                int((r.get("tokens") or {}).get("output_tokens") or 0) for r in rows
            ),
            "total_tokens": sum(
                int((r.get("tokens") or {}).get("total_tokens") or 0) for r in rows
            ),
            "total_llm_calls": sum(int(r.get("llm_calls") or 0) for r in rows),
            "total_real_mcp_executions": sum(
                int(r.get("real_mcp_execution_count") or 0) for r in rows
            ),
            "total_embedding_calls": 0,
            "total_retrieval_calls": 0,
            "total_find_operations": 0,
            "total_describe_operation": 0,
            "schema_bundle_char_count": run_meta.get("serialized_schema_char_count"),
            "schema_bundle_token_estimate": run_meta.get("schema_bundle_token_estimate"),
            "mean_latency_ms": summary.get("mean_latency_ms"),
            "mock": self.mock,
            "condition": self.condition,
            "note": "Token totals only; USD cost requires provider price table.",
        }
        atomic_write_json(self.output_dir / "cost_summary.json", cost)
        atomic_write_json(
            self.output_dir / "retrieval_analysis.json",
            {
                "retrieval_metrics": RETRIEVAL_NA,
                "retrieval_calls": 0,
                "embedding_calls": 0,
                "note": (
                    "Flat All Tools has no retrieval stage; "
                    "metrics are N/A (not scored as 1.0)."
                ),
                "per_task": [
                    {
                        "task_id": r["task_id"],
                        "retrieved_candidates": RETRIEVAL_NA,
                        "predicted_operations": r.get("predicted_operations"),
                        "gold_operations": r.get("gold_operations"),
                        "selected_operations": r.get("selected_operations"),
                        "invoked_operations": r.get("invoked_operations"),
                    }
                    for r in rows
                ],
            },
        )
        from hk_equity_mcp.experiment.unified_evaluator import evaluator_identity
        from hk_equity_mcp.experiment.unified_grading import summarize_grading
        from hk_equity_mcp.experiment.unified_schema import METRIC_FIELD_DEFINITIONS

        grading_summary = summarize_grading(rows)
        grouped = build_flat_grouped_aggregates(rows)
        atomic_write_json(
            self.output_dir / "error_summary.json",
            {
                "error_counts": summary.get("error_counts") or {},
                "result_status_counts": summary.get("result_status_counts") or {},
                "definitions": {
                    "error_counts": METRIC_FIELD_DEFINITIONS["error_counts"],
                    "result_status_counts": METRIC_FIELD_DEFINITIONS[
                        "result_status_counts"
                    ],
                    "tool_execution_success": METRIC_FIELD_DEFINITIONS[
                        "tool_execution_success"
                    ],
                    "recovered_after_tool_error": METRIC_FIELD_DEFINITIONS[
                        "recovered_after_tool_error"
                    ],
                },
                "evaluator": evaluator_identity(),
                "aggregate": summary,
                "grading": grading_summary,
                "grouped_aggregates": grouped,
                "retrieval_metrics": RETRIEVAL_NA,
                "reporting_axes": {
                    "A_final_answer_accuracy": grading_summary,
                    "B_exact_tool_set_success": summary.get("exact_tool_set_rate"),
                    "C_strict_overall_task_success": (
                        "manual overall_task_success_0_or_1"
                    ),
                },
            },
        )
        atomic_write_json(self.output_dir / "grouped_aggregates.json", grouped)
        atomic_write_json(self.output_dir / "grading_summary.json", grading_summary)
        atomic_write_json(
            self.output_dir / "evaluation_metadata.json",
            {
                "condition": self.condition,
                "evaluator": evaluator_identity(),
                "system_prompt_hash": run_meta.get("system_prompt_hash"),
                "schema_bundle_sha256": run_meta.get("schema_bundle_sha256"),
                "catalog_fingerprint": run_meta.get("catalog_fingerprint"),
                "action_mapping_fingerprint": run_meta.get(
                    "action_mapping_fingerprint"
                ),
                "retrieval_metrics": RETRIEVAL_NA,
                "generation_model": run_meta.get("generation_model"),
                "temperature": run_meta.get("temperature"),
                "max_tokens": run_meta.get("max_tokens"),
                "max_llm_turns": run_meta.get("max_llm_turns"),
            },
        )

    def _assert_output_dir_safe(
        self, *, meta_path: Path, predictions_path: Path
    ) -> Optional[Dict[str, Any]]:
        existing: Optional[Dict[str, Any]] = None
        if meta_path.exists():
            existing = json.loads(meta_path.read_text(encoding="utf-8"))
            existing_mode = existing.get("run_mode") or _run_mode(
                dry_run=bool(existing.get("dry_run")),
                mock=bool(existing.get("mock")),
            )
            current_mode = _run_mode(dry_run=self.dry_run, mock=self.mock)
            if existing_mode != current_mode:
                raise RuntimeError(
                    "Output directory mode mismatch: "
                    f"existing={existing_mode} current={current_mode}. "
                    "Dry-run, mock, and live must use separate --output-dir paths."
                )

        if self.resume:
            return existing

        if predictions_path.exists():
            raise RuntimeError(
                f"Refusing to start without --resume: {predictions_path} already exists. "
                "Use --resume to continue, or choose a new --output-dir."
            )
        if existing is not None and not self.dry_run:
            raise RuntimeError(
                f"Refusing to start without --resume: {meta_path} already exists. "
                "Use a new --output-dir (do not reuse a dry-run directory for mock/live)."
            )
        return existing

    def run(self) -> Dict[str, Any]:
        prepared = self.prepare()
        items: List[Dict[str, Any]] = prepared["items"]
        run_meta = prepared["run_metadata"]
        meta_path = self.output_dir / "run_metadata.json"
        predictions_path = self.output_dir / "predictions.jsonl"

        self._assert_output_dir_safe(
            meta_path=meta_path, predictions_path=predictions_path
        )

        if self.dry_run:
            atomic_write_json(meta_path, run_meta)
            print("DRY-RUN OK (Flat All Tools)")
            print(f"condition={run_meta['condition']}")
            print(f"execution_backend={run_meta['execution_backend']}")
            print(f"split={self.split}")
            print(f"question_count={run_meta['question_count']}")
            print(f"domain_counts={run_meta['domain_counts']}")
            print(f"generation_model={run_meta['generation_model']}")
            print(f"system_prompt_hash={run_meta['system_prompt_hash']}")
            print(f"schema_bundle_sha256={run_meta['schema_bundle_sha256']}")
            print(
                f"serialized_schema_char_count={run_meta['serialized_schema_char_count']}"
            )
            print(
                f"schema_bundle_token_estimate={run_meta['schema_bundle_token_estimate']}"
            )
            print(
                f"action_mapping_fingerprint={run_meta['action_mapping_fingerprint']}"
            )
            print(f"catalog_fingerprint={run_meta['catalog_fingerprint']}")
            print(
                "catalog_stats="
                + json.dumps(run_meta["catalog_stats"], ensure_ascii=False)
            )
            print(f"retrieval_metrics={run_meta['retrieval_metrics']}")
            print(f"branch={run_meta['branch']}")
            print(f"git_commit={run_meta['git_commit']}")
            print(f"output_dir={self.output_dir}")
            return run_meta

        if self.resume and meta_path.exists():
            existing = json.loads(meta_path.read_text(encoding="utf-8"))
            assert_resume_compatible_flat(existing, run_meta)
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
                try:
                    result = agent.run(item["question"])
                    # Fail closed: Flat must never invent embedding/retrieval calls.
                    if (
                        int(result.get("embedding_calls") or 0) != 0
                        or int(result.get("retrieval_calls") or 0) != 0
                        or int(
                            (result.get("resolver_calls") or {}).get("find_operations")
                            or 0
                        )
                        != 0
                        or int(
                            (result.get("resolver_calls") or {}).get(
                                "describe_operation"
                            )
                            or 0
                        )
                        != 0
                    ):
                        raise RuntimeError(
                            "Flat All Tools invariant violated: "
                            "embedding/retrieval/find/describe calls must be 0"
                        )
                    row = self._build_prediction_row(item, result)
                except Exception as exc:  # noqa: BLE001
                    row = self._build_prediction_row(
                        item,
                        {
                            "final_answer": "",
                            "predicted_status": "unavailable",
                            "status": "failed",
                            "trace": [],
                            "llm_calls": 0,
                            "resolver_calls": {
                                "find_operations": 0,
                                "describe_operation": 0,
                            },
                            "underlying_operation_calls": 0,
                            "embedding_calls": 0,
                            "embedding_tokens": 0,
                            "retrieval_calls": 0,
                            "tokens": {},
                            "latency_ms": None,
                            "provider_metadata": {
                                "llm_provider": run_meta.get("generation_provider"),
                                "llm_model": run_meta.get("generation_model"),
                                "embedding_model": None,
                            },
                            "error": type(exc).__name__,
                            "selected_operations": [],
                            "invoked_operations": [],
                            "successful_invoked_operations": [],
                            "invalid_selections": [],
                        },
                    )
                    row["error"] = type(exc).__name__
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
