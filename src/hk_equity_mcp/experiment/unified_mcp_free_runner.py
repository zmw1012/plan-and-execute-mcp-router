"""Unified HK MCP-Free (closed-book) baseline runner.

condition=mcp_free, execution_backend=none.
No MCP / Embedding / catalog / RAG / web. Offline-safe --dry-run / --mock.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import yaml

from hk_equity_mcp.agent.llm_client import OpenAICompatibleChatClient, ScriptedLLMClient
from hk_equity_mcp.agent.mcp_free import MCPFreeAgent
from hk_equity_mcp.common.env_loader import GENERATION_REQUIRED, load_generation_profile, missing_keys
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
from hk_equity_mcp.experiment.unified_mcp_free_metrics import (
    TOOL_NA,
    build_mcp_free_grouped_aggregates,
    evaluate_mcp_free_task,
)
from hk_equity_mcp.experiment.unified_mcp_free_prompt import (
    UNIFIED_HK_MCP_FREE_SYSTEM_PROMPT,
    unified_mcp_free_prompt_hash,
)
from hk_equity_mcp.experiment.unified_runner import (
    append_jsonl,
    atomic_write_json,
    fsync_file,
)
from hk_equity_mcp.experiment.unified_schema import (
    BANNED_PROMPT_SUBSTRINGS,
    MANUAL_GRADING_FIELDS,
    PER_TASK_METRIC_FIELDS,
    PREDICTION_FIELDS,
)

RESUME_IDENTITY_KEYS = (
    "condition",
    "execution_backend",
    "generation_model",
    "dataset_split",
    "system_prompt_hash",
    "temperature",
    "max_tokens",
    "catalog_fingerprint",
    "exploratory_manifest_hash",
    "run_order_hash",
    "retirement_policy_version",
)

DEFAULT_OUTPUT_DIR = "outputs/hk_unified_exploratory155_mcp_free_v1"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_mcp_free_config(
    path: Optional[Path] = None, root: Optional[Path] = None
) -> Dict[str, Any]:
    root = root or repo_root()
    cfg_path = path or (root / "configs" / "hk_unified_mcp_free_v1.yaml")
    if not cfg_path.exists():
        return {
            "condition": "mcp_free",
            "execution_backend": "none",
            "temperature": 0.0,
            "max_tokens": 1024,
            "json_mode": True,
            "generation_model": "deepseek-v4.1-flash",
            "generation_provider": "aliyun",
            "limits": {"bounded_retry": 1, "timeout_sec": 120},
        }
    with cfg_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def questions_identity_hash(tasks: Sequence[Dict[str, Any]]) -> str:
    """Stable hash over locked task_id + question text (no gold)."""
    payload = [
        {"task_id": t["task_id"], "question": t["question"]}
        for t in tasks
    ]
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


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


def _mock_mcp_free_llm() -> ScriptedLLMClient:
    answer = {
        "status": "unavailable",
        "final_answer": "Exact figures are not verifiable without external sources.",
        "reasoning_brief": "MCP-Free offline mock; closed-book refusal.",
    }
    return _CyclingScriptedLLM([json.dumps(answer, ensure_ascii=False)])


def assert_resume_compatible_mcp_free(
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


class UnifiedHkMcpFreeRunner:
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
        self.config = config or load_mcp_free_config(root=self.root)
        self.json_mode = json_mode
        self.temperature = float(self.config.get("temperature", 0.0))
        self.max_tokens = int(self.config.get("max_tokens", 1024))
        self.limits = dict(self.config.get("limits") or {})
        self.condition = "mcp_free"
        self.execution_backend = "none"
        self.system_prompt = UNIFIED_HK_MCP_FREE_SYSTEM_PROMPT
        self.gen_meta: Dict[str, str] = {}
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
                raise RuntimeError(f"MCP-Free system prompt unexpectedly contains {banned}")

        prompt_h = unified_mcp_free_prompt_hash()
        tasks = self._load_tasks()

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
            if self.dry_run and not self.mock:
                self._dry_run_missing = {
                    "generation": missing_keys(GENERATION_REQUIRED),
                    "embedding": ["N/A — MCP-Free does not use embedding"],
                }
        else:
            self.gen_meta = load_generation_profile(self.root, self.env_file)

        expected_model = str(
            self.config.get("generation_model") or "deepseek-v4.1-flash"
        )
        expected_provider = str(self.config.get("generation_provider") or "aliyun")
        if not self.mock:
            actual_model = self.gen_meta.get("model") or os.environ.get("LLM_MODEL")
            actual_provider = self.gen_meta.get("provider") or os.environ.get(
                "LLM_PROVIDER"
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

        self.output_dir.mkdir(parents=True, exist_ok=True)
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
                catalog_fingerprint="none",
                manifest_hash=str(exploratory_manifest_hash),
                run_order_hash=str(run_order_hash),
                timestamp=datetime.now(timezone.utc).isoformat(),
                existing_first_real=existing_first,
            )

        q_hash = questions_identity_hash(tasks) if tasks else None
        meta = {
            "condition": self.condition,
            "execution_backend": self.execution_backend,
            "real_mcp": False,
            "embedding_enabled": False,
            "catalog_exposed": False,
            "mock": self.mock,
            "dry_run": self.dry_run,
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
            "questions_identity_hash": q_hash,
            "domain_counts": domain_counts,
            "source_split_counts": source_split_counts or None,
            "generation_provider": self.gen_meta.get("provider") or expected_provider,
            "generation_model": self.gen_meta.get("model") or expected_model,
            "generation_profile_file": self.gen_meta.get("profile_file"),
            "generation_base_url_host": gen_host,
            "embedding_model": None,
            "embedding_profile_file": None,
            "embedding_base_url_host": None,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "json_mode": self.json_mode,
            "limits": self.limits,
            "system_prompt_hash": prompt_h,
            "system_prompt": self.system_prompt,
            "catalog_fingerprint": None,
            "catalog_fingerprint_note": "none — MCP-Free has no operation catalog",
            "branch": git_branch(self.root),
            "git_commit": git_commit(self.root),
            "git_dirty": dirty,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "resume": self.resume,
            "allow_dirty": self.allow_dirty,
            "max_tasks": self.max_tasks,
            "ids": self.ids,
            "run_status": "dry_run" if self.dry_run else ("mock" if self.mock else "running"),
            "fidelity_note": (
                "unified HK MCP-Free closed-book baseline; "
                "model-internal knowledge only; no tools/MCP/embedding/RAG/web"
            ),
            "dataset_note": (
                "Same exploratory_all 155 identity as Operation-Visible. "
                "exploratory_all requires --confirm-retire-heldout."
            ),
            "dry_run_missing_env": getattr(self, "_dry_run_missing", None),
            "tool_policy": {
                "tool_calls": 0,
                "resolver_calls": 0,
                "real_mcp_executions": 0,
                "embedding_calls": 0,
                "tool_metrics": TOOL_NA,
                "fallback_to_tools_on_failure": False,
            },
        }
        return {"items": tasks, "run_metadata": redact_secrets(meta)}

    def build_agent(self) -> MCPFreeAgent:
        if self.dry_run:
            raise RuntimeError("build_agent must not be called during dry-run.")
        if self.mock:
            llm: Any = _mock_mcp_free_llm()
        else:
            llm = OpenAICompatibleChatClient(
                base_url=os.environ["LLM_BASE_URL"],
                api_key=os.environ["LLM_API_KEY"],
                model=os.environ["LLM_MODEL"],
                provider=os.environ.get("LLM_PROVIDER") or "openai_compatible",
                json_mode=self.json_mode,
            )
        bounded = int((self.limits or {}).get("bounded_retry", 1))
        return MCPFreeAgent(
            llm=llm,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            system_prompt=self.system_prompt,
            max_parse_repairs=bounded,
        )

    def _build_prediction_row(
        self, item: Dict[str, Any], result: Dict[str, Any]
    ) -> Dict[str, Any]:
        from hk_equity_mcp.experiment.unified_grading import grade_task

        metrics = evaluate_mcp_free_task(item=item, result=result)
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
            "latency_ms": result.get("latency_ms"),
            "retry_count": result.get("retry_count", 0),
            "timeout": result.get("error") == "timeout",
            "trace": result.get("trace") or [],
            "resolver_calls": result.get("resolver_calls")
            or {"find_operations": 0, "describe_operation": 0},
            "underlying_operation_calls": 0,
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
                        "operation_precision": pred.get("operation_precision", TOOL_NA),
                        "operation_recall": pred.get("operation_recall", TOOL_NA),
                        "operation_f1": pred.get("operation_f1", TOOL_NA),
                        "exact_tool_set": pred.get("exact_tool_set", TOOL_NA),
                        "graph_dependency_accuracy": pred.get(
                            "graph_dependency_accuracy", TOOL_NA
                        ),
                        "tool_execution_success": pred.get(
                            "tool_execution_success", TOOL_NA
                        ),
                        "all_tool_invocations_successful": pred.get(
                            "all_tool_invocations_successful", TOOL_NA
                        ),
                        "successful_invocation_count": pred.get(
                            "successful_invocation_count", 0
                        ),
                        "failed_invocation_count": pred.get(
                            "failed_invocation_count", 0
                        ),
                        "recovered_after_tool_error": pred.get(
                            "recovered_after_tool_error", TOOL_NA
                        ),
                        "protocol_success_rate": pred.get(
                            "protocol_success_rate", TOOL_NA
                        ),
                        "wrong_tool_rate": pred.get("wrong_tool_rate", TOOL_NA),
                        "refusal_correctness": pred.get("refusal_correctness"),
                        "llm_calls": pred.get("llm_calls"),
                        "input_tokens": tokens.get("input_tokens"),
                        "output_tokens": tokens.get("output_tokens"),
                        "total_tokens": tokens.get("total_tokens"),
                        "embedding_calls": 0,
                        "embedding_tokens": 0,
                        "latency_ms": pred.get("latency_ms"),
                        "real_mcp_execution_count": 0,
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
                        "predicted_operations": "[]",
                        "evidence": "[]",
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
        grouped = build_mcp_free_grouped_aggregates(rows)
        overall = grouped.get("overall") or {}
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
            "total_real_mcp_executions": 0,
            "total_embedding_calls": 0,
            "total_resolver_calls": 0,
            "mean_latency_ms": overall.get("mean_latency_ms"),
            "mock": self.mock,
            "condition": self.condition,
            "execution_backend": self.execution_backend,
            "note": "Token totals only; USD cost requires provider price table.",
        }
        atomic_write_json(self.output_dir / "cost_summary.json", cost)

        from hk_equity_mcp.experiment.unified_evaluator import evaluator_identity
        from hk_equity_mcp.experiment.unified_grading import summarize_grading

        grading_summary = summarize_grading(rows)
        error_counts: Dict[str, int] = {}
        result_status_counts: Dict[str, int] = {}
        for r in rows:
            status = r.get("status")
            status_key = (
                str(status) if status is not None and str(status).strip() else "missing"
            )
            result_status_counts[status_key] = result_status_counts.get(status_key, 0) + 1
            err = r.get("error")
            if err is not None and str(err).strip() and str(err).lower() not in {
                "none",
                "null",
            }:
                key = str(err)
                error_counts[key] = error_counts.get(key, 0) + 1

        atomic_write_json(
            self.output_dir / "error_summary.json",
            {
                "error_counts": error_counts,
                "result_status_counts": result_status_counts,
                "tool_metrics": TOOL_NA,
                "tool_metrics_note": (
                    "MCP-Free: tool metrics are N/A; zero calls are not failures."
                ),
                "evaluator": evaluator_identity(),
                "grading": grading_summary,
                "mcp_free_overall": overall,
                "grouped_aggregates": grouped,
            },
        )
        atomic_write_json(self.output_dir / "grouped_aggregates.json", grouped)
        atomic_write_json(self.output_dir / "grading_summary.json", grading_summary)
        # Silence unused run_meta for API parity with OV runner.
        _ = run_meta

    def run(self) -> Dict[str, Any]:
        prepared = self.prepare()
        items: List[Dict[str, Any]] = prepared["items"]
        run_meta = prepared["run_metadata"]
        meta_path = self.output_dir / "run_metadata.json"
        predictions_path = self.output_dir / "predictions.jsonl"

        if self.dry_run:
            atomic_write_json(meta_path, run_meta)
            print("DRY-RUN OK (MCP-Free)")
            print(f"condition={run_meta['condition']}")
            print(f"execution_backend={run_meta['execution_backend']}")
            print(f"split={self.split}")
            print(f"question_count={run_meta['question_count']}")
            print(f"domain_counts={run_meta['domain_counts']}")
            print(f"source_split_counts={run_meta.get('source_split_counts')}")
            print(f"exploratory_manifest_hash={run_meta.get('exploratory_manifest_hash')}")
            print(f"run_order_hash={run_meta.get('run_order_hash')}")
            print(f"questions_identity_hash={run_meta.get('questions_identity_hash')}")
            print(f"generation_model={run_meta['generation_model']}")
            print(f"embedding_model={run_meta['embedding_model']}")
            print(f"catalog_fingerprint={run_meta['catalog_fingerprint']}")
            print(f"system_prompt_hash={run_meta['system_prompt_hash']}")
            print(f"branch={run_meta['branch']}")
            print(f"git_commit={run_meta['git_commit']}")
            print(f"output_dir={self.output_dir}")
            return run_meta

        if self.resume and meta_path.exists():
            existing = json.loads(meta_path.read_text(encoding="utf-8"))
            assert_resume_compatible_mcp_free(existing, run_meta)
        elif self.resume and not meta_path.exists():
            raise RuntimeError(
                "Resume refused: run_metadata.json is missing. "
                "Cannot verify experiment identity."
            )

        run_meta["run_status"] = "running" if not self.mock else "mock_running"
        atomic_write_json(meta_path, run_meta)

        agent = self.build_agent()
        done = already_done_ids(predictions_path) if self.resume else set()
        if self.resume and predictions_path.exists():
            done |= {
                row.get("task_id") or row.get("id")
                for row in (
                    json.loads(line)
                    for line in predictions_path.read_text(encoding="utf-8").splitlines()
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
            try:
                # Model receives ONLY the question string (no gold leakage).
                result = agent.run(item["question"])
                # Hard fail-closed: never attach tools after LLM failure.
                if result.get("error") and (
                    result.get("underlying_operation_calls")
                    or result.get("real_mcp_execution_count")
                    or result.get("embedding_calls")
                ):
                    raise RuntimeError(
                        "MCP-Free invariant violated: tools/embedding used after failure"
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
                        "parse_repair_count": 0,
                        "resolver_calls": {
                            "find_operations": 0,
                            "describe_operation": 0,
                        },
                        "underlying_operation_calls": 0,
                        "embedding_calls": 0,
                        "embedding_tokens": 0,
                        "real_mcp_execution_count": 0,
                        "tokens": {},
                        "latency_ms": None,
                        "provider_metadata": {
                            "llm_provider": run_meta.get("generation_provider"),
                            "llm_model": run_meta.get("generation_model"),
                            "embedding_model": None,
                        },
                        "error": type(exc).__name__,
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
        atomic_write_json(meta_path, run_meta)
        print(f"Predictions: {predictions_path}")
        print(f"Run metadata: {meta_path}")
        return run_meta
