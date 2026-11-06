"""Unified HK MCP-Zero baseline runner (Real-MCP execution).

Offline-safe modes: --dry-run and --mock. Never calls paid APIs in those modes.
Shares dataset / evaluator / gateway with Operation-Visible; the primary
experiment variable is the MCP-Zero router / orchestration method.
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
from hk_equity_mcp.mcp_zero import (
    CONDITION,
    DEFAULT_MAX_EXTENSION_ROUNDS,
    DEFAULT_SERVER_TOP_K,
    DEFAULT_SIMILARITY_THRESHOLD,
    DEFAULT_TOOL_TOP_K,
    FIDELITY_NAME,
    assert_frozen_config,
    frozen_method_identity,
    method_identity_sha256,
)
from hk_equity_mcp.mcp_zero.agent import McpZeroAgent
from hk_equity_mcp.mcp_zero.embedding import (
    CACHE_NAMESPACE,
    CountingEmbedder,
    build_embedding_table,
)
from hk_equity_mcp.mcp_zero.metadata import build_hk_metadata
from hk_equity_mcp.mcp_zero.prompt import HK_MCP_ZERO_SYSTEM_PROMPT, mcp_zero_prompt_hash
from hk_equity_mcp.mcp_zero.router import HierarchicalRouter
from hk_equity_mcp.mcp_zero.scoring import SCORE_OFFICIAL_COMBINED
from hk_equity_mcp.operation_resolver.retrieval import (
    DeterministicHashEmbeddingBackend,
    OpenAICompatibleEmbeddingBackend,
)
from hk_equity_mcp.real_mcp.unified_catalog import build_unified_catalog
from hk_equity_mcp.real_mcp.unified_gateway import (
    ScriptedMockTransport,
    UnifiedRealMcpGateway,
)

# max_tasks is intentionally absent: it is the cumulative locked-order prefix
# (20, then 50, then 100, then 155) and must be allowed to grow on resume.
MAX_TASKS_SEMANTICS = "cumulative_locked_order_prefix"

MCP_ZERO_RESUME_IDENTITY_KEYS = RESUME_IDENTITY_KEYS + (
    "server_top_k",
    "tool_top_k",
    "score_mode",
    "max_extension_rounds",
    "similarity_threshold",
    "method_identity_sha256",
    "max_tasks_semantics",
    "embedding_cache_namespace",
)


def load_mcp_zero_config(
    path: Optional[Path] = None, root: Optional[Path] = None
) -> Dict[str, Any]:
    root = root or repo_root()
    cfg_path = path or (root / "configs" / "mcp_zero_hk_v1.yaml")
    if not cfg_path.exists():
        return {
            "condition": CONDITION,
            "execution_backend": "real_mcp",
            "generation_provider": "aliyun",
            "generation_model": "deepseek-v4.1-flash",
            "embedding_model": "text-embedding-3-small",
            "temperature": 0.0,
            "max_tokens": 1024,
            "retrieval": {
                "server_top_k": DEFAULT_SERVER_TOP_K,
                "tool_top_k": DEFAULT_TOOL_TOP_K,
                "similarity_threshold": DEFAULT_SIMILARITY_THRESHOLD,
                "score_mode": SCORE_OFFICIAL_COMBINED,
                "use_summary": True,
            },
            "limits": {
                "max_llm_turns": 8,
                "max_invocations": 6,
                "max_extension_rounds": DEFAULT_MAX_EXTENSION_ROUNDS,
                "timeout_sec": 120,
            },
        }
    with cfg_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


class _McpZeroMockLLM(ScriptedLLMClient):
    """Offline MCP-Zero trajectory that invokes a *retrieved* operation_id."""

    PREFERRED_OP = "aktools.stock_financial_hk_analysis_indicator_em"

    def __init__(self) -> None:
        super().__init__([], model="mock-llm")
        self.provider = "mock"
        self._phase = 0

    def complete(
        self,
        messages,
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ):
        del temperature, max_tokens
        phase = self._phase
        self._phase = (self._phase + 1) % 3
        if phase == 0:
            content = (
                "Need HK equity financial indicators.\n"
                "<tool_assistant>\n"
                "server: AKTools MCP server for Hong Kong equity financial data "
                "and analysis indicators\n"
                "tool: stock financial hk analysis indicator em\n"
                "</tool_assistant>"
            )
        elif phase == 1:
            available = _extract_available_operation_ids(messages)
            oid = (
                self.PREFERRED_OP
                if self.PREFERRED_OP in available
                else (available[0] if available else self.PREFERRED_OP)
            )
            content = json.dumps(
                {
                    "thought": "invoke retrieved tool",
                    "tool_calls": [
                        {
                            "name": "invoke_api",
                            "arguments": {
                                "operation_id": oid,
                                "arguments": {"symbol": "0700"},
                            },
                        }
                    ],
                    "final_answer": None,
                },
                ensure_ascii=False,
            )
        else:
            content = json.dumps(
                {
                    "thought": "answer from evidence",
                    "tool_calls": [],
                    "final_answer": {
                        "status": "answered",
                        "final_answer": "mock answer",
                        "reasoning_brief": "offline mcp-zero mock trajectory",
                        "evidence": [
                            {
                                "operation_id": self.PREFERRED_OP,
                                "fields": ["mock"],
                                "note": "mock transport",
                            }
                        ],
                    },
                },
                ensure_ascii=False,
            )
        return {
            "content": content,
            "usage": {"input_tokens": 1, "output_tokens": 1},
            "latency_ms": 0.0,
            "provider": self.provider,
            "model": self.model,
        }


def _extract_available_operation_ids(messages: Sequence[Dict[str, Any]]) -> List[str]:
    for message in reversed(list(messages)):
        if message.get("role") != "user":
            continue
        content = str(message.get("content") or "")
        marker = "Available operation_ids now:"
        if marker not in content:
            continue
        tail = content.split(marker, 1)[1].strip()
        line = tail.splitlines()[0].strip()
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
    return []


def _mock_scripted_llm() -> ScriptedLLMClient:
    return _McpZeroMockLLM()


class UnifiedHkMcpZeroRunner:
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
        self.config = config or load_mcp_zero_config(root=self.root)
        self.allow_hkex_download = allow_hkex_download
        self.json_mode = json_mode
        self.temperature = float(self.config.get("temperature", 0.0))
        self.max_tokens = int(self.config.get("max_tokens", 1024))
        self.limits = dict(self.config.get("limits") or {})
        self.retrieval = dict(self.config.get("retrieval") or {})
        self.condition = CONDITION
        self.execution_backend = "real_mcp"
        self.system_prompt = HK_MCP_ZERO_SYSTEM_PROMPT
        self.gen_meta: Dict[str, str] = {}
        self.emb_meta: Optional[Dict[str, str]] = None
        self._gateway: Optional[UnifiedRealMcpGateway] = None
        self.bundle = build_unified_catalog(root=self.root)
        self.catalog = build_hk_metadata(
            self.bundle,
            include_summary=bool(self.retrieval.get("use_summary", True)),
        )
        self._exploratory_manifest: Optional[Dict[str, Any]] = None
        self._exploratory_run_order: Optional[Dict[str, Any]] = None
        self.embedding_index_report: Optional[Dict[str, Any]] = None

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

        assert_frozen_config(self.config)
        for banned in BANNED_PROMPT_SUBSTRINGS:
            if banned in self.system_prompt:
                raise RuntimeError(f"MCP-Zero system prompt unexpectedly contains {banned}")

        prompt_h = mcp_zero_prompt_hash()
        tasks = self._load_tasks()
        stats = self.bundle.stats

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
            actual_emb = self.emb_meta.get("model") if self.emb_meta else None
            actual_emb = actual_emb or os.environ.get("EMBEDDING_MODEL")
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

        # Fail closed: never silently write Cursor / Claude into experiment meta.
        for forbidden in ("claude", "sonnet", "cursor-agent", "anthropic"):
            blob = f"{expected_model} {self.gen_meta.get('model')}".lower()
            if forbidden in blob and "deepseek" not in blob:
                raise RuntimeError(
                    f"Refusing experiment generation model containing {forbidden!r}"
                )

        self.output_dir.mkdir(parents=True, exist_ok=True)
        fingerprint = catalog_fingerprint(stats, self.bundle.source_catalogs)
        gen_host = url_host(
            self.gen_meta.get("base_url") or os.environ.get("LLM_BASE_URL")
        )
        emb_host = url_host(
            (self.emb_meta or {}).get("base_url")
            or os.environ.get("EMBEDDING_BASE_URL")
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

        server_top_k = int(self.retrieval.get("server_top_k", DEFAULT_SERVER_TOP_K))
        tool_top_k = int(self.retrieval.get("tool_top_k", DEFAULT_TOOL_TOP_K))
        score_mode = str(self.retrieval.get("score_mode", SCORE_OFFICIAL_COMBINED))
        similarity_threshold = float(
            self.retrieval.get("similarity_threshold", DEFAULT_SIMILARITY_THRESHOLD)
        )
        max_extension_rounds = int(
            self.limits.get("max_extension_rounds", DEFAULT_MAX_EXTENSION_ROUNDS)
        )

        meta = {
            "condition": self.condition,
            "execution_backend": self.execution_backend,
            "real_mcp": True,
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
            "domain_counts": domain_counts,
            "source_split_counts": source_split_counts or None,
            "generation_provider": self.gen_meta.get("provider") or expected_provider,
            "generation_model": self.gen_meta.get("model") or expected_model,
            "generation_profile_file": self.gen_meta.get("profile_file"),
            "generation_base_url_host": gen_host,
            "embedding_model": (self.emb_meta or {}).get("model") or expected_emb,
            "embedding_profile_file": (self.emb_meta or {}).get("profile_file"),
            "embedding_base_url_host": emb_host,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "json_mode": self.json_mode,
            "server_top_k": server_top_k,
            "tool_top_k": tool_top_k,
            "score_mode": score_mode,
            "similarity_threshold": similarity_threshold,
            "max_extension_rounds": max_extension_rounds,
            "retrieval_top_k": tool_top_k,  # schema compat alias
            "lexical_weight": None,  # MCP-Zero does not use lexical blend
            "limits": self.limits,
            "system_prompt_hash": prompt_h,
            "system_prompt": self.system_prompt,
            "catalog_fingerprint": fingerprint,
            "mcp_zero_catalog_fingerprint": self.catalog.fingerprint(),
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
                "mcp_zero_servers": self.catalog.server_ids(),
                "mcp_zero_tool_count": len(self.catalog),
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
            "max_tasks_semantics": MAX_TASKS_SEMANTICS,
            "max_tasks_in_resume_identity": False,
            "ids": self.ids,
            "method_identity": frozen_method_identity(),
            "method_identity_sha256": method_identity_sha256(),
            "embedding_cache_namespace": CACHE_NAMESPACE,
            "embedding_index": None,
            "run_status": "dry_run" if self.dry_run else ("mock" if self.mock else "running"),
            "fidelity_note": FIDELITY_NAME,
            "config_path": "configs/mcp_zero_hk_v1.yaml",
            "config_sha256": (
                file_sha256(self.root / "configs" / "mcp_zero_hk_v1.yaml")
                if (self.root / "configs" / "mcp_zero_hk_v1.yaml").exists()
                else None
            ),
            "dataset_note": (
                "HK-155 MCP-Zero adapted baseline. Same Real MCP Gateway + "
                "evaluator v1.2 as Operation-Visible. Primary variable is the "
                "MCP-Zero hierarchical router / orchestration."
            ),
            "dry_run_missing_env": getattr(self, "_dry_run_missing", None),
            "cursor_coding_model_note": (
                "Cursor coding agents are not an experiment variable; "
                "generation_model must remain deepseek-v4.1-flash."
            ),
        }
        return {"items": tasks, "run_metadata": redact_secrets(meta)}

    def build_agent(self) -> McpZeroAgent:
        if self.dry_run:
            raise RuntimeError("build_agent must not be called during dry-run.")

        use_summary = bool(self.retrieval.get("use_summary", True))
        if self.mock:
            llm: Any = _mock_scripted_llm()
            backend: Any = DeterministicHashEmbeddingBackend()
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
            backend = OpenAICompatibleEmbeddingBackend(
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
        embedder = CountingEmbedder(
            backend,
            model_name=getattr(backend, "model_name", None)
            or (self.emb_meta or {}).get("model")
            or "text-embedding-3-small",
        )
        cache_dir = self.output_dir / ".cache" / CACHE_NAMESPACE
        server_top_k = int(self.retrieval.get("server_top_k", DEFAULT_SERVER_TOP_K))
        tool_top_k = int(self.retrieval.get("tool_top_k", DEFAULT_TOOL_TOP_K))
        score_mode = str(self.retrieval.get("score_mode", SCORE_OFFICIAL_COMBINED))
        similarity_threshold = float(
            self.retrieval.get("similarity_threshold", DEFAULT_SIMILARITY_THRESHOLD)
        )
        catalog_fp = catalog_fingerprint(self.bundle.stats, self.bundle.source_catalogs)
        table = build_embedding_table(
            self.catalog,
            embedder,
            cache_dir=cache_dir,
            use_summary=use_summary,
            catalog_fingerprint=catalog_fp,
            score_mode=score_mode,
            server_top_k=server_top_k,
            tool_top_k=tool_top_k,
            similarity_threshold=similarity_threshold,
        )
        if not table.complete:
            raise RuntimeError(
                "Real MCP-Zero run requires a complete embedding index. "
                f"index_action={table.index_action}."
            )
        self.embedding_index_report = {
            "namespace": CACHE_NAMESPACE,
            "identity": table.identity,
            "identity_sha256": table.identity_sha256,
            "cache_path": table.cache_path,
            "cache_reused": table.cache_reused,
            "index_action": table.index_action,
            "build_calls": table.build_calls,
            "build_tokens": table.build_tokens,
            "build_latency_ms": table.build_latency_ms,
            "complete": True,
            "embedding_model": table.model_name,
        }
        # Reset query-time counters after table build (build tokens tracked in table).
        embedder.calls = 0
        embedder.tokens = 0
        router = HierarchicalRouter(
            self.catalog,
            table,
            embedder,
            server_top_k=int(self.retrieval.get("server_top_k", DEFAULT_SERVER_TOP_K)),
            tool_top_k=int(self.retrieval.get("tool_top_k", DEFAULT_TOOL_TOP_K)),
            similarity_threshold=float(
                self.retrieval.get(
                    "similarity_threshold", DEFAULT_SIMILARITY_THRESHOLD
                )
            ),
            score_mode=str(
                self.retrieval.get("score_mode", SCORE_OFFICIAL_COMBINED)
            ),
            use_summary=use_summary,
        )
        return McpZeroAgent(
            llm=llm,
            router=router,
            gateway=gateway,
            max_llm_turns=int(self.limits.get("max_llm_turns", 8)),
            max_invocations=int(self.limits.get("max_invocations", 6)),
            max_extension_rounds=int(
                self.limits.get("max_extension_rounds", DEFAULT_MAX_EXTENSION_ROUNDS)
            ),
            timeout_sec=float(self.limits.get("timeout_sec", 120)),
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            system_prompt=self.system_prompt,
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
            "previously_reserved": item.get("previously_reserved"),
            "benchmark_role": item.get("benchmark_role"),
            "company": item.get("company"),
            "gold_status": item.get("gold_status"),
            "gold_values": item.get("gold_values"),
            "llm_calls": result.get("llm_calls"),
            "tokens": tokens,
            "embedding_calls": result.get("embedding_calls"),
            "embedding_tokens": result.get("embedding_tokens"),
            "latency_ms": result.get("latency_ms"),
            "generation_latency_ms": result.get("generation_latency_ms"),
            "retrieval_latency_ms": result.get("retrieval_latency_ms"),
            "retry_count": result.get("retry_count", 0),
            "timeout": result.get("error") == "timeout",
            "trace": result.get("trace") or [],
            "resolver_calls": result.get("resolver_calls")
            or {"find_operations": 0, "describe_operation": 0},
            "underlying_operation_calls": result.get("underlying_operation_calls", 0),
            "provider_metadata": result.get("provider_metadata") or {},
            "condition": self.condition,
            "active_tool_request": result.get("active_tool_request"),
            "retrieved_servers": result.get("retrieved_servers"),
            "retrieved_tools": result.get("retrieved_tools"),
            "selected_operations": result.get("selected_operations"),
            "invoked_operations": result.get("invoked_operations"),
            "extension_rounds": result.get("extension_rounds"),
            "number_of_extension_rounds": result.get("number_of_extension_rounds"),
            "termination_reason": result.get("termination_reason"),
            "mcp_zero_trace": result.get("mcp_zero_trace"),
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
                        "failed_invocation_count": pred.get("failed_invocation_count"),
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
        summary = summarize_predictions(rows)
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
            "total_embedding_calls": sum(int(r.get("embedding_calls") or 0) for r in rows),
            "total_embedding_tokens": sum(
                int(r.get("embedding_tokens") or 0) for r in rows
            ),
            "total_real_mcp_executions": sum(
                int(r.get("real_mcp_execution_count") or 0) for r in rows
            ),
            "mean_latency_ms": summary.get("mean_latency_ms"),
            "mean_retrieval_latency_ms": (
                sum(float(r.get("retrieval_latency_ms") or 0) for r in rows) / len(rows)
                if rows
                else None
            ),
            "mean_generation_latency_ms": (
                sum(float(r.get("generation_latency_ms") or 0) for r in rows) / len(rows)
                if rows
                else None
            ),
            "mock": self.mock,
            "embedding_index_build_calls": (run_meta.get("embedding_index") or {}).get(
                "build_calls"
            ),
            "embedding_index_build_tokens": (run_meta.get("embedding_index") or {}).get(
                "build_tokens"
            ),
            "embedding_index_build_latency_ms": (
                run_meta.get("embedding_index") or {}
            ).get("build_latency_ms"),
            "embedding_index_cache_reused": (run_meta.get("embedding_index") or {}).get(
                "cache_reused"
            ),
            "embedding_index_action": (run_meta.get("embedding_index") or {}).get(
                "index_action"
            ),
            "embedding_cache_namespace": CACHE_NAMESPACE,
            "note": "Token totals only; USD cost requires provider price table. "
            "embedding_calls on each task are query-time. Index build calls, "
            "tokens, latency, and cache_reused are the embedding_index_* fields.",
        }
        atomic_write_json(self.output_dir / "cost_summary.json", cost)
        atomic_write_json(
            self.output_dir / "retrieval_analysis.json",
            {
                "condition": self.condition,
                "server_top_k": run_meta.get("server_top_k"),
                "tool_top_k": run_meta.get("tool_top_k"),
                "score_mode": run_meta.get("score_mode"),
                "per_task": [
                    {
                        "task_id": r["task_id"],
                        "active_tool_request": r.get("active_tool_request"),
                        "retrieved_servers": r.get("retrieved_servers"),
                        "retrieved_tools": r.get("retrieved_tools"),
                        "retrieved_candidates": r.get("retrieved_candidates"),
                        "selected_operations": r.get("selected_operations"),
                        "invoked_operations": r.get("invoked_operations"),
                        "predicted_operations": r.get("predicted_operations"),
                        "gold_operations": r.get("gold_operations"),
                        "extension_rounds": r.get("extension_rounds"),
                        "termination_reason": r.get("termination_reason"),
                        "retrieval_latency_ms": r.get("retrieval_latency_ms"),
                        "generation_latency_ms": r.get("generation_latency_ms"),
                        "embedding_calls": r.get("embedding_calls"),
                        "embedding_tokens": r.get("embedding_tokens"),
                        "llm_calls": r.get("llm_calls"),
                    }
                    for r in rows
                ],
            },
        )
        from hk_equity_mcp.experiment.unified_aggregates import build_grouped_aggregates
        from hk_equity_mcp.experiment.unified_evaluator import evaluator_identity
        from hk_equity_mcp.experiment.unified_grading import summarize_grading
        from hk_equity_mcp.experiment.unified_schema import METRIC_FIELD_DEFINITIONS

        grading_summary = summarize_grading(rows)
        grouped = build_grouped_aggregates(rows)
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
            },
        )
        atomic_write_json(self.output_dir / "grouped_aggregates.json", grouped)
        atomic_write_json(self.output_dir / "grading_summary.json", grading_summary)

    def run(self) -> Dict[str, Any]:
        prepared = self.prepare()
        items: List[Dict[str, Any]] = prepared["items"]
        run_meta = prepared["run_metadata"]
        meta_path = self.output_dir / "run_metadata.json"
        predictions_path = self.output_dir / "predictions.jsonl"

        if self.dry_run:
            atomic_write_json(meta_path, run_meta)
            print("DRY-RUN OK (mcp_zero)")
            print(f"split={self.split}")
            print(f"smoke={self.smoke}")
            print(f"mock={self.mock}")
            print(f"generation_model={run_meta['generation_model']}")
            print(f"embedding_model={run_meta['embedding_model']}")
            print(f"server_top_k={run_meta['server_top_k']}")
            print(f"tool_top_k={run_meta['tool_top_k']}")
            print(f"max_extension_rounds={run_meta['max_extension_rounds']}")
            print(f"system_prompt_hash={run_meta['system_prompt_hash']}")
            print(f"unique_operation_ids={run_meta['catalog_stats']['unique_operation_ids']}")
            print(
                "runtime_exposed_unique="
                f"{run_meta['catalog_stats']['runtime_exposed_unique']}"
            )
            print(f"mcp_zero_tool_count={run_meta['catalog_stats']['mcp_zero_tool_count']}")
            print(f"branch={run_meta['branch']}")
            print(f"git_commit={run_meta['git_commit']}")
            print(f"output_dir={self.output_dir}")
            return run_meta

        if self.resume and meta_path.exists():
            existing = json.loads(meta_path.read_text(encoding="utf-8"))
            assert_resume_compatible_mcp_zero(existing, run_meta)
        elif self.resume and not meta_path.exists():
            raise RuntimeError(
                "Resume refused: run_metadata.json is missing. "
                "Cannot verify experiment identity."
            )

        run_meta["run_status"] = "running" if not self.mock else "mock_running"
        atomic_write_json(meta_path, run_meta)

        try:
            agent = self.build_agent()
            run_meta["embedding_index"] = self.embedding_index_report
            atomic_write_json(meta_path, run_meta)
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
                                "mcp_zero_retrieve": 0,
                            },
                            "underlying_operation_calls": 0,
                            "tokens": {},
                            "latency_ms": None,
                            "provider_metadata": {
                                "llm_provider": run_meta.get("generation_provider"),
                                "llm_model": run_meta.get("generation_model"),
                                "embedding_model": run_meta.get("embedding_model"),
                            },
                            "error": type(exc).__name__,
                        },
                    )
                    row["error"] = type(exc).__name__
                append_jsonl(predictions_path, row)
                predictions_by_id[qid] = row
                run_meta["completed_count"] = len(predictions_by_id)
                run_meta["last_completed_task_id"] = qid
                run_meta["run_status"] = "mock_running" if self.mock else "running"
                self._write_side_artifacts(
                    items=items,
                    predictions_by_id=predictions_by_id,
                    run_meta=run_meta,
                )
                atomic_write_json(meta_path, run_meta)

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


def assert_resume_compatible_mcp_zero(
    existing: Dict[str, Any], current: Dict[str, Any]
) -> None:
    mismatches = []
    for key in MCP_ZERO_RESUME_IDENTITY_KEYS:
        if existing.get(key) != current.get(key):
            mismatches.append(
                f"{key}: existing={existing.get(key)!r} current={current.get(key)!r}"
            )
    old_ids = existing.get("question_ids")
    new_ids = current.get("question_ids")
    if not isinstance(old_ids, list) or not isinstance(new_ids, list):
        mismatches.append("question_ids: missing on existing or current metadata")
    elif new_ids[: len(old_ids)] != old_ids:
        mismatches.append(
            "question_ids: current list is not a cumulative prefix extension of "
            f"the previous run (previous={len(old_ids)} current={len(new_ids)}). "
            "max_tasks is the cumulative locked-order target, not a count of new tasks."
        )
    if mismatches:
        raise RuntimeError(
            "Resume refused: MCP-Zero experiment identity mismatch:\n"
            + "\n".join(mismatches)
        )
