"""MCP-Zero × ETOM pilot-10 / formal-100 routing-only runner."""

from __future__ import annotations

import csv
import json
import os
import signal
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import yaml

from hk_equity_mcp.etom.cost import summarize_costs, task_variable_cost, usd
from hk_equity_mcp.etom.evaluator import evaluate_prediction
from hk_equity_mcp.etom.formal_metrics import dependency_accuracy
from hk_equity_mcp.experiment.runner import git_branch, git_commit, git_dirty
from hk_equity_mcp.mcp_zero import method_identity_sha256
from hk_equity_mcp.mcp_zero.embedding import (
    CountingEmbedder,
    build_embedding_table,
    default_shared_cache_dir,
    load_complete_table,
    table_status,
)
from hk_equity_mcp.mcp_zero.router import HierarchicalRouter
from hk_equity_mcp.mcp_zero_etom import (
    CONDITION,
    EMBEDDING_NAMESPACE,
    EXPERIMENT_FORMAL,
    EXPERIMENT_PILOT,
    EXPECTED_HK_METHOD_CARD_SHA256,
    assert_frozen_etom_config,
    assert_shared_knobs_match_hk,
    etom_method_identity_sha256,
    frozen_etom_method_identity,
    frozen_etom_runtime_env_identity,
    retrieval_method_sha256,
)
from hk_equity_mcp.mcp_zero_etom.agent import EtomMcpZeroAgent
from hk_equity_mcp.mcp_zero_etom.identity import (
    assert_resume_compatible,
    evaluator_file_hashes,
    pricing_sha256,
    resume_identity_sha256,
)
from hk_equity_mcp.mcp_zero_etom.l4_metrics import evaluate_l4_separate
from hk_equity_mcp.mcp_zero_etom.metadata import build_etom_catalog
from hk_equity_mcp.mcp_zero_etom.metrics import (
    build_error_summary,
    build_grouped_with_strict,
    build_protocol_summary,
    mcp_zero_retrieval_analysis,
)
from hk_equity_mcp.mcp_zero_etom.mock_llm import EtomMockLLM, build_route_script
from hk_equity_mcp.mcp_zero_etom.prompt import (
    ETOM_MCP_ZERO_SYSTEM_PROMPT,
    ETOM_PARSER_PROTOCOL,
    ETOM_PROMPT_VERSION,
    assert_etom_prompt_safe,
    etom_prompt_hash,
)
from hk_equity_mcp.mcp_zero_etom.recorder import EtomRouteRecorder
from hk_equity_mcp.mcp_zero_etom.source import (
    assert_formal_lock_identity,
    build_source_manifest,
    load_registry_payload,
    load_tasks_from_queries,
    resolve_source_config,
    select_tasks,
    verify_source_config,
)
from hk_equity_mcp.operation_resolver.retrieval import DeterministicHashEmbeddingBackend


def load_etom_mcp_zero_config(root: Path, path: Optional[Path] = None) -> Dict[str, Any]:
    cfg_path = path or (root / "configs" / "mcp_zero_etom_v1.yaml")
    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    assert_frozen_etom_config(data)
    assert_shared_knobs_match_hk()
    assert_etom_prompt_safe()
    return data


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


class EtomMcpZeroRunner:
    def __init__(
        self,
        root: Path,
        output_dir: Path,
        *,
        split: str,
        mode: str,
        resume: bool = False,
        allow_dirty: bool = False,
        max_tasks: Optional[int] = None,
        ids: Optional[Sequence[str]] = None,
        config_path: Optional[Path] = None,
        etom_source_dir: Optional[str] = None,
        fixture_dir: Optional[str] = None,
        embedding_cache_dir: Optional[Path] = None,
        require_complete_index: bool = False,
        index_status_only: bool = False,
        build_index_only: bool = False,
    ):
        self.root = Path(root)
        self.output_dir = Path(output_dir)
        self.split = split
        if split not in {"pilot10", "formal100"}:
            raise ValueError("split must be pilot10 or formal100")
        self.mode = mode
        self.resume = bool(resume)
        self.allow_dirty = bool(allow_dirty)
        self.max_tasks = max_tasks
        self.ids = list(ids) if ids else None
        self.config = load_etom_mcp_zero_config(self.root, config_path)
        self.require_complete_index = bool(require_complete_index)
        self.index_status_only = bool(index_status_only)
        self.build_index_only = bool(build_index_only)
        self.source_cfg = resolve_source_config(
            self.root,
            etom_source_dir=etom_source_dir,
            fixture_dir=fixture_dir,
            mode=mode if mode != "live" else "live",
        )
        if mode == "live" and self.source_cfg.source_kind == "fixture":
            raise RuntimeError("Fixture source refused for live runs")
        self.embedding_cache_dir = Path(
            embedding_cache_dir
            or default_shared_cache_dir(self.root, namespace=EMBEDDING_NAMESPACE)
        )
        self._interrupt = False
        self.catalog = None
        self.official = None
        self.registry_meta = None
        self.tasks = None
        self.selected = None
        self.lock = None
        self.table = None
        self.verified = None
        self.source_manifest = None

    def _install_signal_handlers(self) -> None:
        def _handler(signum, frame):  # noqa: ARG001
            self._interrupt = True
        signal.signal(signal.SIGINT, _handler)
        signal.signal(signal.SIGTERM, _handler)

    def prepare(self) -> Dict[str, Any]:
        self.verified = verify_source_config(self.source_cfg)
        expect = None if self.source_cfg.source_kind == "fixture" else 2375
        self.registry_meta = load_registry_payload(
            self.source_cfg, expect_tool_count=expect
        )
        self.official = self.registry_meta["official"]
        self.catalog = build_etom_catalog(
            self.registry_meta["payload"],
            official=self.official,
            include_summary=bool(self.config["retrieval"].get("use_summary", True)),
        )
        formal_identity = assert_formal_lock_identity(self.root)
        self.tasks = load_tasks_from_queries(self.source_cfg)
        if self.source_cfg.source_kind == "fixture":
            selected = list(self.tasks)
            if self.ids:
                by_id = {t.task_id: t for t in selected}
                missing = [i for i in self.ids if i not in by_id]
                if missing:
                    raise RuntimeError(f"fixture --ids not found: {missing}")
                selected = [by_id[i] for i in self.ids]
            if self.max_tasks is not None:
                selected = selected[: int(self.max_tasks)]
            self.selected = selected
            self.lock = {
                "dataset_split": "fixture",
                "ordered_task_ids": [t.task_id for t in selected],
                "sample_size": len(selected),
                "seed": None,
                "note": "fixture mode; formal lock identity still verified separately",
                "sample_sha256": formal_identity["sample_sha256"],
                "ordered_task_ids_sha256": formal_identity["ordered_task_ids_sha256"],
            }
        else:
            self.selected, self.lock = select_tasks(
                self.source_cfg,
                self.tasks,
                split=self.split,
                max_tasks=self.max_tasks,
                ids=self.ids,
            )
        self.source_manifest = build_source_manifest(
            self.source_cfg,
            verified=self.verified,
            lock=self.lock,
            registry_meta=self.registry_meta,
            catalog_fingerprint=self.catalog.fingerprint(),
            split=self.split,
        )
        self.source_manifest["formal_identity_check"] = formal_identity
        return {
            "split": self.split,
            "mode": self.mode,
            "selected_tasks": len(self.selected),
            "tool_count": self.registry_meta["tool_count"],
            "server_count": self.registry_meta["server_count"],
            "source_kind": self.source_cfg.source_kind,
            "formal_identity": formal_identity,
        }

    def index_status_report(self) -> Dict[str, Any]:
        if self.catalog is None:
            self.prepare()
        retrieval = self.config["retrieval"]
        status = table_status(
            self.catalog,
            model_name=str(self.config["embedding_model"]),
            cache_dir=self.embedding_cache_dir,
            use_summary=bool(retrieval.get("use_summary", True)),
            catalog_fingerprint=self.catalog.fingerprint(),
            score_mode=str(retrieval["score_mode"]),
            server_top_k=int(retrieval["server_top_k"]),
            tool_top_k=int(retrieval["tool_top_k"]),
            similarity_threshold=float(retrieval["similarity_threshold"]),
            namespace=EMBEDDING_NAMESPACE,
        )
        status["units"] = status["total"]
        status["server_units"] = len(self.catalog.servers())
        status["tool_units"] = len(self.catalog)
        return status

    def build_index(self, *, allow_build: bool = True) -> Dict[str, Any]:
        if self.catalog is None:
            self.prepare()
        status = self.index_status_report()
        if status["final_complete"]:
            self.table = load_complete_table(
                self.catalog,
                model_name=str(self.config["embedding_model"]),
                cache_dir=self.embedding_cache_dir,
                use_summary=bool(self.config["retrieval"].get("use_summary", True)),
                catalog_fingerprint=self.catalog.fingerprint(),
                score_mode=str(self.config["retrieval"]["score_mode"]),
                server_top_k=int(self.config["retrieval"]["server_top_k"]),
                tool_top_k=int(self.config["retrieval"]["tool_top_k"]),
                similarity_threshold=float(self.config["retrieval"]["similarity_threshold"]),
                namespace=EMBEDDING_NAMESPACE,
            )
            return {"index_action": "reuse", **status}
        if self.require_complete_index or not allow_build:
            raise RuntimeError(
                "Complete hierarchical index required but missing; refusing silent rebuild."
            )
        if self.mode == "live":
            raise RuntimeError(
                "Refusing paid live index build from runner.run; use --build-index-only."
            )
        backend = DeterministicHashEmbeddingBackend(model_name=str(self.config["embedding_model"]))
        embedder = CountingEmbedder(backend, model_name=str(self.config["embedding_model"]))
        retrieval = self.config["retrieval"]
        self.table = build_embedding_table(
            self.catalog,
            embedder,
            cache_dir=self.embedding_cache_dir,
            use_summary=bool(retrieval.get("use_summary", True)),
            catalog_fingerprint=self.catalog.fingerprint(),
            score_mode=str(retrieval["score_mode"]),
            server_top_k=int(retrieval["server_top_k"]),
            tool_top_k=int(retrieval["tool_top_k"]),
            similarity_threshold=float(retrieval["similarity_threshold"]),
            namespace=EMBEDDING_NAMESPACE,
        )
        return {
            "index_action": self.table.index_action,
            "identity_sha256": self.table.identity_sha256,
            "complete": True,
            "build_calls": self.table.build_calls,
            "build_tokens": self.table.build_tokens,
            "units": self.table.size(),
        }

    def _ensure_table(self) -> None:
        if self.table is not None:
            return
        status = self.index_status_report()
        if status["final_complete"]:
            self.table = load_complete_table(
                self.catalog,
                model_name=str(self.config["embedding_model"]),
                cache_dir=self.embedding_cache_dir,
                use_summary=bool(self.config["retrieval"].get("use_summary", True)),
                catalog_fingerprint=self.catalog.fingerprint(),
                score_mode=str(self.config["retrieval"]["score_mode"]),
                server_top_k=int(self.config["retrieval"]["server_top_k"]),
                tool_top_k=int(self.config["retrieval"]["tool_top_k"]),
                similarity_threshold=float(self.config["retrieval"]["similarity_threshold"]),
                namespace=EMBEDDING_NAMESPACE,
            )
            return
        if self.mode in {"mock", "dry-run"} and not self.require_complete_index:
            self.build_index(allow_build=True)
            return
        raise RuntimeError("Complete index missing; refuse silent rebuild.")

    def build_agent(self) -> EtomMcpZeroAgent:
        self._ensure_table()
        retrieval = self.config["retrieval"]
        limits = self.config["limits"]
        if self.mode != "mock":
            raise RuntimeError("Live LLM disabled in offline construction round; use --mock.")
        llm: Any = EtomMockLLM()
        embedder = CountingEmbedder(
            DeterministicHashEmbeddingBackend(model_name=str(self.config["embedding_model"])),
            model_name=str(self.config["embedding_model"]),
        )
        router = HierarchicalRouter(
            self.catalog,
            self.table,
            embedder,
            server_top_k=int(retrieval["server_top_k"]),
            tool_top_k=int(retrieval["tool_top_k"]),
            similarity_threshold=float(retrieval["similarity_threshold"]),
            score_mode=str(retrieval["score_mode"]),
            use_summary=bool(retrieval.get("use_summary", True)),
        )
        recorder = EtomRouteRecorder(self.official)
        return EtomMcpZeroAgent(
            llm,
            router,
            recorder,
            max_llm_turns=int(limits["max_llm_turns"]),
            max_invocations=int(limits["max_invocations"]),
            max_extension_rounds=int(limits["max_extension_rounds"]),
            timeout_sec=float(limits["timeout_sec"]),
            temperature=float(self.config["temperature"]),
            max_tokens=int(self.config["max_tokens"]),
            system_prompt=ETOM_MCP_ZERO_SYSTEM_PROMPT,
            debug_dir=self.output_dir / "debug",
        )

    def _run_metadata_base(self) -> Dict[str, Any]:
        retrieval = self.config["retrieval"]
        limits = self.config["limits"]
        runtime = frozen_etom_runtime_env_identity()
        status = self.index_status_report()
        experiment = EXPERIMENT_FORMAL if self.split == "formal100" else EXPERIMENT_PILOT
        pricing = dict(self.config.get("pricing") or {})
        meta = {
            "experiment": experiment,
            "condition": CONDITION,
            "method_identity_sha256": method_identity_sha256(),
            "hk_method_identity_sha256": EXPECTED_HK_METHOD_CARD_SHA256,
            "retrieval_method_sha256": retrieval_method_sha256(),
            "etom_method_identity_sha256": etom_method_identity_sha256(),
            "system_prompt_sha256": etom_prompt_hash(),
            "prompt_version": ETOM_PROMPT_VERSION,
            "parser_protocol_version": ETOM_PARSER_PROTOCOL,
            "server_top_k": int(retrieval["server_top_k"]),
            "tool_top_k": int(retrieval["tool_top_k"]),
            "score_mode": str(retrieval["score_mode"]),
            "similarity_threshold": float(retrieval["similarity_threshold"]),
            "max_extension_rounds": int(limits["max_extension_rounds"]),
            "use_summary": bool(retrieval.get("use_summary", True)),
            "generation_model": str(self.config["generation_model"]),
            "generation_provider": runtime["generation_provider"],
            "generation_base_url_host": runtime["generation_base_url_host"],
            "temperature": float(self.config["temperature"]),
            "max_tokens": int(self.config["max_tokens"]),
            "json_mode": bool(self.config.get("json_mode")),
            "limits": {
                "max_llm_turns": int(limits["max_llm_turns"]),
                "max_invocations": int(limits["max_invocations"]),
                "timeout_sec": float(limits["timeout_sec"]),
            },
            "embedding_model": str(self.config["embedding_model"]),
            "embedding_cache_namespace": EMBEDDING_NAMESPACE,
            "embedding_index_identity_sha256": status["identity_sha256"],
            "evaluator_file_sha256s": evaluator_file_hashes(self.root),
            "pricing_sha256": pricing_sha256(pricing),
            "dataset_split": self.split,
            "mock": self.mode == "mock",
            "source_kind": self.source_cfg.source_kind,
            "source_repository": (self.source_manifest or {}).get("source_repository"),
            "source_commit": (self.source_manifest or {}).get("source_commit"),
            "source_pin_sha256": (self.source_manifest or {}).get("source_pin_sha256"),
            "dataset_file_sha256": (self.source_manifest or {}).get("dataset_file_sha256"),
            "registry_sha256": self.registry_meta["registry_sha256"],
            "registry_fingerprint": self.registry_meta["registry_fingerprint"],
            "registry_tool_count": self.registry_meta["tool_count"],
            "registry_server_count": self.registry_meta["server_count"],
            "sample_lock_sha256": (self.source_manifest or {}).get("sample_lock_sha256"),
            "sample_sha256": (self.source_manifest or {}).get("sample_sha256"),
            "ordered_task_ids_sha256": (self.source_manifest or {}).get("ordered_task_ids_sha256"),
            "excluded_cost_pilot_lock_sha256": (self.source_manifest or {}).get(
                "excluded_cost_pilot_lock_sha256"
            ),
            "question_ids": [t.task_id for t in self.selected],
            "max_tasks": self.max_tasks,
            "max_tasks_semantics": "cumulative_locked_order_prefix",
            "frozen_method": frozen_etom_method_identity(),
            "routing_only": True,
            "real_tool_executions": 0,
            "git_commit": git_commit(self.root),
            "git_branch": git_branch(self.root),
            "git_dirty": git_dirty(self.root),
            "output_dir": str(self.output_dir),
        }
        meta["resume_identity_sha256"] = resume_identity_sha256(meta)
        return meta

    def _score_row(self, task, agent_result: Dict[str, Any]) -> Dict[str, Any]:
        predicted_tools = agent_result.get("predicted_tools") or []
        predicted_nodes = agent_result.get("predicted_nodes") or []
        status = agent_result.get("status")
        if status != "completed":
            eval_row = {"exact_match": False, "precision": 0.0, "recall": 0.0, "f1": 0.0, "status": "failed"}
        else:
            eval_row = evaluate_prediction(int(task.level), task.ground_truth_tools, predicted_tools)
        dep = dependency_accuracy(
            level=int(task.level),
            ground_truth_tools=task.ground_truth_tools,
            predicted_nodes=predicted_nodes,
        )
        l4 = {}
        if int(task.level) == 4:
            l4 = evaluate_l4_separate(
                ground_truth_tools=task.ground_truth_tools,
                predicted_tools=predicted_tools,
                predicted_nodes=predicted_nodes,
                status=status if status == "completed" else "failed",
            )
        pricing = dict(self.config.get("pricing") or {})
        row = {
            "task_id": task.task_id,
            "level": int(task.level),
            "query_id": task.query_id,
            "dataset_split": self.split,
            "query": task.query,
            "gold_route": task.ground_truth_tools,
            "gold_operation_ids": list(task.operation_ids),
            "gold_server_names": sorted(task.server_names),
            "status": status,
            "error": agent_result.get("error"),
            "error_type": agent_result.get("error_type"),
            "timeout": bool(agent_result.get("timeout")),
            "exact_match": bool(eval_row.get("exact_match")) if status == "completed" else False,
            "precision": float(eval_row.get("precision") or 0.0),
            "recall": float(eval_row.get("recall") or 0.0),
            "f1": float(eval_row.get("f1") or 0.0),
            "strict_score": 1.0 if status == "completed" and eval_row.get("exact_match") else 0.0,
            "eval_status": eval_row.get("status"),
            "dependency_edge_accuracy": dep.get("dependency_edge_accuracy"),
            "ordered_dependency_accuracy": dep.get("ordered_dependency_accuracy"),
            "predicted_tools": predicted_tools,
            "predicted_nodes": predicted_nodes,
            "predicted_rejection": bool(agent_result.get("predicted_rejection")),
            "retrieved_servers": agent_result.get("retrieved_servers"),
            "retrieved_tools": agent_result.get("retrieved_tools"),
            "expanded_servers": agent_result.get("expanded_servers"),
            "expanded_tools": agent_result.get("expanded_tools"),
            "candidate_operation_ids": agent_result.get("candidate_operation_ids"),
            "selected_operations": agent_result.get("selected_operations"),
            "number_of_extension_rounds": agent_result.get("number_of_extension_rounds"),
            "llm_calls": agent_result.get("llm_calls"),
            "input_tokens": agent_result.get("input_tokens"),
            "output_tokens": agent_result.get("output_tokens"),
            "total_tokens": agent_result.get("total_tokens"),
            "query_embedding_calls": agent_result.get("query_embedding_calls"),
            "query_embedding_tokens": agent_result.get("query_embedding_tokens"),
            "latency_ms": agent_result.get("latency_ms"),
            "parse_repairs": agent_result.get("parse_repairs"),
            "protocol_repair_count": agent_result.get("protocol_repair_count"),
            "http_retries": 0,
            "real_tool_executions": 0,
            "executed_etom_mcp_tool": False,
            "trace": agent_result.get("trace"),
            "raw_assistant_contents": [
                e.get("raw_content")
                for e in (agent_result.get("trace") or [])
                if isinstance(e, dict) and e.get("type") == "llm_turn"
            ],
            "mcp_zero_trace": agent_result.get("mcp_zero_trace"),
            "task_protocol_success": agent_result.get("task_protocol_success"),
            "final_answer_obj": agent_result.get("final_answer_obj"),
            **l4,
        }
        row["variable_cost_usd"] = task_variable_cost(row, pricing)
        row["llm_cost_usd"] = usd(int(row.get("input_tokens") or 0), float(pricing.get("llm_input_usd_per_million") or 0)) + usd(
            int(row.get("output_tokens") or 0), float(pricing.get("llm_output_usd_per_million") or 0)
        )
        row["query_embedding_cost_usd"] = usd(
            int(row.get("query_embedding_tokens") or 0),
            float(pricing.get("embedding_usd_per_million") or 0),
        )
        return row

    def _flush_reports(self, rows: List[Dict[str, Any]], run_meta: Dict[str, Any]) -> None:
        tasks_by_id = {t.task_id: t for t in self.selected}
        grouped = build_grouped_with_strict(rows, expected_per_level=20 if self.split == "formal100" else 2)
        _atomic_write_json(self.output_dir / "grouped_results.json", grouped)
        _atomic_write_json(self.output_dir / "error_summary.json", build_error_summary(rows))
        _atomic_write_json(self.output_dir / "retrieval_analysis.json", mcp_zero_retrieval_analysis(rows, tasks_by_id))
        _atomic_write_json(self.output_dir / "protocol_summary.json", build_protocol_summary(rows))
        pricing = dict(self.config.get("pricing") or {})
        costs = summarize_costs(
            rows,
            pricing=pricing,
            catalog={
                "calls": getattr(self.table, "build_calls", 0) if self.table else 0,
                "retries": 0,
                "tokens": getattr(self.table, "build_tokens", 0) if self.table else 0,
                "token_source": "estimated",
                "cache_reused": bool(getattr(self.table, "cache_reused", False)),
            },
            probe={"usd": 0.0},
        )
        _atomic_write_json(self.output_dir / "cost_summary.json", costs)
        _atomic_write_json(self.output_dir / "run_metadata.json", run_meta)
        _atomic_write_json(self.output_dir / "source_manifest.json", self.source_manifest or {})
        fields = [
            "task_id","level","status","exact_match","strict_score","precision","recall","f1",
            "dependency_edge_accuracy","ordered_dependency_accuracy",
            "official_union_node_set_em","efs_covering_node_set",
            "cross_server_dependency_correctness","complete_route_success",
            "predicted_rejection","llm_calls","input_tokens","output_tokens",
            "query_embedding_calls","query_embedding_tokens","latency_ms",
            "parse_repairs","real_tool_executions","error_type",
        ]
        with (self.output_dir / "per_task_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k) for k in fields})

    def execute(self) -> Dict[str, Any]:
        pre = self.prepare()
        if self.index_status_only:
            status = self.index_status_report()
            self.output_dir.mkdir(parents=True, exist_ok=True)
            _atomic_write_json(self.output_dir / "index_status.json", status)
            return {"action": "index-status", **status}
        if self.build_index_only:
            built = self.build_index(allow_build=True)
            self.output_dir.mkdir(parents=True, exist_ok=True)
            _atomic_write_json(self.output_dir / "index_build.json", built)
            return {"action": "build-index", **built}
        if self.mode == "dry-run":
            self.output_dir.mkdir(parents=True, exist_ok=True)
            report = {
                "action": "dry-run",
                "preflight": pre,
                "index_status": self.index_status_report(),
                "selected_task_ids": [t.task_id for t in self.selected],
                "method_identity_sha256": method_identity_sha256(),
                "etom_method_identity_sha256": etom_method_identity_sha256(),
                "etom_prompt_hash": etom_prompt_hash(),
                "routing_only": True,
                "real_tool_executions": 0,
            }
            _atomic_write_json(self.output_dir / "preflight_report.json", report)
            _atomic_write_json(self.output_dir / "source_manifest.json", self.source_manifest or {})
            print("DRY-RUN OK (mcp_zero_etom)")
            return report
        if self.mode == "preflight":
            self.output_dir.mkdir(parents=True, exist_ok=True)
            report = {"action": "preflight", "preflight": pre, "index_status": self.index_status_report(), "offline": True, "network_calls": 0}
            _atomic_write_json(self.output_dir / "preflight_report.json", report)
            print("PREFLIGHT OK (mcp_zero_etom)")
            return report
        if self.mode != "mock":
            raise RuntimeError(f"Mode {self.mode!r} disabled in offline construction round.")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_table()
        run_meta = self._run_metadata_base()
        pred_path = self.output_dir / "predictions.jsonl"
        existing_rows = _read_jsonl(pred_path) if self.resume else []
        if self.resume and (self.output_dir / "run_metadata.json").exists():
            old = json.loads((self.output_dir / "run_metadata.json").read_text(encoding="utf-8"))
            assert_resume_compatible(old, run_meta)
        completed = {r["task_id"] for r in existing_rows if r.get("status") == "completed"}
        rows = [r for r in existing_rows if r.get("task_id") in completed]
        if self.resume:
            if pred_path.exists():
                pred_path.unlink()
            for row in rows:
                _append_jsonl(pred_path, row)

        agent = self.build_agent()
        mock_llm: EtomMockLLM = agent.llm  # type: ignore[assignment]
        self._install_signal_handlers()
        for task in self.selected:
            if self._interrupt:
                break
            if task.task_id in completed:
                continue
            if int(task.level) == 5 or task.infeasible:
                script = build_route_script(server_desc="none", tool_desc="none", operation_ids=[], infeasible=True)
            else:
                ops = list(task.operation_ids)
                server_desc = sorted(task.server_names)[0] if task.server_names else "server"
                tool_desc = task.tool_names[0] if task.tool_names else "tool"
                script = build_route_script(server_desc=server_desc, tool_desc=tool_desc, operation_ids=ops[:1] if ops else [], infeasible=False)
            mock_llm.set_task(task.task_id, script)
            result = agent.run(task.query, task_label=task.task_id, ground_truth_tools=task.ground_truth_tools)
            row = self._score_row(task, result)
            assert int(row.get("real_tool_executions") or 0) == 0
            _append_jsonl(pred_path, row)
            rows.append(row)
            run_meta["completed_task_ids"] = [r["task_id"] for r in rows if r.get("status") == "completed"]
            run_meta["run_status"] = "interrupted" if self._interrupt else "running"
            self._flush_reports(rows, run_meta)

        run_meta["run_status"] = "interrupted" if self._interrupt else "completed"
        run_meta["completed_task_ids"] = [r["task_id"] for r in rows if r.get("status") == "completed"]
        self._flush_reports(rows, run_meta)
        print(f"MOCK OK (mcp_zero_etom) split={self.split} rows={len(rows)} real_tool_executions=0")
        return {"action": "mock", "rows": len(rows), "output_dir": str(self.output_dir), "resume_identity_sha256": run_meta["resume_identity_sha256"]}
