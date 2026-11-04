"""10-task ETOM cost pilot. Refuses an 11th task and any non-routing mode."""

from __future__ import annotations

import csv
import json
import os
import signal
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import yaml

from hk_equity_mcp.common.env_loader import (
    load_embedding_profile,
    load_generation_profile,
)
from hk_equity_mcp.etom import (
    EXPECTED_EMBEDDING_MODEL,
    EXPECTED_LLM_MODEL,
    EXPECTED_LLM_PROVIDER,
    FIDELITY_NAME,
    PILOT_MAX_TASKS,
)
from hk_equity_mcp.etom.adapter import EtomRoutingResolver
from hk_equity_mcp.etom.agent import EtomRoutingAgent
from hk_equity_mcp.etom.convert import convert_registry
from hk_equity_mcp.etom.cost import summarize_costs, task_variable_cost, usd
from hk_equity_mcp.etom.embeddings import (
    CatalogIndexError,
    HashEmbeddingClient,
    IncompleteCatalogIndexError,
    ModelMismatchError,
    OpenAICompatibleEmbeddingClient,
    build_catalog_index,
    catalog_index_status,
    load_complete_catalog_index,
    scrub_secret,
)
from hk_equity_mcp.etom.evaluator import evaluate_prediction
from hk_equity_mcp.etom.llm import FailClosedChatClient, ScriptedChatClient
from hk_equity_mcp.etom.loader import load_official_registry, load_official_tasks
from hk_equity_mcp.etom.prompts import ETOM_ROUTING_SYSTEM_PROMPT, prompt_hash
from hk_equity_mcp.etom.sampler import assert_lock_matches, select_locked_tasks
from hk_equity_mcp.etom.source import file_sha256, verify_source
from hk_equity_mcp.experiment.runner import git_branch, git_commit, git_dirty

CSV_FIELDS = [
    "task_id",
    "level",
    "query_id",
    "status",
    "exact_match",
    "precision",
    "recall",
    "f1",
    "llm_calls",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "query_embedding_calls",
    "query_embedding_tokens",
    "latency_ms",
    "http_retries",
    "parse_repairs",
    "timeout",
    "error",
    "llm_cost_usd",
    "query_embedding_cost_usd",
    "variable_cost_usd",
    "real_tool_executions",
    "predicted_tool_count",
]


class PilotLimitError(RuntimeError):
    pass


def assert_pilot_task_count(max_tasks: int) -> None:
    if int(max_tasks) != PILOT_MAX_TASKS:
        raise PilotLimitError(
            f"This pilot accepts max_tasks=10 only. Refusing max_tasks={max_tasks}."
        )


def _inside(root: Path, path: Path) -> Path:
    resolved = path.resolve()
    resolved.relative_to(root.resolve())
    return resolved


def _host(url: Optional[str]) -> Optional[str]:
    if not url:
        return None
    return urlparse(url).netloc or None


def _read_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _assert_trace_has_no_execution(trace: List[dict]) -> None:
    for event in trace:
        for item in event.get("results") or []:
            result = item.get("result") or {}
            if result.get("executed") is True or result.get("business_result") not in (None,):
                raise RuntimeError("ETOM trace contains a real tool result.")


def _latest_by_task(rows: List[dict]) -> Dict[str, dict]:
    latest = {}
    for row in rows:
        latest[row["task_id"]] = row
    return latest


class EtomPilotRunner:
    def __init__(
        self,
        root: Path,
        output_dir: Path,
        *,
        mode: str,
        resume: bool = False,
        allow_dirty: bool = False,
        env_file: Optional[str] = None,
        embedding_env_file: Optional[str] = None,
        max_tasks: int = PILOT_MAX_TASKS,
        config_path: Optional[Path] = None,
        build_index_only: bool = False,
        require_complete_index: bool = False,
        index_status_only: bool = False,
    ):
        if mode not in {"dry-run", "mock", "live"}:
            raise PilotLimitError(f"Unknown pilot mode {mode}.")
        if build_index_only and index_status_only:
            raise PilotLimitError("Choose only one of build-index-only or index-status.")
        if (build_index_only or index_status_only) and mode == "dry-run":
            raise PilotLimitError("Index commands cannot combine with dry-run.")
        assert_pilot_task_count(max_tasks)
        self.root = root.resolve()
        self.output_dir = _inside(self.root, output_dir)
        self.mode = mode
        self.resume = resume
        self.allow_dirty = allow_dirty
        self.env_file = env_file
        self.embedding_env_file = embedding_env_file
        self.max_tasks = PILOT_MAX_TASKS
        self.config_path = config_path or (self.root / "configs" / "etom_external_pilot_v1.yaml")
        self.config = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        self.build_index_only = build_index_only
        self.require_complete_index = require_complete_index or mode == "live"
        self.index_status_only = index_status_only
        self._stop = False
        self._cache_dir = self.root / ".cache" / "etom_embeddings"

    def execute(self) -> Dict[str, Any]:
        self._install_signal_handler()
        source = verify_source(self.root)
        if self.index_status_only:
            return self._index_status(source)
        tasks = load_official_tasks(self.root, verify=False)
        lock = assert_lock_matches(self.root, tasks)
        selected = select_locked_tasks(tasks, lock)
        if len(selected) != PILOT_MAX_TASKS:
            raise PilotLimitError("Locked pilot does not contain exactly 10 tasks.")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._write_json("run.pid", {"pid": os.getpid()})
        if self.mode == "dry-run":
            report = self._preflight(source, selected, lock)
            self._write_json("preflight_report.json", report)
            return report
        if self.build_index_only:
            return self._build_index_phase(source)
        return self._agent_run_phase(source, selected)

    def _registry_bundle(self):
        registry_payload = load_official_registry(self.root, verify=False)
        specs, official = convert_registry(registry_payload)
        if len(specs) != 2375:
            raise PilotLimitError(f"Converted registry has {len(specs)} operations, expected 2375.")
        registry_sha = file_sha256(self.root / "data" / "etom" / "raw" / "data" / "mcp_registry.json")
        return specs, official, registry_sha

    def _index_status(self, source) -> Dict[str, Any]:
        specs, _official, registry_sha = self._registry_bundle()
        if self.mode == "live":
            model_name = EXPECTED_EMBEDDING_MODEL
        else:
            model_name = "mock-hash-embedding-v1"
        status = catalog_index_status(
            registry=specs,
            model_name=model_name,
            registry_sha256=registry_sha,
            cache_dir=self._cache_dir,
        )
        status.update(
            {
                "mode": "index-status",
                "fidelity": FIDELITY_NAME,
                "network_calls": 0,
                "etom_commit": source["pin"]["commit_sha"],
                "expected_total": 2375,
            }
        )
        print(
            "index-status model={model} completed={completed}/{total} "
            "final_complete={final} partial={partial}".format(
                model=status["model"],
                completed=status["completed"],
                total=status["total"],
                final=status["final_complete"],
                partial=status["partial_completed_count"],
            ),
            flush=True,
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._write_json("index_status.json", status)
        return status

    def _build_index_phase(self, source) -> Dict[str, Any]:
        """Phase A: embedding provider only. Never instantiate or call DeepSeek."""
        if self.mode == "live":
            dirty = git_dirty(self.root)
            if dirty and not self.allow_dirty:
                raise PilotLimitError(
                    "Live build-index refused because the worktree is dirty. "
                    "Commit first, or pass --allow-dirty."
                )
            embedding = load_embedding_profile(self.root, self.embedding_env_file)
            self._assert_embedding_model()
            embedder = OpenAICompatibleEmbeddingClient(
                model_name=EXPECTED_EMBEDDING_MODEL,
                timeout_sec=float(self.config["limits"]["request_timeout_sec"]),
                max_retries=int(self.config["limits"]["max_http_retries"]),
                batch_size=int(self.config["embedding"]["batch_size"]),
            )
        else:
            embedding = {
                "model": "mock-hash-embedding-v1",
                "profile_file": None,
                "base_url": None,
            }
            embedder = HashEmbeddingClient(
                batch_size=int(self.config["embedding"]["batch_size"]),
            )
        specs, _official, registry_sha = self._registry_bundle()
        print("build-index start (embedding only; no generation LLM)", flush=True)
        index = build_catalog_index(
            registry=specs,
            embedder=embedder,
            registry_sha256=registry_sha,
            cache_dir=self._cache_dir,
        )
        report = {
            "mode": "build-index",
            "fidelity": FIDELITY_NAME,
            "generation_llm_instantiated": False,
            "generation_llm_calls": 0,
            "embedding_model": embedder.model_name,
            "embedding_base_url_host": _host(embedding.get("base_url") or os.environ.get("EMBEDDING_BASE_URL")),
            "registry_sha256": registry_sha,
            "completed_count": len(index.vectors),
            "total_count": len(specs),
            "final_complete": len(index.vectors) == len(specs),
            "cache_reused": index.cache_reused,
            "cache_path": index.cache_path,
            "build_usage": index.build_usage.to_dict(),
            "etom_commit": source["pin"]["commit_sha"],
            "branch": git_branch(self.root),
            "git_commit": git_commit(self.root),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._write_json("build_index_report.json", report)
        print(
            "build-index done completed={completed}/{total} cache_reused={reused}".format(
                completed=report["completed_count"],
                total=report["total_count"],
                reused=report["cache_reused"],
            ),
            flush=True,
        )
        return report

    def _agent_run_phase(self, source, selected) -> Dict[str, Any]:
        """Phase B: require complete final index, then probe generation, then 10 tasks."""
        if self.mode == "live":
            dirty = git_dirty(self.root)
            if dirty and not self.allow_dirty:
                raise PilotLimitError(
                    "Live pilot refused because the worktree is dirty. "
                    "Commit first, or pass --allow-dirty."
                )
            generation = load_generation_profile(self.root, self.env_file)
            embedding = load_embedding_profile(self.root, self.embedding_env_file)
            self._assert_env_models()
            model_name = EXPECTED_EMBEDDING_MODEL
            specs, official, registry_sha = self._registry_bundle()
            print("require complete catalog index", flush=True)
            index = load_complete_catalog_index(
                registry=specs,
                model_name=model_name,
                registry_sha256=registry_sha,
                cache_dir=self._cache_dir,
            )
            print(
                f"catalog index ready {len(index.vectors)}/{len(specs)}",
                flush=True,
            )
            print("verification probe start (generation only)", flush=True)
            llm = FailClosedChatClient(
                timeout_sec=float(self.config["limits"]["request_timeout_sec"]),
                max_retries=int(self.config["limits"]["max_http_retries"]),
                json_mode=bool(self.config["llm"]["json_mode"]),
            )
            embedder = OpenAICompatibleEmbeddingClient(
                model_name=EXPECTED_EMBEDDING_MODEL,
                timeout_sec=float(self.config["limits"]["request_timeout_sec"]),
                max_retries=int(self.config["limits"]["max_http_retries"]),
                batch_size=int(self.config["embedding"]["batch_size"]),
            )
            probe = self._probe_generation(llm)
        else:
            generation = {
                "provider": "mock",
                "model": "mock-llm",
                "profile_file": None,
                "base_url": None,
            }
            embedding = {"model": "mock-hash-embedding-v1", "profile_file": None, "base_url": None}
            llm = ScriptedChatClient()
            embedder = HashEmbeddingClient(
                batch_size=int(self.config["embedding"]["batch_size"]),
            )
            specs, official, registry_sha = self._registry_bundle()
            if self.require_complete_index:
                index = load_complete_catalog_index(
                    registry=specs,
                    model_name=embedder.model_name,
                    registry_sha256=registry_sha,
                    cache_dir=self._cache_dir,
                )
            else:
                index = build_catalog_index(
                    registry=specs,
                    embedder=embedder,
                    registry_sha256=registry_sha,
                    cache_dir=self._cache_dir,
                )
            probe = {
                "llm_calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "embedding_calls": 0,
                "embedding_tokens": 0,
                "usd": 0.0,
                "skipped": True,
            }
        resolver = EtomRoutingResolver(
            specs,
            official,
            index,
            embedder,
            lexical_weight=float(self.config["embedding"]["lexical_weight"]),
            top_k_default=int(self.config["embedding"]["top_k_default"]),
            top_k_max=int(self.config["embedding"]["top_k_max"]),
        )
        agent = EtomRoutingAgent(
            llm,
            resolver,
            temperature=float(self.config["llm"]["temperature"]),
            max_tokens=int(self.config["llm"]["max_tokens"]),
            max_llm_turns=int(self.config["limits"]["max_llm_turns"]),
            max_search_reformulations=int(self.config["limits"]["max_search_reformulations"]),
            max_invocations=int(self.config["limits"]["max_invocations"]),
            timeout_sec=float(self.config["limits"]["timeout_sec"]),
            max_parse_repairs=int(self.config["limits"]["max_parse_repairs"]),
        )
        self._write_bundle(selected, source)
        predictions = self.output_dir / "predictions.jsonl"
        existing = _read_jsonl(predictions)
        latest = _latest_by_task(existing)
        run_status = "completed"
        try:
            for index_number, task in enumerate(selected, start=1):
                if index_number > PILOT_MAX_TASKS:
                    raise PilotLimitError("Refusing to execute task 11.")
                if self._stop:
                    run_status = "interrupted"
                    break
                prior = latest.get(task.task_id)
                if self.resume and prior and prior.get("status") == "completed":
                    print(f"resume skip {task.task_id}", flush=True)
                    continue
                print(
                    f"task {index_number}/10 {task.task_id} L{task.level} start",
                    flush=True,
                )
                row = self._run_one(agent, resolver, task)
                print(
                    "task {task_id} status={status} exact_match={exact} "
                    "llm_calls={calls} latency_ms={latency} error={error}".format(
                        task_id=row["task_id"],
                        status=row["status"],
                        exact=row["exact_match"],
                        calls=row["llm_calls"],
                        latency=row["latency_ms"],
                        error=row["error"],
                    ),
                    flush=True,
                )
                self._append_jsonl(predictions, row)
                existing.append(row)
                latest[task.task_id] = row
                self._flush_reports(
                    existing,
                    source,
                    selected,
                    generation,
                    embedding,
                    index,
                    probe,
                    run_status="running",
                )
        except (ModelMismatchError, IncompleteCatalogIndexError, CatalogIndexError):
            run_status = "fail_closed"
            self._flush_reports(
                existing,
                source,
                selected,
                generation,
                embedding,
                index,
                probe,
                run_status=run_status,
            )
            raise
        except KeyboardInterrupt:
            run_status = "interrupted"
            self._flush_reports(
                existing,
                source,
                selected,
                generation,
                embedding,
                index,
                probe,
                run_status=run_status,
            )
            raise
        else:
            if run_status != "interrupted":
                finished = [
                    latest[task.task_id]
                    for task in selected
                    if task.task_id in latest and latest[task.task_id].get("status") == "completed"
                ]
                run_status = "completed" if len(finished) == PILOT_MAX_TASKS else "partial"
        summary = self._flush_reports(
            existing,
            source,
            selected,
            generation,
            embedding,
            index,
            probe,
            run_status=run_status,
        )
        return summary

    def _run_one(self, agent, resolver, task) -> Dict[str, Any]:
        query_calls_before = resolver.index.query_usage.calls
        query_tokens_before = resolver.index.query_usage.tokens
        http_before = int(getattr(agent.llm, "retries", 0) or 0)
        try:
            result = agent.run(task.query, task.ground_truth_tools)
            error = result.get("error")
            status = result.get("status") or "failed"
        except ModelMismatchError:
            raise
        except Exception as exc:  # noqa: BLE001
            result = {
                "status": "failed",
                "error": type(exc).__name__,
                "timeout": False,
                "predicted_tools": [],
                "predicted_nodes": [],
                "dependency_edges": [],
                "llm_calls": 0,
                "tokens": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                "parse_repairs": 0,
                "real_tool_executions": resolver.real_tool_executions,
                "latency_ms": None,
                "final_answer": None,
                "trace": [],
            }
            error = scrub_secret(f"{type(exc).__name__}: {exc}")
            status = "failed"
        metrics = evaluate_prediction(
            task.level, task.ground_truth_tools, result.get("predicted_tools") or []
        )
        if any(node.get("business_result") not in (None, {}) for node in result.get("predicted_nodes") or []):
            raise RuntimeError("ETOM invoke recorded a business result.")
        if resolver.real_tool_executions != 0:
            raise RuntimeError("ETOM routing mode executed a real tool.")
        _assert_trace_has_no_execution(result.get("trace") or [])
        tokens = result.get("tokens") or {}
        pricing = self.config["pricing"]
        query_embedding_calls = resolver.index.query_usage.calls - query_calls_before
        query_embedding_tokens = resolver.index.query_usage.tokens - query_tokens_before
        row = {
            "task_id": task.task_id,
            "level": task.level,
            "query_id": task.query_id,
            "dataset_split": "cost_pilot",
            "exclude_from_final_etom_sample": True,
            "fidelity": FIDELITY_NAME,
            "status": status,
            "error": error,
            "timeout": bool(result.get("timeout")),
            "exact_match": metrics.get("exact_match"),
            "precision": metrics.get("precision"),
            "recall": metrics.get("recall"),
            "f1": metrics.get("f1"),
            "graph_f1_score": metrics.get("graph_f1_score"),
            "eval_status": metrics.get("status"),
            "llm_calls": int(result.get("llm_calls") or 0),
            "input_tokens": int(tokens.get("input_tokens") or 0),
            "output_tokens": int(tokens.get("output_tokens") or 0),
            "total_tokens": int(tokens.get("total_tokens") or 0),
            "query_embedding_calls": query_embedding_calls,
            "query_embedding_tokens": query_embedding_tokens,
            "latency_ms": result.get("latency_ms"),
            "http_retries": int(getattr(agent.llm, "retries", 0) or 0) - http_before,
            "parse_repairs": int(result.get("parse_repairs") or 0),
            "real_tool_executions": int(result.get("real_tool_executions") or 0),
            "predicted_tools": result.get("predicted_tools") or [],
            "predicted_nodes": result.get("predicted_nodes") or [],
            "dependency_edges": result.get("dependency_edges") or [],
            "predicted_tool_count": len(result.get("predicted_tools") or []),
            "final_answer": result.get("final_answer"),
            "trace": result.get("trace") or [],
            "executed_etom_mcp_tool": False,
        }
        for node in row["predicted_nodes"]:
            if node.get("business_result") not in (None,):
                raise RuntimeError("Predicted node contains a business result.")
        row["llm_cost_usd"] = round(
            usd(row["input_tokens"], float(pricing["llm_input_usd_per_million"]))
            + usd(row["output_tokens"], float(pricing["llm_output_usd_per_million"])),
            6,
        )
        row["query_embedding_cost_usd"] = round(
            usd(row["query_embedding_tokens"], float(pricing["embedding_usd_per_million"])),
            6,
        )
        row["variable_cost_usd"] = round(task_variable_cost(row, pricing), 6)
        return row

    def _probe_generation(self, llm) -> Dict[str, Any]:
        response = llm.complete(
            [{"role": "user", "content": 'Reply with JSON {"ok": true}'}],
            temperature=float(self.config["llm"]["temperature"]),
            max_tokens=32,
        )
        pricing = self.config["pricing"]
        llm_usd = usd(
            int(response["usage"]["input_tokens"]), float(pricing["llm_input_usd_per_million"])
        ) + usd(
            int(response["usage"]["output_tokens"]), float(pricing["llm_output_usd_per_million"])
        )
        return {
            "llm_calls": 1,
            "input_tokens": int(response["usage"]["input_tokens"]),
            "output_tokens": int(response["usage"]["output_tokens"]),
            "llm_model": response.get("model"),
            "model_echo_absent": response.get("model_echo_absent"),
            "embedding_calls": 0,
            "embedding_tokens": 0,
            "usd": round(llm_usd, 6),
            "skipped": False,
            "probe_kind": "generation_only",
        }

    def _preflight(self, source, selected, lock) -> Dict[str, Any]:
        generation = load_generation_profile(self.root, self.env_file)
        embedding = load_embedding_profile(self.root, self.embedding_env_file)
        self._assert_env_models()
        specs, _official, registry_sha = self._registry_bundle()
        status = catalog_index_status(
            registry=specs,
            model_name=EXPECTED_EMBEDDING_MODEL,
            registry_sha256=registry_sha,
            cache_dir=self._cache_dir,
        )
        return {
            "mode": "dry-run",
            "fidelity": FIDELITY_NAME,
            "network_calls": 0,
            "max_tasks": PILOT_MAX_TASKS,
            "task_ids": [task.task_id for task in selected],
            "levels": {f"L{level}": 2 for level in range(1, 6)},
            "generation_model": generation["model"],
            "generation_provider": generation["provider"],
            "generation_base_url_host": _host(generation.get("base_url")),
            "embedding_model": embedding["model"],
            "embedding_base_url_host": _host(embedding.get("base_url")),
            "etom_commit": source["pin"]["commit_sha"],
            "dataset_hashes_ok": True,
            "lock_matches_sampler": True,
            "exclude_from_final_etom_sample": True,
            "seed": lock["seed"],
            "real_tool_execution": False,
            "catalog_index": {
                "completed": status["completed"],
                "total": status["total"],
                "final_complete": status["final_complete"],
                "partial_completed_count": status["partial_completed_count"],
            },
            "parameters": self._parameters(),
        }

    def _assert_embedding_model(self) -> None:
        embedding_model = os.environ.get("EMBEDDING_MODEL")
        if embedding_model != EXPECTED_EMBEDDING_MODEL:
            raise ModelMismatchError(
                f"Embedding model mismatch: expected {EXPECTED_EMBEDDING_MODEL}."
            )
        if str(self.config["embedding"]["model"]) != EXPECTED_EMBEDDING_MODEL:
            raise ModelMismatchError("Pilot config embedding model drifted.")

    def _assert_env_models(self) -> None:
        provider = os.environ.get("LLM_PROVIDER")
        model = os.environ.get("LLM_MODEL")
        embedding_model = os.environ.get("EMBEDDING_MODEL")
        if (
            provider != EXPECTED_LLM_PROVIDER
            or model != EXPECTED_LLM_MODEL
            or embedding_model != EXPECTED_EMBEDDING_MODEL
        ):
            raise ModelMismatchError(
                "Provider/model mismatch: expected "
                f"{EXPECTED_LLM_PROVIDER}/{EXPECTED_LLM_MODEL} and "
                f"{EXPECTED_EMBEDDING_MODEL}."
            )
        if str(self.config["llm"]["model"]) != EXPECTED_LLM_MODEL:
            raise ModelMismatchError("Pilot config generation model drifted.")
        if str(self.config["embedding"]["model"]) != EXPECTED_EMBEDDING_MODEL:
            raise ModelMismatchError("Pilot config embedding model drifted.")

    def _write_bundle(self, selected, source) -> None:
        payload = {
            "fidelity": FIDELITY_NAME,
            "dataset_split": "cost_pilot",
            "exclude_from_final_etom_sample": True,
            "etom_commit": source["pin"]["commit_sha"],
            "model_input_note": (
                "Gold graphs in this file are for scoring only. "
                "The agent receives the query text and not this file."
            ),
            "tasks": [
                {
                    "task_id": task.task_id,
                    "level": task.level,
                    "query_id": task.query_id,
                    "query": task.query,
                    "ground_truth_tools": task.ground_truth_tools,
                    "dataset_split": "cost_pilot",
                    "exclude_from_final_etom_sample": True,
                    "server_names": sorted(task.server_names),
                    "single_server_sequential": task.single_server_sequential,
                    "cross_server_dependency": task.cross_server_dependency,
                    "infeasible": task.infeasible,
                }
                for task in selected
            ],
        }
        self._write_json("etom_cost_pilot_10_v1.json", payload)

    def _flush_reports(
        self,
        rows,
        source,
        selected,
        generation,
        embedding,
        index,
        probe,
        *,
        run_status: str,
    ) -> Dict[str, Any]:
        latest = _latest_by_task(rows)
        ordered = [latest[task.task_id] for task in selected if task.task_id in latest]
        catalog = {
            "calls": index.build_usage.calls,
            "retries": index.build_usage.retries,
            "tokens": index.build_usage.tokens,
            "token_source": index.build_usage.token_source,
            "cache_reused": index.cache_reused,
        }
        summary = summarize_costs(
            ordered, pricing=self.config["pricing"], catalog=catalog, probe=probe
        )
        summary["run_status"] = run_status
        summary["fidelity"] = FIDELITY_NAME
        summary["real_mcp_tool_executed"] = summary["real_tool_executions"] != 0
        self._write_json("cost_summary.json", summary)
        self._write_csv(ordered)
        metadata = {
            "fidelity": FIDELITY_NAME,
            "mode": self.mode,
            "run_status": run_status,
            "resume": self.resume,
            "require_complete_index": self.require_complete_index,
            "max_tasks": PILOT_MAX_TASKS,
            "task_ids": [task.task_id for task in selected],
            "completed_task_ids": [
                row["task_id"] for row in ordered if row.get("status") == "completed"
            ],
            "etom_repository": source["pin"]["repository_url"],
            "etom_commit": source["pin"]["commit_sha"],
            "etom_license": source["pin"]["license"],
            "dataset_sha256": source["sha256"],
            "registry_tool_count": 2375,
            "generation_provider": generation.get("provider"),
            "generation_model": generation.get("model"),
            "generation_base_url_host": _host(generation.get("base_url") or os.environ.get("LLM_BASE_URL")),
            "embedding_model": embedding.get("model"),
            "embedding_base_url_host": _host(
                embedding.get("base_url") or os.environ.get("EMBEDDING_BASE_URL")
            ),
            "parameters": self._parameters(),
            "system_prompt_sha256": prompt_hash(ETOM_ROUTING_SYSTEM_PROMPT),
            "catalog_cache_reused": index.cache_reused,
            "catalog_cache_path": index.cache_path,
            "branch": git_branch(self.root),
            "git_commit": git_commit(self.root),
            "git_dirty": git_dirty(self.root),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "real_tool_execution": False,
            "output_dir": str(self.output_dir),
        }
        self._write_json("run_metadata.json", metadata)
        return summary

    def _parameters(self) -> Dict[str, Any]:
        return {
            "temperature": self.config["llm"]["temperature"],
            "max_tokens": self.config["llm"]["max_tokens"],
            "top_p": self.config["llm"].get("top_p"),
            "generation_top_k": self.config["llm"].get("top_k"),
            "json_mode": self.config["llm"]["json_mode"],
            "retrieval_top_k": self.config["embedding"]["top_k_default"],
            "lexical_weight": self.config["embedding"]["lexical_weight"],
            "max_llm_turns": self.config["limits"]["max_llm_turns"],
            "max_search_reformulations": self.config["limits"]["max_search_reformulations"],
            "max_invocations": self.config["limits"]["max_invocations"],
            "timeout_sec": self.config["limits"]["timeout_sec"],
            "max_parse_repairs": self.config["limits"]["max_parse_repairs"],
            "max_http_retries": self.config["limits"]["max_http_retries"],
            "embedding_batch_size": self.config["embedding"]["batch_size"],
            "seed": self.config["sampler"]["seed"],
        }

    def _write_csv(self, rows: List[dict]) -> None:
        path = self.output_dir / "per_task_metrics.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
            handle.flush()
            os.fsync(handle.fileno())

    def _append_jsonl(self, path: Path, row: dict) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def _write_json(self, name: str, payload: dict) -> None:
        path = self.output_dir / name
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        temporary.replace(path)

    def _install_signal_handler(self) -> None:
        def _handle(signum, _frame):
            self._stop = True
            raise KeyboardInterrupt

        signal.signal(signal.SIGINT, _handle)
        signal.signal(signal.SIGTERM, _handle)
