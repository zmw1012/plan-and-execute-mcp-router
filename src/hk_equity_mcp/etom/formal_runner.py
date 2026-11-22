"""Operation-Visible ETOM-100 formal runner.

Reads the frozen formal_100_v1.lock.json only. Never resamples.
Requires a complete catalog embedding index for live runs.
"""

from __future__ import annotations

import csv
import hashlib
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
    FORMAL_100_MAX_TASKS,
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
from hk_equity_mcp.etom.formal_metrics import (
    dependency_accuracy,
    error_summary,
    grouped_results,
    retrieval_analysis,
)
from hk_equity_mcp.etom.formal_sampler import (
    assert_formal_100_lock_matches,
    load_formal_100_lock,
    select_formal_locked_tasks,
)
from hk_equity_mcp.etom.llm import FailClosedChatClient, ScriptedChatClient
from hk_equity_mcp.etom.loader import load_official_registry, load_official_tasks
from hk_equity_mcp.etom.prompts import ETOM_ROUTING_SYSTEM_PROMPT, prompt_hash
from hk_equity_mcp.etom.source import file_sha256, verify_source
from hk_equity_mcp.experiment.runner import git_branch, git_commit, git_dirty

CSV_FIELDS = [
    "task_id",
    "level",
    "query_id",
    "selected_order",
    "status",
    "exact_match",
    "precision",
    "recall",
    "f1",
    "dependency_edge_accuracy",
    "ordered_dependency_accuracy",
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


class FormalPilotError(RuntimeError):
    pass


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


def _latest_by_task(rows: List[dict]) -> Dict[str, dict]:
    latest = {}
    for row in rows:
        latest[row["task_id"]] = row
    return latest


def _assert_trace_has_no_execution(trace: List[dict]) -> None:
    for event in trace:
        for item in event.get("results") or []:
            result = item.get("result") or {}
            if result.get("executed") is True or result.get("business_result") not in (None,):
                raise RuntimeError("ETOM trace contains a real tool result.")


def resume_identity(metadata: Dict[str, Any]) -> str:
    """Identity used for resume continuity. Intentionally omits max_tasks."""
    keys = [
        "source_commit",
        "source_pin_sha256",
        "formal_sample_sha256",
        "ordered_task_ids_sha256",
        "condition",
        "generation_model",
        "embedding_model",
        "system_prompt_sha256",
        "registry_sha256",
        "retrieval_top_k",
        "lexical_weight",
        "limits",
    ]
    payload = {key: metadata.get(key) for key in keys}
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class EtomFormal100Runner:
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
        max_tasks: int = FORMAL_100_MAX_TASKS,
        config_path: Optional[Path] = None,
        require_complete_index: bool = False,
        index_status_only: bool = False,
    ):
        if mode not in {"dry-run", "mock", "live"}:
            raise FormalPilotError(f"Unknown formal mode {mode}.")
        if int(max_tasks) < 1 or int(max_tasks) > FORMAL_100_MAX_TASKS:
            raise FormalPilotError(
                f"Formal runner accepts max_tasks in 1..{FORMAL_100_MAX_TASKS}."
            )
        self.root = root.resolve()
        self.output_dir = _inside(self.root, output_dir)
        self.mode = mode
        self.resume = resume
        self.allow_dirty = allow_dirty
        self.env_file = env_file
        self.embedding_env_file = embedding_env_file
        self.max_tasks = int(max_tasks)
        self.config_path = config_path or (
            self.root / "configs" / "etom_external_formal_100_v1.yaml"
        )
        self.config = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        self.require_complete_index = require_complete_index or mode == "live"
        self.index_status_only = index_status_only
        self._stop = False
        self._cache_dir = self.root / ".cache" / "etom_embeddings"

    def execute(self) -> Dict[str, Any]:
        self._install_signal_handler()
        source = verify_source(self.root)
        lock = assert_formal_100_lock_matches(
            self.root, load_official_tasks(self.root, verify=False)
        )
        if self.index_status_only:
            return self._index_status(source, lock)
        tasks = load_official_tasks(self.root, verify=False)
        selected = select_formal_locked_tasks(tasks, lock, max_tasks=self.max_tasks)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._write_json("run.pid", {"pid": os.getpid(), "max_tasks": self.max_tasks})
        if self.mode == "dry-run":
            report = self._preflight(source, selected, lock)
            self._write_json("preflight_report.json", report)
            return report
        return self._agent_run(source, selected, lock)

    def _registry_bundle(self):
        registry_payload = load_official_registry(self.root, verify=False)
        specs, official = convert_registry(registry_payload)
        if len(specs) != 2375:
            raise FormalPilotError(
                f"Converted registry has {len(specs)} operations, expected 2375."
            )
        registry_sha = file_sha256(
            self.root / "data" / "etom" / "raw" / "data" / "mcp_registry.json"
        )
        return specs, official, registry_sha

    def _index_status(self, source, lock) -> Dict[str, Any]:
        specs, _official, registry_sha = self._registry_bundle()
        model_name = (
            EXPECTED_EMBEDDING_MODEL if self.mode == "live" else "mock-hash-embedding-v1"
        )
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
                "dataset_split": "external_formal",
                "formal_sample_sha256": lock["sample_sha256"],
                "network_calls": 0,
                "etom_commit": source["pin"]["commit_sha"],
            }
        )
        print(
            "index-status model={model} completed={completed}/{total} final_complete={final}".format(
                model=status["model"],
                completed=status["completed"],
                total=status["total"],
                final=status["final_complete"],
            ),
            flush=True,
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._write_json("index_status.json", status)
        return status

    def _agent_run(self, source, selected, lock) -> Dict[str, Any]:
        if self.mode == "live":
            if git_dirty(self.root) and not self.allow_dirty:
                raise FormalPilotError(
                    "Live formal run refused because the worktree is dirty."
                )
            generation = load_generation_profile(self.root, self.env_file)
            embedding = load_embedding_profile(self.root, self.embedding_env_file)
            self._assert_env_models()
            specs, official, registry_sha = self._registry_bundle()
            print("require complete catalog index", flush=True)
            index = load_complete_catalog_index(
                registry=specs,
                model_name=EXPECTED_EMBEDDING_MODEL,
                registry_sha256=registry_sha,
                cache_dir=self._cache_dir,
            )
            print(f"catalog index ready {len(index.vectors)}/{len(specs)}", flush=True)
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
            embedding = {
                "model": "mock-hash-embedding-v1",
                "profile_file": None,
                "base_url": None,
            }
            llm = ScriptedChatClient()
            embedder = HashEmbeddingClient(
                batch_size=int(self.config["embedding"]["batch_size"])
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
                "probe_kind": "mock_skipped",
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
        order_map = {
            row["task_id"]: row["selected_order"] for row in lock.get("tasks") or []
        }
        predictions = self.output_dir / "predictions.jsonl"
        existing = _read_jsonl(predictions)
        # Never mix cost-pilot predictions into formal outputs.
        existing = [
            row
            for row in existing
            if row.get("dataset_split") == "external_formal"
            or row.get("task_id") in {task.task_id for task in selected}
        ]
        latest = _latest_by_task(existing)
        identity_meta = self._identity_metadata(
            source, lock, generation, embedding, registry_sha, index
        )
        identity_hash = resume_identity(identity_meta)
        prior_meta_path = self.output_dir / "run_metadata.json"
        if self.resume and prior_meta_path.exists():
            prior = json.loads(prior_meta_path.read_text(encoding="utf-8"))
            prior_hash = prior.get("resume_identity_sha256")
            if prior_hash and prior_hash != identity_hash:
                raise FormalPilotError(
                    "Resume identity mismatch. Refusing to continue with a different "
                    "formal sample / model / prompt / registry configuration."
                )
        run_status = "completed"
        try:
            for index_number, task in enumerate(selected, start=1):
                if self._stop:
                    run_status = "interrupted"
                    break
                prior = latest.get(task.task_id)
                if self.resume and prior and prior.get("status") == "completed":
                    print(f"resume skip {task.task_id}", flush=True)
                    continue
                print(
                    f"task {index_number}/{len(selected)} {task.task_id} "
                    f"L{task.level} start",
                    flush=True,
                )
                row = self._run_one(
                    agent,
                    resolver,
                    task,
                    selected_order=int(order_map.get(task.task_id) or index_number),
                )
                print(
                    "task {task_id} status={status} exact_match={exact} "
                    "llm_calls={calls} error={error}".format(
                        task_id=row["task_id"],
                        status=row["status"],
                        exact=row["exact_match"],
                        calls=row["llm_calls"],
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
                    lock,
                    generation,
                    embedding,
                    index,
                    probe,
                    registry_sha,
                    identity_hash,
                    run_status="running",
                )
        except (ModelMismatchError, IncompleteCatalogIndexError, CatalogIndexError):
            run_status = "fail_closed"
            self._flush_reports(
                existing,
                source,
                selected,
                lock,
                generation,
                embedding,
                index,
                probe,
                registry_sha,
                identity_hash,
                run_status=run_status,
            )
            raise
        except KeyboardInterrupt:
            run_status = "interrupted"
            self._flush_reports(
                existing,
                source,
                selected,
                lock,
                generation,
                embedding,
                index,
                probe,
                registry_sha,
                identity_hash,
                run_status=run_status,
            )
            raise
        else:
            if run_status != "interrupted":
                finished = [
                    latest[task.task_id]
                    for task in selected
                    if task.task_id in latest
                    and latest[task.task_id].get("status") == "completed"
                ]
                run_status = (
                    "completed" if len(finished) == len(selected) else "partial"
                )
        return self._flush_reports(
            existing,
            source,
            selected,
            lock,
            generation,
            embedding,
            index,
            probe,
            registry_sha,
            identity_hash,
            run_status=run_status,
        )

    def _run_one(self, agent, resolver, task, *, selected_order: int) -> Dict[str, Any]:
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
        dep = dependency_accuracy(
            level=task.level,
            ground_truth_tools=task.ground_truth_tools,
            predicted_nodes=result.get("predicted_nodes") or [],
        )
        if any(
            node.get("business_result") not in (None, {})
            for node in result.get("predicted_nodes") or []
        ):
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
            "selected_order": selected_order,
            "dataset_split": "external_formal",
            "benchmark_role": "external_validation",
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
            "dependency_edge_accuracy": dep["dependency_edge_accuracy"],
            "ordered_dependency_accuracy": dep["ordered_dependency_accuracy"],
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
            int(response["usage"]["input_tokens"]),
            float(pricing["llm_input_usd_per_million"]),
        ) + usd(
            int(response["usage"]["output_tokens"]),
            float(pricing["llm_output_usd_per_million"]),
        )
        return {
            "llm_calls": 1,
            "input_tokens": int(response["usage"]["input_tokens"]),
            "output_tokens": int(response["usage"]["output_tokens"]),
            "llm_model": response.get("model"),
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
            "dataset_split": "external_formal",
            "benchmark_role": "external_validation",
            "network_calls": 0,
            "max_tasks": self.max_tasks,
            "locked_sample_size": lock["sample_size"],
            "task_ids": [task.task_id for task in selected],
            "formal_sample_sha256": lock["sample_sha256"],
            "ordered_task_ids_sha256": lock["ordered_task_ids_sha256"],
            "generation_model": generation["model"],
            "generation_provider": generation["provider"],
            "generation_base_url_host": _host(generation.get("base_url")),
            "embedding_model": embedding["model"],
            "embedding_base_url_host": _host(embedding.get("base_url")),
            "etom_commit": source["pin"]["commit_sha"],
            "catalog_index": {
                "completed": status["completed"],
                "total": status["total"],
                "final_complete": status["final_complete"],
            },
            "require_complete_index": True,
            "real_tool_execution": False,
            "parameters": self._parameters(),
        }

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
            raise ModelMismatchError("Formal config generation model drifted.")
        if str(self.config["embedding"]["model"]) != EXPECTED_EMBEDDING_MODEL:
            raise ModelMismatchError("Formal config embedding model drifted.")

    def _parameters(self) -> Dict[str, Any]:
        return {
            "temperature": self.config["llm"]["temperature"],
            "max_tokens": self.config["llm"]["max_tokens"],
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

    def _identity_metadata(
        self, source, lock, generation, embedding, registry_sha, index
    ) -> Dict[str, Any]:
        return {
            "source_commit": source["pin"]["commit_sha"],
            "source_pin_sha256": file_sha256(
                self.root / "benchmarks" / "etom" / "etom_source_pin_v1.json"
            ),
            "formal_sample_sha256": lock["sample_sha256"],
            "ordered_task_ids_sha256": lock["ordered_task_ids_sha256"],
            "condition": self.config.get("condition"),
            "generation_model": generation.get("model"),
            "embedding_model": embedding.get("model"),
            "system_prompt_sha256": prompt_hash(ETOM_ROUTING_SYSTEM_PROMPT),
            "registry_sha256": registry_sha,
            "retrieval_top_k": self.config["embedding"]["top_k_default"],
            "lexical_weight": self.config["embedding"]["lexical_weight"],
            "limits": {
                "max_llm_turns": self.config["limits"]["max_llm_turns"],
                "max_search_reformulations": self.config["limits"][
                    "max_search_reformulations"
                ],
                "max_invocations": self.config["limits"]["max_invocations"],
                "timeout_sec": self.config["limits"]["timeout_sec"],
                "max_parse_repairs": self.config["limits"]["max_parse_repairs"],
                "max_http_retries": self.config["limits"]["max_http_retries"],
            },
            "catalog_cache_path": getattr(index, "cache_path", None),
        }

    def _flush_reports(
        self,
        rows,
        source,
        selected,
        lock,
        generation,
        embedding,
        index,
        probe,
        registry_sha,
        identity_hash,
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
        summary["dataset_split"] = "external_formal"
        summary["real_mcp_tool_executed"] = False
        self._write_json("cost_summary.json", summary)
        self._write_csv(ordered)
        grouped = grouped_results(ordered)
        self._write_json("grouped_results.json", grouped)
        errors = error_summary(ordered)
        self._write_json("error_summary.json", errors)
        retrieval = retrieval_analysis(ordered)
        self._write_json("retrieval_analysis.json", retrieval)
        identity = self._identity_metadata(
            source, lock, generation, embedding, registry_sha, index
        )
        metadata = {
            **identity,
            "fidelity": FIDELITY_NAME,
            "mode": self.mode,
            "run_status": run_status,
            "resume": self.resume,
            "max_tasks": self.max_tasks,
            "max_tasks_excluded_from_resume_identity": True,
            "resume_identity_sha256": identity_hash,
            "require_complete_index": self.require_complete_index,
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
            "generation_base_url_host": _host(
                generation.get("base_url") or os.environ.get("LLM_BASE_URL")
            ),
            "embedding_base_url_host": _host(
                embedding.get("base_url") or os.environ.get("EMBEDDING_BASE_URL")
            ),
            "parameters": self._parameters(),
            "catalog_cache_reused": index.cache_reused,
            "branch": git_branch(self.root),
            "git_commit": git_commit(self.root),
            "git_dirty": git_dirty(self.root),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "real_tool_execution": False,
            "output_dir": str(self.output_dir),
            "pilot_predictions_mixed": False,
        }
        self._write_json("run_metadata.json", metadata)
        return summary

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
