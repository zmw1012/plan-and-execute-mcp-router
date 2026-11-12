"""Independent ETOM PE-Fin RePlan runner (mock / dry-run / future live gate)."""

from __future__ import annotations

import json
import os
import signal
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from hk_equity_mcp.etom.embeddings import HashEmbeddingClient
from hk_equity_mcp.etom.evaluator import evaluate_prediction
from hk_equity_mcp.experiment.runner import git_branch, git_commit, git_dirty
from hk_equity_mcp.operation_resolver.schemas import OperationSpec, ParameterSpec
from hk_equity_mcp.pe_fin_replan_etom.capability_profile import (
    capability_profile_hash,
    load_capability_profile,
)
from hk_equity_mcp.pe_fin_replan_etom.dataset import assert_frozen_etom100_identity
from hk_equity_mcp.pe_fin_replan_etom import EMBEDDING_CACHE_NAMESPACE
from hk_equity_mcp.pe_fin_replan_etom.identity import (
    DEFAULT_BUDGETS,
    FROZEN_REGISTRY_SHA256,
    FROZEN_CATALOG_TOTAL_TOOLS,
    evaluator_identity,
    protocol_identity,
    resume_identity_hash,
)
from hk_equity_mcp.pe_fin_replan_etom.operation_documents import (
    documents_hash,
)
from hk_equity_mcp.pe_fin_replan_etom.operation_embedding_index import (
    LiveEmbeddingGatedError,
    assert_embedder_allowed,
    build_index_identity,
    build_index_with_embedder,
    index_status,
    namespace_inventory,
    resolve_embedding_cache_dir,
)


class EtomPeFinReplanRunnerError(RuntimeError):
    pass


def _fsync_write_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with tmp.open("rb") as handle:
        os.fsync(handle.fileno())
    tmp.replace(path)


def _mini_registry_from_official(official: Dict[str, Dict[str, Any]]) -> Dict[str, OperationSpec]:
    registry: Dict[str, OperationSpec] = {}
    for oid, meta in official.items():
        schema = meta.get("input_schema") or {}
        props = schema.get("properties") or {}
        required = set(schema.get("required") or [])
        params = []
        for name, pmeta in props.items():
            pmeta = pmeta or {}
            params.append(
                ParameterSpec(
                    name=str(name),
                    type=str(pmeta.get("type") or "string"),
                    required=name in required,
                    description=str(pmeta.get("description") or ""),
                    enum=pmeta.get("enum"),
                )
            )
        registry[oid] = OperationSpec(
            operation_id=oid,
            service_id=str(meta.get("server_name") or ""),
            name=str(meta.get("tool_name") or ""),
            summary=f"{meta.get('tool_name')} on {meta.get('server_name')}",
            description=str(meta.get("description") or ""),
            action_terms=[str(meta.get("tool_name") or "").replace("_", " ")],
            parameters=params,
            underlying_tool=str(meta.get("tool_name") or ""),
        )
    return registry


class UnifiedPeFinReplanEtomRunner:
    def __init__(
        self,
        root: Path,
        output_dir: Path,
        *,
        mode: str = "dry-run",
        formal100: bool = True,
        resume: bool = False,
        env_file: Optional[str] = None,
        embedding_env_file: Optional[str] = None,
        embedding_cache_dir: Optional[str] = None,
        require_complete_index: bool = False,
        index_status_only: bool = False,
        build_index_only: bool = False,
        mock_embeddings: bool = False,
        max_tasks: int = 100,
        config_path: Optional[Path] = None,
    ):
        if mode not in {"dry-run", "mock", "live"}:
            raise EtomPeFinReplanRunnerError(f"unknown mode {mode}")
        self.root = Path(root).resolve()
        self.output_dir = Path(output_dir)
        self.mode = mode
        self.formal100 = formal100
        self.resume = resume
        self.env_file = env_file
        self.embedding_env_file = embedding_env_file
        self.require_complete_index = require_complete_index
        self.index_status_only = index_status_only
        self.build_index_only = build_index_only
        self.mock_embeddings = mock_embeddings or mode == "mock"
        self.max_tasks = int(max_tasks)
        self.config_path = config_path or (
            self.root / "configs" / "pe_fin_replan_etom_formal100_v1.yaml"
        )
        self.config = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        self.cache_dir = resolve_embedding_cache_dir(self.root, embedding_cache_dir)
        self._stop = False
        self.run_status = "running"

    def _install_signal_handler(self) -> None:
        def _handler(signum, frame):  # noqa: ANN001
            self._stop = True
            self.run_status = "aborted"

        signal.signal(signal.SIGINT, _handler)
        signal.signal(signal.SIGTERM, _handler)

    def execute(self) -> Dict[str, Any]:
        self._install_signal_handler()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        try:
            return self._execute_inner()
        except (KeyboardInterrupt, SystemExit):
            self.run_status = "aborted"
            self._finalize_metadata(extra={"abort_reason": "signal"})
            raise
        except Exception as exc:
            self.run_status = "aborted"
            self._finalize_metadata(
                extra={
                    "abort_reason": "unknown_runner_error",
                    "error": str(exc),
                    "traceback": traceback.format_exc()[-4000:],
                }
            )
            raise

    def _load_mini_official(self) -> Dict[str, Dict[str, Any]]:
        fixture = (
            self.root / "tests" / "fixtures" / "pe_fin_replan_etom" / "mini_official.json"
        )
        if fixture.exists():
            return json.loads(fixture.read_text(encoding="utf-8"))
        return {
            "Demo Server::echo": {
                "server_name": "Demo Server",
                "tool_name": "echo",
                "description": "echo text",
                "input_schema": {
                    "type": "object",
                    "required": ["query"],
                    "properties": {"query": {"type": "string"}},
                },
            }
        }

    def _execute_inner(self) -> Dict[str, Any]:
        identity = assert_frozen_etom100_identity(self.root)
        profile = load_capability_profile(self.root)
        profile_hash = capability_profile_hash(profile)
        inventory = namespace_inventory(self.root)

        if self.index_status_only:
            payload = {
                "mode": self.mode,
                "dataset_identity": identity,
                "namespace": EMBEDDING_CACHE_NAMESPACE,
                "cache_dir": str(self.cache_dir),
                "namespace_inventory": inventory,
                "live_pass_claimed": False,
            }
            _write_json(self.output_dir / "index_status.json", payload)
            self.run_status = "completed"
            return payload

        if self.build_index_only:
            if not self.mock_embeddings:
                raise LiveEmbeddingGatedError(
                    "Refusing live embedding API build in offline scaffolding. "
                    "Pass --mock-embeddings for deterministic hash index."
                )
            official = self._load_mini_official()
            registry = _mini_registry_from_official(official)
            embedder = HashEmbeddingClient(model_name="mock-hash-embedding-v1")
            assert_embedder_allowed(embedder, allow_live=False)
            identity_idx = build_index_identity(
                registry=registry,
                catalog_fingerprint=documents_hash(registry),
                embedding_model=embedder.model_name,
                embedding_provider_host="local_hash_mock",
            )
            built = build_index_with_embedder(
                registry=registry,
                identity=identity_idx,
                cache_dir=self.cache_dir,
                embedder=embedder,
                allow_live=False,
            )
            status = index_status(
                registry=registry, identity=identity_idx, cache_dir=self.cache_dir
            )
            payload = {
                "mode": "mock",
                "build_result": built,
                "index_status": status.to_dict(),
                "live_pass_claimed": False,
            }
            _write_json(self.output_dir / "build_index_result.json", payload)
            self.run_status = "completed"
            return payload

        protocol = protocol_identity()
        metadata = {
            "condition": protocol["condition"],
            "method_version": protocol["method_version"],
            "protocol_version": protocol["protocol_version"],
            "architecture": protocol["architecture"],
            "mode": self.mode,
            "run_mode": self.mode,
            "formal100": self.formal100,
            "dataset_identity": identity,
            "dataset_sample_sha256": identity["sample_sha256"],
            "dataset_manifest_sha256": identity["lock_sha256"],
            "ordered_task_ids_sha256": identity["ordered_task_ids_sha256"],
            "capability_profile_hash": profile_hash,
            "prompt_hash": protocol.get("prompt_hash"),
            "schema_set_hash": protocol.get("schema_set_hash"),
            "replanner_schema_hash": protocol.get("replanner_schema_hash"),
            "grounding_contract_hash": protocol.get("grounding_contract_hash"),
            "catalog_registry_sha256": FROZEN_REGISTRY_SHA256,
            "operation_documents_sha256": None,
            "operation_index_identity": {"namespace": EMBEDDING_CACHE_NAMESPACE},
            "embedding_cache_namespace": EMBEDDING_CACHE_NAMESPACE,
            "generation_provider": (self.config.get("llm") or {}).get("provider"),
            "generation_model": (self.config.get("llm") or {}).get("model"),
            "generation_host": None,
            "embedding_model": (self.config.get("embedding") or {}).get("model"),
            "embedding_host": None,
            "budgets": dict(DEFAULT_BUDGETS),
            "evaluator_identity": evaluator_identity(),
            "namespace_inventory": inventory,
            "protocol": protocol,
            "git_commit": git_commit(self.root),
            "git_branch": git_branch(self.root),
            "git_dirty": git_dirty(self.root),
            "started_at": datetime.now(timezone.utc).isoformat(),
            "live_pass_claimed": False,
            "run_status": self.run_status,
            "evaluator": evaluator_identity().get("module"),
            "env_file_set": bool(self.env_file),
            "embedding_env_file_set": bool(self.embedding_env_file),
            "require_complete_index": self.require_complete_index,
            "tool_count": FROZEN_CATALOG_TOTAL_TOOLS,
        }
        metadata["resume_identity_sha256"] = resume_identity_hash(metadata)
        _write_json(self.output_dir / "run_metadata.json", metadata)

        if self.mode == "dry-run":
            preflight = {
                "ok": True,
                "mode": "dry-run",
                "live_pass_claimed": False,
                "dataset_identity": identity,
                "capability_profile_hash": profile_hash,
                "namespace_inventory": inventory,
                "resume_identity_sha256": metadata["resume_identity_sha256"],
                "note": "dry-run only; no LLM/MCP/embedding API calls",
            }
            if self.require_complete_index:
                # Offline dry-run can require the flag but must not invent a live pass.
                preflight["require_complete_index"] = True
                preflight["complete_index_present"] = False
                # Do not fail dry-run solely on missing live index during scaffolding.
            _write_json(self.output_dir / "preflight_report.json", preflight)
            self.run_status = "completed"
            metadata["run_status"] = self.run_status
            _write_json(self.output_dir / "run_metadata.json", metadata)
            return preflight

        if self.mode == "live":
            raise EtomPeFinReplanRunnerError(
                "Live ETOM formal100 is gated until HK-155 freeze + explicit authorization. "
                "Use --mock or --dry-run for offline engineering."
            )

        lock_tasks = identity["ordered_task_ids"][: self.max_tasks]
        pred_path = self.output_dir / "predictions.jsonl"
        if not self.resume and pred_path.exists():
            pred_path.unlink()
        completed = set()
        if self.resume and pred_path.exists():
            for line in pred_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    completed.add(json.loads(line)["task_id"])
        for task_id in lock_tasks:
            if self._stop:
                break
            if task_id in completed:
                continue
            # Keep evaluator import live for identity without gold-driven scoring here.
            _ = evaluate_prediction
            row = {
                "task_id": task_id,
                "mode": "mock",
                "live_pass_claimed": False,
                "status": "mock_placeholder",
                "predicted_tools": [],
                "exact_match": False,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "note": "mock formal100 placeholder; not a live pass",
                "budgets": dict(DEFAULT_BUDGETS),
            }
            _fsync_write_jsonl(pred_path, row)

        self.run_status = "aborted" if self._stop else "completed"
        metadata["run_status"] = self.run_status
        metadata["completed_at"] = datetime.now(timezone.utc).isoformat()
        metadata["live_pass_claimed"] = False
        _write_json(self.output_dir / "run_metadata.json", metadata)
        return metadata

    def _finalize_metadata(self, extra: Optional[Dict[str, Any]] = None) -> None:
        path = self.output_dir / "run_metadata.json"
        meta: Dict[str, Any] = {}
        if path.exists():
            meta = json.loads(path.read_text(encoding="utf-8"))
        meta["run_status"] = self.run_status
        meta["live_pass_claimed"] = False
        meta["finalized_at"] = datetime.now(timezone.utc).isoformat()
        if extra:
            meta.update(extra)
        _write_json(path, meta)
