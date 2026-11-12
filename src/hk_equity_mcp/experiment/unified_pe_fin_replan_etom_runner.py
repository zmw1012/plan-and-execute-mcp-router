"""PE-Fin RePlan-ETOM formal-100 runner: mock / dry-run / index tooling / gated live.

Modes
-----
* ``dry-run``   identity + config + lock + capability profile + index inventory.
                No LLM, no MCP, no embedding API. Writes ``preflight_report.json``.
* ``mock``      ScriptedEtomPeFinLLM + HashEmbeddingClient + routing-only resolver.
                Runs the real agent loop offline, scores with the pinned ETOM
                evaluator. NEVER claims a live pass.
* ``live``      gated. Requires ``allow_live=True`` AND
                ``PE_FIN_REPLAN_ETOM_ALLOW_LIVE=1``, a clean tree, matching
                provider/model env, and a COMPLETE ``pe_fin_replan_etom`` index.
                The tests never exercise this path.

Predicted tools come only from route nodes recorded by the gateway.
Gold is read ONLY by the evaluator after the agent has returned.
"""

from __future__ import annotations

import csv
import json
import os
import signal
import threading
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

import yaml

from hk_equity_mcp.etom.convert import convert_registry
from hk_equity_mcp.etom.embeddings import (
    CatalogIndex,
    EmbeddingUsage,
    HashEmbeddingClient,
    ModelMismatchError,
    scrub_secret,
)
from hk_equity_mcp.etom.evaluator import evaluate_prediction
from hk_equity_mcp.etom.formal_metrics import (
    dependency_accuracy,
    error_summary,
    grouped_results,
    retrieval_analysis,
)
from hk_equity_mcp.experiment.runner import git_branch, git_commit, git_dirty
from hk_equity_mcp.pe_fin_replan_etom import EMBEDDING_CACHE_NAMESPACE
from hk_equity_mcp.pe_fin_replan_etom.capability_profile import (
    capability_profile_hash,
    load_capability_profile,
)
from hk_equity_mcp.pe_fin_replan_etom.dataset import assert_frozen_etom100_identity
from hk_equity_mcp.pe_fin_replan_etom.identity import (
    DEFAULT_BUDGETS,
    FROZEN_CATALOG_TOTAL_TOOLS,
    FROZEN_REGISTRY_SHA256,
    assert_budgets,
    evaluator_identity,
    protocol_identity,
    resume_identity_hash,
)
from hk_equity_mcp.pe_fin_replan_etom.operation_documents import (
    documents_for_registry,
    documents_hash,
    registry_catalog_fingerprint,
)
from hk_equity_mcp.pe_fin_replan_etom.operation_embedding_index import (
    LiveEmbeddingGatedError,
    assert_embedder_allowed,
    build_index_identity,
    build_index_with_embedder,
    identity_digest,
    index_status,
    load_catalog_index,
    namespace_inventory,
    resolve_embedding_cache_dir,
)

LIVE_ENV_GATE = "PE_FIN_REPLAN_ETOM_ALLOW_LIVE"
MOCK_EMBEDDING_MODEL = "mock-hash-embedding-v1"

CSV_FIELDS = [
    "task_id",
    "level",
    "query_id",
    "selected_order",
    "mode",
    "status",
    "agent_status",
    "finalized_by_replanner",
    "completion",
    "error_type",
    "exact_match",
    "precision",
    "recall",
    "f1",
    "eval_status",
    "dependency_edge_accuracy",
    "ordered_dependency_accuracy",
    "llm_calls",
    "mcp_invocations",
    "execution_round_count",
    "executed_step_count",
    "replan_calls",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "query_embedding_calls",
    "query_embedding_tokens",
    "latency_ms",
    "http_retries",
    "parse_repairs",
    "real_tool_executions",
    "predicted_tool_count",
    "live_pass_claimed",
    "error",
]


class EtomPeFinReplanRunnerError(RuntimeError):
    pass


class ResumeIdentityMismatch(EtomPeFinReplanRunnerError):
    pass


class LiveGateError(EtomPeFinReplanRunnerError):
    pass


# ---------------------------------------------------------------------------
# io helpers
# ---------------------------------------------------------------------------


def _fsync_write_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _latest_by_task(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    latest: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        latest[row["task_id"]] = row
    return latest


def _host(url: Optional[str]) -> Optional[str]:
    if not url:
        return None
    return urlparse(url).netloc or None


# ---------------------------------------------------------------------------
# small runtime adapters
# ---------------------------------------------------------------------------


class CountingEmbedder:
    """Counts query-time embedding calls/tokens; delegates everything else."""

    def __init__(self, inner: Any):
        self._inner = inner
        self.model_name = inner.model_name
        self.usage = EmbeddingUsage()

    def embed(self, texts: List[str]) -> tuple:
        vectors, usage = self._inner.embed(texts)
        self.usage.add(usage)
        return vectors, usage


class StageChatClient:
    """Adds the ``stage`` kwarg expected by BudgetedLLMClient to a plain client."""

    def __init__(self, inner: Any):
        self._inner = inner
        self.provider = getattr(inner, "provider", "unknown")
        self.model = getattr(inner, "model", "unknown")

    @property
    def retries(self) -> int:
        return int(getattr(self._inner, "retries", 0) or 0)

    def complete(self, messages, *, temperature=0.0, max_tokens=1024, stage=None):
        return self._inner.complete(messages, temperature=temperature, max_tokens=max_tokens)


class FixtureTask:
    """Minimal task for fixture-based mock runs (tests only)."""

    def __init__(self, row: Dict[str, Any]):
        self.task_id = str(row["task_id"])
        self.level = int(row["level"])
        self.query_id = int(row.get("query_id") or 0)
        self.query = str(row["query"])
        self.ground_truth_tools = row.get("ground_truth_tools") or []


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------


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
        allow_live: bool = False,
        allow_dirty: bool = False,
        llm_override: Any = None,
        fixture_registry_payload: Optional[Dict[str, Any]] = None,
        fixture_tasks: Optional[Sequence[Dict[str, Any]]] = None,
    ):
        if mode not in {"dry-run", "mock", "live"}:
            raise EtomPeFinReplanRunnerError(f"unknown mode {mode}")
        if int(max_tasks) < 1 or int(max_tasks) > 100:
            raise EtomPeFinReplanRunnerError("max_tasks must be in 1..100")
        if build_index_only and index_status_only:
            raise EtomPeFinReplanRunnerError("choose one of --build-index-only / --index-status")
        if (fixture_registry_payload or fixture_tasks) and mode != "mock":
            raise EtomPeFinReplanRunnerError("fixture catalogs/tasks are mock-only")
        self.root = Path(root).resolve()
        out = Path(output_dir)
        self.output_dir = out if out.is_absolute() else (self.root / out)
        self.mode = mode
        self.formal100 = formal100
        self.resume = resume
        self.env_file = env_file
        self.embedding_env_file = embedding_env_file
        self.embedding_cache_dir_arg = embedding_cache_dir
        self.require_complete_index = bool(require_complete_index or mode == "live")
        self.index_status_only = index_status_only
        self.build_index_only = build_index_only
        self.mock_embeddings = bool(mock_embeddings or mode == "mock")
        self.max_tasks = int(max_tasks)
        self.allow_live = allow_live
        self.allow_dirty = allow_dirty
        self.llm_override = llm_override
        self.fixture_registry_payload = fixture_registry_payload
        self.fixture_tasks = list(fixture_tasks) if fixture_tasks else None
        self.config_path = Path(config_path) if config_path else (
            self.root / "configs" / "pe_fin_replan_etom_formal100_v1.yaml"
        )
        self.config = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        self.limits = dict(DEFAULT_BUDGETS)
        for key, value in (self.config.get("limits") or {}).items():
            if key in DEFAULT_BUDGETS:
                self.limits[key] = value
        assert_budgets(self.limits)
        self._stop = False
        self.run_status = "running"
        self._metadata: Dict[str, Any] = {}

    # ----------------------------------------------------------------- cache
    def _cache_dir(self) -> Path:
        return resolve_embedding_cache_dir(self.root, self.embedding_cache_dir_arg)

    def _mock_build_cache_dir(self) -> Path:
        if self.embedding_cache_dir_arg:
            return resolve_embedding_cache_dir(self.root, self.embedding_cache_dir_arg)
        # Mock hash indexes never touch the shared .cache namespace by default.
        return resolve_embedding_cache_dir(
            self.root,
            self.output_dir / "mock_embedding_cache" / EMBEDDING_CACHE_NAMESPACE,
        )

    # --------------------------------------------------------------- signals
    def _install_signal_handler(self) -> None:
        if threading.current_thread() is not threading.main_thread():
            return

        def _handler(signum, frame):  # noqa: ANN001
            self._stop = True
            raise KeyboardInterrupt

        signal.signal(signal.SIGINT, _handler)
        signal.signal(signal.SIGTERM, _handler)

    # --------------------------------------------------------------- execute
    def execute(self) -> Dict[str, Any]:
        self._install_signal_handler()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        try:
            return self._execute_inner()
        except (KeyboardInterrupt, SystemExit):
            self.run_status = "aborted"
            self._finalize_metadata(extra={"abort_reason": "signal_or_exit"})
            raise
        except (LiveGateError, ResumeIdentityMismatch, LiveEmbeddingGatedError):
            # Fail-closed configuration errors: nothing ran; keep artifacts as-is.
            self.run_status = "fail_closed"
            self._finalize_metadata(extra={"abort_reason": "fail_closed"})
            raise
        except Exception as exc:  # noqa: BLE001
            self.run_status = "aborted"
            self._finalize_metadata(
                extra={
                    "abort_reason": "unknown_runner_error",
                    "error": scrub_secret(f"{type(exc).__name__}: {exc}"),
                    "traceback": scrub_secret(traceback.format_exc()[-4000:]),
                }
            )
            raise

    # -------------------------------------------------------------- catalog
    def _catalog(self) -> Tuple[Dict[str, Any], Dict[str, Any], str, bool]:
        """(specs, official, registry_sha256, is_fixture)."""
        if self.fixture_registry_payload is not None:
            specs, official = convert_registry(self.fixture_registry_payload)
            return specs, official, "fixture:" + registry_catalog_fingerprint(official), True
        from hk_equity_mcp.etom.loader import load_official_registry
        from hk_equity_mcp.etom.source import file_sha256, raw_root

        payload = load_official_registry(self.root, verify=True)
        specs, official = convert_registry(payload)
        if len(specs) != FROZEN_CATALOG_TOTAL_TOOLS:
            raise EtomPeFinReplanRunnerError(
                f"converted registry has {len(specs)} operations, expected {FROZEN_CATALOG_TOTAL_TOOLS}"
            )
        sha = file_sha256(raw_root(self.root) / "data" / "mcp_registry.json")
        if sha != FROZEN_REGISTRY_SHA256:
            raise EtomPeFinReplanRunnerError("registry sha256 drifted from the frozen pin")
        return specs, official, sha, False

    def _tasks(self, lock_identity: Dict[str, Any]) -> Tuple[List[Any], Dict[str, int]]:
        if self.fixture_tasks is not None:
            rows = self.fixture_tasks[: self.max_tasks]
            tasks = [FixtureTask(r) for r in rows]
            return tasks, {t.task_id: i for i, t in enumerate(tasks, start=1)}
        from hk_equity_mcp.etom.formal_sampler import (
            assert_formal_100_lock_matches,
            select_formal_locked_tasks,
        )
        from hk_equity_mcp.etom.loader import load_official_tasks

        official_tasks = load_official_tasks(self.root, verify=True)
        lock = assert_formal_100_lock_matches(self.root, official_tasks)
        selected = select_formal_locked_tasks(official_tasks, lock, max_tasks=self.max_tasks)
        order = {
            row["task_id"]: int(row.get("selected_order") or i)
            for i, row in enumerate(lock.get("tasks") or [], start=1)
        }
        return selected, order

    # ---------------------------------------------------------------- index
    def _index_identity(
        self, specs, official, *, embedding_model: str, host: Optional[str]
    ) -> Dict[str, Any]:
        return build_index_identity(
            registry=specs,
            catalog_fingerprint=registry_catalog_fingerprint(official),
            embedding_model=embedding_model,
            embedding_provider_host=host,
        )

    def _index_status_payload(self, specs, official) -> Dict[str, Any]:
        model = str((self.config.get("embedding") or {}).get("model"))
        host = None
        if self.embedding_env_file:
            try:
                from hk_equity_mcp.common.env_loader import load_embedding_profile

                profile = load_embedding_profile(self.root, self.embedding_env_file)
                host = _host(profile.get("base_url") or os.environ.get("EMBEDDING_BASE_URL"))
            except Exception:  # noqa: BLE001 - inventory only; never fatal
                host = None
        identity = self._index_identity(specs, official, embedding_model=model, host=host)
        status = index_status(registry=specs, identity=identity, cache_dir=self._cache_dir())
        payload = status.to_dict()
        payload["identity"] = {
            k: v for k, v in payload["identity"].items() if k != "ordered_operation_ids"
        }
        payload["embedding_host_known"] = host is not None
        payload["network_calls"] = 0
        return payload

    def _mock_index(self, specs) -> CatalogIndex:
        embedder = HashEmbeddingClient(model_name=MOCK_EMBEDDING_MODEL)
        rows = documents_for_registry(specs)
        texts = [doc for _oid, doc, _h in rows]
        vectors_list, usage = embedder.embed(texts)
        vectors = {oid: list(vec) for (oid, _d, _h), vec in zip(rows, vectors_list)}
        return CatalogIndex(
            model_name=MOCK_EMBEDDING_MODEL,
            registry_sha256="mock",
            vectors=vectors,
            build_usage=usage,
            cache_reused=False,
            cache_path=None,
            completed_count=len(vectors),
            total_count=len(specs),
        )

    # ---------------------------------------------------------------- inner
    def _execute_inner(self) -> Dict[str, Any]:
        profile = load_capability_profile(self.root)
        profile_hash = capability_profile_hash(profile)
        inventory = namespace_inventory(self.root)

        if self.fixture_tasks is not None:
            lock_identity = {
                "sample_sha256": "fixture",
                "lock_sha256": "fixture",
                "ordered_task_ids_sha256": "fixture",
                "ordered_task_ids": [t["task_id"] for t in self.fixture_tasks],
                "fixture": True,
            }
        else:
            lock_identity = assert_frozen_etom100_identity(self.root)

        if self.index_status_only:
            catalog_error = None
            index_payload: Optional[Dict[str, Any]] = None
            registry_sha = None
            try:
                specs, official, registry_sha, _fx = self._catalog()
                index_payload = self._index_status_payload(specs, official)
            except Exception as exc:  # noqa: BLE001 - inventory only
                catalog_error = scrub_secret(f"{type(exc).__name__}: {exc}")
            payload = {
                "mode": "index-status",
                "run_mode": self.mode,
                "dataset_identity": {
                    k: v for k, v in lock_identity.items() if k != "ordered_task_ids"
                },
                "namespace": EMBEDDING_CACHE_NAMESPACE,
                "cache_dir": str(self._cache_dir()),
                "index": index_payload,
                "catalog_error": catalog_error,
                "namespace_inventory": inventory,
                "registry_sha256": registry_sha,
                "network_calls": 0,
                "live_pass_claimed": False,
                "ok": catalog_error is None,
            }
            _write_json(self.output_dir / "index_status.json", payload)
            self.run_status = "completed"
            return payload

        if self.build_index_only:
            return self._build_index_phase(profile_hash, inventory)

        if self.mode == "dry-run":
            try:
                specs, official, registry_sha, fixture = self._catalog()
                catalog_error = None
            except Exception as exc:  # noqa: BLE001 - dry-run reports, never fails on raw data
                specs, official, registry_sha, fixture = None, None, None, False
                catalog_error = scrub_secret(f"{type(exc).__name__}: {exc}")
        else:
            specs, official, registry_sha, fixture = self._catalog()
            catalog_error = None
        protocol = protocol_identity()
        gen_cfg = self.config.get("llm") or {}
        emb_cfg = self.config.get("embedding") or {}
        mock = self.mode != "live"
        metadata: Dict[str, Any] = {
            "condition": protocol["condition"],
            "method_version": protocol["method_version"],
            "protocol_version": protocol["protocol_version"],
            "architecture": protocol["architecture"],
            "mode": self.mode,
            "run_mode": self.mode,
            "formal100": self.formal100,
            "fixture_run": fixture or self.fixture_tasks is not None,
            "dataset_identity": {
                k: v for k, v in lock_identity.items() if k != "ordered_task_ids"
            },
            "dataset_sample_sha256": lock_identity["sample_sha256"],
            "dataset_manifest_sha256": lock_identity["lock_sha256"],
            "ordered_task_ids_sha256": lock_identity["ordered_task_ids_sha256"],
            "capability_profile_hash": profile_hash,
            "prompt_hash": protocol.get("prompt_hash"),
            "schema_set_hash": protocol.get("schema_set_hash"),
            "replanner_schema_hash": protocol.get("replanner_schema_hash"),
            "grounding_contract_hash": protocol.get("grounding_contract_hash"),
            "catalog_registry_sha256": registry_sha,
            "operation_documents_sha256": documents_hash(specs) if specs is not None else None,
            "embedding_cache_namespace": EMBEDDING_CACHE_NAMESPACE,
            "generation_provider": "mock" if mock else gen_cfg.get("provider"),
            "generation_model": "mock-pe-fin-replan-etom-llm" if mock else gen_cfg.get("model"),
            "generation_host": None,
            "embedding_model": MOCK_EMBEDDING_MODEL if mock else emb_cfg.get("model"),
            "embedding_host": None,
            "budgets": dict(self.limits),
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
            "max_tasks": self.max_tasks,
            "require_complete_index": self.require_complete_index,
            "tool_count": len(specs) if specs is not None else None,
            "catalog_error": catalog_error,
            "real_tool_execution": False,
            "posthoc_exploratory": True,
            "formal_primary_result": False,
        }

        if self.mode == "dry-run":
            idx_status = (
                self._index_status_payload(specs, official)
                if specs is not None
                else {"complete": False, "identity_sha256": None, "catalog_error": catalog_error}
            )
            metadata["operation_index_identity"] = idx_status.get("identity_sha256")
            metadata["resume_identity_sha256"] = resume_identity_hash(metadata)
            preflight = {
                "ok": True,
                "mode": "dry-run",
                "live_pass_claimed": False,
                "network_calls": 0,
                "llm_calls": 0,
                "mcp_calls": 0,
                "dataset_identity": metadata["dataset_identity"],
                "capability_profile_hash": profile_hash,
                "protocol": protocol,
                "budgets": dict(self.limits),
                "index_status": idx_status,
                "require_complete_index": self.require_complete_index,
                "complete_index_present": bool(idx_status.get("complete")),
                "namespace_inventory": inventory,
                "resume_identity_sha256": metadata["resume_identity_sha256"],
                "task_ids": list(lock_identity.get("ordered_task_ids") or [])[: self.max_tasks],
                "note": "dry-run only; no LLM/MCP/embedding API calls",
            }
            if self.require_complete_index and not preflight["complete_index_present"]:
                preflight["warning"] = "complete live index not present; live run would fail closed"
            _write_json(self.output_dir / "preflight_report.json", preflight)
            self.run_status = "completed"
            metadata["run_status"] = self.run_status
            metadata["completed_at"] = datetime.now(timezone.utc).isoformat()
            _write_json(self.output_dir / "run_metadata.json", metadata)
            return preflight

        if self.mode == "live":
            return self._live_run(specs, official, registry_sha, metadata, lock_identity)

        # ------------------------------------------------------------- mock
        index = self._mock_index(specs)
        embedder = CountingEmbedder(HashEmbeddingClient(model_name=MOCK_EMBEDDING_MODEL))
        idx_identity = self._index_identity(
            specs, official, embedding_model=MOCK_EMBEDDING_MODEL, host=None
        )
        metadata["operation_index_identity"] = identity_digest(idx_identity)
        from hk_equity_mcp.pe_fin_replan_etom.mock_llm import ScriptedEtomPeFinLLM

        llm = self.llm_override or ScriptedEtomPeFinLLM()
        return self._run_tasks(
            specs=specs,
            official=official,
            index=index,
            embedder=embedder,
            llm=llm,
            profile=profile,
            metadata=metadata,
            lock_identity=lock_identity,
        )

    # ---------------------------------------------------------------- build
    def _build_index_phase(self, profile_hash: str, inventory: Dict[str, Any]) -> Dict[str, Any]:
        specs, official, registry_sha, fixture = self._catalog()
        if not self.mock_embeddings:
            if not (self.allow_live and os.environ.get(LIVE_ENV_GATE) == "1"):
                raise LiveEmbeddingGatedError(
                    "Live embedding index build is future-only and fails closed offline. "
                    "Use --mock-embeddings for a deterministic hash index, or follow "
                    "docs/pe_fin_replan_etom_v1/FUTURE_LIVE_COMMANDS.md."
                )
            return self._live_build_index(specs, official, registry_sha)
        embedder = HashEmbeddingClient(model_name=MOCK_EMBEDDING_MODEL)
        assert_embedder_allowed(embedder, allow_live=False)
        identity = self._index_identity(
            specs, official, embedding_model=MOCK_EMBEDDING_MODEL, host=None
        )
        cache_dir = self._mock_build_cache_dir()
        built = build_index_with_embedder(
            registry=specs, identity=identity, cache_dir=cache_dir, embedder=embedder
        )
        status = index_status(registry=specs, identity=identity, cache_dir=cache_dir)
        payload = {
            "mode": "mock",
            "run_mode": "build-index-only",
            "mock_embeddings": True,
            "cache_dir": str(cache_dir),
            "build_result": built,
            "index_status": {
                k: v for k, v in status.to_dict().items() if k != "identity"
            },
            "live_pass_claimed": False,
            "ok": True,
        }
        _write_json(self.output_dir / "build_index_result.json", payload)
        self.run_status = "completed"
        return payload

    def _live_build_index(self, specs, official, registry_sha) -> Dict[str, Any]:  # pragma: no cover
        from hk_equity_mcp.common.env_loader import load_embedding_profile
        from hk_equity_mcp.etom import EXPECTED_EMBEDDING_MODEL
        from hk_equity_mcp.etom.embeddings import OpenAICompatibleEmbeddingClient

        profile = load_embedding_profile(self.root, self.embedding_env_file)
        if os.environ.get("EMBEDDING_MODEL") != EXPECTED_EMBEDDING_MODEL:
            raise ModelMismatchError("EMBEDDING_MODEL mismatch")
        embedder = OpenAICompatibleEmbeddingClient(
            model_name=EXPECTED_EMBEDDING_MODEL,
            timeout_sec=float((self.config.get("limits") or {}).get("request_timeout_sec", 120)),
            max_retries=1,
            batch_size=16,
        )
        identity = self._index_identity(
            specs,
            official,
            embedding_model=EXPECTED_EMBEDDING_MODEL,
            host=_host(profile.get("base_url") or os.environ.get("EMBEDDING_BASE_URL")),
        )
        built = build_index_with_embedder(
            registry=specs,
            identity=identity,
            cache_dir=self._cache_dir(),
            embedder=embedder,
            allow_live=True,
        )
        payload = {"mode": "live-build-index", "build_result": built, "live_pass_claimed": False}
        _write_json(self.output_dir / "build_index_result.json", payload)
        self.run_status = "completed"
        return payload

    # ----------------------------------------------------------------- live
    def _live_run(self, specs, official, registry_sha, metadata, lock_identity):  # pragma: no cover
        if not (self.allow_live and os.environ.get(LIVE_ENV_GATE) == "1"):
            raise LiveGateError(
                "Live ETOM formal100 is gated. It needs allow_live=True (CLI "
                "--allow-live-formal100) AND env PE_FIN_REPLAN_ETOM_ALLOW_LIVE=1, "
                "and is intended only after HK-155 freeze. Use --mock or --dry-run offline."
            )
        if git_dirty(self.root) and not self.allow_dirty:
            raise LiveGateError("Live formal run refused: worktree is dirty.")
        from hk_equity_mcp.common.env_loader import (
            load_embedding_profile,
            load_generation_profile,
        )
        from hk_equity_mcp.etom import (
            EXPECTED_EMBEDDING_MODEL,
            EXPECTED_LLM_MODEL,
            EXPECTED_LLM_PROVIDER,
        )
        from hk_equity_mcp.etom.embeddings import OpenAICompatibleEmbeddingClient
        from hk_equity_mcp.etom.llm import FailClosedChatClient

        generation = load_generation_profile(self.root, self.env_file)
        embedding = load_embedding_profile(self.root, self.embedding_env_file)
        if (
            os.environ.get("LLM_PROVIDER") != EXPECTED_LLM_PROVIDER
            or os.environ.get("LLM_MODEL") != EXPECTED_LLM_MODEL
            or os.environ.get("EMBEDDING_MODEL") != EXPECTED_EMBEDDING_MODEL
        ):
            raise ModelMismatchError("provider/model env does not match the frozen ETOM models")
        gen_host = _host(generation.get("base_url") or os.environ.get("LLM_BASE_URL"))
        emb_host = _host(embedding.get("base_url") or os.environ.get("EMBEDDING_BASE_URL"))
        idx_identity = self._index_identity(
            specs, official, embedding_model=EXPECTED_EMBEDDING_MODEL, host=emb_host
        )
        index = load_catalog_index(
            registry=specs,
            identity=idx_identity,
            cache_dir=self._cache_dir(),
            registry_sha256=registry_sha,
        )
        metadata.update(
            {
                "generation_provider": generation.get("provider"),
                "generation_model": generation.get("model"),
                "generation_host": gen_host,
                "embedding_model": EXPECTED_EMBEDDING_MODEL,
                "embedding_host": emb_host,
                "operation_index_identity": identity_digest(idx_identity),
            }
        )
        timeout = float((self.config.get("limits") or {}).get("request_timeout_sec", 120))
        llm = StageChatClient(
            FailClosedChatClient(
                timeout_sec=timeout,
                max_retries=int((self.config.get("limits") or {}).get("max_http_retries", 1)),
                json_mode=bool((self.config.get("llm") or {}).get("json_mode", True)),
            )
        )
        embedder = CountingEmbedder(
            OpenAICompatibleEmbeddingClient(
                model_name=EXPECTED_EMBEDDING_MODEL, timeout_sec=timeout, max_retries=1, batch_size=16
            )
        )
        profile = load_capability_profile(self.root)
        return self._run_tasks(
            specs=specs,
            official=official,
            index=index,
            embedder=embedder,
            llm=llm,
            profile=profile,
            metadata=metadata,
            lock_identity=lock_identity,
        )

    # ------------------------------------------------------------ task loop
    def _run_tasks(
        self, *, specs, official, index, embedder, llm, profile, metadata, lock_identity
    ) -> Dict[str, Any]:
        from hk_equity_mcp.etom.adapter import EtomRoutingResolver
        from hk_equity_mcp.pe_fin_replan_etom.agent import PeFinReplanEtomAgent

        selected, order_map = self._tasks(lock_identity)
        metadata["task_ids"] = [t.task_id for t in selected]
        metadata["resume_identity_sha256"] = resume_identity_hash(metadata)
        pred_path = self.output_dir / "predictions.jsonl"
        meta_path = self.output_dir / "run_metadata.json"

        existing = _read_jsonl(pred_path)
        if existing and not self.resume:
            raise EtomPeFinReplanRunnerError(
                f"{pred_path} already has predictions; pass --resume or use a new output dir."
            )
        if self.resume and meta_path.exists():
            prior = json.loads(meta_path.read_text(encoding="utf-8"))
            prior_hash = prior.get("resume_identity_sha256")
            if prior_hash != metadata["resume_identity_sha256"]:
                raise ResumeIdentityMismatch(
                    "Resume identity mismatch (condition/method/protocol, dataset, "
                    "capability profile, prompts/schemas, catalog/index, provider/model/host, "
                    "budgets, evaluator, mode, or git commit/dirty changed). Refusing to resume."
                )
        elif self.resume and existing:
            raise ResumeIdentityMismatch("Cannot resume predictions without prior run_metadata.json")

        latest = _latest_by_task(existing)
        _write_json(self.output_dir / "run.pid.json", {"pid": os.getpid()})
        _write_json(meta_path, {**metadata, "run_status": "running"})

        resolver = EtomRoutingResolver(
            specs,
            official,
            index,
            embedder,
            lexical_weight=float((self.config.get("embedding") or {}).get("lexical_weight", 0.12)),
            top_k_default=int(self.limits["retrieval_top_k"]),
            top_k_max=20,
        )
        agent = PeFinReplanEtomAgent(
            llm,
            resolver,
            capability_profile=profile,
            registry=specs,
            official=official,
            vectors=index.vectors,
            embedder=embedder,
            limits=dict(self.limits),
            temperature=float((self.config.get("llm") or {}).get("temperature", 0.0)),
            max_tokens=int((self.config.get("llm") or {}).get("max_tokens", 2048)),
            planner_max_tokens=int((self.config.get("llm") or {}).get("planner_max_tokens", 4096)),
        )
        rows: List[Dict[str, Any]] = list(existing)
        run_status = "completed"
        try:
            for position, task in enumerate(selected, start=1):
                if self._stop:
                    run_status = "aborted"
                    break
                prior_row = latest.get(task.task_id)
                if self.resume and prior_row and prior_row.get("status") == "completed":
                    continue
                row = self._run_one(
                    agent,
                    resolver,
                    embedder,
                    task,
                    selected_order=int(order_map.get(task.task_id) or position),
                    llm=llm,
                )
                _fsync_write_jsonl(pred_path, row)
                rows.append(row)
                latest[task.task_id] = row
                self._write_reports(rows, selected, metadata, run_status="running")
                if "ModelMismatchError" in str(row.get("error") or ""):
                    raise ModelMismatchError(str(row.get("error")))
        except (KeyboardInterrupt, SystemExit):
            self.run_status = "aborted"
            self._write_reports(rows, selected, metadata, run_status="aborted")
            raise
        except Exception:
            self.run_status = "aborted"
            self._write_reports(rows, selected, metadata, run_status="aborted")
            raise

        done = [
            t for t in selected if latest.get(t.task_id, {}).get("status") == "completed"
        ]
        if run_status != "aborted":
            run_status = "completed" if len(done) == len(selected) else "partial"
        self.run_status = run_status
        return self._write_reports(rows, selected, metadata, run_status=run_status)

    # --------------------------------------------------------------- one row
    def _run_one(self, agent, resolver, embedder, task, *, selected_order: int, llm) -> Dict[str, Any]:
        emb_calls_before = embedder.usage.calls
        emb_tokens_before = embedder.usage.tokens
        http_before = int(getattr(llm, "retries", 0) or 0)
        # The agent only ever sees the question text.
        result = agent.run(task.query)
        if resolver.real_tool_executions != 0 or int(result.get("real_tool_executions") or 0) != 0:
            raise RuntimeError("ETOM routing mode executed a real tool.")
        for node in result.get("predicted_nodes") or []:
            if node.get("business_result") not in (None, {}):
                raise RuntimeError("ETOM invoke recorded a business result.")
        predicted_tools = result.get("predicted_tools") or []
        # Gold is consumed only here, after the agent returned.
        metrics = evaluate_prediction(task.level, task.ground_truth_tools, predicted_tools)
        dep = dependency_accuracy(
            level=task.level,
            ground_truth_tools=task.ground_truth_tools,
            predicted_nodes=result.get("predicted_nodes") or [],
        )
        finalized = bool(result.get("final_response_source") == "replanner")
        status = "completed" if finalized else "failed"
        tokens = result.get("tokens") or {}
        in_tok = int(tokens.get("input_tokens") or tokens.get("prompt_tokens") or 0)
        out_tok = int(tokens.get("output_tokens") or tokens.get("completion_tokens") or 0)
        row: Dict[str, Any] = {
            "task_id": task.task_id,
            "level": task.level,
            "query_id": task.query_id,
            "selected_order": selected_order,
            "dataset_split": "external_formal",
            "benchmark_role": "external_validation",
            "mode": self.mode,
            "fixture_run": self.fixture_tasks is not None,
            "live_pass_claimed": False,
            "condition": result.get("condition"),
            "method_version": result.get("method_version"),
            "protocol_version": result.get("protocol_version"),
            "posthoc_exploratory": True,
            "formal_primary_result": False,
            "status": status,
            "agent_status": result.get("status"),
            "finalized_by_replanner": finalized,
            "completion": result.get("completion"),
            "error": scrub_secret(str(result.get("error"))) if result.get("error") else None,
            "error_type": result.get("error_type"),
            "stopping_reason": result.get("stopping_reason"),
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
            "llm_calls_by_stage": result.get("llm_calls_by_stage"),
            "mcp_invocations": int(result.get("mcp_invocations") or 0),
            "execution_round_count": result.get("execution_round_count"),
            "executed_step_count": result.get("executed_step_count"),
            "replan_calls": result.get("replan_calls"),
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "total_tokens": int(tokens.get("total_tokens") or (in_tok + out_tok)),
            "query_embedding_calls": embedder.usage.calls - emb_calls_before,
            "query_embedding_tokens": embedder.usage.tokens - emb_tokens_before,
            "latency_ms": result.get("latency_ms"),
            "http_retries": int(getattr(llm, "retries", 0) or 0) - http_before,
            "parse_repairs": int(result.get("replan_repairs") or 0),
            "real_tool_executions": int(result.get("real_tool_executions") or 0),
            "predicted_tools": predicted_tools,
            "predicted_nodes": result.get("predicted_nodes") or [],
            "dependency_edges": result.get("dependency_edges") or [],
            "predicted_tool_count": len(predicted_tools),
            "final_answer": result.get("final_answer"),
            "initial_plan": result.get("initial_plan"),
            "replan_decisions": result.get("replan_decisions"),
            "retrieval_funnels": result.get("retrieval_funnels"),
            "failed_signature_blocks": result.get("failed_signature_blocks"),
            "forced_final_round": result.get("forced_final_round"),
            "executor_to_executor_edge": result.get("executor_to_executor_edge"),
            "graph_edges": result.get("graph_edges"),
            "trace": result.get("trace") or [],
            "evidence_store": result.get("evidence_store"),
            "executed_etom_mcp_tool": False,
        }
        return row

    # --------------------------------------------------------------- reports
    @staticmethod
    def _failures_as_zero(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Controlled failures count as wrong (L5 empty route is not a free pass)."""
        out: List[Dict[str, Any]] = []
        for row in _latest_by_task(rows).values():
            if row.get("status") == "completed":
                out.append(row)
                continue
            zero = dict(row)
            zero.update(
                {
                    "status": "completed",
                    "exact_match": False,
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                    "dependency_edge_accuracy": 0.0,
                    "ordered_dependency_accuracy": 0.0,
                    "scored_as": "failure_as_zero",
                }
            )
            out.append(zero)
        return out

    def _write_reports(
        self, rows, selected, metadata, *, run_status: str
    ) -> Dict[str, Any]:
        latest = _latest_by_task(rows)
        ordered = [latest[t.task_id] for t in selected if t.task_id in latest]
        path = self.output_dir / "per_task_metrics.csv"
        tmp = path.with_suffix(".csv.tmp")
        with tmp.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
            writer.writeheader()
            for row in ordered:
                writer.writerow(row)
            handle.flush()
            os.fsync(handle.fileno())
        tmp.replace(path)
        _write_json(self.output_dir / "grouped_results.json", grouped_results(ordered))
        _write_json(
            self.output_dir / "grouped_results_failures_as_zero.json",
            grouped_results(self._failures_as_zero(ordered)),
        )
        _write_json(self.output_dir / "error_summary.json", error_summary(ordered))
        _write_json(self.output_dir / "retrieval_analysis.json", retrieval_analysis(ordered))
        completed_ids = [r["task_id"] for r in ordered if r.get("status") == "completed"]
        if self.mode == "mock":
            _write_json(
                self.output_dir / "mock_summary.json",
                {
                    "mode": "mock",
                    "run_status": run_status,
                    "live_pass_claimed": False,
                    "engineering_only": True,
                    "note": "Scripted LLM + hash embeddings; not a live or primary result.",
                    "n_rows": len(ordered),
                    "completed": len(completed_ids),
                    "failed": len(ordered) - len(completed_ids),
                    "max_llm_calls_observed": max(
                        [int(r.get("llm_calls") or 0) for r in ordered] or [0]
                    ),
                    "max_mcp_invocations_observed": max(
                        [int(r.get("mcp_invocations") or 0) for r in ordered] or [0]
                    ),
                    "real_tool_executions": sum(
                        int(r.get("real_tool_executions") or 0) for r in ordered
                    ),
                },
            )
        final_meta = {
            **metadata,
            "run_status": run_status,
            "resume": self.resume,
            "completed_task_ids": completed_ids,
            "n_rows": len(ordered),
            "live_pass_claimed": False,
            "real_tool_execution": False,
            "max_tasks_excluded_from_resume_identity": True,
            "finalized_at": datetime.now(timezone.utc).isoformat(),
            "output_dir": str(self.output_dir),
        }
        _write_json(self.output_dir / "run_metadata.json", final_meta)
        self._metadata = final_meta
        return final_meta

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
        self.output_dir.mkdir(parents=True, exist_ok=True)
        _write_json(path, meta)
