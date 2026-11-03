"""Unified financial experiment runner (MCP-Free / Flat / Operation-Visible)."""

from __future__ import annotations

import csv
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set
from urllib.parse import urlparse

import yaml

from hk_equity_mcp.agent.flat import FlatAgent
from hk_equity_mcp.agent.llm_client import OpenAICompatibleChatClient
from hk_equity_mcp.agent.mcp_free import MCPFreeAgent
from hk_equity_mcp.agent.operation_visible import OperationVisibleAgent
from hk_equity_mcp.common.env_loader import (
    EMBEDDING_REQUIRED,
    GENERATION_REQUIRED,
    load_embedding_profile,
    load_generation_profile,
    missing_keys,
)
from hk_equity_mcp.experiment.prompts import (
    MCP_FREE_SYSTEM_PROMPT,
    OPERATION_VISIBLE_SYSTEM_PROMPT,
    build_flat_system_prompt,
    prompt_hash,
)
from hk_equity_mcp.experiment.schema import (
    CONDITIONS,
    MANUAL_GRADING_FIELDS,
    PREDICTION_FIELDS,
)
from hk_equity_mcp.operation_resolver.registry import build_default_registry
from hk_equity_mcp.operation_resolver.retrieval import OpenAICompatibleEmbeddingBackend
from hk_equity_mcp.operation_resolver.server import OperationResolverServer

RESUME_IDENTITY_KEYS = (
    "condition",
    "generation_model",
    "questions_file",
    "system_prompt_hash",
    "temperature",
    "max_tokens",
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_experiment_config(path: Optional[Path] = None) -> Dict[str, Any]:
    cfg_path = path or (repo_root() / "configs" / "financial_experiment.yaml")
    if not cfg_path.exists():
        return {
            "temperature": 0.0,
            "max_tokens": 1024,
            "limits": {
                "max_llm_turns": 8,
                "max_search_reformulations": 3,
                "max_invocations": 6,
                "timeout_sec": 120,
            },
        }
    with cfg_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    return rows


def git_commit(root: Path) -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except Exception:  # noqa: BLE001
        return None


def git_branch(root: Path) -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=str(root),
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except Exception:  # noqa: BLE001
        return None


def git_dirty(root: Path) -> bool:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=str(root),
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return bool(out.strip())
    except Exception:  # noqa: BLE001
        return True


def url_host(url: Optional[str]) -> Optional[str]:
    if not url:
        return None
    parsed = urlparse(url)
    return parsed.netloc or None


def system_prompt_for(condition: str) -> str:
    if condition == "mcp_free":
        return MCP_FREE_SYSTEM_PROMPT
    if condition == "flat":
        return build_flat_system_prompt()
    if condition == "operation_visible":
        return OPERATION_VISIBLE_SYSTEM_PROMPT
    raise ValueError(f"Unknown condition: {condition}")


def catalog_size_for(condition: str) -> int:
    if condition == "mcp_free":
        return 0
    if condition == "flat":
        return 2
    if condition == "operation_visible":
        return len(build_default_registry())
    raise ValueError(f"Unknown condition: {condition}")


def validate_questions(items: List[Dict[str, Any]]) -> None:
    if not items:
        raise ValueError("Question file is empty.")
    for item in items:
        if "id" not in item or "question" not in item:
            raise ValueError("Each question row must include id and question.")


def parse_ids(raw: Optional[str]) -> Optional[List[str]]:
    if raw is None or str(raw).strip() == "":
        return None
    ids = [part.strip() for part in str(raw).split(",") if part.strip()]
    if not ids:
        raise ValueError("--ids was provided but no valid ids were parsed.")
    return ids


def select_questions_by_ids(
    items: Sequence[Dict[str, Any]], ids: Sequence[str]
) -> List[Dict[str, Any]]:
    """Keep original benchmark order; require every requested id to exist."""
    wanted = list(ids)
    by_id = {item["id"]: item for item in items}
    missing = [qid for qid in wanted if qid not in by_id]
    if missing:
        raise ValueError(f"Unknown question id(s): {missing}")
    wanted_set = set(wanted)
    return [item for item in items if item["id"] in wanted_set]


def already_done_ids(predictions_path: Path) -> Set[str]:
    done: Set[str] = set()
    if not predictions_path.exists():
        return done
    with predictions_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("id"):
                done.add(row["id"])
    return done


def redact_secrets(obj: Any) -> Any:
    """Recursively drop keys that look like secrets (defense in depth)."""
    secret_keys = {
        "api_key",
        "apikey",
        "llm_api_key",
        "embedding_api_key",
        "authorization",
        "password",
        "secret",
        "access_token",
        "refresh_token",
    }
    if isinstance(obj, dict):
        out = {}
        for key, value in obj.items():
            lowered = str(key).lower().replace("-", "_")
            if lowered in secret_keys or lowered.endswith("_api_key"):
                continue
            out[key] = redact_secrets(value)
        return out
    if isinstance(obj, list):
        return [redact_secrets(x) for x in obj]
    return obj


def build_prediction_row(
    *,
    item: Dict[str, Any],
    condition: str,
    result: Dict[str, Any],
) -> Dict[str, Any]:
    row = {
        "id": item["id"],
        "condition": condition,
        "question": item["question"],
        "final_answer": result.get("final_answer"),
        "predicted_status": result.get("predicted_status"),
        "trace": result.get("trace") or [],
        "llm_calls": result.get("llm_calls"),
        "resolver_calls": result.get("resolver_calls")
        or {"find_operations": 0, "describe_operation": 0},
        "underlying_operation_calls": result.get("underlying_operation_calls", 0),
        "tokens": result.get("tokens") or {},
        "latency_ms": result.get("latency_ms"),
        "provider_metadata": result.get("provider_metadata") or {},
        "error": result.get("error"),
    }
    return redact_secrets(row)


def write_manual_grading(
    path: Path,
    *,
    condition: str,
    items: List[Dict[str, Any]],
    predictions_by_id: Dict[str, Dict[str, Any]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(MANUAL_GRADING_FIELDS))
        writer.writeheader()
        for item in items:
            pred = predictions_by_id.get(item["id"], {})
            writer.writerow(
                {
                    "id": item["id"],
                    "condition": condition,
                    "company": item.get("company", ""),
                    "task_type": item.get("task_type", ""),
                    "question": item["question"],
                    "gold_status": item.get("gold_status", ""),
                    "reference_answer": item.get("reference_answer", ""),
                    "predicted_status": pred.get("predicted_status", ""),
                    "predicted_answer": pred.get("final_answer", ""),
                    "correct_0_or_1": "",
                    "error_type": "",
                    "notes": "",
                }
            )


def assert_resume_compatible(existing: Dict[str, Any], current: Dict[str, Any]) -> None:
    mismatches = []
    for key in RESUME_IDENTITY_KEYS:
        if existing.get(key) != current.get(key):
            mismatches.append(
                f"{key}: existing={existing.get(key)!r} current={current.get(key)!r}"
            )
    if mismatches:
        raise RuntimeError(
            "Resume refused: run_metadata identity mismatch. "
            "Do not mix different experiments into one predictions.jsonl. "
            + "; ".join(mismatches)
        )


class FinancialExperimentRunner:
    def __init__(
        self,
        *,
        condition: str,
        questions_path: Path,
        output_dir: Path,
        limit: Optional[int] = None,
        ids: Optional[Sequence[str]] = None,
        resume: bool = False,
        dry_run: bool = False,
        allow_dirty: bool = False,
        env_file: Optional[str] = None,
        embedding_env_file: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        root: Optional[Path] = None,
        json_mode: bool = True,
    ):
        if condition not in CONDITIONS:
            raise ValueError(
                f"Unsupported condition {condition!r}. Choose from {list(CONDITIONS)}"
            )
        self.condition = condition
        self.root = root or repo_root()
        self.questions_path = questions_path
        self.output_dir = output_dir
        self.limit = limit
        self.ids = list(ids) if ids else None
        self.resume = resume
        self.dry_run = dry_run
        self.allow_dirty = allow_dirty
        self.env_file = env_file
        self.embedding_env_file = embedding_env_file
        self.config = config or load_experiment_config()
        self.temperature = float(self.config.get("temperature", 0.0))
        self.max_tokens = int(self.config.get("max_tokens", 1024))
        self.limits = self.config.get("limits") or {}
        self.json_mode = bool(json_mode)
        self.gen_meta: Dict[str, str] = {}
        self.emb_meta: Optional[Dict[str, str]] = None

    def prepare(self) -> Dict[str, Any]:
        dirty = git_dirty(self.root)
        if not self.dry_run and dirty and not self.allow_dirty:
            raise RuntimeError(
                "Paid/formal run refused: git working tree is dirty. "
                "Commit changes first, or pass --allow-dirty explicitly "
                "(not recommended for reproducible experiments)."
            )

        items = load_jsonl(self.questions_path)
        validate_questions(items)
        if self.ids is not None:
            items = select_questions_by_ids(items, self.ids)
        if self.limit is not None:
            items = items[: max(0, int(self.limit))]

        if self.dry_run:
            try:
                self.gen_meta = load_generation_profile(self.root, self.env_file)
            except FileNotFoundError as exc:
                raise RuntimeError(str(exc)) from exc
            missing_gen = missing_keys(GENERATION_REQUIRED)
            if missing_gen:
                raise RuntimeError(
                    f"Dry-run: missing generation env var names: {missing_gen}"
                )
        else:
            self.gen_meta = load_generation_profile(self.root, self.env_file)

        if self.condition == "operation_visible":
            if self.dry_run:
                try:
                    self.emb_meta = load_embedding_profile(
                        self.root, self.embedding_env_file
                    )
                except FileNotFoundError as exc:
                    raise RuntimeError(str(exc)) from exc
                missing_emb = missing_keys(EMBEDDING_REQUIRED)
                if missing_emb:
                    raise RuntimeError(
                        f"Dry-run: missing embedding env var names: {missing_emb}"
                    )
            else:
                self.emb_meta = load_embedding_profile(
                    self.root, self.embedding_env_file
                )
        else:
            self.emb_meta = None

        catalog_size = catalog_size_for(self.condition)
        if self.condition == "operation_visible":
            registry = build_default_registry()
            if len(registry) != catalog_size:
                raise RuntimeError("Operation registry size mismatch.")

        self.output_dir.mkdir(parents=True, exist_ok=True)

        prompt = system_prompt_for(self.condition)
        gen_host = url_host(self.gen_meta.get("base_url") or os.environ.get("LLM_BASE_URL"))
        emb_host = None
        if self.emb_meta is not None:
            emb_host = url_host(
                self.emb_meta.get("base_url") or os.environ.get("EMBEDDING_BASE_URL")
            )

        meta = {
            "condition": self.condition,
            "questions_file": str(self.questions_path),
            "question_ids": [item["id"] for item in items],
            "question_count": len(items),
            "generation_provider": self.gen_meta.get("provider"),
            "generation_model": self.gen_meta.get("model"),
            "generation_profile_file": self.gen_meta.get("profile_file"),
            "generation_base_url_host": gen_host,
            "embedding_model": None
            if self.emb_meta is None
            else self.emb_meta.get("model"),
            "embedding_profile_file": None
            if self.emb_meta is None
            else self.emb_meta.get("profile_file"),
            "embedding_base_url_host": emb_host,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "json_mode": self.json_mode,
            "catalog_size": catalog_size,
            "system_prompt_hash": prompt_hash(prompt),
            "system_prompt": prompt,
            "branch": git_branch(self.root),
            "git_commit": git_commit(self.root),
            "git_dirty": dirty,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "dry_run": self.dry_run,
            "resume": self.resume,
            "allow_dirty": self.allow_dirty,
            "run_status": "dry_run" if self.dry_run else "running",
            "fidelity_note": (
                "simplified reimplementation adapted from Ghoshal et al. (2026)"
                if self.condition == "operation_visible"
                else None
            ),
        }
        return {"items": items, "run_metadata": redact_secrets(meta)}

    def build_agent(self):
        if self.dry_run:
            raise RuntimeError("build_agent must not be called during dry-run.")
        llm = OpenAICompatibleChatClient(
            base_url=os.environ["LLM_BASE_URL"],
            api_key=os.environ["LLM_API_KEY"],
            model=os.environ["LLM_MODEL"],
            provider=os.environ.get("LLM_PROVIDER") or "openai_compatible",
            json_mode=self.json_mode,
        )
        if self.condition == "mcp_free":
            return MCPFreeAgent(
                llm, temperature=self.temperature, max_tokens=self.max_tokens
            )
        if self.condition == "flat":
            return FlatAgent(
                llm,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                max_llm_turns=int(self.limits.get("max_llm_turns", 8)),
                max_invocations=int(self.limits.get("max_invocations", 6)),
                timeout_sec=float(self.limits.get("timeout_sec", 120)),
            )
        embedding = OpenAICompatibleEmbeddingBackend(
            base_url=os.environ["EMBEDDING_BASE_URL"],
            api_key=os.environ["EMBEDDING_API_KEY"],
            model=os.environ["EMBEDDING_MODEL"],
        )
        resolver = OperationResolverServer(embedding_backend=embedding)
        cfg = {
            "llm": {
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            },
            "limits": dict(self.limits),
        }
        return OperationVisibleAgent(
            llm=llm, embedding_backend=embedding, config=cfg, resolver=resolver
        )

    def run(self) -> Dict[str, Any]:
        prepared = self.prepare()
        items: List[Dict[str, Any]] = prepared["items"]
        run_meta = prepared["run_metadata"]
        meta_path = self.output_dir / "run_metadata.json"
        predictions_path = self.output_dir / "predictions.jsonl"
        grading_path = self.output_dir / "manual_grading.csv"

        if self.dry_run:
            meta_path.write_text(
                json.dumps(run_meta, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print("DRY-RUN OK")
            print(f"condition={self.condition}")
            print(f"questions={self.questions_path}")
            print(f"question_count={run_meta['question_count']}")
            print(f"question_ids={run_meta.get('question_ids')}")
            print(f"generation_model={run_meta['generation_model']}")
            print(f"generation_provider={run_meta['generation_provider']}")
            print(f"generation_base_url_host={run_meta['generation_base_url_host']}")
            print(f"embedding_model={run_meta['embedding_model']}")
            print(f"embedding_base_url_host={run_meta['embedding_base_url_host']}")
            print(f"catalog_size={run_meta['catalog_size']}")
            print(f"temperature={run_meta['temperature']}")
            print(f"max_tokens={run_meta['max_tokens']}")
            print(f"json_mode={run_meta['json_mode']}")
            print(f"branch={run_meta['branch']}")
            print(f"git_commit={run_meta['git_commit']}")
            print(f"git_dirty={run_meta['git_dirty']}")
            print(f"output_dir={self.output_dir}")
            print(f"generation_profile_file={run_meta['generation_profile_file']}")
            print(f"embedding_profile_file={run_meta['embedding_profile_file']}")
            return run_meta

        if self.resume and meta_path.exists():
            existing = json.loads(meta_path.read_text(encoding="utf-8"))
            assert_resume_compatible(existing, run_meta)
        elif self.resume and not meta_path.exists():
            raise RuntimeError(
                "Resume refused: run_metadata.json is missing. "
                "Cannot verify experiment identity."
            )

        # Write metadata before the first paid question.
        run_meta["run_status"] = "running"
        meta_path.write_text(
            json.dumps(run_meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        agent = self.build_agent()
        done = already_done_ids(predictions_path) if self.resume else set()
        if not self.resume and predictions_path.exists():
            predictions_path.unlink()

        predictions_by_id: Dict[str, Dict[str, Any]] = {}
        if self.resume and predictions_path.exists():
            for line in predictions_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                predictions_by_id[row["id"]] = row

        mode = "a" if self.resume else "w"
        with predictions_path.open(mode, encoding="utf-8") as pred_file:
            for index, item in enumerate(items, start=1):
                qid = item["id"]
                if qid in done:
                    print(f"[{index:02d}/{len(items)}] skip {qid} (resume)", flush=True)
                    continue
                print(f"[{index:02d}/{len(items)}] {qid}", flush=True)
                try:
                    result = agent.run(item["question"])
                    row = build_prediction_row(
                        item=item, condition=self.condition, result=result
                    )
                except Exception as exc:  # noqa: BLE001
                    row = build_prediction_row(
                        item=item,
                        condition=self.condition,
                        result={
                            "final_answer": "",
                            "predicted_status": "unavailable",
                            "trace": [],
                            "llm_calls": 0,
                            "resolver_calls": {
                                "find_operations": 0,
                                "describe_operation": 0,
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
                for key in PREDICTION_FIELDS:
                    row.setdefault(key, None)
                pred_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                pred_file.flush()
                predictions_by_id[qid] = row

        write_manual_grading(
            grading_path,
            condition=self.condition,
            items=items,
            predictions_by_id=predictions_by_id,
        )
        run_meta["run_status"] = "completed"
        run_meta["completed_at"] = datetime.now(timezone.utc).isoformat()
        meta_path.write_text(
            json.dumps(run_meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"Predictions: {predictions_path}")
        print(f"Manual grading: {grading_path}")
        print(f"Run metadata: {meta_path}")
        return run_meta
