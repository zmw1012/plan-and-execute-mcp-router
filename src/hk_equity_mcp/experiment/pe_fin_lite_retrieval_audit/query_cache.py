"""Question-query and planner-step-query vector caches (checkpoint/resume)."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from hk_equity_mcp.experiment.pe_fin_lite_retrieval_audit import (
    QUERY_CONSTRUCTION_VERSION,
)
from hk_equity_mcp.experiment.pe_fin_lite_retrieval_audit.identity import (
    sha256_text,
    stable_json_hash,
)
from hk_equity_mcp.experiment.pe_fin_lite_retrieval_audit.ranking import (
    build_question_query,
    build_step_query,
)


EmbedFn = Callable[[List[str]], List[List[float]]]


class QueryCacheError(RuntimeError):
    pass


def _fsync_append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    data = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    with tmp.open("w", encoding="utf-8") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def question_query_cache_identity(
    *,
    dataset_identities: Dict[str, Any],
    embedding_model: str,
    embedding_provider_host: Optional[str],
    embedding_dim: Optional[int],
    query_construction_version: str = QUERY_CONSTRUCTION_VERSION,
) -> Dict[str, Any]:
    payload = {
        "kind": "question_query_vector_cache",
        "manifest_sha256": dataset_identities.get("manifest_sha256"),
        "ordered_task_ids_hash": dataset_identities.get("ordered_task_ids_hash"),
        "question_identity_hash": dataset_identities.get("question_identity_hash"),
        "query_construction_version": query_construction_version,
        "embedding_model": embedding_model,
        "embedding_provider_host": embedding_provider_host,
        "embedding_dim": embedding_dim,
    }
    return {**payload, "identity_sha256": stable_json_hash(payload)}


def step_query_cache_identity(
    *,
    dataset_identities: Dict[str, Any],
    planner_cache_sha256: str,
    embedding_model: str,
    embedding_provider_host: Optional[str],
    embedding_dim: Optional[int],
    query_construction_version: str = QUERY_CONSTRUCTION_VERSION,
) -> Dict[str, Any]:
    payload = {
        "kind": "planner_step_query_vector_cache",
        "manifest_sha256": dataset_identities.get("manifest_sha256"),
        "ordered_task_ids_hash": dataset_identities.get("ordered_task_ids_hash"),
        "question_identity_hash": dataset_identities.get("question_identity_hash"),
        "planner_cache_sha256": planner_cache_sha256,
        "query_construction_version": query_construction_version,
        "embedding_model": embedding_model,
        "embedding_provider_host": embedding_provider_host,
        "embedding_dim": embedding_dim,
    }
    return {**payload, "identity_sha256": stable_json_hash(payload)}


def _load_partial_vectors(partial: Path) -> Dict[str, Dict[str, Any]]:
    if not partial.exists():
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    lines = partial.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            if index == len(lines) - 1:
                continue
            raise QueryCacheError(f"Corrupt query cache JSONL line {index + 1}: {exc}") from exc
        key = str(row.get("cache_key") or "")
        if not key:
            raise QueryCacheError("query cache row missing cache_key")
        if key in out:
            raise QueryCacheError(f"duplicate cache_key in partial: {key}")
        vec = row.get("vector")
        if not isinstance(vec, list) or not vec:
            raise QueryCacheError(f"missing vector for {key}")
        out[key] = row
    return out


def question_cache_key(task_id: str, query: str) -> str:
    return sha256_text(f"question|{task_id}|{query}")


def step_cache_key(task_id: str, step_id: str, query: str) -> str:
    return sha256_text(f"step|{task_id}|{step_id}|{query}")


def embed_question_queries(
    *,
    tasks: Sequence[Dict[str, Any]],
    output_dir: Path,
    identity: Dict[str, Any],
    embed_fn: EmbedFn,
    batch_size: int = 16,
    resume: bool = True,
) -> Dict[str, Any]:
    """Generate/resume question-query vectors. Calls embed_fn only (no planner/MCP)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    partial = output_dir / "question_query_vectors.partial.jsonl"
    final = output_dir / "question_query_vectors.json"
    id_path = output_dir / "question_query_cache_identity.json"

    if id_path.exists() and resume:
        prev = json.loads(id_path.read_text(encoding="utf-8"))
        if prev.get("identity_sha256") != identity.get("identity_sha256"):
            # Allow dim=None → set later by first batch: compare without dim if needed
            prev_core = {k: v for k, v in prev.items() if k not in {"embedding_dim", "identity_sha256"}}
            new_core = {
                k: v for k, v in identity.items() if k not in {"embedding_dim", "identity_sha256"}
            }
            if stable_json_hash(prev_core) != stable_json_hash(new_core):
                raise QueryCacheError("question query cache identity mismatch (fail-closed)")
    else:
        _atomic_write_json(id_path, identity)

    existing = _load_partial_vectors(partial) if resume else {}
    pending: List[Tuple[str, str, str]] = []  # cache_key, task_id, query
    for task in tasks:
        tid = str(task.get("task_id") or task.get("id"))
        query = build_question_query(str(task.get("question") or ""))
        key = question_cache_key(tid, query)
        if key not in existing:
            pending.append((key, tid, query))

    dim = identity.get("embedding_dim")
    newly = 0
    for i in range(0, len(pending), max(1, int(batch_size))):
        batch = pending[i : i + max(1, int(batch_size))]
        texts = [q for _k, _t, q in batch]
        vectors = embed_fn(texts)
        if len(vectors) != len(batch):
            raise QueryCacheError("embed_fn returned unexpected batch size")
        for (key, tid, query), vec in zip(batch, vectors):
            if not vec:
                raise QueryCacheError(f"empty vector for task {tid}")
            if dim is None:
                dim = len(vec)
            elif len(vec) != int(dim):
                raise QueryCacheError(
                    f"dimension drift for {tid}: {len(vec)} != {dim}"
                )
            row = {
                "cache_key": key,
                "task_id": tid,
                "query": query,
                "query_sha256": sha256_text(query),
                "vector": [float(x) for x in vec],
                "vector_dim": int(dim),
                "model": identity.get("embedding_model"),
            }
            _fsync_append_jsonl(partial, row)
            existing[key] = row
            newly += 1

    # Finalize when complete.
    expected = len(tasks)
    complete = len(existing) == expected
    payload = {
        "complete": complete,
        "n": len(existing),
        "expected": expected,
        "identity": {**identity, "embedding_dim": dim},
        "vectors_by_task_id": {
            str(row["task_id"]): row["vector"] for row in existing.values()
        },
        "queries_by_task_id": {
            str(row["task_id"]): row["query"] for row in existing.values()
        },
    }
    if complete:
        _atomic_write_json(final, payload)
        # Refresh identity with discovered dim.
        _atomic_write_json(id_path, {**identity, "embedding_dim": dim,
                                       "identity_sha256": stable_json_hash(
                                           {**{k: v for k, v in identity.items() if k != "identity_sha256"},
                                            "embedding_dim": dim}
                                       )})
    return {
        "complete": complete,
        "completed": len(existing),
        "expected": expected,
        "newly_written": newly,
        "embedding_dim": dim,
        "final_path": str(final) if final.exists() else None,
        "partial_path": str(partial),
        "identity_sha256": identity.get("identity_sha256"),
    }


def embed_planner_step_queries(
    *,
    planner_rows: Sequence[Dict[str, Any]],
    output_dir: Path,
    identity: Dict[str, Any],
    embed_fn: EmbedFn,
    batch_size: int = 16,
    resume: bool = True,
) -> Dict[str, Any]:
    """Embed all retrieve-step queries from frozen planner cache."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    partial = output_dir / "step_query_vectors.partial.jsonl"
    final = output_dir / "step_query_vectors.json"
    id_path = output_dir / "step_query_cache_identity.json"

    if id_path.exists() and resume:
        prev = json.loads(id_path.read_text(encoding="utf-8"))
        prev_core = {k: v for k, v in prev.items() if k not in {"embedding_dim", "identity_sha256"}}
        new_core = {
            k: v for k, v in identity.items() if k not in {"embedding_dim", "identity_sha256"}
        }
        if stable_json_hash(prev_core) != stable_json_hash(new_core):
            raise QueryCacheError("step query cache identity mismatch (fail-closed)")
    else:
        _atomic_write_json(id_path, identity)

    existing = _load_partial_vectors(partial) if resume else {}
    pending: List[Tuple[str, str, str, str]] = []  # key, task_id, step_id, query
    for prow in planner_rows:
        tid = str(prow.get("task_id") or "")
        for step in prow.get("retrieve_steps") or []:
            if not isinstance(step, dict):
                continue
            sid = str(step.get("id") or step.get("step_id") or "")
            query = str(step.get("search_query") or build_step_query(step))
            key = step_cache_key(tid, sid, query)
            if key not in existing:
                pending.append((key, tid, sid, query))

    expected_keys = set()
    for prow in planner_rows:
        tid = str(prow.get("task_id") or "")
        for step in prow.get("retrieve_steps") or []:
            if not isinstance(step, dict):
                continue
            sid = str(step.get("id") or step.get("step_id") or "")
            query = str(step.get("search_query") or build_step_query(step))
            expected_keys.add(step_cache_key(tid, sid, query))

    dim = identity.get("embedding_dim")
    newly = 0
    for i in range(0, len(pending), max(1, int(batch_size))):
        batch = pending[i : i + max(1, int(batch_size))]
        texts = [q for _k, _t, _s, q in batch]
        vectors = embed_fn(texts)
        if len(vectors) != len(batch):
            raise QueryCacheError("embed_fn returned unexpected batch size")
        for (key, tid, sid, query), vec in zip(batch, vectors):
            if dim is None:
                dim = len(vec)
            elif len(vec) != int(dim):
                raise QueryCacheError(f"dimension drift for {tid}/{sid}")
            row = {
                "cache_key": key,
                "task_id": tid,
                "step_id": sid,
                "query": query,
                "query_sha256": sha256_text(query),
                "vector": [float(x) for x in vec],
                "vector_dim": int(dim),
                "model": identity.get("embedding_model"),
            }
            _fsync_append_jsonl(partial, row)
            existing[key] = row
            newly += 1

    complete = set(existing.keys()) == expected_keys and len(expected_keys) == len(existing)
    # Index by task/step for ranking consumers.
    by_task_step: Dict[str, Dict[str, Any]] = {}
    for row in existing.values():
        by_task_step[f"{row['task_id']}::{row['step_id']}"] = {
            "vector": row["vector"],
            "query": row["query"],
            "cache_key": row["cache_key"],
        }
    payload = {
        "complete": complete,
        "n": len(existing),
        "expected": len(expected_keys),
        "identity": {**identity, "embedding_dim": dim},
        "vectors_by_task_step": by_task_step,
    }
    if complete:
        _atomic_write_json(final, payload)
        refreshed = {
            **{k: v for k, v in identity.items() if k != "identity_sha256"},
            "embedding_dim": dim,
        }
        refreshed["identity_sha256"] = stable_json_hash(refreshed)
        _atomic_write_json(id_path, refreshed)
    return {
        "complete": complete,
        "completed": len(existing),
        "expected": len(expected_keys),
        "newly_written": newly,
        "embedding_dim": dim,
        "final_path": str(final) if final.exists() else None,
        "partial_path": str(partial),
    }


def load_complete_question_query_cache(path: Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise QueryCacheError(f"Missing complete question query cache: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not payload.get("complete"):
        raise QueryCacheError("question query cache incomplete (fail-closed)")
    return payload


def load_complete_step_query_cache(path: Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise QueryCacheError(f"Missing complete step query cache: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not payload.get("complete"):
        raise QueryCacheError("step query cache incomplete (fail-closed)")
    return payload


def query_cache_status(
    *,
    final_path: Path,
    partial_path: Path,
    expected: int,
) -> Dict[str, Any]:
    completed = 0
    complete = False
    if final_path.exists():
        payload = json.loads(final_path.read_text(encoding="utf-8"))
        complete = bool(payload.get("complete"))
        completed = int(payload.get("n") or 0)
    elif partial_path.exists():
        completed = len(_load_partial_vectors(partial_path))
    return {
        "completed": completed,
        "expected": expected,
        "complete": complete and completed == expected,
        "final_path": str(final_path) if final_path.exists() else None,
        "partial_path": str(partial_path) if partial_path.exists() else None,
    }
