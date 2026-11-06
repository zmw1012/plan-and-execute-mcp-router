"""Embedding utilities for MCP-Zero: counting wrapper + server/tool vector table.

Two embedder interfaces exist in this repo:
  * HK ``EmbeddingBackend.embed(texts) -> List[vector]``
  * ETOM clients ``embed(texts) -> (vectors, usage)``
``CountingEmbedder`` accepts both and exposes ``embed(texts) -> List[vector]``
plus call / token counters (query-time accounting).
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from hk_equity_mcp.mcp_zero.metadata import McpZeroCatalog


class IncompleteEmbeddingTableError(RuntimeError):
    """A complete table was required but is missing (never builds silently)."""


def estimate_tokens(texts: Sequence[str]) -> int:
    return sum(max(1, (len(t) + 3) // 4) for t in texts)


class CountingEmbedder:
    """Wrap an embedder; count calls/tokens; normalise the return type."""

    def __init__(self, inner: Any, model_name: Optional[str] = None):
        self.inner = inner
        self.model_name = model_name or getattr(inner, "model_name", "unknown")
        self.calls = 0
        self.tokens = 0
        self.token_source = "provider"

    def snapshot(self) -> Dict[str, int]:
        return {"calls": self.calls, "tokens": self.tokens}

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        texts = list(texts)
        if not texts:
            return []
        result = self.inner.embed(texts)
        if isinstance(result, tuple):
            vectors, usage = result
            calls = int(getattr(usage, "calls", 0) or 1)
            tokens = int(getattr(usage, "tokens", 0) or 0)
            if getattr(usage, "token_source", "provider") != "provider":
                self.token_source = "estimated"
        else:
            vectors = result
            calls = 1
            tokens = 0
        if tokens <= 0:
            tokens = estimate_tokens(texts)
            self.token_source = "estimated"
        if len(vectors) != len(texts):
            raise RuntimeError("Embedding backend returned unexpected batch size.")
        self.calls += calls
        self.tokens += tokens
        return [list(map(float, v)) for v in vectors]


@dataclass
class EmbeddingTable:
    model_name: str
    fingerprint: str
    server_desc: Dict[str, List[float]] = field(default_factory=dict)
    server_summary: Dict[str, List[float]] = field(default_factory=dict)
    tool: Dict[str, List[float]] = field(default_factory=dict)
    build_calls: int = 0
    build_tokens: int = 0
    cache_reused: bool = False
    cache_path: Optional[str] = None

    def size(self) -> int:
        return len(self.server_desc) + len(self.server_summary) + len(self.tool)


def _units(catalog: McpZeroCatalog, use_summary: bool):
    units = []
    for server in catalog.servers():
        units.append(("sd:" + server.server_id, server.description))
        if use_summary and server.summary:
            units.append(("ss:" + server.server_id, server.summary))
        for tool in server.tools:
            units.append(("t:" + tool.operation_id, tool.text))
    return units


def _digest(model_name: str, fingerprint: str, use_summary: bool) -> str:
    raw = f"{model_name}|{fingerprint}|summary={int(use_summary)}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def table_paths(cache_dir: Path, model_name: str, fingerprint: str, use_summary: bool):
    key = _digest(model_name, fingerprint, use_summary)
    final = Path(cache_dir) / f"mcp_zero_table_{key}.json"
    partial = Path(cache_dir) / f"mcp_zero_table_{key}.partial.jsonl"
    return final, partial


def _split_table(
    vectors: Dict[str, List[float]], model_name: str, fingerprint: str
) -> EmbeddingTable:
    table = EmbeddingTable(model_name=model_name, fingerprint=fingerprint)
    for key, vec in vectors.items():
        kind, _, ident = key.partition(":")
        if kind == "sd":
            table.server_desc[ident] = vec
        elif kind == "ss":
            table.server_summary[ident] = vec
        elif kind == "t":
            table.tool[ident] = vec
    return table


def table_status(
    catalog: McpZeroCatalog,
    *,
    model_name: str,
    cache_dir: Path,
    use_summary: bool = False,
) -> Dict[str, Any]:
    """Offline status of the final cache + partial checkpoint (no network)."""
    fingerprint = catalog.fingerprint()
    final, partial = table_paths(cache_dir, model_name, fingerprint, use_summary)
    expected = {k for k, _ in _units(catalog, use_summary)}
    final_complete = False
    final_count = 0
    if final.exists():
        try:
            payload = json.loads(final.read_text(encoding="utf-8"))
            vectors = payload.get("vectors") or {}
            final_count = len(vectors)
            final_complete = (
                payload.get("model") == model_name
                and payload.get("fingerprint") == fingerprint
                and bool(payload.get("complete"))
                and set(vectors) == expected
            )
        except (json.JSONDecodeError, TypeError, ValueError):
            final_complete = False
    partial_count = 0
    if partial.exists():
        partial_count = sum(
            1 for line in partial.read_text(encoding="utf-8").splitlines() if line.strip()
        )
    return {
        "model": model_name,
        "catalog_fingerprint": fingerprint,
        "total": len(expected),
        "final_cache_path": str(final),
        "final_complete": final_complete,
        "final_completed_count": final_count,
        "partial_path": str(partial),
        "partial_completed_count": partial_count,
        "completed": final_count if final_complete else partial_count,
        "cache_dir": str(cache_dir),
    }


def load_complete_table(
    catalog: McpZeroCatalog,
    *,
    model_name: str,
    cache_dir: Path,
    use_summary: bool = False,
) -> EmbeddingTable:
    status = table_status(
        catalog, model_name=model_name, cache_dir=cache_dir, use_summary=use_summary
    )
    if not status["final_complete"]:
        raise IncompleteEmbeddingTableError(
            "Complete MCP-Zero embedding table required but missing "
            f"({status['completed']}/{status['total']}) at {status['final_cache_path']}. "
            "Refusing to rebuild implicitly."
        )
    payload = json.loads(Path(status["final_cache_path"]).read_text(encoding="utf-8"))
    table = _split_table(payload["vectors"], model_name, catalog.fingerprint())
    table.cache_reused = True
    table.cache_path = status["final_cache_path"]
    return table


def build_embedding_table(
    catalog: McpZeroCatalog,
    embedder: CountingEmbedder,
    *,
    cache_dir: Optional[Path] = None,
    use_summary: bool = True,
    batch_size: int = 64,
    progress: Optional[Callable[[int, int], None]] = None,
) -> EmbeddingTable:
    """Embed servers + tools. With ``cache_dir``: checkpointed + reusable.

    Never called implicitly for live ETOM runs (those use
    ``load_complete_table``); the explicit ``--build-index`` path calls this.
    """
    fingerprint = catalog.fingerprint()
    units = _units(catalog, use_summary)
    model_name = embedder.model_name
    vectors: Dict[str, List[float]] = {}
    final = partial = None
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        final, partial = table_paths(cache_dir, model_name, fingerprint, use_summary)
        status = table_status(
            catalog, model_name=model_name, cache_dir=cache_dir, use_summary=use_summary
        )
        if status["final_complete"]:
            return load_complete_table(
                catalog,
                model_name=model_name,
                cache_dir=cache_dir,
                use_summary=use_summary,
            )
        if partial.exists():
            for line in partial.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                vectors[row["k"]] = row["v"]
    calls_before, tokens_before = embedder.calls, embedder.tokens
    pending = [(k, t) for k, t in units if k not in vectors]
    step = max(1, int(batch_size))
    for start in range(0, len(pending), step):
        chunk = pending[start : start + step]
        vecs = embedder.embed([t for _, t in chunk])
        for (key, _), vec in zip(chunk, vecs):
            vectors[key] = vec
        if partial is not None:
            with partial.open("a", encoding="utf-8") as handle:
                for (key, _), vec in zip(chunk, vecs):
                    handle.write(json.dumps({"k": key, "v": vec}) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        if progress:
            progress(len(vectors), len(units))
    table = _split_table(vectors, model_name, fingerprint)
    table.build_calls = embedder.calls - calls_before
    table.build_tokens = embedder.tokens - tokens_before
    if final is not None:
        tmp = final.with_suffix(final.suffix + ".tmp")
        tmp.write_text(
            json.dumps(
                {
                    "model": model_name,
                    "fingerprint": fingerprint,
                    "complete": True,
                    "vectors": vectors,
                }
            ),
            encoding="utf-8",
        )
        tmp.replace(final)
        if partial is not None and partial.exists():
            partial.unlink()
        table.cache_path = str(final)
    return table
