"""Resume identity for MCP-Zero × ETOM runners."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

from hk_equity_mcp.etom.source import file_sha256

ETOM_RESUME_IDENTITY_KEYS: Tuple[str, ...] = (
    "experiment",
    "condition",
    "method_identity_sha256",
    "hk_method_identity_sha256",
    "retrieval_method_sha256",
    "etom_method_identity_sha256",
    "system_prompt_sha256",
    "prompt_version",
    "parser_protocol_version",
    "server_top_k",
    "tool_top_k",
    "score_mode",
    "similarity_threshold",
    "max_extension_rounds",
    "use_summary",
    "generation_model",
    "generation_provider",
    "generation_base_url_host",
    "temperature",
    "max_tokens",
    "json_mode",
    "limits",
    "embedding_model",
    "embedding_provider",
    "embedding_base_url_host",
    "embedding_cache_namespace",
    "embedding_index_identity_sha256",
    "evaluator_file_sha256s",
    "pricing_sha256",
    "dataset_split",
    "mock",
    "source_kind",
    "source_repository",
    "source_commit",
    "source_pin_sha256",
    "dataset_file_sha256",
    "registry_sha256",
    "registry_fingerprint",
    "registry_tool_count",
    "registry_server_count",
    "sample_lock_sha256",
    "sample_sha256",
    "ordered_task_ids_sha256",
    "excluded_cost_pilot_lock_sha256",
    "question_ids",
)


def _canonical(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _canonical(value[k]) for k in sorted(value)}
    if isinstance(value, list):
        return [_canonical(v) for v in value]
    return value


def resume_identity_payload(meta: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key in ETOM_RESUME_IDENTITY_KEYS:
        if key in meta:
            out[key] = _canonical(meta[key])
    return out


def resume_identity_sha256(meta: Mapping[str, Any]) -> str:
    blob = json.dumps(
        resume_identity_payload(meta),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def assert_resume_compatible(existing: Mapping[str, Any], current: Mapping[str, Any]) -> None:
    """Fail closed if resume identity drifts. ``max_tasks`` is intentionally excluded."""
    mismatches = []
    for key in ETOM_RESUME_IDENTITY_KEYS:
        if key == "question_ids":
            continue
        if key not in existing and key not in current:
            continue
        if existing.get(key) != current.get(key):
            mismatches.append(key)
    prev_ids = list(existing.get("question_ids") or [])
    curr_ids = list(current.get("question_ids") or [])
    if len(curr_ids) < len(prev_ids):
        mismatches.append("question_ids_shorter")
    elif prev_ids != curr_ids[: len(prev_ids)]:
        mismatches.append("question_ids_prefix")
    if mismatches:
        raise RuntimeError(
            "Resume identity mismatch (fail closed): " + ", ".join(mismatches)
        )


def evaluator_file_hashes(root: Path) -> Dict[str, str]:
    files = {
        "evaluator.py": root
        / "src"
        / "hk_equity_mcp"
        / "etom"
        / "evaluator.py",
        "formal_metrics.py": root
        / "src"
        / "hk_equity_mcp"
        / "etom"
        / "formal_metrics.py",
        "cost.py": root / "src" / "hk_equity_mcp" / "etom" / "cost.py",
        "l4_metrics.py": root
        / "src"
        / "hk_equity_mcp"
        / "mcp_zero_etom"
        / "l4_metrics.py",
    }
    return {name: file_sha256(path) for name, path in files.items() if path.exists()}


def pricing_sha256(pricing: Mapping[str, Any]) -> str:
    blob = json.dumps(_canonical(dict(pricing)), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()
