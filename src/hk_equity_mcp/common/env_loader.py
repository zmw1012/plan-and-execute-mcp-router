"""Dual-profile env loading for generation vs embedding.

Shell-exported variables take precedence over values in env files.
Never print or return API key values.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence


GENERATION_REQUIRED = (
    "LLM_PROVIDER",
    "LLM_BASE_URL",
    "LLM_API_KEY",
    "LLM_MODEL",
)

EMBEDDING_REQUIRED = (
    "EMBEDDING_BASE_URL",
    "EMBEDDING_API_KEY",
    "EMBEDDING_MODEL",
)


def load_env_file(path: Path) -> None:
    """Load KEY=VALUE pairs with setdefault (shell wins). Never logs values."""
    if not path.exists():
        raise FileNotFoundError(
            f"Env profile not found: {path}. Provide a local git-ignored profile."
        )
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[7:].strip()
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                value = value[1:-1]
            if key:
                os.environ.setdefault(key, value)


def resolve_env_file(
    explicit: Optional[str],
    env_var_name: str,
    default_name: str,
    repo_root: Path,
) -> Path:
    raw = explicit or os.environ.get(env_var_name) or default_name
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = (repo_root / path).resolve()
    return path


def missing_keys(required: Iterable[str]) -> List[str]:
    missing: List[str] = []
    for key in required:
        value = os.environ.get(key)
        if value is None or str(value).strip() == "":
            missing.append(key)
    return missing


def require_keys(required: Sequence[str], profile_label: str) -> None:
    missing = missing_keys(required)
    if missing:
        raise RuntimeError(
            f"Missing required {profile_label} env var(s): {missing}. "
            "No silent fallback to another provider or model."
        )


def load_generation_profile(
    repo_root: Path,
    env_file: Optional[str] = None,
) -> Dict[str, str]:
    """Load ENV_FILE (.env.aliyun by default) and require generation keys."""
    path = resolve_env_file(env_file, "ENV_FILE", ".env.aliyun", repo_root)
    load_env_file(path)
    require_keys(GENERATION_REQUIRED, "generation")
    model = os.environ["LLM_MODEL"].strip()
    if model.lower() == "auto":
        raise RuntimeError("LLM_MODEL=auto is not allowed; pin an explicit model.")
    return {
        "profile_file": path.name,
        "provider": os.environ["LLM_PROVIDER"].strip(),
        "base_url": os.environ["LLM_BASE_URL"].strip().rstrip("/"),
        "model": model,
        # api_key intentionally omitted from returned metadata
    }


def load_embedding_profile(
    repo_root: Path,
    env_file: Optional[str] = None,
) -> Dict[str, str]:
    """Load EMBEDDING_ENV_FILE (.env.embedding-relay) and require embedding keys."""
    path = resolve_env_file(
        env_file, "EMBEDDING_ENV_FILE", ".env.embedding-relay", repo_root
    )
    load_env_file(path)
    require_keys(EMBEDDING_REQUIRED, "embedding")
    model = os.environ["EMBEDDING_MODEL"].strip()
    if model.lower() == "auto":
        raise RuntimeError("EMBEDDING_MODEL=auto is not allowed; pin an explicit model.")
    return {
        "profile_file": path.name,
        "base_url": os.environ["EMBEDDING_BASE_URL"].strip().rstrip("/"),
        "model": model,
        # api_key intentionally omitted
    }


def generation_public_metadata() -> Dict[str, Optional[str]]:
    return {
        "llm_provider": os.environ.get("LLM_PROVIDER"),
        "llm_model": os.environ.get("LLM_MODEL"),
        "llm_base_url": os.environ.get("LLM_BASE_URL"),
    }


def embedding_public_metadata() -> Dict[str, Optional[str]]:
    return {
        "embedding_model": os.environ.get("EMBEDDING_MODEL"),
        "embedding_base_url": os.environ.get("EMBEDDING_BASE_URL"),
    }
