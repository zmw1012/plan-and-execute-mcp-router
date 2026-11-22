"""Proposed-method generation provider identity for pe_fin_replan_etom.

Independent of ``hk_equity_mcp.etom.EXPECTED_LLM_PROVIDER`` (historical aliyun
Operation-Visible baseline). Live generation for this condition must match the
versioned YAML profile exactly: openai_compatible + deepseek-v4.1-flash +
api.zhizengzeng.com.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import yaml

PROFILE_REL = "configs/operations/pe_fin_replan_etom_generation_provider_v1.yaml"
PROFILE_ID = "pe_fin_replan_etom_zhizengzeng_v1"
EXPECTED_PROVIDER = "openai_compatible"
EXPECTED_HOST = "api.zhizengzeng.com"
EXPECTED_MODEL = "deepseek-v4.1-flash"
EXPECTED_ENV_ROLE = "relay_v41"
EXPECTED_ENV_FILE_NAME = ".env.relay-v41"

# Forbidden for proposed live identity (historical baseline / mocks).
FORBIDDEN_LIVE_PROVIDERS = frozenset(
    {
        "aliyun",
        "mock",
        "hash",
        "scripted",
        "fixture",
    }
)


class GenerationProviderIdentityError(RuntimeError):
    """Fail-closed generation provider / host / model mismatch."""


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_obj(obj: Any) -> str:
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return _sha256_bytes(blob.encode("utf-8"))


def host_from_base_url(base_url: Optional[str]) -> Optional[str]:
    if not base_url:
        return None
    parsed = urlparse(str(base_url).strip())
    host = (parsed.netloc or "").strip().lower()
    if not host and "://" not in str(base_url):
        # bare host or host/path
        host = str(base_url).strip().split("/", 1)[0].lower()
    return host or None


def profile_path(root: Path) -> Path:
    return Path(root) / PROFILE_REL


def load_generation_provider_profile(root: Path) -> Dict[str, Any]:
    path = profile_path(root)
    raw = path.read_bytes()
    data = yaml.safe_load(raw.decode("utf-8"))
    if not isinstance(data, dict):
        raise GenerationProviderIdentityError(f"invalid provider profile: {path}")
    required = {
        "profile_id": PROFILE_ID,
        "provider": EXPECTED_PROVIDER,
        "provider_host": EXPECTED_HOST,
        "model": EXPECTED_MODEL,
        "env_profile_role": EXPECTED_ENV_ROLE,
    }
    for key, want in required.items():
        got = data.get(key)
        if got != want:
            raise GenerationProviderIdentityError(
                f"provider profile {key}={got!r} != frozen {want!r}"
            )
    scope = list(data.get("condition_scope") or [])
    if "pe_fin_replan_etom" not in scope:
        raise GenerationProviderIdentityError(
            "provider profile condition_scope must include pe_fin_replan_etom"
        )
    out = dict(data)
    try:
        out["profile_path"] = str(path.relative_to(Path(root)))
    except ValueError:
        out["profile_path"] = str(path)
    out["profile_file_sha256"] = _sha256_bytes(raw)
    out["profile_hash"] = generation_provider_profile_hash(out)
    return out


def generation_provider_profile_hash(profile: Dict[str, Any]) -> str:
    """Content hash of the semantic identity fields (not path/file sha)."""
    payload = {
        "profile_id": profile.get("profile_id"),
        "profile_version": profile.get("profile_version"),
        "provider": profile.get("provider"),
        "provider_host": profile.get("provider_host"),
        "model": profile.get("model"),
        "env_profile_role": profile.get("env_profile_role"),
        "expected_env_file_name": profile.get("expected_env_file_name"),
        "condition_scope": list(profile.get("condition_scope") or []),
    }
    return _sha256_obj(payload)


def assert_live_generation_identity(
    *,
    provider: str,
    model: str,
    host: Optional[str],
    profile: Optional[Dict[str, Any]] = None,
    env_file_name: Optional[str] = None,
) -> None:
    """Fail closed unless provider/model/host match the proposed relay profile."""
    if profile is not None:
        if profile.get("profile_id") != PROFILE_ID:
            raise GenerationProviderIdentityError(
                f"generation_provider_profile_id mismatch: {profile.get('profile_id')!r}"
            )
        if profile.get("provider") != EXPECTED_PROVIDER:
            raise GenerationProviderIdentityError("profile provider drifted")
        if profile.get("provider_host") != EXPECTED_HOST:
            raise GenerationProviderIdentityError("profile host drifted")
        if profile.get("model") != EXPECTED_MODEL:
            raise GenerationProviderIdentityError("profile model drifted")

    prov = str(provider or "").strip()
    if prov in FORBIDDEN_LIVE_PROVIDERS or prov != EXPECTED_PROVIDER:
        raise GenerationProviderIdentityError(
            f"generation provider mismatch for pe_fin_replan_etom: expected "
            f"{EXPECTED_PROVIDER!r}, got {prov!r} (no Aliyun / mock live fallback)"
        )
    mod = str(model or "").strip()
    if mod != EXPECTED_MODEL:
        raise GenerationProviderIdentityError(
            f"generation model mismatch: expected {EXPECTED_MODEL!r}, got {mod!r}"
        )
    if host is None or str(host).strip() == "":
        raise GenerationProviderIdentityError(
            "generation host is null/empty; refusing live run"
        )
    if str(host).strip().lower() != EXPECTED_HOST:
        raise GenerationProviderIdentityError(
            f"generation host mismatch: expected {EXPECTED_HOST!r}, got {host!r}"
        )
    if env_file_name is not None:
        name = Path(env_file_name).name
        if name != EXPECTED_ENV_FILE_NAME:
            raise GenerationProviderIdentityError(
                f"generation env file must be {EXPECTED_ENV_FILE_NAME!r}, got {name!r}"
            )


def public_generation_identity(profile: Dict[str, Any], *, host: str) -> Dict[str, Any]:
    """Metadata-safe identity fields (never includes API keys)."""
    return {
        "generation_provider_profile_id": profile["profile_id"],
        "generation_provider_profile_hash": profile["profile_hash"],
        "generation_provider_profile_file_sha256": profile["profile_file_sha256"],
        "generation_provider": profile["provider"],
        "generation_model": profile["model"],
        "generation_host": host,
        "generation_env_profile_role": profile.get("env_profile_role"),
    }
