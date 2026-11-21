"""Shared pytest fixtures for clean-clone offline suites.

Generation/embedding loaders use ``os.environ.setdefault``, so one test that
loads a fixture ``.env`` can leak ``LLM_*`` / ``EMBEDDING_*`` into later tests.
Clear those keys before every test so dry-run/mock host checks stay deterministic.
"""

from __future__ import annotations

import pytest

_PROFILE_ENV_KEYS = (
    "LLM_PROVIDER",
    "LLM_BASE_URL",
    "LLM_API_KEY",
    "LLM_MODEL",
    "EMBEDDING_BASE_URL",
    "EMBEDDING_API_KEY",
    "EMBEDDING_MODEL",
    "ENV_FILE",
    "EMBEDDING_ENV_FILE",
)


@pytest.fixture(autouse=True)
def _isolate_llm_embedding_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in _PROFILE_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
