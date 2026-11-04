"""Resolve Real-MCP stdio executables without copying venvs or embedding secrets.

Lookup order (fail-closed when nothing resolves):
1. explicit environment variable
2. current worktree discovery venv
3. sibling main repo ``hk-equity-mcp-agent`` discovery venv
4. PATH
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import List, Optional, Sequence

from hk_equity_mcp.real_mcp.client import RealMcpError, resolve_command


def package_root() -> Path:
    """Repo root for the current worktree (…/hk-equity-mcp-agent-hk-unified)."""
    return Path(__file__).resolve().parents[3]


def sibling_main_repo(root: Optional[Path] = None) -> Path:
    root = root or package_root()
    return root.parent / "hk-equity-mcp-agent"


def candidate_mcp_commands(
    *,
    env_var: str,
    binary_name: str,
    venv_dirname: str,
    root: Optional[Path] = None,
) -> List[str]:
    """Ordered candidate command strings (absolute paths preferred)."""
    root = root or package_root()
    candidates: List[str] = []
    env = os.environ.get(env_var)
    if env and str(env).strip():
        candidates.append(str(env).strip())

    local = root / venv_dirname / "bin" / binary_name
    candidates.append(str(local))

    sibling = sibling_main_repo(root) / venv_dirname / "bin" / binary_name
    candidates.append(str(sibling))

    candidates.append(binary_name)
    # Preserve order, drop exact duplicates.
    seen = set()
    out: List[str] = []
    for c in candidates:
        if c in seen:
            continue
        seen.add(c)
        out.append(c)
    return out


def resolve_mcp_executable(
    *,
    env_var: str,
    binary_name: str,
    venv_dirname: str,
    root: Optional[Path] = None,
    label: Optional[str] = None,
    require: bool = True,
) -> Optional[str]:
    """Return an executable path/command.

    When ``require=True`` (default), raise RealMcpError if none resolve.
    When ``require=False``, return None (for pytest skip probes).
    """
    errors: List[str] = []
    for candidate in candidate_mcp_commands(
        env_var=env_var,
        binary_name=binary_name,
        venv_dirname=venv_dirname,
        root=root,
    ):
        try:
            return resolve_command(candidate)
        except RealMcpError as exc:
            errors.append(f"{candidate}: {exc}")
            continue
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{candidate}: {type(exc).__name__}: {exc}")
            continue
    if not require:
        return None
    name = label or binary_name
    raise RealMcpError(
        f"{name} executable not found (env={env_var}, local venv, sibling "
        f"hk-equity-mcp-agent venv, PATH). Tried: "
        + "; ".join(errors),
        error_type="command_not_found",
    )


def which_or_none(command: str) -> Optional[str]:
    if os.path.sep in command or command.startswith("."):
        path = Path(command)
        if not path.is_absolute():
            path = Path.cwd() / path
        return str(path) if path.exists() else None
    return shutil.which(command)


def first_existing(candidates: Sequence[str]) -> Optional[str]:
    for c in candidates:
        found = which_or_none(c)
        if found:
            return found
    return None
