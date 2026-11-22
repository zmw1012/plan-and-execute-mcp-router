"""Offline release documentation contract — no network, no live APIs."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"

PUBLIC_RUNNERS = [
    "run_hk_unified_mcp_free_experiment.py",
    "run_hk_unified_flat_all_tools_experiment.py",
    "run_hk_unified_mcp_zero_experiment.py",
    "run_hk_unified_operation_visible_experiment.py",
    "run_hk_unified_pe_fin_lite_experiment.py",
    "run_etom_mcp_zero.py",
    "run_etom_operation_visible_formal_100.py",
    "run_etom_pe_fin_replan_experiment.py",
]

# Flags that public docs recommend; must appear in the corresponding --help.
REQUIRED_FLAGS = {
    "run_hk_unified_mcp_free_experiment.py": [
        "--smoke",
        "--mock",
        "--dry-run",
        "--split",
        "--confirm-retire-heldout",
        "--output-dir",
        "--resume",
    ],
    "run_hk_unified_flat_all_tools_experiment.py": [
        "--smoke",
        "--mock",
        "--dry-run",
        "--confirm-retire-heldout",
        "--output-dir",
        "--allow-hkex-download",
    ],
    "run_hk_unified_mcp_zero_experiment.py": [
        "--smoke",
        "--mock",
        "--index-status",
        "--build-index-only",
        "--require-complete-index",
        "--confirm-retire-heldout",
    ],
    "run_hk_unified_operation_visible_experiment.py": [
        "--smoke",
        "--mock",
        "--dry-run",
        "--confirm-retire-heldout",
        "--output-dir",
    ],
    "run_hk_unified_pe_fin_lite_experiment.py": [
        "--smoke5",
        "--replan",
        "--replan-smoke5",
        "--replan-round-cap",
        "--mock",
        "--confirm-retire-heldout",
        "--require-complete-index",
        "--index-status",
        "--build-index-only",
    ],
    "run_etom_mcp_zero.py": [
        "--split",
        "--dry-run",
        "--mock",
        "--live",
        "--build-index-only",
        "--require-complete-index",
    ],
    "run_etom_operation_visible_formal_100.py": [
        "--mock",
        "--dry-run",
        "--live",
        "--require-complete-index",
        "--resume",
    ],
    "run_etom_pe_fin_replan_experiment.py": [
        "--formal100",
        "--allow-live-formal100",
        "--require-complete-index",
        "--engineering-smoke5",
        "--dry-run",
        "--mock",
        "--index-status",
        "--build-index-only",
        "--resume",
    ],
}

# ETOM RePlan must NOT advertise unsupported --live (dual-gate uses env + allow flag).
ETOM_REPLAN_FORBIDDEN_IN_HELP = ["--live"]

AUDIT_SCRIPTS = [
    "run_hk_unified_mcp_free_final_audit_v1.py",
    "run_hk_unified_flat_all_tools_final_audit_v1.py",
    "run_hk_unified_mcp_zero_final_audit_v1.py",
    "run_pe_fin_lite_r5_hk155_final_audit_v1.py",
    "run_pe_fin_replan_r7_hk155_final_audit_v1.py",
    "run_etom_mcp_zero_formal100_final_audit_v1.py",
    "run_pe_fin_replan_etom_formal100_final_audit_v1.py",
]

CANONICAL_METHODS = [
    "MCP-Free",
    "Flat-All-Tools",
    "MCP-Zero",
    "Operation-Visible",
    "MCP-PlanRoute Lite",
    "MCP-PlanRoute RePlan",
    "HK-155",
    "ETOM-100",
]

PUBLIC_DOCS = [
    ROOT / "README.md",
    ROOT / "REPRODUCIBILITY.md",
    ROOT / "DATA.md",
    ROOT / "CODE_MAP.md",
    ROOT / "scripts" / "README.md",
    ROOT / "docs" / "README.md",
]


def _help_text(script_name: str) -> str:
    import os

    path = SCRIPTS / script_name
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src") + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    proc = subprocess.run(
        [sys.executable, str(path), "--help"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert proc.returncode == 0, f"{script_name} --help failed: {proc.stderr}"
    return proc.stdout + proc.stderr


@pytest.mark.parametrize("name", PUBLIC_RUNNERS)
def test_public_runner_exists(name: str) -> None:
    assert (SCRIPTS / name).is_file(), f"missing public runner {name}"


@pytest.mark.parametrize("name", PUBLIC_RUNNERS)
def test_public_runner_help_exposes_required_flags(name: str) -> None:
    help_text = _help_text(name)
    for flag in REQUIRED_FLAGS[name]:
        assert flag in help_text, f"{name} --help missing required flag {flag}"


def test_etom_replan_help_has_no_live_flag() -> None:
    help_text = _help_text("run_etom_pe_fin_replan_experiment.py")
    # Lone --live must not appear; --allow-live-formal100 is the live gate flag.
    tokens = {tok.strip(",.") for tok in help_text.replace("|", " ").split()}
    assert "--live" not in tokens
    assert "--allow-live-formal100" in help_text
    assert "--formal100" in help_text


def test_reproducibility_etom_replan_command_has_no_unsupported_live() -> None:
    text = (ROOT / "REPRODUCIBILITY.md").read_text(encoding="utf-8")
    # Disallow the known bad pattern: pe_fin_replan ETOM formal100 with --live
    bad = re.compile(
        r"run_etom_pe_fin_replan_experiment\.py[^\n]*--live\b|"
        r"run_etom_pe_fin_replan_experiment\.py[\s\S]{0,200}?--formal100[\s\S]{0,120}?--live\b",
        re.MULTILINE,
    )
    assert not bad.search(text), "REPRODUCIBILITY.md still documents unsupported --live for ETOM RePlan"
    assert "PE_FIN_REPLAN_ETOM_ALLOW_LIVE=1" in text
    assert "--allow-live-formal100" in text
    assert "--formal100" in text


@pytest.mark.parametrize("name", AUDIT_SCRIPTS)
def test_documented_audit_scripts_exist(name: str) -> None:
    assert (SCRIPTS / name).is_file(), f"missing audit script {name}"


def test_canonical_method_names_in_code_map_and_readme() -> None:
    code_map = (ROOT / "CODE_MAP.md").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for name in CANONICAL_METHODS:
        assert name in code_map, f"CODE_MAP.md missing {name}"
        assert name in readme, f"README.md missing {name}"


def test_distribution_name_and_cli_aliases_documented() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'name = "plan-and-execute-mcp-router"' in pyproject
    assert "mcp-router-demo" in pyproject
    assert "mcp-router-eval" in pyproject
    assert "hk-equity-demo" in pyproject
    assert "hk-equity-eval" in pyproject
    # Import package path unchanged
    assert (ROOT / "src" / "hk_equity_mcp").is_dir()
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "plan-and-execute-mcp-router" in readme
    assert "hk_equity_mcp" in readme
    assert "mcp-router-demo" in readme


def test_public_docs_have_no_machine_home_paths() -> None:
    for path in PUBLIC_DOCS:
        text = path.read_text(encoding="utf-8")
        assert "/Users/" not in text, f"{path.name} contains absolute /Users/ path"
        assert "<<<<<<<<" not in text
        assert ">>>>>>>" not in text


def test_readme_citation_has_no_internal_editor_voice() -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "do not invent bibliographic fields" not in text.lower()
    assert "Related work and source repositories" in text
