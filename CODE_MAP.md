# Code map

Canonical **public** method names vs **legacy** import paths. Legacy packages are **not** renamed in this release (import compatibility, historical commits, audit identities).

## Distribution / CLI

| Layer | Name |
|-------|------|
| Setuptools / distribution | `plan-and-execute-mcp-router` |
| Canonical console scripts | `mcp-router-demo`, `mcp-router-eval` |
| Legacy console scripts | `hk-equity-demo`, `hk-equity-eval` |
| Python import package | `hk_equity_mcp` (under `src/hk_equity_mcp/`; not physically renamed) |

---

## Baselines

| Canonical name | Primary package / module path | Notes |
|----------------|-------------------------------|-------|
| MCP-Free | `src/hk_equity_mcp/experiment/unified_mcp_free_runner.py` (+ agent closed-book path) | No MCP tools |
| Flat-All-Tools | `src/hk_equity_mcp/experiment/unified_flat_runner.py` | Full schema exposure |
| MCP-Zero | `src/hk_equity_mcp/mcp_zero/` | Hierarchical Server→Tool retrieval |
| Operation-Visible | `src/hk_equity_mcp/agent/operation_visible.py` (+ unified OV runner) | Find→Describe→Invoke ReAct |

ETOM adapters for baselines:

- MCP-Zero × ETOM: `src/hk_equity_mcp/mcp_zero_etom/`, `src/hk_equity_mcp/etom/`
- Operation-Visible × ETOM: `src/hk_equity_mcp/etom/` formal OV runner + `scripts/run_etom_operation_visible_*.py`

---

## Proposed methods

| Canonical name | Legacy package path | Why path kept |
|----------------|---------------------|---------------|
| MCP-PlanRoute Lite | `src/hk_equity_mcp/pe_fin_lite/` | Historical `method_version` / imports / audits |
| MCP-PlanRoute RePlan (HK-155) | `src/hk_equity_mcp/pe_fin_replan/` | Same |
| MCP-PlanRoute RePlan (ETOM) | `src/hk_equity_mcp/pe_fin_replan_etom/` | ETOM routing adapter |

Unified experiment runners:

- Lite / RePlan HK: `src/hk_equity_mcp/experiment/unified_pe_fin_lite_runner.py` (RePlan via `--replan`)
- RePlan ETOM: `src/hk_equity_mcp/experiment/unified_pe_fin_replan_etom_runner.py`

Shared experiment infrastructure historically labeled **hk_unified** maps to **HK-155** shared runners under `src/hk_equity_mcp/experiment/unified_*.py`.

---

## Shared modules

| Concern | Location |
|---------|----------|
| Retrieval (R5 lexical+cosine) | `pe_fin_lite` retrieval + OV lexical helpers; retrieval audit under `experiment/pe_fin_lite_retrieval_audit/` |
| Real MCP gateways / catalogs | `src/hk_equity_mcp/real_mcp/` |
| Evaluation / graders | `src/hk_equity_mcp/evaluation/` |
| Benchmark manifests | `benchmarks/` |
| Audit scripts | `scripts/run_*_final_audit*.py`, `scripts/audit_*.py` |
| Tests | `tests/` (+ `tests/artifact_gates.py`) |
| Configs | `configs/` |
| ETOM source helpers | `src/hk_equity_mcp/etom/` |

---

## Entry-point matrix

| Condition | Runner script | Config / identity | Benchmark | Evaluator / audit | Principal tests | Result / audit docs |
|-----------|---------------|-------------------|-----------|-------------------|-----------------|---------------------|
| MCP-Free HK | `scripts/run_hk_unified_mcp_free_experiment.py` | unified MCP-Free config | HK-155 exploratory | `run_hk_unified_mcp_free_final_audit_v1.py` | `tests/test_*mcp_free*` | `docs/` MCP-Free final audit |
| Flat HK | `scripts/run_hk_unified_flat_all_tools_experiment.py` | flat unified config | HK-155 | `run_hk_unified_flat_all_tools_final_audit_v1.py` | `tests/test_*flat*` | Flat final audit docs |
| MCP-Zero HK | `scripts/run_hk_unified_mcp_zero_experiment.py` | `mcp_zero` method card | HK-155 | `run_hk_unified_mcp_zero_final_audit_v1.py` | `tests/test_mcp_zero_*` | `docs/mcp_zero_*` |
| OV HK | `scripts/run_hk_unified_operation_visible_experiment.py` | OV configs | HK-155 | unified OV audits in `docs/` | `tests/test_operation_visible_*` | `docs/operation_visible_*` |
| PlanRoute Lite HK | `scripts/run_hk_unified_pe_fin_lite_experiment.py` | `pe_fin_lite` | HK-155 | `run_pe_fin_lite_r5_hk155_final_audit_v1.py` | `tests/test_pe_fin_lite_*` | `docs/pe_fin_lite_*` |
| PlanRoute RePlan HK | same + `--replan` | `pe_fin_replan` | HK-155 | `run_pe_fin_replan_r7_hk155_final_audit_v1.py` | `tests/test_pe_fin_replan_*` | `docs/pe_fin_replan_*` |
| MCP-Zero ETOM | `scripts/run_etom_mcp_zero.py` / `run_etom_mcp_zero_formal_100.sh` | mcp_zero_etom | ETOM-100 lock | `run_etom_mcp_zero_formal100_final_audit_v1.py` | `tests/test_mcp_zero_etom*` / etom | `docs/mcp_zero_etom_*` |
| OV ETOM | `scripts/run_etom_operation_visible_formal_100.py` (+ `.sh`) | OV etom | ETOM-100 lock | OV formal100 audit docs | `tests/test_etom*` OV | `docs/` OV ETOM audit |
| PlanRoute RePlan ETOM | `scripts/run_etom_pe_fin_replan_experiment.py` | `pe_fin_replan_etom` | ETOM-100 lock | `run_pe_fin_replan_etom_formal100_final_audit_v1.py` | `tests/test_pe_fin_replan_etom*` | `docs/pe_fin_replan_etom_*` |

There is **no** single unified multi-method CLI in this release; use the matrix above. Do not invent a behavior-changing wrapper solely for naming.

---

## Legacy / superseded code

| Name | Status |
|------|--------|
| **APV-Fin** (`apv_fin`, `run_hk_unified_apv_fin_experiment.py`) | Superseded exploratory lineage — **not** a final reported method |
| Intermediate diagnostics (Diagnostic-12/20, EEA-only ablations, invalidated pilots) | Historical; keep for provenance |
| Old README “HK Equity MCP Agent” mock framing | Replaced by root [README.md](README.md) |

Do not delete these packages in the documentation phase; mark them superseded when navigating.
