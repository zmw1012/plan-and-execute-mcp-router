# Scripts index

Historical runner script **filenames are not renamed** in this phase. Use canonical method names below; map to files via [CODE_MAP.md](../CODE_MAP.md).

**Recommended release entry points** are marked with ★.

---

## HK-155 baselines

| Method | ★ Entry point | Related |
|--------|---------------|---------|
| MCP-Free | ★ `run_hk_unified_mcp_free_experiment.py` | `run_hk_unified_mcp_free_final_audit_v1.py`, `run_mcp_free_baseline.py` (older) |
| Flat-All-Tools | ★ `run_hk_unified_flat_all_tools_experiment.py` | `run_hk_unified_flat_all_tools_final_audit_v1.py` |
| MCP-Zero | ★ `run_hk_unified_mcp_zero_experiment.py` | `run_hk_unified_mcp_zero_final_audit_v1.py` |
| Operation-Visible | ★ `run_hk_unified_operation_visible_experiment.py` | OV audit docs under `docs/` |

Shared postprocess: `postprocess_hk_unified_run.py`, `run_hk_unified_final_audit_v2.py`.

---

## HK-155 proposed methods

| Method | ★ Entry point | Notes |
|--------|---------------|-------|
| MCP-PlanRoute Lite | ★ `run_hk_unified_pe_fin_lite_experiment.py` | Default Lite path |
| MCP-PlanRoute RePlan | ★ same + `--replan` / `--replan-smoke5` | Round cap `--replan-round-cap {4,7}` |

Audits: `run_pe_fin_lite_r5_hk155_final_audit_v1.py`, `run_pe_fin_replan_r7_hk155_final_audit_v1.py`, `audit_pe_fin_lite_*.py`, `run_pe_fin_lite_retrieval_audit.py`, `run_pe_fin_replan_tool_readiness_audit_v1.py`.

---

## ETOM-100 baselines

| Method | ★ Entry point | Related |
|--------|---------------|---------|
| MCP-Zero | ★ `run_etom_mcp_zero.py` / `run_etom_mcp_zero_formal_100.sh` | `run_etom_mcp_zero_formal100_final_audit_v1.py`, pilot scripts |
| Operation-Visible | ★ `run_etom_operation_visible_formal_100.py` / `.sh` | `run_etom_operation_visible_pilot*.py`, `audit_etom_operation_visible_formal100_final_v1.py` |

Helpers: `sample_etom_formal_100.py`, `sample_etom_cost_pilot.py`, `convert_etom_registry.py`, `evaluate_etom_empty_route_control.py`.

---

## ETOM-100 proposed method

| Method | ★ Entry point | Audit |
|--------|---------------|-------|
| MCP-PlanRoute RePlan | ★ `run_etom_pe_fin_replan_experiment.py` | `run_pe_fin_replan_etom_formal100_final_audit_v1.py` |

---

## Audits

All `*_final_audit*.py` and `audit_*.py` scripts. Prefer the method-specific final audit named in [RESULTS.md](../RESULTS.md) provenance. Many require `MCP_ROUTER_*_ARTIFACT_ROOT`.

---

## Index building

| Area | Script / flag |
|------|----------------|
| HK MCP-Zero | `run_hk_unified_mcp_zero_experiment.py --build-index-only` / `--index-status` |
| HK PlanRoute | `run_hk_unified_pe_fin_lite_experiment.py --build-index-only` / `--index-status` |
| ETOM MCP-Zero | `run_etom_mcp_zero_formal_100.sh build-index` / `index-status` |
| ETOM PlanRoute | `run_etom_pe_fin_replan_experiment.py --build-index-only` / `--index-status` |

ETOM source fetch (network): `fetch_etom_source.sh`.

---

## Development diagnostics

Benchmark builders/validators (`build_*`, `validate_*`), MCP discovery smokes (`*_mcp_discovery*.py`, `run_real_mcp_smoke.py`), and **superseded** APV-Fin (`run_hk_unified_apv_fin_experiment.py`, `run_apv_fin_diagnostic12_final_audit_v1.py`) — historical only; not final methods.

---

## CLI discovery

```bash
"$PYTHON" scripts/<runner>.py --help
```

Do not invent unsupported flags. See [REPRODUCIBILITY.md](../REPRODUCIBILITY.md).
