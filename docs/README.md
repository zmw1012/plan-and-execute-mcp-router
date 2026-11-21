# Documentation index

## Current release documentation (start here)

| Doc | Role |
|-----|------|
| [../README.md](../README.md) | Project overview, methods, quick start |
| [../RESULTS.md](../RESULTS.md) | Authoritative metrics + provenance |
| [../REPRODUCIBILITY.md](../REPRODUCIBILITY.md) | Offline / live command matrix |
| [../CODE_MAP.md](../CODE_MAP.md) | Names ↔ packages ↔ entry points |
| [../DATA.md](../DATA.md) | Committed vs external data |
| [../scripts/README.md](../scripts/README.md) | Script groups |

Method cards still useful for implementers:

- [operation_visible_baseline.md](operation_visible_baseline.md)
- [mcp_zero_hk_baseline.md](mcp_zero_hk_baseline.md) / [mcp_zero_fidelity_audit.md](mcp_zero_fidelity_audit.md)
- [pe_fin_lite_method_v1.md](pe_fin_lite_method_v1.md) / [pe_fin_replan_v1/](pe_fin_replan_v1/)
- [experiment_design.md](experiment_design.md), [architecture.md](architecture.md)

---

## Benchmark documentation

HK-155 domain designs and freezes: `financial_*`, `market_*`, `news_*`, `cross_domain_*`.  
ETOM sampling / fidelity: `etom_*`, `mcp_zero_etom_fidelity.md`.

---

## Final result audits (sealed)

Prefer these folders for reported tips (see RESULTS provenance):

| Condition | Folder |
|-----------|--------|
| MCP-Free HK | `hk_unified_mcp_free_final_audit_v1/` |
| Flat HK | `hk_unified_flat_all_tools_final_audit_v1/` |
| MCP-Zero HK | `hk_unified_mcp_zero_final_audit_v1/` |
| OV / MCP-Zero / baselines rollup | `hk_unified_baseline_final_audit_v2/` |
| PlanRoute Lite HK | `pe_fin_lite_r5_hk155_final_audit_v1/` |
| PlanRoute RePlan HK | under `pe_fin_replan_v1/` + RePlan-7 audit scripts/docs |
| MCP-Zero ETOM | `etom_mcp_zero_formal100_final_audit_v1/` |
| OV ETOM | `etom_operation_visible_formal100_final_audit_v1/` |
| PlanRoute RePlan ETOM | `pe_fin_replan_etom_formal100_final_audit_v1/` |

---

## Frozen historical diagnostics

Checkpoint freezes, planner diagnostics, smoke engineering audits, retrieval readiness trees (`pe_fin_lite_diagnostic12_*`, `pe_fin_lite_planner_*`, `*_smoke5_*`, readiness folders). **Not** recommended usage guides—provenance only.

### Historical local absolute paths

Some frozen command/audit Markdown files still contain machine-local absolute paths from the original experiment host. Those strings are retained **only as provenance**. They are **not** the recommended reproduction interface—use `$PROJECT_ROOT` and env vars in [REPRODUCIBILITY.md](../REPRODUCIBILITY.md). Do not rewrite hash-sensitive audit payloads to “clean” those paths.

---

## Superseded experiments

| Area | Status |
|------|--------|
| `apv_fin_*` | Superseded exploratory lineage (not a final method) |
| Early financial MCP-Free 30-question pilots | Historical scaffolding |
| Invalidated / post-hoc-only ablations labeled as such in-folder | Keep; do not promote to headline results |

Do not delete historical files in the release documentation phase.
