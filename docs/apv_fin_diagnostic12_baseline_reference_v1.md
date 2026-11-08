# APV-Fin Diagnostic-12 Baseline Reference v1

**Descriptive frozen baseline reference only. Not an APV-Fin result. Do not treat as APV-Fin accuracy.**

Offline extraction from sibling worktree raw/audit artifacts for the same Diagnostic-12 IDs. Baselines were **not** re-run.

## Diagnostic-12 identity

Ordered IDs:

1. `fin_cf_003`
2. `fin_cf_011`
3. `fin_cf_020`
4. `fin_cf_023`
5. `fin_ht_016`
6. `xd_dev_002`
7. `xd_ht_007`
8. `xd_ht_014`
9. `mkt_dev_014`
10. `fin_cf_024`
11. `mkt_dev_001`
12. `news_ht_004`

- `manifest_hash`: `3b67b1f03746181bcb856bda60f8954a0db0ca8c8f42a881e9c93ff032ac5f17`
- `ordered_id_hash`: `96b633f00f2ec67daa47deae5c1e4252fff6cdb111a5759a121d66f5d09153d9`

## Fixture

`benchmarks/manifests/hk_apv_fin_diagnostic12_baseline_reference_v1.json`

## Sibling sources (read-only)

| Condition | Path | Diagnostic-12 coverage |
|-----------|------|------------------------|
| Operation-Visible | `../hk-equity-mcp-agent-hk-unified/outputs/hk_unified_exploratory155_operation_visible_v1` | 12/12 |
| MCP-Zero | `../hk-equity-mcp-agent-mcp-zero/outputs/hk_unified_mcp_zero_exploratory155_v2` | 12/12 |
| Flat All Tools | `../hk-equity-mcp-agent-flat-all-tools/outputs/hk_unified_flat_all_tools_155_v1` | 12/12 |
| MCP-Free | `../hk-equity-mcp-agent-hk-mcp-free/outputs/hk_unified_exploratory155_mcp_free_live_v1` | 12/12 |
| OV audit CSV | `docs/hk_unified_baseline_final_audit_v2/per_task_audit_v2.csv` | descriptive |

Raw prediction SHA256 and per-task status / grade fields are recorded in the JSON fixture. Paths are absolute in the fixture for provenance; relative sibling layout is assumed on this machine.
