# APV-Fin Real Smoke-4 Selection v1

Pointer-only preflight subset of Diagnostic-12. **Not a formal accuracy result.**

## Parent identity (unchanged)

- Parent: `benchmarks/manifests/hk_apv_fin_diagnostic12_v1.json`
- `parent_manifest_hash`: `3b67b1f03746181bcb856bda60f8954a0db0ca8c8f42a881e9c93ff032ac5f17`
- `parent_ordered_id_hash`: `96b633f00f2ec67daa47deae5c1e4252fff6cdb111a5759a121d66f5d09153d9`

## Selection policy

Deterministic: first item in Diagnostic-12 order for each required bucket.

| Bucket | Task ID | Reason |
|--------|---------|--------|
| Financial multi-step evidence chain | `fin_cf_003` | First evidence-chain item in Diagnostic-12 order |
| Financial calculation / period / basis | `fin_cf_023` | First calculation/period/basis item in Diagnostic-12 order |
| Cross-domain sequential dependency | `xd_dev_002` | First cross-domain sequential item in Diagnostic-12 order |
| Refusal / clarification | `mkt_dev_014` | First refusal/clarification item in Diagnostic-12 order |

## Smoke-4 hashes

- Manifest: `benchmarks/manifests/hk_apv_fin_real_smoke4_v1.json`
- `ordered_task_ids`: `fin_cf_003`, `fin_cf_023`, `xd_dev_002`, `mkt_dev_014`
- `ordered_id_hash`: `ff9d5baf493fdc91670a038e8458397a633d20351ebfa4765a707ea59d2a03bf`
- `manifest_hash`: `c675d5d59ac1289fc1e3d4b6fb45e7c836ffb0d02f339805b9088cc3c9c87c8b`

Questions/gold are not copied; loader selects frozen exploratory_all rows by ID.

## Formal output directory

`outputs/apv_fin_real_smoke4_v1` (do not mix with Diagnostic-12 live or dry-run dirs).
