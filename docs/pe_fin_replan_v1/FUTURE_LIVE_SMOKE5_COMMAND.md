# Future LIVE RePlan Smoke-5 — do not run until approved

**Live engineering smoke only.**  
**Not accuracy. Cannot replace Diagnostic-20. Cannot replace HK-155.**

Prior Diagnostic-20 live v1 (`outputs/pe_fin_replan_diagnostic20_live_v1/`) remains
**INVALIDATED** and must not be resumed.

## Identity

- `condition=pe_fin_replan`
- `method_version=pe_fin_replan_v1_2`
- `protocol_version=planner_executor_replanner_contract_v2`
- `subset_kind=replan_smoke5`
- Manifest: `benchmarks/manifests/hk_pe_fin_replan_smoke5_v1.json`

Ordered IDs:

1. `mkt_dev_003`
2. `news_dev_004`
3. `fin_cf_019`
4. `xd_dev_008`
5. `xd_dev_004`

## Exact command (when approved)

```bash
cd "/Users/wzm/Desktop/科大/课程/MSBD5018_NLP/group project/hk-equity-mcp-agent-pe-fin-lite"

PYTHONPATH=src python scripts/run_hk_unified_pe_fin_lite_experiment.py \
  --split exploratory_all \
  --replan-smoke5 \
  --replan \
  --confirm-retire-heldout \
  --allow-hkex-download \
  --env-file .env \
  --embedding-env-file .env \
  --require-complete-index \
  --output-dir outputs/pe_fin_replan_smoke5_live_v2
```

## Resume isolation

Refuse mixing:

- Smoke-5 ↔ Diagnostic-20
- invalidated live v1 ↔ v2 (`method_version` / `protocol_version`)
- mock/dry-run ↔ live (`mock` / `dry_run` resume keys)

## Explicitly not this round

- No real Smoke-5 until user confirms
- No Diagnostic-20 live v2 yet
- No HK-155 / ETOM / push / PR / merge
