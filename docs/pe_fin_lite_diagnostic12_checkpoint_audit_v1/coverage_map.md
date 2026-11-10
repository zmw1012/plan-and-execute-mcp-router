# Diagnostic-12 → Checkpoint coverage map (v1)

Manifest: `benchmarks/manifests/hk_pe_fin_lite_diagnostic12_v1.json`  
`manifest_hash`: `6aaae881a38746fce65459232dab3c54af1d9d4f69b5a77a5ed56b10162f391f`  
`ordered_id_hash`: `96b633f00f2ec67daa47deae5c1e4252fff6cdb111a5759a121d66f5d09153d9`

IDs/order frozen (pointer-only into HK-155; no gold/question mutation).

| Need | Covered by |
|------|------------|
| Financial | fin_cf_003, fin_cf_011, fin_cf_020, fin_cf_023, fin_ht_016, fin_cf_024 |
| Market | mkt_dev_014, mkt_dev_001 |
| News | news_ht_004 |
| Cross-domain | xd_dev_002, xd_ht_007, xd_ht_014 |
| Retrieve | all evidence-answerable / structured retrieve tasks |
| Compute | fin_cf_023 (numerical_reasoning), fin_ht_016 (multi_step_calculation) — natural trigger; else N/A + unit fixture |
| Reason | may be N/A on some plans; unit fixtures cover unsupported-claim |
| Disclosure composite | fin_cf_011 / xd_ht_007 (HKEX/annual-report style) when planner emits disclosure workflow |
| Fallback / unavailable | mkt_dev_014, fin_cf_024 (refusal/clarification buckets) |
| Final synthesis | all 12 |

If a checkpoint is not naturally triggered on a live task, status=`N/A` and offline fixtures/unit tests cover the failure modes. **Do not** rewrite Diagnostic-12 questions to force triggers.
