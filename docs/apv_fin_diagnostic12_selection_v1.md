# APV-Fin Diagnostic-12 Selection v1

Pointer-only subset of frozen HK-155 (`hk_unified_exploratory_all_v1`).
**Does not copy or modify** question/gold text.

- Manifest: `benchmarks/manifests/hk_apv_fin_diagnostic12_v1.json`
- Baseline audit source: `docs/hk_unified_baseline_final_audit_v2`

## Distribution

| Bucket | N | IDs |
|---|---:|---|
| Financial multi-step evidence chain | 3 | `fin_cf_003`, `fin_cf_011`, `fin_cf_020` |
| Financial calculation / period / basis | 2 | `fin_cf_023`, `fin_ht_016` |
| Cross-domain sequential dependency | 2 | `xd_dev_002`, `xd_ht_007` |
| Cross-domain parallel / mixed | 1 | `xd_ht_014` |
| Refusal / clarification / temporal gap | 2 | `mkt_dev_014`, `fin_cf_024` |
| Market easy regression sentinel | 1 | `mkt_dev_001` |
| News easy regression sentinel | 1 | `news_ht_004` |

## Issuer coverage

Tencent, HSBC, Alibaba are all represented.

## Selection principles applied

1. Prefer shared OV failure modes: `agent_selection`, `answer_synthesis_calculation`, `resource_limit`, `feasibility_refusal`.
2. No task-id-specific APV-Fin logic may be written from these answers.
3. Include successful exact-tool sentinels (`mkt_dev_001`, `news_ht_004`) for regression.
4. Record domain / difficulty / answerability / dependency / historical OV outcomes in the manifest.
5. Deterministic `manifest_hash` and `ordered_id_hash` stored in the JSON.

## Hashes

See manifest fields:

- `ordered_id_hash`
- `manifest_hash`

## Usage (offline)

```bash
PYTHONPATH=src python scripts/run_hk_unified_apv_fin_experiment.py \
  --diagnostic12 --dry-run --allow-dirty \
  --split exploratory_all --confirm-retire-heldout \
  --output-dir outputs/apv_fin_diagnostic12_dry_v1
```

Formal Diagnostic-12 Agent runs are **out of scope** for this implementation round.
