# PE-Fin Lite Diagnostic-12 Checkpoint Live v1 — Freeze + Hardening

## Freeze verdict

**engineering-observable but not method-valid for full HK-155**

| Flag | Value |
|---|---|
| `engineering_pass` | `true` |
| `explicitly_not_formal_accuracy` | `true` |
| method-valid for HK-155 | **false** |

## Actual result freeze (v1 live)

| Metric | Value |
|---|---|
| rows | 12 |
| full / partial / failed | 4 / 4 / 4 |
| answer_correct | 2/12 |
| LLM calls | 38 |
| tokens | 88726 |
| MCP | 28 |
| embedding queries | 12 |

## Raw artifact SHA256 (immutable)

| File | SHA256 |
|---|---|
| `predictions.jsonl` | `a84762f8d5123db68eaf0445794adfb1c8bced90cbc4b61e2bdddd3b39683131` |
| `run_metadata.json` | `14a2f5531a1a04e211c3e4fa4b2ef127eae2d1ec5a967c7b39cd5407e2238e61` |
| `error_summary.json` | `8e351c3e1c920ad1d6d9d075eb795c8638d0122173c9672ff0683ed9a23fc1dc` |
| `cost_summary.json` | `80dfb182536efd5dbaaafca2a78d34f09981c17437308300f987ae874bbeb300` |
| `grading_summary.json` | `f73939c82f618eace9f75ab1a27914db191a361c56c612cac4f7fd7109b78418` |
| `per_task_metrics.csv` | `0c3a8bedf0ed1380b9a1748994ed65329dbbe54c6877ef3f3d36349c4190ab79` |

Before == after for all raw artifacts. Raw predictions / run_metadata were **not** mutated.

## Runtime identity after hardening

- Method version bumped: `pe_fin_lite_v1` → `pe_fin_lite_v1.1`
- Prompt/schema hashes change only when content changes (prompt hardened for compute whitelist + temporal + reason JSON contract).
- Unchanged protocol knobs: TopK=5, Describe 2–3, fallback≤1, LLM≤8, MCP≤8, Monitor=0, global replan=0.
- Diagnostic-12 IDs/order/gold unchanged.

## Future Diagnostic-12 v2 command (DO NOT RUN in this round)

```bash
PYTHONPATH=src python scripts/run_hk_unified_pe_fin_lite_experiment.py \
  --diagnostic12 \
  --split exploratory_all \
  --require-complete-index \
  --allow-dirty \
  --env-file ../hk-equity-mcp-agent/.env.relay-v41 \
  --embedding-env-file ../hk-equity-mcp-agent/.env.embedding-relay \
  --output-dir outputs/pe_fin_lite_diagnostic12_checkpoint_live_v2
```

Method version will be read from runtime (`pe_fin_lite_v1.1`). Do not mutate Diagnostic-12 IDs/order/gold.
