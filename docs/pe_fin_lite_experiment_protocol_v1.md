# PE-Fin Lite v1 — Experiment Protocol

## Offline this round (required)

```bash
# Dry-run Smoke-5
PYTHONPATH=src python scripts/run_hk_unified_pe_fin_lite_experiment.py \
  --smoke5 --dry-run --allow-dirty \
  --output-dir outputs/pe_fin_lite_smoke5_dry_run_v1

# Mock Smoke-5 (offline LLM + hash embedding + mock MCP)
PYTHONPATH=src python scripts/run_hk_unified_pe_fin_lite_experiment.py \
  --smoke5 --mock --allow-dirty --confirm-retire-heldout \
  --output-dir outputs/pe_fin_lite_smoke5_mock_v1

# Dry-run Diagnostic-12
PYTHONPATH=src python scripts/run_hk_unified_pe_fin_lite_experiment.py \
  --diagnostic12 --dry-run --allow-dirty \
  --output-dir outputs/pe_fin_lite_diagnostic12_dry_run_v1
```

## Future real Smoke-5 (DO NOT RUN this round)

```bash
# 1) Build complete catalog embedding index (paid embedding only)
PYTHONPATH=src ../hk-equity-mcp-agent/.venv-aktools-discovery/bin/python \
  scripts/run_hk_unified_pe_fin_lite_experiment.py \
  --build-index-only \
  --embedding-env-file ../hk-equity-mcp-agent/.env.embedding-relay

# 2) Live Smoke-5
PYTHONPATH=src ../hk-equity-mcp-agent/.venv-aktools-discovery/bin/python \
  scripts/run_hk_unified_pe_fin_lite_experiment.py \
  --smoke5 --confirm-retire-heldout --require-complete-index \
  --env-file ../hk-equity-mcp-agent/.env.relay-v41 \
  --embedding-env-file ../hk-equity-mcp-agent/.env.embedding-relay \
  --output-dir outputs/pe_fin_lite_smoke5_live_v3
```

Never default to `.env.aliyun`.

## Smoke-5 live v3 offline audit (done)

Offline audit artifacts (raw predictions/metadata unchanged):

- `docs/pe_fin_lite_smoke5_live_v3_final_audit_v1/`
- `outputs/pe_fin_lite_smoke5_live_v3/final_audit_v1/` (local mirror; `outputs/` gitignored)

Frozen raw hashes:

- predictions.jsonl `512e9e59a68ab192b6c0818c25be7a60a0ab4ef0d73549852a889a74abc6122f`
- run_metadata.json `520cd970c783de85c50184767b1342e1a4e487adaa36ed478d737fac6d68e83a`

## Future Disclosure single-task live probe (DO NOT RUN until confirmed)

```bash
PYTHONPATH=src ../hk-equity-mcp-agent/.venv-aktools-discovery/bin/python \
  scripts/run_hk_unified_pe_fin_lite_experiment.py \
  --smoke5 --confirm-retire-heldout --require-complete-index \
  --ids news_ht_004 \
  --env-file ../hk-equity-mcp-agent/.env.relay-v41 \
  --embedding-env-file ../hk-equity-mcp-agent/.env.embedding-relay \
  --output-dir outputs/pe_fin_lite_disclosure_probe_news_ht_004_v1
```

Do not mutate Smoke-5/D12 manifests. Optional second probe: `--ids fin_cf_003` with a separate output dir.

## Identity locks

- generation_provider=`openai_compatible`
- generation_host=`api.zhizengzeng.com`
- generation_model=`deepseek-v4.1-flash`
- embedding_model=`text-embedding-3-small`

## Output isolation

Refuse dry-run→live resume, mock→live resume, subset mixing, and identity drift. Refuse overwrite of existing predictions without `--resume`.

## Manifests

- Smoke-5: `benchmarks/manifests/hk_pe_fin_lite_smoke5_v1.json`
- Diagnostic-12: `benchmarks/manifests/hk_pe_fin_lite_diagnostic12_v1.json`  
  (same ordered IDs / `ordered_id_hash` as APV-Fin Diagnostic-12)
