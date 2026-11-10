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
  --output-dir outputs/pe_fin_lite_smoke5_live_v1
```

Never default to `.env.aliyun`.

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
