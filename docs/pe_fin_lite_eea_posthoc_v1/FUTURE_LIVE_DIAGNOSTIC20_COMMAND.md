# Future LIVE Diagnostic-20 (EEA) — do not run until approved

**PE-Fin Lite v1.2 formal HK-155 remains the primary frozen result.**  
**PE-Fin Lite v1.2-EEA is a post-hoc exploratory engineering ablation.**  
Diagnostic-20 is **not** formal accuracy and **cannot** replace HK-155 or reserved85.

## Prerequisites

- Clean git tree (or explicit `--allow-dirty` only if intentionally used)
- Complete shared embedding index (same identity as formal R5 run)
- Valid generation + embedding env files
- `--allow-hkex-download` only if disclosure download is required and approved

## Exact command (when approved)

```bash
cd "/Users/wzm/Desktop/科大/课程/MSBD5018_NLP/group project/hk-equity-mcp-agent-pe-fin-lite"

PYTHONPATH=src python scripts/run_hk_unified_pe_fin_lite_experiment.py \
  --split exploratory_all \
  --diagnostic20-eea \
  --eea \
  --confirm-retire-heldout \
  --allow-hkex-download \
  --env-file .env \
  --embedding-env-file .env \
  --require-complete-index \
  --output-dir outputs/pe_fin_lite_eea_diagnostic20_live_v1
```

Adjust `--env-file` / `--embedding-env-file` to the project’s live credential paths.

## Explicitly not this round

- No live Diagnostic-20 until user confirms
- No HK-155 rerun
- No ETOM
- No reserved85 development
