# Data

What is committed in-repo vs what must be supplied externally for full reproduction.

---

## Committed data

| Category | Location | Notes |
|----------|----------|-------|
| Benchmark manifests / locks | `benchmarks/` | HK-155 pointers, smoke/diagnostic subsets, ETOM `formal_100_v1.lock.json`, source pins |
| Frozen sample IDs / run orders | under `benchmarks/` and audit JSON in `docs/` | Do not resample formal locks |
| Small audit summaries | `docs/*_final_audit*/`, identity check JSON | Metrics + hashes; not full raw predictions |
| Schemas / configs | `configs/`, operation YAMLs | Runtime method cards |
| Offline fixtures | `tests/` fixtures; limited `data/mock/` | For mock/dry-run |

---

## External data

| Category | Typical location | How to attach |
|----------|------------------|---------------|
| ETOM source checkout | `data/etom/raw` (gitignored) or external path | `scripts/fetch_etom_source.sh` → pin `790f28727f567b88918420d163c524d46644a694`; set `MCP_ROUTER_ETOM_DATA_ROOT` |
| HK market / disclosure caches | local caches under worktree / artifact root | Produced by live MCP + optional `--allow-hkex-download` |
| Frozen raw experiment artifacts | large `outputs/` trees | Set `MCP_ROUTER_HK_ARTIFACT_ROOT` / `MCP_ROUTER_ETOM_ARTIFACT_ROOT` |
| Embedding indexes | `.cache/mcp_zero_*`, PE-Fin Lite cache dirs | Build with `--build-index-only` or copy frozen cache |
| Large PDFs / downloaded filings | outside git | Not required for offline unit tests |

---

## Data availability

Frozen experiment artifacts for course submission are published outside Git (HKUST OneDrive). Download the ZIP from the shared folder below, verify the SHA256, then point the `MCP_ROUTER_*` environment variables at the extracted trees.

### Final archive (published)

```text
ARCHIVE_URL: https://hkustconnect-my.sharepoint.com/:f:/r/personal/zwanas_connect_ust_hk/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fzwanas%5Fconnect%5Fust%5Fhk%2FDocuments%2Fnlp%5Fgroup%5Fproject%5Fmcp%5Frouter&share=cgq4kpr%5FeXFMQJtANKIfdI9QEgUBWU4Ayqt3ZHyIG9UwNN%2DB%2Dw
ARCHIVE_FILE: mcp-router-release-artifacts-v1.zip
ARCHIVE_SHA256: 8c883917580e372dc814169b4b81b24aa315e580bf54b6f815808f21ba7c069b
```

Verify the downloaded ZIP before extract:

```bash
echo "8c883917580e372dc814169b4b81b24aa315e580bf54b6f815808f21ba7c069b  mcp-router-release-artifacts-v1.zip" | shasum -a 256 -c -
```

### Archive contents / layout

The published ZIP expands to `mcp-router-release-artifacts-v1/` with sealed raw runs and final audits for all nine formal conditions (no INVALIDATED / smoke / mock / dry-run trees; no credentials):

```text
mcp-router-release-artifacts-v1/
  README.txt
  ARTIFACT_INVENTORY.tsv
  ARTIFACT_INVENTORY.json
  ARCHIVE_MANIFEST.json
  SHA256SUMS
  hk155/
    mcp-free/{raw,audit}/
    flat-all-tools/{raw,audit}/
    mcp-zero/{raw,audit}/
    operation-visible/{raw,audit}/
    planroute-lite/{raw,audit}/
    planroute-replan/{raw,audit}/
  etom100/
    mcp-zero/{raw,audit}/
    operation-visible/{raw,audit}/
    planroute-replan/{raw,audit}/
```

ETOM upstream source is **not** bundled. Clone and pin separately (see Data integrity).

### Environment variables after extract

```bash
export MCP_ROUTER_HK_ARTIFACT_ROOT="/path/to/mcp-router-release-artifacts-v1/hk155"
export MCP_ROUTER_ETOM_ARTIFACT_ROOT="/path/to/mcp-router-release-artifacts-v1/etom100"
# Optional ETOM source checkout (not in the ZIP):
# export MCP_ROUTER_ETOM_DATA_ROOT="/path/to/ETOM"   # commit 790f28727f567b88918420d163c524d46644a694
```

Method raw trees live under each method folder’s `raw/` directory. For audits that expect a flat run directory, point tools at the relevant `.../<method>/raw` path (or symlink as needed).

### SHA256SUMS verify (inside extracted archive)

```bash
cd /path/to/mcp-router-release-artifacts-v1
shasum -a 256 -c SHA256SUMS
```

### Forbidden in the archive (policy)

- `.env`, `.env.*`, API keys, tokens, private relay credentials  
- Personal absolute home paths as the only documentation (prefer relative layout + env vars)  
- Unrelated personal files  

Do **not** commit raw `outputs/` into Git; use the published archive instead.

---

## Data integrity

| Pin | Value |
|-----|-------|
| ETOM source commit | `790f28727f567b88918420d163c524d46644a694` (`scripts/fetch_etom_source.sh`, `benchmarks/.../etom_source_pin_v1.json`) |
| HK-155 cutoff date | `2026-08-12` (`benchmarks/manifests/hk_unified_dataset_v1.json`) |
| Formal ETOM-100 lock | `benchmarks/.../formal_100_v1.lock.json` (sha recorded in audit `identity_checks.json`) |
| Method / protocol versions | Frozen inside packages and audit JSON — **do not edit** |

### How to verify

```bash
# ETOM pin
git -C "${MCP_ROUTER_ETOM_DATA_ROOT:-$PROJECT_ROOT/data/etom/raw}" rev-parse HEAD
# expect 790f28727f567b88918420d163c524d46644a694

# Offline audit identity scripts (when artifacts attached)
"$PYTHON" scripts/run_pe_fin_replan_etom_formal100_final_audit_v1.py --help
# Compare reported sha256 fields to docs/*/identity_checks.json
```

Manifest `*_sha256` fields in `benchmarks/manifests/` and audit folders are the source of truth for committed pointer integrity.
