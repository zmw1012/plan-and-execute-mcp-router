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

> **Release checklist (before course submission):**  
> **Add the final HKUST OneDrive or approved cloud archive URL before submission.**

Do **not** invent a cloud URL in this repository until the archive is published.

### Final archive URL (fill before submission)

Paste the approved cloud link here after the archive is uploaded:

```text
ARCHIVE_URL: <REPLACE_WITH_HKUST_ONEDRIVE_OR_APPROVED_CLOUD_URL>
```

Until `ARCHIVE_URL` is a real HTTPS link (not the placeholder above), the release is **not** submission-complete.

### Recommended archive contents / layout

Upload these **outside Git** (do not `git add outputs/`):

1. **HK-155 frozen run trees** referenced by sealed audits (predictions, run metadata, human grades pointers as applicable).  
2. **ETOM-100 frozen run trees** for MCP-Zero, Operation-Visible, and PlanRoute RePlan formal-100.  
3. **Optional embedding caches** needed for offline index-status / audit parity.  
4. **Optional pinned ETOM source checkout** at commit `790f28727f567b88918420d163c524d46644a694`.  
5. Top-level `README.txt` (archive version, matching git tip of `release/code-submission-v1`, contact).  
6. `SHA256SUMS` for key files / directories.

```text
mcp-router-release-artifacts/
  README.txt
  hk155/
    <method_run_dirs>/
  etom100/
    <method_run_dirs>/
  caches/
    mcp_zero_hk_v1/
    mcp_zero_etom_v1/
    pe_fin_lite_*/
  etom_source/
    raw/
  SHA256SUMS
```

### Environment variables after extract

```bash
export MCP_ROUTER_HK_ARTIFACT_ROOT="/path/to/mcp-router-release-artifacts/hk155"
export MCP_ROUTER_ETOM_ARTIFACT_ROOT="/path/to/mcp-router-release-artifacts/etom100"
export MCP_ROUTER_ETOM_DATA_ROOT="/path/to/mcp-router-release-artifacts/etom_source/raw"
```

### SHA256SUMS generate and verify

```bash
# From inside the archive root (example: hash key files / top-level dirs)
cd /path/to/mcp-router-release-artifacts
find hk155 etom100 caches etom_source -type f | sort | xargs shasum -a 256 > SHA256SUMS
# Verify later:
shasum -a 256 -c SHA256SUMS
```

### Forbidden in the archive

- `.env`, `.env.*`, API keys, tokens, private relay credentials  
- Personal absolute home paths as the only documentation (prefer relative layout + env vars)  
- Unrelated personal files

### User operation checklist

1. Assemble the directory layout above from the sealed experiment machines.  
2. Generate `SHA256SUMS` and spot-check a few hashes.  
3. Upload to HKUST OneDrive or another approved cloud store.  
4. Replace `ARCHIVE_URL` in this file with the real HTTPS link.  
5. Re-run offline tests with the three `MCP_ROUTER_*` roots set.  
6. Do **not** commit raw `outputs/` into Git.

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
