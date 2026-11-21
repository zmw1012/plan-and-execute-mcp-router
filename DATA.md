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

### Recommended archive contents / layout

```text
mcp-router-release-artifacts/
  README.txt                 # archive version, git tip, contact
  hk155/
    <method_run_dirs>/       # frozen outputs referenced by audits
  etom100/
    <method_run_dirs>/
  caches/
    mcp_zero_hk_v1/          # optional embedding indexes
    mcp_zero_etom_v1/
    pe_fin_lite_*/           # if required by audits
  etom_source/
    raw/                     # optional copy of pinned ETOM checkout
  SHA256SUMS                 # hashes of top-level trees / key files
```

Point env vars at the extracted roots:

```bash
export MCP_ROUTER_HK_ARTIFACT_ROOT="/path/to/mcp-router-release-artifacts/hk155"
export MCP_ROUTER_ETOM_ARTIFACT_ROOT="/path/to/mcp-router-release-artifacts/etom100"
export MCP_ROUTER_ETOM_DATA_ROOT="/path/to/mcp-router-release-artifacts/etom_source/raw"
```

**Never** place API keys, relay tokens, or private credentials in the archive.

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
