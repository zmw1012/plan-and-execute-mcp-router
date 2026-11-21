# Reproducibility

Executable guide for another machine. Use shell variables—**never** hard-code `/Users/…` paths.

```bash
export PROJECT_ROOT="$(pwd)"          # clone / release worktree root
export PYTHON="${PYTHON:-python3}"
export GEN_ENV="${GEN_ENV:-$PROJECT_ROOT/.env.generation}"       # LLM_* placeholders
export EMBEDDING_ENV="${EMBEDDING_ENV:-$PROJECT_ROOT/.env.embedding}"
# Optional external frozen trees (see DATA.md):
export HK_ARTIFACT_ROOT="${MCP_ROUTER_HK_ARTIFACT_ROOT:-}"
export ETOM_ARTIFACT_ROOT="${MCP_ROUTER_ETOM_ARTIFACT_ROOT:-}"
export ETOM_DATA_ROOT="${MCP_ROUTER_ETOM_DATA_ROOT:-}"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
```

---

## Supported environment

| Item | Value |
|------|-------|
| Tested Python | **3.9+** (`requires-python >=3.9`; offline suite verified on CPython 3.9.6 / macOS) |
| OS notes | macOS and Linux expected; Windows untested. Live HK helpers may use `caffeinate` on macOS only. |
| Install | `pip install -e ".[dev]"` for offline tests; add `.[real_mcp]` for live MCP stdio clients |
| Lock | See [requirements-release.txt](requirements-release.txt) (pinned from the verified env). Full cross-platform lock resolution may need network; this pin is the documented fallback. |

```bash
"$PYTHON" -m pip install -U pip setuptools wheel
"$PYTHON" -m pip install -e ".[dev]"
# Optional live MCP:
# "$PYTHON" -m pip install -e ".[real_mcp]"
# Or: "$PYTHON" -m pip install -r requirements-release.txt
```

---

## Offline validation (no LLM / embedding / MCP / market APIs)

```bash
# Installation + import
"$PYTHON" -c "import hk_equity_mcp; print('import ok')"

# Core contract suite (representative)
"$PYTHON" -m pytest \
  tests/test_mcp_zero_protocol_v2.py \
  tests/test_pe_fin_replan_contract_v1_4_1.py \
  tests/test_operation_visible_agent.py -q

# Full offline suite (skips when external artifacts / ETOM checkout / caches absent)
# Live APIs are not invoked; gated tests call pytest.skip via tests/artifact_gates.py
"$PYTHON" -m pytest -q

# Mock / dry-run smoke (no paid APIs)
"$PYTHON" scripts/run_hk_unified_mcp_free_experiment.py --smoke --mock \
  --output-dir "$PROJECT_ROOT/outputs/_offline_mcp_free_smoke"
"$PYTHON" scripts/run_hk_unified_flat_all_tools_experiment.py --smoke --mock \
  --output-dir "$PROJECT_ROOT/outputs/_offline_flat_smoke"
"$PYTHON" scripts/run_hk_unified_mcp_zero_experiment.py --smoke --mock \
  --output-dir "$PROJECT_ROOT/outputs/_offline_mcp_zero_smoke"
"$PYTHON" scripts/run_hk_unified_operation_visible_experiment.py --smoke --mock \
  --output-dir "$PROJECT_ROOT/outputs/_offline_ov_smoke"
"$PYTHON" scripts/run_hk_unified_pe_fin_lite_experiment.py \
  --split exploratory_all --confirm-retire-heldout --smoke5 --mock \
  --output-dir "$PROJECT_ROOT/outputs/_offline_planroute_lite_smoke"
"$PYTHON" scripts/run_hk_unified_pe_fin_lite_experiment.py \
  --split exploratory_all --confirm-retire-heldout --replan-smoke5 --mock \
  --output-dir "$PROJECT_ROOT/outputs/_offline_planroute_replan_smoke"

# ETOM offline mock/dry-run (PlanRoute RePlan supports fixture-free dry-run;
# MCP-Zero / OV formal runners require pinned ETOM checkout or fixture dir)
"$PYTHON" scripts/run_etom_pe_fin_replan_experiment.py --dry-run --engineering-smoke5 \
  --output-dir "$PROJECT_ROOT/outputs/_offline_etom_replan_dry"
# After `bash scripts/fetch_etom_source.sh` (or fixtures):
# "$PYTHON" scripts/run_etom_mcp_zero.py --split formal100 --mock \
#   --output-dir "$PROJECT_ROOT/outputs/_offline_etom_mcp_zero_mock"
# "$PYTHON" scripts/run_etom_operation_visible_formal_100.py --mock \
#   --output-dir "$PROJECT_ROOT/outputs/_offline_etom_ov_mock"

# Final audits when external frozen raw trees are attached
# (skip cleanly if MCP_ROUTER_*_ARTIFACT_ROOT unset / incomplete)
if [[ -n "${MCP_ROUTER_HK_ARTIFACT_ROOT:-}" ]]; then
  "$PYTHON" scripts/run_pe_fin_lite_r5_hk155_final_audit_v1.py --help >/dev/null
  "$PYTHON" scripts/run_pe_fin_replan_r7_hk155_final_audit_v1.py --help >/dev/null
fi
if [[ -n "${MCP_ROUTER_ETOM_ARTIFACT_ROOT:-}" ]]; then
  "$PYTHON" scripts/run_pe_fin_replan_etom_formal100_final_audit_v1.py --help >/dev/null
fi
```

---

## External artifact variables

| Variable | Purpose |
|----------|---------|
| `MCP_ROUTER_HK_ARTIFACT_ROOT` | Root of frozen HK-155 raw run trees (predictions, caches, indexes) |
| `MCP_ROUTER_ETOM_ARTIFACT_ROOT` | Root of frozen ETOM-100 raw run trees |
| `MCP_ROUTER_ETOM_DATA_ROOT` | Pinned ETOM git checkout (or path containing `data/etom/raw`) |

Tests that need these roots **skip** (not fail) when unset or incomplete. See `tests/artifact_gates.py`.

---

## HK-155 execution

Canonical runners live under `scripts/run_hk_unified_*.py`.  
Formal sealed results used `--split exploratory_all --confirm-retire-heldout` (see historical audits). Prefer **smoke / mock** first.

### Shared warnings

| Flag / action | Cost / network |
|---------------|----------------|
| Default live agent run | **Generation LLM** + (where used) **embedding API** + **real MCP** + possible HKEX/Sina/Eastmoney downloads |
| `--allow-hkex-download` | Explicit HKEX / disclosure download |
| `--build-index-only` | **Embedding API** spend |
| `--mock` / `--dry-run` / `--index-status` | Offline / no paid generation |

### MCP-Free

```bash
# Smoke (mock)
"$PYTHON" scripts/run_hk_unified_mcp_free_experiment.py --smoke --mock \
  --output-dir "$PROJECT_ROOT/outputs/hk_mcp_free_smoke_mock"

# Full exploratory-155 (LIVE — generation LLM)
"$PYTHON" scripts/run_hk_unified_mcp_free_experiment.py \
  --split exploratory_all --confirm-retire-heldout \
  --env-file "$GEN_ENV" \
  --output-dir "$PROJECT_ROOT/outputs/hk_unified_exploratory155_mcp_free_live_v1" \
  --resume

# Offline audit (needs frozen outputs under artifact root or local outputs/)
"$PYTHON" scripts/run_hk_unified_mcp_free_final_audit_v1.py --help
```

### Flat-All-Tools

```bash
"$PYTHON" scripts/run_hk_unified_flat_all_tools_experiment.py --smoke --mock \
  --output-dir "$PROJECT_ROOT/outputs/hk_flat_smoke_mock"

# LIVE — generation + real MCP (+ optional --allow-hkex-download)
"$PYTHON" scripts/run_hk_unified_flat_all_tools_experiment.py \
  --split exploratory_all --confirm-retire-heldout \
  --env-file "$GEN_ENV" \
  --output-dir "$PROJECT_ROOT/outputs/hk_unified_exploratory155_flat_all_tools_live_v1" \
  --resume

"$PYTHON" scripts/run_hk_unified_flat_all_tools_final_audit_v1.py --help
```

### MCP-Zero

```bash
# Index status (offline)
"$PYTHON" scripts/run_hk_unified_mcp_zero_experiment.py --index-status

# Build index only (LIVE embedding API)
"$PYTHON" scripts/run_hk_unified_mcp_zero_experiment.py --build-index-only \
  --embedding-env-file "$EMBEDDING_ENV" \
  --embedding-cache-dir "$PROJECT_ROOT/.cache/mcp_zero_hk_v1"

# Smoke mock
"$PYTHON" scripts/run_hk_unified_mcp_zero_experiment.py --smoke --mock \
  --output-dir "$PROJECT_ROOT/outputs/hk_mcp_zero_smoke_mock"

# Full LIVE — generation + embedding index + real MCP
"$PYTHON" scripts/run_hk_unified_mcp_zero_experiment.py \
  --split exploratory_all --confirm-retire-heldout \
  --env-file "$GEN_ENV" --embedding-env-file "$EMBEDDING_ENV" \
  --require-complete-index \
  --output-dir "$PROJECT_ROOT/outputs/hk_unified_exploratory155_mcp_zero_live_v1" \
  --resume

"$PYTHON" scripts/run_hk_unified_mcp_zero_final_audit_v1.py --help
```

### Operation-Visible

```bash
"$PYTHON" scripts/run_hk_unified_operation_visible_experiment.py --smoke --mock \
  --output-dir "$PROJECT_ROOT/outputs/hk_ov_smoke_mock"

# LIVE
"$PYTHON" scripts/run_hk_unified_operation_visible_experiment.py \
  --split exploratory_all --confirm-retire-heldout \
  --env-file "$GEN_ENV" --embedding-env-file "$EMBEDDING_ENV" \
  --output-dir "$PROJECT_ROOT/outputs/hk_unified_exploratory155_operation_visible_live_v1" \
  --resume
```

### MCP-PlanRoute Lite

Release entry point: `scripts/run_hk_unified_pe_fin_lite_experiment.py` (internal package `pe_fin_lite`).

```bash
"$PYTHON" scripts/run_hk_unified_pe_fin_lite_experiment.py --index-status
"$PYTHON" scripts/run_hk_unified_pe_fin_lite_experiment.py --build-index-only \
  --embedding-env-file "$EMBEDDING_ENV"

"$PYTHON" scripts/run_hk_unified_pe_fin_lite_experiment.py \
  --split exploratory_all --confirm-retire-heldout --smoke5 --mock \
  --output-dir "$PROJECT_ROOT/outputs/hk_planroute_lite_smoke_mock"

# LIVE full exploratory (generation + embedding + real MCP)
"$PYTHON" scripts/run_hk_unified_pe_fin_lite_experiment.py \
  --split exploratory_all --confirm-retire-heldout \
  --env-file "$GEN_ENV" --embedding-env-file "$EMBEDDING_ENV" \
  --require-complete-index \
  --output-dir "$PROJECT_ROOT/outputs/hk_unified_exploratory155_pe_fin_lite_live_v1" \
  --resume

"$PYTHON" scripts/run_pe_fin_lite_r5_hk155_final_audit_v1.py --help
```

### MCP-PlanRoute RePlan (HK-155)

Same unified Lite runner with `--replan` (internal `pe_fin_replan`). Default `--replan-round-cap 7` preserves sealed identity.

```bash
"$PYTHON" scripts/run_hk_unified_pe_fin_lite_experiment.py \
  --split exploratory_all --confirm-retire-heldout --replan-smoke5 --mock \
  --output-dir "$PROJECT_ROOT/outputs/hk_planroute_replan_smoke_mock"

# LIVE
"$PYTHON" scripts/run_hk_unified_pe_fin_lite_experiment.py \
  --replan --replan-round-cap 7 \
  --split exploratory_all --confirm-retire-heldout \
  --env-file "$GEN_ENV" --embedding-env-file "$EMBEDDING_ENV" \
  --require-complete-index \
  --output-dir "$PROJECT_ROOT/outputs/hk_unified_exploratory155_pe_fin_replan_live_v1" \
  --resume

"$PYTHON" scripts/run_pe_fin_replan_r7_hk155_final_audit_v1.py --help
```

---

## ETOM-100 execution

Pinned source: `snooow1029/ETOM` @ `790f28727f567b88918420d163c524d46644a694` via `scripts/fetch_etom_source.sh` (network clone; gitignored under `data/etom/raw`).

```bash
# Source checkout (NETWORK)
bash scripts/fetch_etom_source.sh
export MCP_ROUTER_ETOM_DATA_ROOT="$PROJECT_ROOT/data/etom/raw"
# or: export ETOM_SOURCE_DIR=...

# Frozen sample / lock verification (offline once checkout or fixtures present)
"$PYTHON" -c "from hk_equity_mcp.etom import source; print('etom source helpers ok')"
```

### MCP-Zero × ETOM

```bash
# Prefer shell helper modes: dry-run | preflight | mock | index-status | build-index | run
OUT="$PROJECT_ROOT/outputs/etom_external_formal100_mcp_zero_v1" \
  bash scripts/run_etom_mcp_zero_formal_100.sh mock

# LIVE index build (embedding API) then run (generation LLM) — set ENV_FILE / EMBEDDING_ENV_FILE / ETOM_SOURCE_DIR
ENV_FILE="$GEN_ENV" EMBEDDING_ENV_FILE="$EMBEDDING_ENV" \
  ETOM_SOURCE_DIR="${MCP_ROUTER_ETOM_DATA_ROOT:-$PROJECT_ROOT/data/etom/raw}" \
  OUT="$PROJECT_ROOT/outputs/etom_external_formal100_mcp_zero_v1" \
  bash scripts/run_etom_mcp_zero_formal_100.sh build-index

ENV_FILE="$GEN_ENV" EMBEDDING_ENV_FILE="$EMBEDDING_ENV" \
  ETOM_SOURCE_DIR="${MCP_ROUTER_ETOM_DATA_ROOT:-$PROJECT_ROOT/data/etom/raw}" \
  OUT="$PROJECT_ROOT/outputs/etom_external_formal100_mcp_zero_v1" \
  bash scripts/run_etom_mcp_zero_formal_100.sh run

"$PYTHON" scripts/run_etom_mcp_zero_formal100_final_audit_v1.py --help
```

Equivalent Python entry: `scripts/run_etom_mcp_zero.py --split formal100 …`.

### Operation-Visible × ETOM

```bash
OUT="$PROJECT_ROOT/outputs/etom_external_formal100_operation_visible_v1" \
  bash scripts/run_etom_operation_visible_formal_100.sh mock

# LIVE (set ENV_FILE / EMBEDDING_ENV_FILE; wrapper defaults must be overridden off this machine)
ENV_FILE="$GEN_ENV" EMBEDDING_ENV_FILE="$EMBEDDING_ENV" \
  OUT="$PROJECT_ROOT/outputs/etom_external_formal100_operation_visible_v1" \
  bash scripts/run_etom_operation_visible_formal_100.sh run
# resume: ... sh resume
```

### MCP-PlanRoute RePlan × ETOM

```bash
"$PYTHON" scripts/run_etom_pe_fin_replan_experiment.py --index-status
"$PYTHON" scripts/run_etom_pe_fin_replan_experiment.py --mock --engineering-smoke5 \
  --output-dir "$PROJECT_ROOT/outputs/etom_planroute_replan_smoke_mock"

# LIVE formal100 — dual gate: --allow-live-formal100 + env PE_FIN_REPLAN_ETOM_ALLOW_LIVE=1
# (generation + embedding; routing-only — no business MCP execution)
PE_FIN_REPLAN_ETOM_ALLOW_LIVE=1 \
"$PYTHON" scripts/run_etom_pe_fin_replan_experiment.py --formal100 --live \
  --allow-live-formal100 --require-complete-index \
  --env-file "$GEN_ENV" --embedding-env-file "$EMBEDDING_ENV" \
  --output-dir "$PROJECT_ROOT/outputs/etom_external_formal100_pe_fin_replan_v1" \
  --resume

"$PYTHON" scripts/run_pe_fin_replan_etom_formal100_final_audit_v1.py --help
```

---

## Expected output structure

```text
outputs/<run_name>/
  run_manifest.json / resume state
  predictions/ or per-task JSON
  summary / metrics JSON (method-specific)
  run.log (some wrappers)
docs/<method>_…_final_audit_v1/   # sealed audit summaries (committed)
.cache/                           # embedding indexes (local, usually gitignored)
```

Exact filenames vary by method; sealed audits under `docs/` record the expected layout for each formal tip.

---

## Cost and network warning

| Command class | Generation LLM | Embedding API | Real MCP | Market / HKEX download | Cost |
|---------------|----------------|---------------|----------|------------------------|------|
| `--mock`, `--dry-run`, `--index-status`, most unit tests | No | No | No | No | None |
| `--build-index-only` | No | **Yes** | No | No | Embedding |
| HK live agent (`exploratory_all`, etc.) | **Yes** | Often **Yes** | **Yes** | If `--allow-hkex-download` | High |
| ETOM live formal100 | **Yes** | **Yes** | No business tools | No | Medium–high |
| `fetch_etom_source.sh` | No | No | No | Git clone | Network |

Never commit filled env files. Never print API keys.

---

## Historical reproduction

Exact reported numbers correspond to **frozen run commits** listed in [RESULTS.md](RESULTS.md) (Provenance). Check out those tips (or the release merge commits that contain them) to recover the implementation state used for each sealed audit.

Live LLM outputs are **non-deterministic**; re-running live will not guarantee bit-identical answers. Offline audits over frozen raw trees **are** meant to be deterministic given the same artifacts and grader identity.

Do not change frozen `method_version`, `protocol_version`, manifest IDs, or hashes when reproducing.
