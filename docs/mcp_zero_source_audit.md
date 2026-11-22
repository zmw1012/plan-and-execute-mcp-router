# MCP-Zero Source Audit (HK-155 baseline)

Date: 2026-11-06  
Worktree: `hk-equity-mcp-agent-mcp-zero`  
Branch: `feature/mcp-zero-baseline-v1`  
Base commit: `1eb3315`

## Policy

- Prefer **reimplementation** from the paper + pinned reference behaviour.
- Do **not** commit absolute-path symlinks.
- Do **not** silently vendor entire third-party trees into `src/`.
- Cursor coding models are **not** experiment variables.

## Source repositories

| Source | URL | Commit | License |
|---|---|---|---|
| Official MCP-Zero | https://github.com/xfey/MCP-Zero | `fd666c44c9290a671949b974444c30f4ab161622` | see upstream repo |
| Paper | https://arxiv.org/abs/2506.01056 | — | arXiv |
| ETOM (MCP0 pin) | https://github.com/snooow1029/ETOM | `790f28727f567b88918420d163c524d46644a694` | MIT (per pin metadata) |

Pinned file: `benchmarks/mcp_zero_source_pins_v1.json`.

## Classification of untracked files from prior round

### 1. This project's original implementation (commit for HK-155)

| Path | Notes |
|---|---|
| `src/hk_equity_mcp/mcp_zero/*.py` | Reimplemented hierarchical router, agent, runner helpers |
| `configs/mcp_zero_hk_v1.yaml` | HK experiment config |
| `scripts/run_hk_unified_mcp_zero_experiment.py` | HK runner CLI |
| `src/hk_equity_mcp/experiment/unified_mcp_zero_runner.py` | Runner (shares OV artifacts/evaluator) |
| `docs/mcp_zero_*.md` | Audit / fidelity / baseline docs |
| `tests/test_mcp_zero_*.py` | Offline unit + mock/dry-run tests |

### 2. Reimplemented from paper / pinned behaviour (not verbatim copy)

| Module | Reference behaviour |
|---|---|
| `active_request.py` | `<tool_assistant>` parse (paper + matcher) |
| `scoring.py` | Official combined score + ETOM tool-only mode |
| `server_retriever.py` / `tool_retriever.py` | Two-stage hierarchical routing |
| `extension.py` | Iterative capability extension termination |
| `agent.py` / `prompt.py` | Adapted agent loop + HK answer contract |

### 3. Third-party copied / adapted reference material (NOT committed as runnable `src`)

Local pin copies under `data/etom/raw_mcp0_pin/` (reference only; do not import from production HK runner):

| File | SHA256 | Status |
|---|---|---|
| `README_MCP0_excerpt.md` | `eea451ae18b7815d09759cc3ad028846af5ddd0d5a8f954fe0013f4314c61aa3` | excerpt / adapted notes |
| `eval.py` | `d3452a7513aa0ce1fa36075f90d2c72b826f3b32a4d58514570921facc6ae7a9` | copied pin (ETOM) |
| `official_matcher.py` | `fe22e63c28af9cdc41ceaace0ca507722a445ebb0a96428b4cf19c37fb6989fd` | copied pin (official matcher) |
| `orchestrator_core.py` | `f19998d66f03c88aad62e192c32dac78c247e433efff48e04d5f141c3cc272aa` | copied pin (ETOM) |
| `orchestrator_mcp0.py` | `5d09ba3338fb2b36cc630b5e44308bac9a58d68c103620f3879abc2e2f0dd9ec` | copied pin (ETOM) |
| `orchestrator_utils.py` | `0e74de5b679c149032c072ab44889bb4cc968fe44934f9702458e2d2868dd0f1` | copied pin (ETOM) |
| `schemas.py` | `91807b6ce3e09ef1604794932c15aafcf3903f628b91dd6266e3418ab2e4d44d` | copied pin (ETOM) |

Attribution: copyright remains with upstream authors. These pins are for fidelity audit / offline comparison only.

`prompts/mcp_zero/*.prompt` — adapted from official / ETOM tool-request templates (reference prompts; HK runtime uses `src/hk_equity_mcp/mcp_zero/prompt.py`).

### 4. Benchmark lock / source pins (may commit)

| Path | Role |
|---|---|
| `benchmarks/mcp_zero_source_pins_v1.json` | MCP-Zero + ETOM pin metadata |
| `benchmarks/etom/*.lock.json` / `etom_source_pin_v1.json` | ETOM locks (deferred; not required for HK-155 MCP-Zero) |

### 5. Temporary / must-not-commit

| Path | Action |
|---|---|
| `data/etom/raw` → absolute symlink to ETOM worktree | **Deleted** (symlink only; target untouched) |
| `src/hk_equity_mcp/etom/**` | Prior-round ETOM runner half-work — **excluded from this commit** |
| `data/etom/raw_mcp0_pin/**` | Third-party pin copies — **excluded from this commit**; hashes recorded above |

## ETOM source path policy

No absolute symlink. Future ETOM source must be passed explicitly:

- CLI: `--etom-source-dir`
- Env: `ETOM_SOURCE_DIR`

Missing path → fail-closed.

## License / attribution status

- HK original code: project license.
- Official MCP-Zero / ETOM pins: retain upstream copyright; reimplementation preferred for runnable code.
- Paper concepts attributed to Fei et al. (2025).
