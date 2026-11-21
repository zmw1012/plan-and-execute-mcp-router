# Plan-and-Execute MCP Router

This project studies whether **explicit Plan-and-Execute control** can improve multi-tool **MCP (Model Context Protocol) routing** relative to direct answering, flat schema exposure, hierarchical retrieval (MCP-Zero), and ReAct-style Operation-Visible control. Evaluation uses two complementary benchmarks: **ETOM-100** (external, routing-only) and **HK-155** (Hong Kong equity **application** scenario with real MCP execution). HK-155 is an application benchmark, not the project title.

## Research question

Can explicit planning improve multi-tool MCP routing while reducing repeated LLM control calls and full-schema exposure?

## Methods

| Method | Control structure | HK-155 | ETOM-100 |
|--------|-------------------|--------|----------|
| MCP-Free | Direct LLM answer, no MCP | Yes | N/A |
| Flat-All-Tools | All schemas exposed | Yes | Not run |
| MCP-Zero | Hierarchical Server → Tool retrieval | Yes | Yes |
| Operation-Visible | Find → Describe → Invoke with ReAct | Yes | Yes |
| MCP-PlanRoute Lite | One-shot plan, no global replan | Yes | Not run |
| MCP-PlanRoute RePlan | Planner → Execute one step → Replanner | Yes | Yes |

### Proposed methods

**MCP-PlanRoute Lite** builds a single plan, then executes step-level retrieval and MCP calls without a global replanner.

**MCP-PlanRoute RePlan** interleaves Planner → execute one step → Replanner until finish, abort, or budget/step caps.

Both share **R5 retrieval**:

```text
score = cosine_similarity + 0.12 × lexical_score
```

- no hard domain filter  
- Top-K = 5  
- Describe-N = 3  

## Benchmarks

### HK-155 (application / real execution)

- **155** tasks: Financial 60, Market 35, News 35, Cross-domain 25  
- Real MCP tool execution against Hong Kong equity data providers  
- Graded answer accuracy + tool-set metrics  

### ETOM-100 (routing-only)

- **100** tasks (20 per official L1–L5)  
- Source registry: **491** servers, **2,375** tools  
- Route Exact Match / F1 only — **no** real business-tool execution  

## Main results

Full tables, metric definitions, ablations, and provenance: **[RESULTS.md](RESULTS.md)**.

### HK-155 (headline)

| Method | Answer Accuracy | Common Strict | Notes |
|--------|-----------------|---------------|-------|
| Operation-Visible | **61.29%** | 29.03% | Best answer accuracy |
| MCP-Zero | 58.06% | **58.06%** | Best common strict |
| MCP-PlanRoute Lite | 22.58% lower / 23.23% upper | 5.81% | One manual-pending bound |
| MCP-PlanRoute RePlan | **25.16%** conservative | 11.61% | Engineering-valid; not formal-accuracy-valid alone |

### ETOM-100 (headline)

| Method | Route EM | Notes |
|--------|----------|-------|
| Operation-Visible | **66%** | Best routing EM |
| MCP-Zero | 49% | |
| MCP-PlanRoute RePlan | **35%** conservative / 45% raw | Failures-as-zero conservative |

**Negative result (do not hide):** R5 improves candidate retrieval, but the rigid Planner–Executor–Replanner implementation does **not** outperform ReAct-style Operation-Visible or MCP-Zero on either benchmark under conservative scoring.

## Key conclusion

These experiments reject the **current rigid PlanRoute implementation**, not Plan-and-Execute as a general design idea. Recommended future work: **Plan-guided ReAct**.

## Repository navigation

| Doc | Purpose |
|-----|---------|
| [RESULTS.md](RESULTS.md) | Authoritative metrics, ablations, error analysis, provenance |
| [REPRODUCIBILITY.md](REPRODUCIBILITY.md) | Offline and live reproduction commands |
| [CODE_MAP.md](CODE_MAP.md) | Canonical names ↔ package paths ↔ entry points |
| [DATA.md](DATA.md) | Committed vs external data; archive checklist |
| [docs/](docs/README.md) | Release vs audit vs historical documentation |
| [benchmarks/](benchmarks/) | HK-155 / ETOM manifests and schemas |
| [scripts/](scripts/README.md) | Runner and audit script index |
| [tests/](tests/) | Offline contract and integration tests |

## Quick start (offline only)

```bash
export PROJECT_ROOT="$(pwd)"   # clone root
export PYTHON="${PYTHON:-python3}"
cd "$PROJECT_ROOT"
"$PYTHON" -m pip install -e ".[dev]"
"$PYTHON" -c "import hk_equity_mcp; print('ok')"
"$PYTHON" -m pytest tests/test_mcp_zero_protocol_v2.py tests/test_pe_fin_replan_contract_v1_4_1.py tests/test_operation_visible_agent.py -q
# Full offline suite (skips when external artifacts absent):
"$PYTHON" -m pytest -q
```

No generation LLM, embedding API, or live MCP is required for the offline path.

### Live runs (credentials / network / cost)

Live HK-155 and ETOM-100 runners call generation LLMs, may call embedding APIs, and (for HK-155) invoke real MCP tools and market/disclosure downloads. They can incur **API cost**.

See **[REPRODUCIBILITY.md](REPRODUCIBILITY.md)** for exact CLI flags, smoke vs full runs, resume, and audit steps. Copy [.env.example](.env.example) to a local env file and fill placeholders—never commit secrets.

## Citation and related work

Use the titles and URLs already documented in-repo (do not invent bibliographic fields):

| Work | Title / location |
|------|------------------|
| **ETOM** | Source: [snooow1029/ETOM](https://github.com/snooow1029/ETOM); paper also cited in-repo via [ACL Anthology 2026.findings-eacl.75](https://aclanthology.org/2026.findings-eacl.75/) |
| **MCP-Zero** | Fei et al., *MCP-Zero: Active Tool Discovery for Autonomous LLM Agents* — [arXiv:2506.01056](https://arxiv.org/abs/2506.01056) |
| **Operation-Visible** | Ghoshal et al. (2026), *Scaling Enterprise Cloud Agents to Thousands of APIs…* (EMNLP 2026 Industry Track); see [docs/operation_visible_baseline.md](docs/operation_visible_baseline.md) |
| **ReAct** | Yao et al., *ReAct: Synergizing Reasoning and Acting in Language Models* — [arXiv:2210.03629](https://arxiv.org/abs/2210.03629) |
| **Plan-and-Execute / Plan-and-Solve** | Wang et al., *Plan-and-Solve Prompting: Improving Zero-Shot Chain-of-Thought Reasoning by Large Language Models* — [arXiv:2305.04091](https://arxiv.org/abs/2305.04091) |

Internal package paths (`pe_fin_lite`, `pe_fin_replan`, …) are **legacy import identities**; public names above are canonical. See [CODE_MAP.md](CODE_MAP.md).

## License / course note

Course research code for MSBD5018 NLP. Do not redistribute private API keys, local absolute paths, or unreleased market data.
