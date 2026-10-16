# HK Equity MCP Agent

Course project framework for comparing **flat**, **routed**, and **routed_structured** agent settings on Hong Kong equity questions, using multiple MCP-style tool servers.

This repository currently provides a **mock-only** runnable baseline: no paid APIs, no real LLM calls, and no online deployment.

## Project goals

- Build three mock MCP servers: Financial / Market / News
- Provide a unified agent/router interface with three experiment modes
- Provide a minimal evaluation pipeline (JSONL → CSV + metrics stubs)
- Support three-person collaboration with a clear directory layout

## Suggested team split

| Person | Focus |
|--------|--------|
| Member A | Financial server + related benchmarks |
| Member B | Market server + agent/router modes |
| Member C | News server + evaluation metrics / docs |

Adjust ownership as needed; keep PRs small and reviewable.

## Repository layout

```text
src/hk_equity_mcp/
  agent/                 # router + experiment modes
  common/                # schema + mock data
  mcp_servers/
    financial/           # multi-period annual/interim snapshots
    market/
    news/
  evaluation/            # JSONL runner + metrics interfaces
configs/                 # agent.yaml, evaluation.yaml
data/mock/               # inspectable mock JSON snapshots
benchmarks/questions/    # JSONL test questions
tests/
docs/
```

Financial domain details (periods, sources, verification): see [docs/financial_domain.md](docs/financial_domain.md).

## Local setup

Python 3.9+ recommended.

```bash
cd "hk-equity-mcp-agent"
python3 -m pip install -e ".[dev]"
cp .env.example .env   # optional for later; mock demo does not need real keys
```

## Run mock demo

```bash
python3 -m hk_equity_mcp.agent.router
# or, after install:
hk-equity-demo
```

## Run evaluation (smoke)

```bash
python3 -m hk_equity_mcp.evaluation.runner
# writes outputs/eval_results.csv and .summary.json
```

## Run tests

```bash
pytest
```

## Add a new MCP tool

1. Implement the method on the relevant server under `src/hk_equity_mcp/mcp_servers/<domain>/server.py`
2. Return the unified schema fields: `success`, `source`, `ticker`, `as_of_date`, `data`, `evidence`, `error`
3. Register the tool in the server `call()` map and in `TOOL_DOMAIN` inside `agent/router.py`
4. Add unit tests under `tests/`
5. Optionally add benchmark questions that use the new tool

## Add a benchmark question

Append one JSON object per line to `benchmarks/questions/sample_questions.jsonl` (or a new JSONL file) with:

- `id`
- `question`
- `required_domain`
- `gold_tools`
- `gold_arguments`
- `reference_answer`
- `supporting_evidence`

Then point `configs/evaluation.yaml` at the file and rerun the evaluation runner.

## API key safety rules

- Never commit real API keys, passwords, or tokens
- Keep secrets in local `.env` only (gitignored)
- `.env.example` may contain **variable names and fake placeholders only**
- Do not put real financial data dumps or large files in git
- Do not enable paid GitHub Actions / Codespaces / third-party paid services unless the team explicitly agrees

## Experiment modes

Configured in `configs/agent.yaml`:

- `flat`
- `routed`
- `routed_structured`

Current implementations are interface/placeholder level with mock tool execution.

## License

No open-source license is attached yet (pending course policy confirmation).
