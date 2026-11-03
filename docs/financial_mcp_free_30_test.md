# Financial MCP-Free 30-question test

This pilot is the closed-book baseline for the Financial domain. It sends only
the question to the model and exposes no MCP tools, reports, retrieval context,
web search, or gold answers.

## Research basis

- [FinanceBench](https://arxiv.org/abs/2311.11944): short-form factual QA over financial filings;
- [FinQA](https://arxiv.org/abs/2109.00122) and [TAT-QA](https://arxiv.org/abs/2105.07624): numerical reasoning, comparison, and table/text distinctions;
- [ETOM](https://aclanthology.org/2026.findings-eacl.75/): direct mapping, implicit discrimination, and infeasible-task robustness;
- [FinMCP-Bench](https://arxiv.org/abs/2603.24943): single-tool/multi-turn task structure and clarification behaviour.

The project adapts these task patterns to Tencent, HSBC, and Alibaba. It does not
claim to reproduce any of the original benchmarks.

## Composition

- Tencent: 11 questions;
- HSBC: 10 questions;
- Alibaba: 9 questions;
- direct retrieval, metric/period discrimination, numerical reasoning, temporal
  robustness, unavailable requests, and clarification requests.

All numerical gold answers are based on entries already marked verified in the
repository's formal Annual/Interim Report snapshot.

## Run

Use the [official DeepSeek model identifier](https://api-docs.deepseek.com/news/news260910/)
`deepseek-flash` for the current DeepSeek V4.1 Flash API. The runner automatically
loads the repository's local `.env` file. Put the real key only in that ignored file;
do not paste it into chat, source code, `.env.example`, or git.

Recommended `.env` format:

```dotenv
LLM_BASE_URL=https://api.deepseek.com
LLM_API_KEY=your-local-key
LLM_MODEL=deepseek-flash
```

The runner also accepts the common OpenAI-compatible aliases `OPENAI_BASE_URL`,
`OPENAI_API_KEY`, and `OPENAI_MODEL`, so an existing environment file can usually
be copied directly.

```bash
python3 scripts/run_mcp_free_baseline.py \
  --questions benchmarks/questions/financial_mcp_free_30_v1.jsonl \
  --output-dir outputs/mcp_free_financial_30_v1 \
  --model deepseek-flash
```

If an OpenAI-compatible relay is used instead of the official endpoint, set its
base URL and exact model alias locally:

```bash
export LLM_BASE_URL='https://your-provider.example/v1'
export LLM_MODEL='the-provider-model-alias'
```

After the run, open `manual_grading.csv`. Grade each response as 1 or 0 against
the frozen reference answer. For an initial 30-question test, human grading is
more transparent than asking the tested model to judge itself.

## Frozen scoring rule

- Numeric retrieval: correct value, currency, scale, period, and accounting basis.
- Numerical reasoning: correct operands/formula and answer within the stored tolerance.
- Clarification: must ask for the missing period, business, or metric rather than guess.
- Unavailable: must refuse the requested value and must not substitute an interim,
  fiscal-year, or unrelated metric.
- Accuracy: number of correct examples divided by 30.
