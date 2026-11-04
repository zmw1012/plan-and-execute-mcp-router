"""ETOM external benchmark adapter.

Operation-Visible adapted for ETOM routing-only evaluation.

This package does not execute ETOM MCP tools and does not replace the
HK-equity Financial, Market, News, or held-out benchmarks.
"""

FIDELITY_NAME = "Operation-Visible adapted for ETOM routing-only evaluation"

EXPECTED_LLM_PROVIDER = "aliyun"
EXPECTED_LLM_MODEL = "deepseek-v4.1-flash"
EXPECTED_EMBEDDING_MODEL = "text-embedding-3-small"
PILOT_MAX_TASKS = 10
COST_PILOT_SEED = 5018
