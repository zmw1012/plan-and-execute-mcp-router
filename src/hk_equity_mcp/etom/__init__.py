"""ETOM external benchmark data, sampling, and official routing metrics.

Shared by Operation-Visible and MCP-Zero ETOM runners. Does not execute ETOM
MCP tools and does not replace the HK-equity Financial/Market/News benchmarks.
"""

FIDELITY_NAME = "ETOM routing-only evaluation (shared data/metrics)"

EXPECTED_LLM_PROVIDER = "openai_compatible"
EXPECTED_LLM_MODEL = "deepseek-v4.1-flash"
EXPECTED_EMBEDDING_MODEL = "text-embedding-3-small"
PILOT_MAX_TASKS = 10
COST_PILOT_SEED = 5018
FORMAL_100_SEED = 20261106
FORMAL_100_MAX_TASKS = 100
FORMAL_100_SAMPLER = "etom_stratified_external_100_v1"
FORMAL_100_QUOTAS = {1: 20, 2: 20, 3: 20, 4: 20, 5: 20}
EXPECTED_TOOLS = 2375
EXPECTED_SERVERS = 491
