"""ETOM external benchmark data, sampling, metrics, and OV routing runner.

Shared data/metrics modules are used by Operation-Visible, MCP-Zero, and
PlanRoute RePlan ETOM runners. This package does not execute ETOM MCP tools
and does not replace the HK-equity Financial, Market, News, or held-out
benchmarks.

``EXPECTED_LLM_PROVIDER`` remains the historical Operation-Visible generation
identity (``aliyun``). MCP-Zero and PlanRoute RePlan keep condition-specific
provider identities in their own packages.
"""

FIDELITY_NAME = "Operation-Visible adapted for ETOM routing-only evaluation"

EXPECTED_LLM_PROVIDER = "aliyun"
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
