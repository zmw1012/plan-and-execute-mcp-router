# PE-Fin Lite Planner-only All-155 Offline Final Audit (v1)

## Status mark

`invalid_for_formal_step_level_retrieval_due_to_planner_protocol_failure_rate`

- R0 question-level embedding/ranking: **still valid**
- Catalog embedding index: **still valid**
- Question-query embedding cache: **still valid**
- Current 41 Planner failures: **protocol/schema failures, not Embedding failures**
- Formal R1–R3 / formula selection: **forbidden until Planner-only v2 completes**

## Freeze

- Planner cache SHA256: `002d60972d13b1e3728642ef7d63eeb1ada543c19bd25129b54ae67be3a312d5`
- Protocol: `pe_fin_lite_planner_only_v1`
- Provider/host/model: `openai_compatible` / `api.zhizengzeng.com` / `deepseek-v4.1-flash`
- Prompt hash: `ed7785886b5441e20218e554757cdeb58a3f9d685176feb66afc6ccfb51716b0`
- Schema hash: `9eb087e2b201784dafb0db7bf89935ba0d2774d8deeb3d145678cf7f27bfc842`
- Capability profile hash: `502446f850f6b389f16cfd1ff97ac0c9f9837efeaef54fa9b2d00ac129e1a7ef`

## Funnel

| Metric | Value |
|--------|------:|
| rows | 155 |
| unique task IDs | 155 |
| ok | 114 |
| failed | 41 |
| repair_count=1 | 119 |
| retrieve steps | 184 |

### Success by source split
{
  "development70": {
    "n": 70,
    "ok": 49,
    "failed": 21,
    "success_rate": 0.7
  },
  "previously_reserved85": {
    "n": 85,
    "ok": 65,
    "failed": 20,
    "success_rate": 0.7647058823529411
  }
}

### Success by domain
{
  "cross_domain": {
    "n": 25,
    "ok": 18,
    "failed": 7,
    "success_rate": 0.72
  },
  "financial": {
    "n": 60,
    "ok": 46,
    "failed": 14,
    "success_rate": 0.7666666666666667
  },
  "market": {
    "n": 35,
    "ok": 23,
    "failed": 12,
    "success_rate": 0.6571428571428571
  },
  "news": {
    "n": 35,
    "ok": 27,
    "failed": 8,
    "success_rate": 0.7714285714285715
  }
}

### Success by difficulty
{
  "D1": {
    "n": 35,
    "ok": 32,
    "failed": 3,
    "success_rate": 0.9142857142857143
  },
  "D2": {
    "n": 24,
    "ok": 17,
    "failed": 7,
    "success_rate": 0.7083333333333334
  },
  "D3": {
    "n": 42,
    "ok": 23,
    "failed": 19,
    "success_rate": 0.5476190476190477
  },
  "D4": {
    "n": 28,
    "ok": 19,
    "failed": 9,
    "success_rate": 0.6785714285714286
  },
  "D5": {
    "n": 26,
    "ok": 23,
    "failed": 3,
    "success_rate": 0.8846153846153846
  }
}

## Final error taxonomy

Expected vs observed:

{
  "none": 114,
  "missing_depends_on_first_step": 19,
  "other_missing_field": 1,
  "invalid_json": 7,
  "empty_or_null_content": 7,
  "invalid_domain_enum": 2,
  "truncated_or_non_json": 3,
  "maxLength_violation": 2
}

Matches expected distribution: **True**

## Repair / first-response (offline v2 canonicalizer replay)

- first-response valid under v2 canonicalizer: 48
- repair-response valid under v2 canonicalizer: 92
- repair recovered (first fail → repair ok) under v2: 80

## Offline structural recovery (failed 41 only)

- recoverable: **14**
- still unrecoverable: **27**
- rules recovery counts: `{'first_step_missing_depends_on_to_empty_array': 12, 'reason_domain_analysis_to_general': 1, 'reason_domain_reasoning_to_general': 1}`
- semantic plan changed beyond allowed rules: 0

Allowed unambiguous rules only (canonicalizer `pe_fin_lite_plan_structural_canonicalizer_v1` / `1c0cf80e1a3940a14983b93fb8938fab74e13f48917f0141d18e86531ba949cd`):

- first step missing `depends_on` → `[]`
- reason domain `analysis|reasoning` → `general`
- compute domain `calculation` → `compute`
- JSON text whitespace / fence / unicode quotes / trailing commas

Forbidden: non-first depends_on invent, required_outputs invent, compute_spec invent, overlength truncation, truncated JSON splice, gold/task-id inference.

## Resources

{
  "llm_calls": {
    "n": 155,
    "sum": 274.0,
    "mean": 1.767741935483871,
    "median": 2.0,
    "min": 1.0,
    "max": 2.0
  },
  "tokens_in": {
    "n": 155,
    "sum": 529656.0,
    "mean": 3417.1354838709676,
    "median": 3788.0,
    "min": 1749.0,
    "max": 4581.0
  },
  "tokens_out": {
    "n": 155,
    "sum": 393228.0,
    "mean": 2536.9548387096775,
    "median": 2557.0,
    "min": 528.0,
    "max": 4096.0
  },
  "tokens_total": {
    "n": 155,
    "sum": 922884.0,
    "mean": 5954.090322580645,
    "median": 6526.0,
    "min": 2277.0,
    "max": 7964.0
  },
  "latency_ms": {
    "n": 155,
    "sum": 3402377.0882985555,
    "mean": 21950.81992450681,
    "median": 22731.65758396499,
    "min": 5322.341041872278,
    "max": 37680.54491607472
  },
  "total_llm_calls": 274,
  "total_tokens_in": 529656,
  "total_tokens_out": 393228,
  "total_tokens": 922884
}

## Next formal run

Use new directory `outputs/pe_fin_lite_retrieval_all155_planner_v2` with planner-only protocol **v2**.
Do not overwrite `live_v1`. Do not resume v1 cache into v2.
