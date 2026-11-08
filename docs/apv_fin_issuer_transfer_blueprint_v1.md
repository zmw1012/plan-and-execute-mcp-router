# APV-Fin Issuer-Transfer Blueprint v1

**Status:** design only. Do **not** generate unverified gold in this round.
Do **not** run Agent experiments on these issuers in this round.

## Candidate issuers

1. **Pop Mart International Group Limited**
   - HK code: `09992`
2. **Xiaomi Corporation**
   - HKD counter: `01810`
   - RMB counter: `81810`

## Planned frozen one-shot suite (6 items)

After planning freeze, run **once**:

1. Pop Mart financial
2. Pop Mart official disclosure
3. Pop Mart disclosure → market cross-domain
4. Xiaomi financial calculation
5. Xiaomi official disclosure
6. Xiaomi disclosure → market cross-domain

## Hard constraints

1. These transfer questions **must not** be used to modify APV-Fin prompts, metadata, budgets, or code paths.
2. Before any gold claim: perform a **direct Real-MCP live provenance audit**.
3. Freeze before formal scoring:
   - cutoff
   - gold values
   - evidence artifacts
   - acceptable operations
   - dependency graph
4. If capability cannot support an item, label it **capability gap** (do not invent substitutes).
5. Do **not** replace failed items to make the method look successful.
6. Prompts and operation metadata must remain **company-generic** (no issuer whitelist for Pop Mart / Xiaomi / baseline three issuers).

## Suggested audit checklist (future)

- [ ] Resolve issuer identity via runtime ops (`lookup_stock_code` / profile ops)
- [ ] Confirm HKD vs RMB counter handling for Xiaomi without special-casing in prompts
- [ ] Verify cutoff `2026-08-12` Asia/Hong_Kong post-filter on market series
- [ ] Capture raw MCP payloads + normalized fields for each gold claim
- [ ] Mark formal_gold_eligible ops only; keep distractors out of gold sets
- [ ] Freeze JSONL + provenance note; then one APV-Fin run; no iterative gold edits

## Non-goals for this round

- No Agent run
- No gold JSONL creation
- No prompt/metadata issuer tokens
- No ETOM run
