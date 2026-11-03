# Real Financial MCP Discovery Audit

**Branch:** `feature/real-financial-mcp-discovery` (from `269b3fa`)  
**Discovery date:** 2026-11-03  
**Method:** real MCP client over **stdio** to official `aktools-mcp` (`io.github.akfamily/aktools`).  
**Not used:** paid LLM calls; direct `import akshare` as a substitute for MCP; full 30-question reruns; mutation of existing `outputs/real_smoke_*`.

Raw MCP JSON dumps (gitignored under `outputs/`):

- `outputs/mcp_discovery/aktools_tools_list.json`
- `outputs/mcp_discovery/aktools_search_results.json`
- `outputs/mcp_discovery/aktools_describe_results.json`
- `outputs/mcp_discovery/aktools_get_data_probes.json`
- `outputs/mcp_discovery/aktools_get_data_probes_v2.json`
- `outputs/mcp_discovery/aktools_report_em_head.json`
- `outputs/mcp_discovery/aktools_availability_summary.json`
- `outputs/mcp_discovery/aktools_discovery_summary.json`

Repro helper (no LLM): `scripts/aktools_mcp_discovery.py` with `.venv-aktools-discovery` (Python ≥3.11; local exclude).

---

## 1. Controlled Local-Tool Pilot vs Real MCP

| Dimension | Current pilot (`269b3fa` / smoke) | Real Financial MCP (this discovery) |
|-----------|-----------------------------------|--------------------------------------|
| Tool surface | Local `FinancialServer` over `data/mock/financial.json` | Remote-backed AKShare interfaces via AKTools MCP |
| Evidence | Hand-verified Annual/Interim PDF snapshots | East Money / vendor tables (unverified provenance for course gold) |
| Catalog size | **2** ops (`get_financial_metric`, `get_filing_summary`) | 1000+ underlying AKShare interfaces behind **3** MCP meta-tools |
| Architecture labels | MCP-Free / Flat / Operation-Visible over **local tools** | Same agent patterns can wrap **real** MCP, but must be labeled separately |
| Claim boundary | Measures “local structured tools help vs closed-book” | Must **not** be reported as “enterprise MCP at thousands of APIs” until real MCP is wired and graded |

**Label for the completed 30×3 (or 3×3 smoke) work:** **Controlled Local-Tool Pilot**.  
It is useful and should be kept, but it is **not** a Real MCP experiment.

---

## 2. AKTools MCP server information

| Field | Observed / official |
|-------|---------------------|
| Registry name | `io.github.akfamily/aktools` |
| Official docs | https://aktools.akfamily.xyz/guide/mcp/ |
| Parent projects | https://github.com/akfamily/aktools , https://github.com/akfamily/akshare |
| Install / run (official) | stdio: `uvx --from aktools aktools-mcp` (or pip `aktools-mcp`); HTTP: `python -m aktools` → `http://127.0.0.1:8080/mcp` |
| This audit runtime | `.venv-aktools-discovery/bin/aktools-mcp` via MCP **stdio** |
| Server name | `AKTools` |
| Server version | `0.1.4` |
| Website | https://aktools.akfamily.xyz/ |
| Protocol version | `2025-11-25` |
| Transport used | **stdio** (not HTTP for this audit) |
| Auth / API key | **Not required** for public East Money-backed calls in this probe |
| AKShare version in venv | `1.20.22` |

Python note: current system `/usr/bin/python3` is 3.9; modern `aktools` MCP wheels need **≥3.11**. Discovery used Miniconda `python3.14` only for the isolated discovery venv.

---

## 3. `tools/list` result (actual MCP)

Exactly **three** tools (all `read_only_hint=True`):

| Tool | Role | Required args |
|------|------|----------------|
| `search_interfaces` | Keyword search over AKShare interfaces (≤50 hits, relevance ranked) | `keyword` |
| `describe_interface` | Params, defaults, return columns, source URL, examples | `name` |
| `get_data` | Invoke interface; `max_rows`≤1000; `offset` (neg = from end) | `name`; optional `params`, `max_rows`, `offset` |

This matches the official “search → describe → get_data” design: underlying financial endpoints are **not** each registered as separate MCP tools.

---

## 4. Underlying HK Financial operation candidates

All eight focus names were **`describe_interface`-reachable** (exist in catalog). Search also retrieved them under queries such as `港股 财务报表`, `港股 财务指标`, `valuation`, `company profile`, `dividend`.

| Operation | Exists | Key params | Returns (high level) | Source | Key? | Fit for 0700/0005/09988 | Fit for current 30Q | Risks |
|-----------|--------|------------|----------------------|--------|------|-------------------------|---------------------|-------|
| `stock_financial_hk_report_em` | Yes | `stock`, `symbol`∈{资产负债表,利润表,现金流量表}, `indicator`∈{年度,报告期} | Long item×period rows: `STD_ITEM_NAME`, `AMOUNT`, `REPORT_DATE`, … | East Money HK F10 | No | Yes (data returned) | Partial — good for statement lines; pagination/sort pitfalls | Wrong `symbol` (“财务报表”) → **0 rows**; newest-first ordering → `offset<0` returns **old** years |
| `stock_financial_hk_analysis_indicator_em` | Yes | `symbol`, `indicator`∈{年度,报告期} | Compact metrics: `OPERATE_INCOME`, `HOLDER_PROFIT`, `BASIC_EPS`, `CURRENCY`, … | East Money | No | Yes | **Best primary** for revenue/profit/EPS/date | **No non-IFRS/non-GAAP** fields; HSBC currency labels unstable on 报告期 |
| `stock_hk_financial_indicator_em` | Yes | `symbol` | Single “latest” snapshot row (mixed headline KPIs) | East Money | No | Yes | Weak for period-specific gold | Easy to confuse TTM/latest with FY; units mixed (港元 vs 元) |
| `stock_hk_security_profile_em` | Yes | `symbol` | Listing metadata, 年结日, ISIN, … | East Money | No | Yes | Supportive (fiscal year-end) | Not metric answers |
| `stock_hk_company_profile_em` | Yes | `symbol` | Company profile, industry, 年结日 | East Money | No | Yes | Supportive | Industry tags may be coarse |
| `stock_hk_dividend_payout_em` | Yes | `symbol` | Dividend history | East Money | No | Yes | Only if questions ask dividends | Fiscal year as string year; HKD wording |
| `stock_hk_growth_comparison_em` | Yes | `symbol` | Peer growth ranks | East Money | No | Yes | Low for factual 30Q | Comparative, not gold-path |
| `stock_hk_valuation_comparison_em` | Yes | `symbol` | Peer valuation ranks | East Money | No | Yes | Low for factual 30Q | Same |

**Param discovery detail (report):** `symbol` must be one of `资产负债表` / `利润表` / `现金流量表` — not a free-text “财务报表”.

**Date coverage (analysis 报告期, observed):**  
Tencent includes `2025-12-31` (annual) and `2026-06-30` (interim-like `DATE_TYPE_CODE=002`); Alibaba includes `2026-03-31` (FY year-end) and quarterly/interim dates; HSBC includes `2025-12-31` and 2026 interim/quarter dates.

---

## 5. Three-company availability matrix (MCP `get_data` only)

Symbols used: Tencent `00700`, HSBC `00005`, Alibaba `09988`.

| Check | Tencent | HSBC | Alibaba |
|-------|---------|------|---------|
| Analysis 年度 reachable | Yes | Yes | Yes |
| Analysis 报告期 reachable | Yes | Yes | Yes |
| Report 利润表 年度 reachable | Yes (`offset=0` head) | Yes | Yes |
| FY2025 / FY2026 / 1H2026-like dates present | FY2025 `2025-12-31`; 1H2026 `2026-06-30` | FY2025 `2025-12-31`; 1H-like `2026-06-30` | FY2026 `2026-03-31` (not calendar YE) |
| Revenue-like field | `OPERATE_INCOME` / 营运收入 | `OPERATE_INCOME` / 经营收入总额 | `OPERATE_INCOME` / 营业额=营运收入 |
| Net profit-like | `HOLDER_PROFIT` / 股东应占溢利 | `HOLDER_PROFIT` | `HOLDER_PROFIT` |
| EPS | `BASIC_EPS`, `DILUTED_EPS` | same | same |
| Currency field present | `人民币` | `美元` on annual; **alternates `HKD`/`美元` on 报告期** | `人民币` |
| Annual vs interim distinction | Via `indicator` + `DATE_TYPE_CODE` / `REPORT_DATE` | Same | Same (fiscal year-end `3-31`) |
| Reported vs adjusted / IFRS vs non-IFRS | **Not exposed** in analysis columns | **Not exposed**; gross margin null (good) | **Not exposed** (no non-GAAP diluted EPS) |

---

## 6. Diff vs local verified snapshot (`data/mock/financial.json`)

Local snapshot is IR/HKEX PDF-backed (`verification_status=verified`). MCP is East Money-derived.

| Company / period | Local verified | AKTools MCP (analysis / report head) | Match? |
|------------------|----------------|--------------------------------------|--------|
| Tencent FY2025 revenue | CNY **751,766,000,000** | `OPERATE_INCOME` / 营运收入 **751,766,000,000**; 营业额 **743,689,000,000** | Revenue **match** on 营运收入; 营业额 differs — do not conflate |
| Tencent FY2025 net profit | CNY **224,842,000,000** | `HOLDER_PROFIT` / 股东应占溢利 **224,842,000,000** | Match |
| Tencent FY2025 basic EPS | **24.749** | **24.749** | Match |
| Tencent FY2025 non-IFRS profit / EPS | 259,626,000,000 / 28.577 | **Absent** | Gap — cannot grade non-IFRS items from this op alone |
| Tencent 1H2026 revenue / profit / EPS | 401,243m / 114,115m / 12.639 | Same on 报告期 `2026-06-30` | Match (reported) |
| Alibaba FY2026 revenue | CNY **1,023,670,000,000** end **2026-03-31** | Same amount; `REPORT_DATE=2026-03-31`, profile 年结日 `03-31` | Date semantics **OK** (not calendar 2026-12-31) |
| Alibaba FY2026 reported basic EPS | **5.70** | **5.7** | Match |
| Alibaba FY2026 non-GAAP diluted EPS | **3.35** | Analysis shows diluted **5.5** (reported-like) | **Mismatch / missing non-GAAP** |
| Alibaba FY2026 reported net profit | **105,904,000,000** | `HOLDER_PROFIT` **103,592,000,000** | **Close but not equal** — needs evidence review before trusting MCP as gold |
| HSBC FY2025 revenue (local) | USD **68,274,000,000** (NOI before ECL) | `OPERATE_INCOME` **452,823,411,200** labeled `美元` | **Severe mismatch** — scale/definition/currency labeling unreliable |
| HSBC FY2025 EPS (local) | USD **1.21** basic | `BASIC_EPS` **8.504848** | **Mismatch** |
| HSBC gross margin | N/A (banking) | `GROSS_PROFIT` / ratio **null** on analysis | Good (does not invent margin) |
| HSBC 报告期 currency | USD in local interim | Alternates **`HKD` and `美元`** by row | High risk of unit errors |

**Known AKShare/East Money risk checklist (observed):**

1. **CNY reporters labeled as 人民币** for Tencent/Alibaba annual — OK here; still watch `IS_CNY_CODE=0` oddity.  
2. **Alibaba FY2026** correctly tied to **2026-03-31**, not calendar year-end — OK in these probes.  
3. **HSBC gross margin** not fabricated in analysis — OK; still must block agents from inventing margin via other ops.  
4. **Adjusted / non-GAAP** largely **missing** — cannot answer fin_cf-style non-IFRS/non-GAAP items from analysis alone.  
5. **`stock_financial_hk_report_em` sort:** head (`offset=0`) is newest years; negative offset returned **2001–2003** windows — opposite of “ascending → use negative offset for latest” guidance for this interface. Agents must filter by `REPORT_DATE`, not assume offset semantics.

---

## 7. Recommended keep-list (6–10 Financial operations)

For a **Real Financial MCP** Flat / Operation-Visible catalog (still behind AKTools’ 3 meta-tools, or promoted as named operations in our registry):

1. `stock_financial_hk_analysis_indicator_em` — **primary** numeric facts (period + annual)  
2. `stock_financial_hk_report_em` — statement line items (利润表/资产负债/现金流)  
3. `stock_hk_company_profile_em` — fiscal year-end / identity  
4. `stock_hk_security_profile_em` — listing / 年结日 corroboration  
5. `stock_hk_dividend_payout_em` — dividend questions only  
6. `stock_hk_financial_indicator_em` — optional latest snapshot (strictly labeled “as-of latest”, not FY)  
7. `stock_hk_growth_comparison_em` — optional comparative (non-gold)  
8. `stock_hk_valuation_comparison_em` — optional comparative (non-gold)

**Do not promote** until repaired/validated: any bank “gross margin” path; any silent non-GAAP claim from analysis columns.

---

## 8. HKEX filing MCP audit (`agentladle/mcp-hkexnews`)

Source: https://github.com/agentladle/mcp-hkexnews (README + package docs). **Code/tool audit only** — no bulk PDF download in this pass.

| Topic | Finding |
|-------|---------|
| Install | Official: `uvx agentladle-mcp-hkexnews` (Python ≥3.10, uv); alt pip `agentladle-mcp-hkexnews` |
| Transport | Local **stdio** MCP server (FastMCP) |
| Tools (6) | `list_hkexnews_announcements`, `download_hkexnews_announcement`, `parse_hkexnews_announcement`, `keyword_search`, `get_announcement_pages`, `lookup_stock_code` |
| Annual/interim **full report PDFs** | Explicitly **out of scope v0.1** (`t1=40000` periodic/ESG reports excluded). Performance announcements (results) are in scope |
| PDF download + parse | Yes (PyMuPDF); page-split JSON |
| Page numbers + text evidence | Yes via parse + `get_announcement_pages` / `keyword_search` |
| API key | Open-source local path: **no key** for HKEX scrape tooling. Related cloud “AgentLadle” products advertise keys — keep local vs cloud distinct |
| Runtime | Local cache under `~/.agentladle/mcp-hkexnews/` |
| Stability / safety risks | Scraping HKEX; rate limits; legal/ToS; incomplete coverage of full Annual Report PDFs; must not bulk-download without confirmation |

**Role in design:** complementary **evidence / filing** layer, not a drop-in replacement for East Money numeric tables. Useful later for finance-aware validation/repair citing pages.

---

## 9. Recommended formal experiment structure

Keep the local pilot archived and labeled. Add a **separate** Real MCP track:

1. **MCP-Free** — same closed-book baseline (shared model/decoding).  
2. **Flat real MCP** — expose either AKTools’ 3 meta-tools **or** a curated Flat list of named HK financial operations implemented as thin MCP wrappers; no hidden planner.  
3. **Operation-Visible real MCP** — paper-style `find_operations` / `describe_operation` / `invoke_api` over a registry whose underlying calls go to real MCP (`get_data` / HKEX tools), not local JSON.  
4. **Proposed finance-aware validation/repair** — currency, fiscal calendar (Alibaba YE), bank metric bans, reported vs non-GAAP checks; optionally HKEX page evidence.

Fairness rules (unchanged intent): same questions, model, temperature, answer contract, grading sheet; never leak gold into prompts.

---

## 10. Blockers and risks

1. **Labeling:** must not call local-tool pilot a “real MCP” result.  
2. **HSBC numeric fidelity:** East Money `OPERATE_INCOME` / EPS vs IR-verified USD figures disagree badly — blocker for bank revenue/EPS gold without a reconciliation layer.  
3. **Non-GAAP / non-IFRS gap:** blocks several current 30Q items (e.g. Alibaba non-GAAP diluted EPS, Tencent non-IFRS profit).  
4. **Alibaba net profit small drift** vs verified snapshot — needs source adjudication.  
5. **Pagination/sort footguns** on `stock_financial_hk_report_em`.  
6. **Environment:** Real MCP needs Python ≥3.11 (+ uv recommended); course `.venv` on 3.9 is insufficient for current `aktools` MCP.  
7. **HKEX MCP** does not yet supply full Annual Report PDFs in v0.1 — evidence path incomplete for FinanceBench-style page cites from full reports.  
8. **No invented interfaces:** only operations confirmed via `search_interfaces` / `describe_interface` / `get_data` should enter the registry.

---

## Next step (recommended)

1. Freeze this audit; keep local-tool pilot results under the Controlled Local-Tool Pilot label.  
2. Decide registry strategy: wrap AKTools 3 meta-tools vs promote 6–8 named HK ops with deterministic schemas.  
3. Build a **non-LLM** numeric reconciliation harness (MCP vs `data/mock/financial.json`) before any paid Real-MCP 30Q run.  
4. Only then implement Flat/OV agents against real MCP on a new branch.
