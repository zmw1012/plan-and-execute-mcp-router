# Real Financial MCP Gap-Driven Discovery

**Branch:** `feature/real-financial-mcp-discovery`  
**Base audit:** [`financial_mcp_candidate_audit.md`](financial_mcp_candidate_audit.md) @ `8b31c6f`  
**Discovery date:** 2026-11-03  
**Method:** real MCP clients only (stdio / Streamable HTTP). No paid LLM. No agent/runner edits. No 30Q run. No push.

Raw evidence (gitignored): `outputs/mcp_discovery_v2/`.

---

## 1. MCP servers searched / audited

| Server | Source | Transport | Runtime used | Key? | Read-only? | Status |
|--------|--------|-----------|--------------|------|------------|--------|
| **AKTools** | `io.github.akfamily/aktools` | stdio `aktools-mcp` | `.venv-aktools-discovery` | No | Meta-tools yes | **Runnable**; prior 8 HK ops + gap probes |
| **FTShare-MCP** | https://github.com/FTShare-Lab/FTShare-MCP → hosted `https://market.ft.tech/gateway/mcp` | Streamable HTTP | `.venv-ftshare-discovery` | Recommended / anonymous quota | Appears read-oriented | **tools/list OK (250)**; **data calls blocked** (anon 500/day exhausted; no local `FTSHARE_API_KEY`) |
| **ah-disclosure-kit** | https://github.com/hc938456/ah-disclosure-kit (MIT, v1.1.3) | stdio `ah-disclosure-mcp` | `.venv-ahdisclosure-discovery` + `PYTHONPATH=vendor/.../src` | No for HKEX list | Mostly read; cleanup/LLM tools separate | **Runnable**; 40 tools |
| **mcp-hkexnews** | https://github.com/agentladle/mcp-hkexnews (`agentladle-mcp-hkexnews` 0.1.0) | stdio | `.venv-hkexnews-discovery` (`mcp<2` for FastMCP) | No | Download writes local cache only | **Runnable**; 6 tools; 1 results PDF sample |

Safety notes: isolated venvs; no curl|bash installers; no trading/order tools invoked; vendor clones + discovery outputs gitignored; PDFs not committed.

---

## 2. Actual `tools/list`

### AKTools (3 exposed MCP tools)
`search_interfaces`, `describe_interface`, `get_data` — underlying HK financial operations discovered via search/describe/get_data (not by importing AKShare).

### FTShare (250 exposed MCP tools; sample finance-relevant)
`ft_v1_finance_income`, `ft_v1_finance_balance`, `ft_v1_finance_cashflow`, `ft_v2_finance_cashflow_stock_code`, `ft_v1_stock_dividends`, `ft_v2_report_stock_reports`, `ft_v1_report_announcements_list`, plus HK market tools (`ft_v2_hk_hk_candlesticks`, …). Full list: `outputs/mcp_discovery_v2/ftshare_audit.json`.

### ah-disclosure-kit (40 tools)
Includes: `search_filings`, `search_annual_report`, `find_filing_source_tool`, `ensure_filing_evidence_tool`, `download_and_ingest_*`, `get_financial_statements_tool`, `get_financial_indicators_tool`, `get_dividends_tool`, `get_company_profile_tool`, local PDF search/pages, plus cleanup/LLM-analysis tools. Full list: `outputs/mcp_discovery_v2/ah_disclosure_tools_list.json`.

### mcp-hkexnews (6 tools)
`lookup_stock_code`, `list_hkexnews_announcements`, `download_hkexnews_announcement`, `parse_hkexnews_announcement`, `keyword_search`, `get_announcement_pages`.

---

## 3. New candidate operations (this round)

Prior accepted baseline: **8** AKTools HK operations (see candidate audit §7).

### 3.1 AKTools / AKShare operations (via meta-tools)

| Operation | Probe | Class | Notes |
|-----------|-------|-------|-------|
| `stock_financial_hk_report_em` **bank IS/BS/CF line items** (same op, new capability evidence) | HSBC 利润表/资产负债/现金流 heads OK | `core_gold` for **line-item identity** (e.g. 净利息收入); **not** gold for HSBC absolute USD amounts | Bank template present; avoids corporate gross-margin path. Currency/unit still unreliable for HSBC. |
| `stock_hk_fhpx_detail_ths` | OK with symbol `0700` | `equivalent_alternative` | Dividend/allotment history (THS). Complements `stock_hk_dividend_payout_em`. |
| `stock_hk_scale_comparison_em` | OK `00700` | `distractor_valid` / comparative | Latest scale ranks (mkt cap / revenue / profit); not FY-labeled gold. |
| `stock_hk_valuation_baidu` | OK | `equivalent_alternative` / supporting | Valuation time series (Baidu). |
| `stock_hk_indicator_eniu` | OK with `hk00700` + indicator (`roe`/`pe`) after earlier JSON failures | `supporting_evidence` | Historical ROE/PE; not a substitute for report-period gold. |
| `stock_hk_gxl_lg` | OK **without** symbol (HSI yield series) | `distractor_valid` | Index dividend yield — not issuer fundamentals. |
| `stock_individual_basic_info_hk_xq` | Error: needs `xq_a_token` | `reject` | Auth wall. |
| Dedicated non-IFRS / segment / reporting-currency interfaces | search keywords empty / A-share BS only | `reject` (gap) | No HK segment or non-IFRS numeric op found in AKTools search. |

### 3.2 FTShare operations

| Operation | Probe | Class | Notes |
|-----------|-------|-------|-------|
| `ft_v1_finance_income` / `balance` / `cashflow` | Schema OK (`stock_code`,`year`,`report_type`); call → `UPSTREAM_UNAVAILABLE` anon quota | `reject` *(blocked)* | Cannot verify HSBC raw USD vs CNY without API key. Keep as **blocked candidate**, not accepted. |
| `ft_v1_stock_dividends`, `ft_v2_report_stock_reports` | Schema inspected; not successfully fetched | `reject` *(blocked)* | Same quota/key blocker. |

### 3.3 ah-disclosure-kit operations

| Operation | Probe | Class | Notes |
|-----------|-------|-------|-------|
| `search_annual_report` | Tencent/HSBC/Alibaba annual titles + HKEXnews URLs | `supporting_evidence` | Full annual report discovery (beyond mcp-hkexnews v0.1 scope). HK local listings OK (`00700`,`00005`,`09988`). |
| `find_filing_source_tool` | OK; returns selected filing metadata | `supporting_evidence` | List/source without bulk download. |
| `get_financial_statements_tool` | OK; wraps `stock_financial_hk_report_em` | `equivalent_alternative` | Same East Money/AKShare pedigree as AKTools — Equal Function Set member, not independent gold. |
| `get_financial_indicators_tool` / `get_dividends_tool` / `get_company_profile_tool` | OK | `equivalent_alternative` | Same. |
| `search_filings` with generic category | Returns recent misc notices | `distractor_valid` unless tightly filtered | Weak category semantics for “业绩公告”. |
| cleanup / LLM analysis tools | Not used for catalog | `reject` | Mutating or LLM-coupled; out of read-only financial catalog. |

### 3.4 mcp-hkexnews operations

| Operation | Probe | Class | Notes |
|-----------|-------|-------|-------|
| `list_hkexnews_announcements` | Tencent/HSBC/Alibaba results titles listed | `supporting_evidence` | FY/interim results announcements discoverable. |
| `download_hkexnews_announcement` | 1× Tencent FY2025 annual results PDF | `supporting_evidence` | **Must** pass `release_date` + short `title_keyword` (and/or `news_id`); `news_id` alone can fall through to wrong first hit. |
| `parse_hkexnews_announcement` | 58 pages parsed | `supporting_evidence` | |
| `keyword_search` | Hits page 2 revenues RMB 751,766m; page 20 **Non-IFRS** | `supporting_evidence` / partial `core_gold` for evidence questions | Locates reported + non-IFRS language + currency. |
| `get_announcement_pages` | Pages 1–2 OK | `supporting_evidence` | Page-level text. |
| `lookup_stock_code` | `00700` → stockId 7609 | `supporting_evidence` | |

Full Annual/Interim **report** PDFs remain out of scope for mcp-hkexnews v0.1 (`t1=40000`); use ah-disclosure for that path.

---

## 4. Accept / reject summary

| Bucket | Count (this round, distinct ops) | Examples |
|--------|----------------------------------|----------|
| **Accepted into recommended catalog** | **11 new** (plus prior 8) | fhpx, valuation_baidu, scale (comparative), eniu (supporting), hkexnews×6, ah `search_annual_report` + `find_filing_source_tool` |
| **Equivalent / shared-pool only** | 4 | ah financial statement/indicator/dividend/profile wrappers |
| **Rejected / blocked** | 6+ | xueqiu token; FTShare data (no key); gxl as issuer tool; LLM/cleanup; invented non-IFRS APIs |

Principle: quantity never overrides reliability. FTShare finance tools stay **out** of the accepted pool until authenticated probes reconcile currency.

---

## 5. Capability matrix (Tencent / HSBC / Alibaba)

| Capability (P0/P1) | Tencent | HSBC | Alibaba | Primary MCP path |
|--------------------|---------|------|---------|------------------|
| Reported revenue / profit / basic EPS (numeric) | FY2025 exact vs mock | **Fail** absolute vs IR USD | FY2026 exact vs mock | AKTools `analysis_indicator_em` / `report_em` |
| Bank-specific IS (NII, PBT, PAT) | n/a | Line items **present**; amounts not gold | n/a | AKTools `report_em` 利润表 |
| Report currency metadata | CNY labeled correctly | Label “美元” but scale unexplained | CNY labeled correctly | analysis_indicator `CURRENCY`; HKEX text |
| Reported vs non-IFRS / non-GAAP | Numeric non-IFRS **missing**; **page evidence** in results PDF | Constant-currency / notable items **missing** numerically | non-GAAP diluted EPS **missing** | mcp-hkexnews keyword/pages (evidence only) |
| Official results announcement + URL/date/pages | Verified FY2025 sample | Titles listed (no extra PDF this round) | FY2026 results title listed earlier | mcp-hkexnews list (+1 download) |
| Full annual report PDF path | Title+URL via ah-disclosure | Title+URL via ah-disclosure | FY2026 annual via ah-disclosure | ah-disclosure `search_annual_report` |
| BS / CF detail | Heads OK | Bank BS/CF heads OK | BS head OK | AKTools `report_em` |
| Segments | **Gap** | **Gap** | **Gap** | none found |
| Dividends / ROE / valuation | Available | Dividends OK; ROE in analysis | Available | dividend / eniu / valuation / analysis |

---

## 6. Numeric reconciliation vs `data/mock/financial.json`

Artifact: `outputs/mcp_discovery_v2/reconciliation_summary.json`.

| Check | Result class |
|-------|----------------|
| Tencent FY2025 revenue 751,766,000,000 & basic EPS 24.749 vs analysis_indicator | **exact match** |
| Tencent FY2025 revenues in HKEX results PDF (RMB millions 751,766) | **exact match** (unit = millions in PDF) |
| Tencent non-IFRS profit (mock 259,626m) via numeric MCP | **source unavailable** (numeric); **supporting evidence** only via PDF “Non-IFRS” section |
| Alibaba FY2026 revenue 1,023,670,000,000 & basic EPS 5.7 | **exact match** |
| Alibaba non-GAAP diluted EPS | **source unavailable** / **reported/adjusted mismatch** risk if using reported EPS |
| HSBC FY2025 mock revenue USD 68.274bn vs MCP OPERATE_INCOME ~4.53e11 | **unexplained mismatch** (label USD) |
| HSBC mock basic EPS USD 1.21 vs MCP BASIC_EPS ~8.50 | **definition mismatch** and/or scale issue |
| HSBC NII line present (净利息收入) | **definition** OK as bank item; absolute amount **not** treated as gold |

External MCP values are **never** auto-promoted to gold.

---

## 7. Equal Function Set candidates (ETOM)

| Set | Members | Role |
|-----|---------|------|
| **HK statement tables** | AKTools `stock_financial_hk_report_em` ≡ ah `get_financial_statements_tool` | Same upstream; interchangeable for coverage, not independent corroboration |
| **HK analysis indicators** | AKTools `stock_financial_hk_analysis_indicator_em` ≡ ah `get_financial_indicators_tool` | Same |
| **Dividends** | `stock_hk_dividend_payout_em` ≡ `stock_hk_fhpx_detail_ths` ≡ ah `get_dividends_tool` | Overlapping dividend histories; different vendors/schemas |
| **Valuation / comparative** | `stock_hk_valuation_comparison_em`, `stock_hk_valuation_baidu`, `stock_hk_growth_comparison_em`, `stock_hk_scale_comparison_em` | Non-gold comparative set |
| **Filing evidence** | mcp-hkexnews `{list,download,parse,keyword,pages}` ≈ ah `{search_annual_report,find_filing_source,ensure_filing_evidence,search_local_document_text,get_document_pages}` | Complementary: results announcements vs full annual PDFs |

---

## 8. Recommended final Financial catalog (shared tool pool)

### MCP servers to keep (≥2)
1. **AKTools** (numeric HK fundamentals)  
2. **mcp-hkexnews** (results-announcement evidence)  
3. *(optional third)* **ah-disclosure-kit** for full annual/interim PDF discovery — or treat as evidence Equal Set with hkexnews

### Shared pool for baseline + proposed method

**A. AKTools meta (always):** `search_interfaces`, `describe_interface`, `get_data`

**B. Named HK numeric operations (promote / keep):**
1. `stock_financial_hk_report_em`  
2. `stock_financial_hk_analysis_indicator_em`  
3. `stock_hk_company_profile_em`  
4. `stock_hk_security_profile_em`  
5. `stock_hk_dividend_payout_em`  
6. `stock_hk_financial_indicator_em` *(latest snapshot — label as-of)*  
7. `stock_hk_growth_comparison_em`  
8. `stock_hk_valuation_comparison_em`  
9. `stock_hk_fhpx_detail_ths` *(new)*  
10. `stock_hk_valuation_baidu` *(new)*  
11. `stock_hk_scale_comparison_em` *(new, comparative only)*  
12. `stock_hk_indicator_eniu` *(new, supporting)*  

**C. mcp-hkexnews evidence operations:**
13. `lookup_stock_code`  
14. `list_hkexnews_announcements`  
15. `download_hkexnews_announcement`  
16. `parse_hkexnews_announcement`  
17. `keyword_search`  
18. `get_announcement_pages`  

**D. Optional ah-disclosure (if third server allowed):** `search_annual_report`, `find_filing_source_tool` (+ local page search after **explicit** single-doc ingest)

**Do not put in shared gold pool:** FTShare until keyed+reconciled; xueqiu; bank gross-margin paths; silent non-GAAP from analysis columns; cleanup/LLM tools.

**L5 boundary:** no reliable MCP numeric for non-IFRS/non-GAAP, segments, HSBC IR-faithful USD revenue/EPS, or CET1/NIM/RoTE as dedicated fields — agent must abstain or use filing evidence with explicit uncertainty.

---

## 9. Totals

| Metric | Count |
|--------|------:|
| MCP servers audited | **4** (AKTools, FTShare, ah-disclosure, mcp-hkexnews) |
| MCP servers with runnable tools/list | **4** |
| MCP servers with accepted data path | **3** (FTShare list-only / data blocked) |
| Exposed MCP tools (sum of tools/list) | 3 + 250 + 40 + 6 = **299** |
| Discoverable HK-relevant operations audited this round | **≥20** named candidates |
| Accepted into recommended Financial catalog (named ops, excl. 3 meta) | **18** (12 AKTools-family + 6 hkexnews) |
| Rejected or blocked | **≥6** (xq, FTShare data×3+, gxl-as-issuer, LLM/cleanup) |
| New candidates with live MCP evidence (≥6 required) | **11+** |

---

## 10. Remaining capability gaps

1. **HSBC (and banks generally):** absolute amounts / EPS not reconciling to IR USD snapshots despite “美元” labels — blocks P0 bank gold from East Money alone.  
2. **Reported vs adjusted:** no numeric non-IFRS / non-GAAP / constant-currency-ex-notable ops; PDF evidence only.  
3. **Segments:** no HK segment revenue MCP found.  
4. **CET1 / NIM / RoTE:** not exposed as dedicated reliable fields in probed ops.  
5. **FTShare:** promising income/balance/cashflow schemas, but **no verified rows** without API key / after anon quota.  
6. **mcp-hkexnews:** full Annual Report PDFs out of scope v0.1; download matching is brittle without date+keyword.  
7. **Alibaba/Tencent interim listing:** some title filters returned empty in narrow windows — use broader `Results` lists.

---

## 11. Smoke / reproducibility

- Prior helper: `scripts/aktools_mcp_discovery.py`  
- This round: offline artifact check — presence of `outputs/mcp_discovery_v2/{aktools_gap_*,ftshare_audit,ah_disclosure_tools_list,hkexnews_tools_list,reconciliation_summary}.json`  
- No LLM; no agent wiring.
