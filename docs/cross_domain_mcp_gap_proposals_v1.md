# Cross-domain MCP Gap Proposals v1

**Status:** proposals only — **do not** modify frozen Financial / Market / News catalogs without user approval.

## Observed gaps (not blocking v1)

1. **No free FX rate tool** for converting USD↔HKD/CNY on cutoff-consistent dates.  
   Impact: cross-currency arithmetic must refuse (see currency-conflict items).  
   Proposal: optional free FX MCP/interface later, or explicit offline FX table freeze.

2. **No historical HK media archive** with date filters (`stock_news_em` floating only).  
   Impact: cannot build News media × Market/Financial dated gold.  
   Already documented in News discovery; unchanged.

3. **ah-disclosure `search_filings` date filter unreliable.**  
   Impact: Cross-domain formal gold uses `list_hkexnews_announcements` / `search_annual_report` instead.

4. **No dedicated corporate-event calendar for HK** (A-share gsrl distractor only).  
   Impact: event dating relies on HKEXnews list titles/categories.

## v1 stance

Ship Cross-domain v1 using existing frozen operations only. Revisit gaps after user confirmation.
