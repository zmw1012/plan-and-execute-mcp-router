"""Versioned, company-generic capability ontology for APV-Fin v1.2."""

from __future__ import annotations

from typing import Dict, List, Set, Tuple

CAPABILITY_ONTOLOGY_VERSION = "apv_fin_capability_ontology_v1"

# Capabilities describe abilities — never companies, task IDs, or gold.
CAPABILITY_IDS: Tuple[str, ...] = (
    "financial.reported_metric",
    "financial.reported_metric_bundle",
    "financial.ratio",
    "financial.company_profile",
    "financial.official_narrative_evidence",
    "market.daily_ohlcv",
    "market.latest_close",
    "market.index_daily",
    "market.trading_calendar",
    "disclosure.list",
    "disclosure.download",
    "disclosure.parse",
    "disclosure.keyword_search",
    "disclosure.page_read",
    "disclosure.company_lookup",
    "news.official_announcement",
    "news.media_latest",
)

# Compatible fallbacks: requesting A may accept B when exact match missing.
CAPABILITY_COMPATIBILITY: Dict[str, Tuple[str, ...]] = {
    "market.latest_close": ("market.daily_ohlcv",),
    "financial.reported_metric": ("financial.reported_metric_bundle",),
    "financial.official_narrative_evidence": (
        "disclosure.list",
        "disclosure.keyword_search",
        "disclosure.parse",
        "disclosure.page_read",
    ),
    "news.official_announcement": ("disclosure.list",),
}

# Disclosure narrative workflow (generic; not task-specific).
DISCLOSURE_NARRATIVE_WORKFLOW: Tuple[str, ...] = (
    "disclosure.list",
    "disclosure.download",
    "disclosure.parse",
    "disclosure.keyword_search",
    "disclosure.page_read",
)

# Fields commonly required by capability family.
DEFAULT_REQUIRED_OUTPUTS: Dict[str, Tuple[str, ...]] = {
    "market.daily_ohlcv": ("trade_date", "close"),
    "market.latest_close": ("trade_date", "close"),
    "market.index_daily": ("trade_date", "close"),
    "financial.reported_metric_bundle": ("revenue", "attributable_profit"),
    "financial.reported_metric": ("value",),
    "financial.ratio": ("value",),
    "financial.company_profile": ("entity",),
    "financial.official_narrative_evidence": ("snippet", "local_key"),
    "disclosure.list": ("local_key", "title"),
    "disclosure.download": ("local_key",),
    "disclosure.parse": ("local_key",),
    "disclosure.keyword_search": ("snippet", "local_key"),
    "disclosure.page_read": ("snippet", "page_number", "local_key"),
    "disclosure.company_lookup": ("issuer_code",),
    "news.media_latest": ("title",),
    "news.official_announcement": ("title", "local_key"),
    "market.trading_calendar": ("trade_date",),
}


def compatible_capabilities(required: str) -> Set[str]:
    """Exact required capability plus declared compatible substitutes."""
    out = {required}
    for alt in CAPABILITY_COMPATIBILITY.get(required, ()):
        out.add(alt)
    return out


def ontology_coverage_checklist() -> List[str]:
    return list(CAPABILITY_IDS)
