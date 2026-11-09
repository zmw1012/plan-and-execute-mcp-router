"""Versioned, company-generic financial long-table item alias map (APV-Fin v1.4).

Maps known STD_ITEM_NAME / STD_ITEM_CODE tokens to canonical metric fields.
Never contains gold numeric values or task/company branches.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Dict, Optional, Tuple

FINANCIAL_ITEM_ALIAS_VERSION = "apv_fin_financial_item_alias_v1"

# Wide-row field aliases (analysis-indicator style).
WIDE_FIELD_ALIASES: Dict[str, str] = {
    "BASIC_EPS": "basic_eps",
    "DILUTED_EPS": "diluted_eps",
    "OPERATE_INCOME": "revenue",
    "HOLDER_PROFIT": "attributable_profit",
    "CURRENCY": "currency",
    "REPORT_DATE": "report_date",
}

# Long-table STD_ITEM_NAME / code → canonical metric.
# Names are generic IFRS/HKFRS presentation tokens, not issuer-specific.
LONG_TABLE_ITEM_ALIASES: Dict[str, str] = {
    # Revenue / turnover
    "revenue": "revenue",
    "turnover": "revenue",
    "operating revenue": "revenue",
    "operating income": "revenue",
    "total revenue": "revenue",
    "营业收入": "revenue",
    "营业额": "revenue",
    "營業額": "revenue",
    "營業收入": "revenue",
    "收入": "revenue",
    # Attributable profit
    "profit attributable to owners of the company": "attributable_profit",
    "profit attributable to equity holders": "attributable_profit",
    "profit attributable to shareholders": "attributable_profit",
    "profit attributable to ordinary shareholders": "attributable_profit",
    "net profit attributable to shareholders": "attributable_profit",
    "net income attributable to shareholders": "attributable_profit",
    "attributable profit": "attributable_profit",
    "归属于母公司股东的净利润": "attributable_profit",
    "歸屬於母公司擁有人應佔溢利": "attributable_profit",
    "母公司股东应占利润": "attributable_profit",
    # EPS
    "basic earnings per share": "basic_eps",
    "basic eps": "basic_eps",
    "earnings per share-basic": "basic_eps",
    "基本每股收益": "basic_eps",
}

LONG_TABLE_CODE_ALIASES: Dict[str, str] = {
    # Keep empty/minimal — codes vary by vendor; prefer name matching.
}


def financial_item_alias_map_hash() -> str:
    blob = json.dumps(
        {
            "version": FINANCIAL_ITEM_ALIAS_VERSION,
            "wide": WIDE_FIELD_ALIASES,
            "long_names": LONG_TABLE_ITEM_ALIASES,
            "long_codes": LONG_TABLE_CODE_ALIASES,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _norm_name(text: str) -> str:
    t = str(text or "").strip().lower()
    t = re.sub(r"\s+", " ", t)
    return t


def map_long_table_item(
    *,
    item_name: Any = None,
    item_code: Any = None,
) -> Optional[str]:
    code = str(item_code or "").strip()
    if code and code in LONG_TABLE_CODE_ALIASES:
        return LONG_TABLE_CODE_ALIASES[code]
    name = _norm_name(item_name or "")
    if not name:
        return None
    if name in LONG_TABLE_ITEM_ALIASES:
        return LONG_TABLE_ITEM_ALIASES[name]
    # Soft contains for long descriptive labels.
    for key, canon in LONG_TABLE_ITEM_ALIASES.items():
        if key in name or name in key:
            return canon
    return None


def map_wide_field(field_name: str) -> Optional[str]:
    raw = str(field_name or "").strip()
    if raw in WIDE_FIELD_ALIASES:
        return WIDE_FIELD_ALIASES[raw]
    upper = raw.upper()
    if upper in WIDE_FIELD_ALIASES:
        return WIDE_FIELD_ALIASES[upper]
    return None
