"""Minimal versioned canonical output vocabulary for HK155 PE-Fin Lite.

Not a capability compiler / ontology. Planner required_outputs must use these
field IDs only. Scope (entity/period/date/basis) lives in scope_constraints.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Sequence, Tuple

CANONICAL_OUTPUT_VOCABULARY_VERSION = "hk155_canonical_outputs_v1"

# Fields the HK runtime adapters / disclosure chain already expose.
CANONICAL_OUTPUT_FIELDS: Tuple[str, ...] = (
    "issuer_code",
    "company_name",
    "report_date",
    "fiscal_period",
    "currency",
    "revenue",
    "attributable_profit",
    "basic_eps",
    "close",
    "volume",
    "trade_date",
    "adjustment",
    "news_id",
    "title",
    "release_date",
    "local_key",
    "snippet",
    "page_number",
    "source_url",
)

SCOPE_CONSTRAINT_KEYS: Tuple[str, ...] = (
    "entity",
    "period",
    "trade_date",
    "from_date",
    "to_date",
    "metric_basis",
    "adjustment",
    "currency",
    "title_keyword",
)

SCOPE_CONTRACT_VERSION = "pe_fin_lite_scope_constraints_v2"


def canonical_output_vocabulary_hash() -> str:
    blob = json.dumps(
        {
            "version": CANONICAL_OUTPUT_VOCABULARY_VERSION,
            "fields": list(CANONICAL_OUTPUT_FIELDS),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def scope_contract_hash() -> str:
    # Include length budgets so protocol identity moves when limits change.
    from hk_equity_mcp.pe_fin_lite.schemas import (
        ADJUSTMENT_MAX_LENGTH,
        CURRENCY_MAX_LENGTH,
        METRIC_BASIS_MAX_LENGTH,
    )

    blob = json.dumps(
        {
            "version": SCOPE_CONTRACT_VERSION,
            "keys": list(SCOPE_CONSTRAINT_KEYS),
            "metric_basis_max_length": METRIC_BASIS_MAX_LENGTH,
            "adjustment_max_length": ADJUSTMENT_MAX_LENGTH,
            "currency_max_length": CURRENCY_MAX_LENGTH,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def is_canonical_output(name: str) -> bool:
    return str(name) in CANONICAL_OUTPUT_FIELDS


def validate_required_outputs(required_outputs: Sequence[str]) -> List[str]:
    """Return list of non-canonical IDs (empty if all valid)."""
    bad: List[str] = []
    for item in required_outputs or []:
        if not is_canonical_output(str(item)):
            bad.append(str(item))
    return bad


def vocabulary_for_profile() -> Dict[str, Any]:
    return {
        "version": CANONICAL_OUTPUT_VOCABULARY_VERSION,
        "fields": list(CANONICAL_OUTPUT_FIELDS),
        "notes": (
            "required_outputs must be canonical field IDs only; "
            "put entity/period/date/basis into scope_constraints"
        ),
    }
