"""PE-Fin Lite v1.2-EEA identity (post-hoc exploratory; not formal primary)."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict

PE_FIN_LITE_EEA_CONDITION = "pe_fin_lite_eea"
PE_FIN_LITE_EEA_METHOD_VERSION = "pe_fin_lite_v1_2_eea"
PE_FIN_LITE_EEA_PROTOCOL_VERSION = "execution_evidence_alignment_posthoc_v1"
PE_FIN_LITE_EEA_ARCHITECTURE = "plan_first_step_level_retrieval_v1"

EEA_FLAGS: Dict[str, Any] = {
    "posthoc_exploratory": True,
    "formal_primary_result": False,
    "developed_after_hk155_result_inspection": True,
    "preserves_formal_v1_2": True,
    "no_global_replan": True,
    "no_monitor": True,
    "r5_unchanged": True,
    "budgets_unchanged": True,
}


def eea_protocol_hash() -> str:
    blob = json.dumps(
        {
            "condition": PE_FIN_LITE_EEA_CONDITION,
            "method_version": PE_FIN_LITE_EEA_METHOD_VERSION,
            "protocol_version": PE_FIN_LITE_EEA_PROTOCOL_VERSION,
            "flags": EEA_FLAGS,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()
