"""Evidence ledger for APV-Fin trajectories."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence
import uuid


SOURCE_TYPES = (
    "mcp_invoke",
    "compute",
    "reason",
    "verify",
    "planner",
    "monitor",
    "user_question",
)

VALIDATION_STATUSES = (
    "unchecked",
    "valid",
    "invalid",
    "ambiguous",
    "unsupported",
)


@dataclass
class EvidenceItem:
    evidence_id: str
    step_id: str
    claim: str
    source_type: str
    operation_id: Optional[str] = None
    entity: Optional[str] = None
    period_or_date: Optional[str] = None
    metric_basis: Optional[str] = None
    currency_or_unit: Optional[str] = None
    raw_value: Any = None
    normalized_value: Any = None
    provenance: Optional[str] = None
    validation_status: str = "unchecked"
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        return payload


class EvidenceLedger:
    def __init__(self) -> None:
        self.items: List[EvidenceItem] = []

    def add(
        self,
        *,
        step_id: str,
        claim: str,
        source_type: str,
        operation_id: Optional[str] = None,
        entity: Optional[str] = None,
        period_or_date: Optional[str] = None,
        metric_basis: Optional[str] = None,
        currency_or_unit: Optional[str] = None,
        raw_value: Any = None,
        normalized_value: Any = None,
        provenance: Optional[str] = None,
        validation_status: str = "unchecked",
        evidence_id: Optional[str] = None,
        **extras: Any,
    ) -> EvidenceItem:
        if source_type not in SOURCE_TYPES:
            raise ValueError(f"invalid source_type: {source_type}")
        if validation_status not in VALIDATION_STATUSES:
            raise ValueError(f"invalid validation_status: {validation_status}")
        item = EvidenceItem(
            evidence_id=evidence_id or f"ev_{uuid.uuid4().hex[:12]}",
            step_id=step_id,
            claim=claim,
            source_type=source_type,
            operation_id=operation_id,
            entity=entity,
            period_or_date=period_or_date,
            metric_basis=metric_basis,
            currency_or_unit=currency_or_unit,
            raw_value=raw_value,
            normalized_value=normalized_value,
            provenance=provenance,
            validation_status=validation_status,
            extras=dict(extras),
        )
        self.items.append(item)
        return item

    def get(self, evidence_id: str) -> Optional[EvidenceItem]:
        for item in self.items:
            if item.evidence_id == evidence_id:
                return item
        return None

    def for_steps(self, step_ids: Sequence[str]) -> List[EvidenceItem]:
        wanted = set(step_ids)
        return [i for i in self.items if i.step_id in wanted]

    def to_list(self) -> List[Dict[str, Any]]:
        return [i.to_dict() for i in self.items]

    def summary(self) -> Dict[str, Any]:
        by_source: Dict[str, int] = {}
        by_validation: Dict[str, int] = {}
        for item in self.items:
            by_source[item.source_type] = by_source.get(item.source_type, 0) + 1
            by_validation[item.validation_status] = (
                by_validation.get(item.validation_status, 0) + 1
            )
        return {
            "n_evidence": len(self.items),
            "by_source_type": by_source,
            "by_validation_status": by_validation,
            "operation_ids": sorted(
                {i.operation_id for i in self.items if i.operation_id}
            ),
        }
