"""Experiment mode identifiers for the agent/router framework."""

from __future__ import annotations

from enum import Enum


class ExperimentMode(str, Enum):
    """Three planned experimental settings.

    - flat: all tools visible in one flat tool list (placeholder)
    - routed: domain router then tool call (placeholder)
    - routed_structured: routed + structured output constraints (placeholder)
    """

    FLAT = "flat"
    ROUTED = "routed"
    ROUTED_STRUCTURED = "routed_structured"


def parse_mode(value: str) -> ExperimentMode:
    try:
        return ExperimentMode(value.strip().lower())
    except ValueError as exc:
        raise ValueError(
            f"Unsupported experiment mode: {value}. "
            f"Choose from {[m.value for m in ExperimentMode]}"
        ) from exc
