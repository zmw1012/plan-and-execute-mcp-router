"""Dataclasses for the MCP-Zero adapted baseline.

All classes are plain data holders with ``to_dict`` helpers so traces can be
written to ``predictions.jsonl`` without custom encoders. No gold / reference
information is ever stored here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ActiveToolRequest:
    """Parsed ``<tool_assistant>`` block: free-text server + tool descriptions."""

    server: str
    tool: str
    raw: str = ""
    round_index: int = 0
    source: str = "llm"  # llm | extension

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def key(self) -> str:
        """Normalised identity used for duplicate-request detection."""
        return f"{self.server.strip().lower()}||{self.tool.strip().lower()}"


@dataclass
class ServerHit:
    server_id: str
    score: float
    description_score: float = 0.0
    summary_score: Optional[float] = None
    rank: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ToolHit:
    server_id: str
    tool_name: str
    operation_id: str
    server_score: float
    tool_score: float
    final_score: float
    description: str = ""
    rank: int = 0

    def to_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        # Descriptions can be long; keep traces compact.
        out["description"] = (self.description or "")[:240]
        return out


@dataclass
class ExtensionRound:
    """One retrieval round. ``round_index == 0`` is the initial request."""

    round_index: int
    request: ActiveToolRequest
    servers: List[ServerHit] = field(default_factory=list)
    tools: List[ToolHit] = field(default_factory=list)
    new_operation_ids: List[str] = field(default_factory=list)
    is_extension: bool = False
    stop_reason: Optional[str] = None
    latency_ms: float = 0.0
    embedding_calls: int = 0
    embedding_tokens: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "round_index": self.round_index,
            "is_extension": self.is_extension,
            "request": self.request.to_dict(),
            "servers": [s.to_dict() for s in self.servers],
            "tools": [t.to_dict() for t in self.tools],
            "new_operation_ids": list(self.new_operation_ids),
            "stop_reason": self.stop_reason,
            "latency_ms": round(float(self.latency_ms), 3),
            "embedding_calls": self.embedding_calls,
            "embedding_tokens": self.embedding_tokens,
        }


@dataclass
class McpZeroTrace:
    """Aggregate retrieval trace for one task."""

    rounds: List[ExtensionRound] = field(default_factory=list)
    termination_reason: Optional[str] = None

    @property
    def initial_round(self) -> Optional[ExtensionRound]:
        return self.rounds[0] if self.rounds else None

    @property
    def extension_rounds(self) -> List[ExtensionRound]:
        return [r for r in self.rounds if r.is_extension]

    @property
    def number_of_extension_rounds(self) -> int:
        return len(self.extension_rounds)

    def active_requests(self) -> List[Dict[str, Any]]:
        return [r.request.to_dict() for r in self.rounds]

    def retrieved_servers(self) -> List[Dict[str, Any]]:
        init = self.initial_round
        return [s.to_dict() for s in init.servers] if init else []

    def retrieved_tools(self) -> List[Dict[str, Any]]:
        init = self.initial_round
        return [t.to_dict() for t in init.tools] if init else []

    def expanded_servers(self) -> List[Dict[str, Any]]:
        seen = {s["server_id"] for s in self.retrieved_servers()}
        out: List[Dict[str, Any]] = []
        for rnd in self.extension_rounds:
            for hit in rnd.servers:
                if hit.server_id not in seen:
                    seen.add(hit.server_id)
                    out.append(hit.to_dict())
        return out

    def expanded_tools(self) -> List[Dict[str, Any]]:
        seen = {t["operation_id"] for t in self.retrieved_tools()}
        out: List[Dict[str, Any]] = []
        for rnd in self.extension_rounds:
            for hit in rnd.tools:
                if hit.operation_id not in seen:
                    seen.add(hit.operation_id)
                    out.append(hit.to_dict())
        return out

    def candidate_operation_ids(self) -> List[str]:
        """Union of tools shown to the model, first-seen order."""
        seen: List[str] = []
        for rnd in self.rounds:
            for hit in rnd.tools:
                if hit.operation_id not in seen:
                    seen.append(hit.operation_id)
        return seen

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rounds": [r.to_dict() for r in self.rounds],
            "termination_reason": self.termination_reason,
            "number_of_extension_rounds": self.number_of_extension_rounds,
        }
