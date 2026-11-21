"""Iterative Capability Extension policy and termination reasons.

Paper: after receiving retrieved tools the model may refine its Active Tool
Request and re-retrieve until suitable tools are found or it proceeds with
parametric knowledge / current candidates.

This module only decides whether a *new* request should trigger another
retrieval round. It never reads gold labels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional, Set

from hk_equity_mcp.mcp_zero import DEFAULT_MAX_EXTENSION_ROUNDS
from hk_equity_mcp.mcp_zero.schemas import ActiveToolRequest, ExtensionRound, McpZeroTrace

# Stable reason codes written into traces / run metadata.
TERMINATION_TASK_COMPLETE = "task_complete"
TERMINATION_MAX_EXTENSION_ROUNDS = "max_extension_rounds"
TERMINATION_DUPLICATE_REQUEST = "duplicate_request"
TERMINATION_NO_NEW_TOOLS = "no_new_tools"
TERMINATION_EMPTY_REQUEST = "empty_request"
TERMINATION_NO_SERVERS = "no_servers_matched"
TERMINATION_MODEL_PROCEEDED = "model_proceeded"
TERMINATION_MAX_LLM_TURNS = "max_llm_turns"
TERMINATION_TIMEOUT = "timeout"


@dataclass
class ExtensionDecision:
    allow: bool
    reason: Optional[str] = None
    is_extension: bool = False


@dataclass
class ExtensionController:
    """Stateful gate for iterative retrieval rounds within one task."""

    max_extension_rounds: int = DEFAULT_MAX_EXTENSION_ROUNDS
    seen_request_keys: Set[str] = field(default_factory=set)
    seen_operation_ids: Set[str] = field(default_factory=set)
    initial_done: bool = False
    extension_count: int = 0
    last_stop_reason: Optional[str] = None

    def decide(self, request: Optional[ActiveToolRequest]) -> ExtensionDecision:
        if request is None or not request.server.strip() or not request.tool.strip():
            self.last_stop_reason = TERMINATION_EMPTY_REQUEST
            return ExtensionDecision(allow=False, reason=TERMINATION_EMPTY_REQUEST)

        key = request.key()
        if key in self.seen_request_keys:
            self.last_stop_reason = TERMINATION_DUPLICATE_REQUEST
            return ExtensionDecision(allow=False, reason=TERMINATION_DUPLICATE_REQUEST)

        if not self.initial_done:
            return ExtensionDecision(allow=True, reason=None, is_extension=False)

        if self.extension_count >= int(self.max_extension_rounds):
            self.last_stop_reason = TERMINATION_MAX_EXTENSION_ROUNDS
            return ExtensionDecision(
                allow=False, reason=TERMINATION_MAX_EXTENSION_ROUNDS, is_extension=True
            )
        return ExtensionDecision(allow=True, reason=None, is_extension=True)

    def commit_round(self, round_: ExtensionRound) -> ExtensionRound:
        """Record a completed retrieval round; may rewrite ``stop_reason``."""
        self.seen_request_keys.add(round_.request.key())
        new_ids = [
            oid
            for oid in round_.new_operation_ids
            if oid not in self.seen_operation_ids
        ]
        round_.new_operation_ids = list(new_ids)
        self.seen_operation_ids.update(new_ids)

        if not self.initial_done:
            self.initial_done = True
        else:
            self.extension_count += 1

        if not round_.servers:
            round_.stop_reason = TERMINATION_NO_SERVERS
            self.last_stop_reason = TERMINATION_NO_SERVERS
        elif not new_ids:
            round_.stop_reason = TERMINATION_NO_NEW_TOOLS
            self.last_stop_reason = TERMINATION_NO_NEW_TOOLS
        return round_

    def mark_model_proceeded(self) -> None:
        if self.last_stop_reason is None:
            self.last_stop_reason = TERMINATION_MODEL_PROCEEDED

    def finalize_trace(self, trace: McpZeroTrace, *, reason: Optional[str] = None) -> McpZeroTrace:
        trace.termination_reason = reason or self.last_stop_reason or TERMINATION_TASK_COMPLETE
        return trace

    def already_seen_ops(self) -> Iterable[str]:
        return list(self.seen_operation_ids)
