"""Sub-agent invocation policy models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

DEPTH_STATE_KEY = "agent_core.subquery_depth"
DEFAULT_METADATA_ALIASES = {
    "agent_core.user_id": "user_id",
    "agent_core.a2a_context_id": "parent_context_id",
    "agent_core.a2a_task_id": "parent_task_id",
    "agent_core.trace_id": "trace_id",
    "agent_core.parent_span_id": "parent_span_id",
    "graph_id": "graph_id",
}


@dataclass(frozen=True)
class SubAgentInvocationPolicy:
    """Policy for resolving sub-agents and forwarding parent session state."""

    allowed_agent_ids: tuple[str, ...] = ()
    allowed_skill_ids: tuple[str, ...] = ()
    max_depth: int = 1
    forwarded_state_keys: tuple[str, ...] = (
        "agent_core.user_id",
        "agent_core.a2a_context_id",
        "agent_core.a2a_task_id",
        "agent_core.trace_id",
        "agent_core.parent_span_id",
        "graph_id",
    )
    metadata_aliases: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_METADATA_ALIASES))
    static_metadata: dict[str, Any] = field(default_factory=dict)
