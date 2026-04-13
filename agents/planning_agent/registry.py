"""Registration helpers for the planning agent backend."""

from __future__ import annotations

from typing import Any, Callable, Mapping

from agents.agent_core.agent_descriptor import (
    AgentDescriptor,
    aggregate_tags,
    normalize_skill_descriptors,
)
from agents.agent_core.agent_registry import DynamicAgentRegistry
from agents.planning_agent.agent_card import agent_card
from agents.planning_agent.start_agent import build_local_a2a_planning_agent


def build_planning_agent_descriptor(
    *,
    agent_id: str = "planning_agent.local",
    local_builder: Callable[[], Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> AgentDescriptor:
    """Build the local planning agent descriptor from the current AgentCard."""
    card_data = agent_card.model_dump(mode="json")
    skills = normalize_skill_descriptors(card_data.get("skills"))
    descriptor = AgentDescriptor(
        agent_id=agent_id,
        agent_name=str(card_data.get("name") or "Planning Agent"),
        description=str(card_data.get("description") or ""),
        skills=skills,
        tags=aggregate_tags(skills),
        local_builder=local_builder or build_local_a2a_planning_agent,
        cached_agent_card=agent_card,
        metadata=dict(metadata or {}),
    )
    return descriptor


def register_local_planning_agent(
    registry: DynamicAgentRegistry,
    *,
    agent_id: str = "planning_agent.local",
    local_builder: Callable[[], Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
    replace: bool = False,
) -> AgentDescriptor:
    """Register the current planning agent as a local builder-backed descriptor."""
    descriptor = build_planning_agent_descriptor(
        agent_id=agent_id,
        local_builder=local_builder,
        metadata=metadata,
    )
    return registry.register_descriptor(descriptor, replace=replace)
