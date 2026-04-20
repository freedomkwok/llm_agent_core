"""Registration helpers for the skill route agent backend."""

from __future__ import annotations

from typing import Any, Callable, Mapping

from vertexai.preview.reasoning_engines import A2aAgent

from agents.agent_core import OrchestrationMode, build_local_a2a_agent
from agents.agent_core.agent_descriptor import AgentDescriptor, build_local_descriptor_from_agent_card
from agents.agent_core.agent_registry import DynamicAgentRegistry
from agents.skill_route_agent.agent_card import agent_card


def build_local_a2a_skill_route_agent(
    *,
    mode: OrchestrationMode = OrchestrationMode.AGENT_INTERNAL,
) -> A2aAgent:
    """Build local A2A skill route agent from the shared executor loader."""
    del mode
    return build_local_a2a_agent(
        agent_card=agent_card,
        executor_module_path="agents.skill_route_agent.a2a_executor",
        executor_class_name="SkillRouteA2aExecutor",
    )


def build_skill_route_agent_descriptor(
    *,
    agent_id: str = "skill_route_agent.local",
    local_builder: Callable[[], Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> AgentDescriptor:
    """Build the local skill route agent descriptor from the current AgentCard."""
    descriptor = build_local_descriptor_from_agent_card(
        agent_id=agent_id,
        agent_card=agent_card,
        local_builder=local_builder or build_local_a2a_skill_route_agent,
        metadata=dict(metadata or {}),
    )
    return descriptor


def register_local_skill_route_agent(
    registry: DynamicAgentRegistry,
    *,
    agent_id: str = "skill_route_agent.local",
    local_builder: Callable[[], Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
    replace: bool = False,
) -> AgentDescriptor:
    """Register the current skill route agent as a local builder-backed descriptor."""
    descriptor = build_skill_route_agent_descriptor(
        agent_id=agent_id,
        local_builder=local_builder,
        metadata=metadata,
    )
    return registry.register_descriptor(descriptor, replace=replace)
