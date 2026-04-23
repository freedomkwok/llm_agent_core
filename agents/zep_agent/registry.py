"""Registration helpers for zep_agent backend."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping

from vertexai.preview.reasoning_engines import A2aAgent

from agents.agent_core import ConfiguredA2aExecutor, OrchestrationMode, build_agent_card_from_yaml
from agents.agent_core.agent_descriptor import AgentDescriptor, build_local_descriptor_from_agent_card
from agents.agent_core.agent_registry import DynamicAgentRegistry
from agents.zep_agent._env import bootstrap_env

bootstrap_env()
config_path = Path(__file__).with_name("config.yaml")
agent_card = build_agent_card_from_yaml(config_path, config_section="card_config")


def build_local_a2a_zep_agent(
    *,
    mode: OrchestrationMode = OrchestrationMode.AGENT_INTERNAL,
) -> A2aAgent:
    """Build local A2A zep agent from shared config-driven executor."""
    del mode
    agent = A2aAgent(
        agent_card=agent_card,
        agent_executor_builder=lambda: ConfiguredA2aExecutor(
            config_path=config_path, config_section="executor_config"
        ),
    )
    agent.set_up()
    return agent


def build_zep_agent_descriptor(
    *,
    agent_id: str = "zep_agent.local",
    local_builder: Callable[[], Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> AgentDescriptor:
    """Build local zep agent descriptor from current AgentCard."""
    return build_local_descriptor_from_agent_card(
        agent_id=agent_id,
        agent_card=agent_card,
        local_builder=local_builder or build_local_a2a_zep_agent,
        metadata=dict(metadata or {}),
    )


def register_local_zep_agent(
    registry: DynamicAgentRegistry,
    *,
    agent_id: str = "zep_agent.local",
    local_builder: Callable[[], Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
    replace: bool = False,
) -> AgentDescriptor:
    """Register zep agent as local descriptor."""
    descriptor = build_zep_agent_descriptor(
        agent_id=agent_id,
        local_builder=local_builder,
        metadata=metadata,
    )
    return registry.register_descriptor(descriptor, replace=replace)

