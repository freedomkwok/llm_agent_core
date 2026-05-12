# SPDX-License-Identifier: Apache-2.0
"""Registration helpers for zep_agent backend."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from imp_agent_core.agents.agent_core.a2a import (
    A2aRuntime,
    LocalA2aAgent,
    OrchestrationMode,
    build_agent_card_from_yaml,
    configured_a2a_runtime,
    set_local_a2a_orchestration_mode,
)
from imp_agent_core.agents.agent_core.adk import ConfiguredA2aExecutor
from imp_agent_core.agents.agent_core.routing import (
    AgentDescriptor,
    DynamicAgentRegistry,
    build_local_descriptor_from_agent_card,
    get_global_agent_registry,
)
from imp_agent_core.agents.zep_agent._env import bootstrap_env

bootstrap_env()
config_path = Path(__file__).with_name("config.yaml")
agent_card = build_agent_card_from_yaml(config_path, config_section="card_config")


def build_local_a2a_zep_agent(
    *,
    mode: OrchestrationMode = OrchestrationMode.AGENT_INTERNAL,
    config_section: str = "executor_config",
) -> Any:
    """Build local A2A zep agent from shared config-driven executor."""
    agent_class: type[Any]
    if configured_a2a_runtime() == A2aRuntime.VERTEXAI:
        from vertexai.preview.reasoning_engines import A2aAgent

        agent_class = A2aAgent
    else:
        agent_class = LocalA2aAgent

    agent = agent_class(
        agent_card=agent_card,
        agent_executor_builder=lambda: ConfiguredA2aExecutor(
            config_path=config_path, config_section=config_section
        ),
    )
    set_local_a2a_orchestration_mode(agent, mode)
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


def register_zep_worker_agent(
    registry: DynamicAgentRegistry | None = None,
    *,
    replace: bool = True,
) -> AgentDescriptor:
    """Register a worker-safe zep agent in the shared or provided registry."""
    target_registry = registry or get_global_agent_registry()
    return register_local_zep_agent(
        target_registry,
        agent_id="zep_agent.worker",
        local_builder=lambda: build_local_a2a_zep_agent(
            mode=OrchestrationMode.AGENT_INTERNAL,
            config_section="worker_executor_config",
        ),
        replace=replace,
    )

