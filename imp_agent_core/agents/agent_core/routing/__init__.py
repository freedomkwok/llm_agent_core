# SPDX-License-Identifier: Apache-2.0
"""Agent catalog, registry, and capability routing helpers."""

from imp_agent_core.agents.agent_core.routing.descriptor import (
    AgentBackendType,
    AgentDescriptor,
    AgentHealthStatus,
    SkillDescriptor,
    aggregate_tags,
    build_local_descriptor_from_agent_card,
    normalize_skill_descriptors,
)
from imp_agent_core.agents.agent_core.routing.handle import (
    AgentInvocationResult,
    BaseAgentHandle,
    LocalA2AHandle,
    RemoteA2AHandle,
    build_agent_handle,
)
from imp_agent_core.agents.agent_core.routing.orchestrator import HostOrchestrator
from imp_agent_core.agents.agent_core.routing.registry import (
    DynamicAgentRegistry,
    get_global_agent_registry,
    register_agent_package,
    reset_global_agent_registry,
)
from imp_agent_core.agents.agent_core.routing.resolver import AgentResolver

__all__ = [
    "AgentBackendType",
    "AgentDescriptor",
    "AgentHealthStatus",
    "AgentInvocationResult",
    "AgentResolver",
    "BaseAgentHandle",
    "DynamicAgentRegistry",
    "HostOrchestrator",
    "LocalA2AHandle",
    "RemoteA2AHandle",
    "SkillDescriptor",
    "aggregate_tags",
    "build_agent_handle",
    "build_local_descriptor_from_agent_card",
    "get_global_agent_registry",
    "normalize_skill_descriptors",
    "register_agent_package",
    "reset_global_agent_registry",
]
