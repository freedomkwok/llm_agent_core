"""Agent catalog, registry, and capability routing helpers."""

from agents.agent_core.routing.descriptor import (
    AgentBackendType,
    AgentDescriptor,
    AgentHealthStatus,
    SkillDescriptor,
    aggregate_tags,
    build_local_descriptor_from_agent_card,
    normalize_skill_descriptors,
)
from agents.agent_core.routing.handle import (
    AgentInvocationResult,
    BaseAgentHandle,
    LocalA2AHandle,
    RemoteA2AHandle,
    build_agent_handle,
)
from agents.agent_core.routing.orchestrator import HostOrchestrator
from agents.agent_core.routing.registry import DynamicAgentRegistry
from agents.agent_core.routing.resolver import AgentResolver

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
    "normalize_skill_descriptors",
]
