"""Shared agent core abstractions."""

from agents.agent_core.adk_a2a_execution_wrapper import AdkA2aExecutionWrapper
from agents.agent_core.a2a_agent_builder import build_local_a2a_agent
from agents.agent_core.a2a_agent_card_loader import build_agent_card_from_yaml
from agents.agent_core.configured_a2a_executor import ConfiguredA2aExecutor
from agents.agent_core.a2a_orchestration import (
    A2AFlowResult,
    OrchestrationMode,
    build_get_task_request,
    build_message_payload,
    build_post_request,
    extract_task_id,
    run_local_a2a_orchestration,
)
from agents.agent_core.agent_descriptor import (
    AgentBackendType,
    AgentDescriptor,
    AgentHealthStatus,
    SkillDescriptor,
)
from agents.agent_core.agent_handle import (
    AgentInvocationResult,
    BaseAgentHandle,
    LocalA2AHandle,
    RemoteA2AHandle,
)
from agents.agent_core.agent_registry import DynamicAgentRegistry
from agents.agent_core.agent_resolver import AgentResolver
from agents.agent_core.host_orchestrator import HostOrchestrator
from agents.agent_core.inference_provider_llm_adapter import InferenceProviderLlmAdapter
from agents.agent_core.adk_llm_agent_base import UserRoleNormalizedLlmAgent

__all__ = [
    "A2AFlowResult",
    "AdkA2aExecutionWrapper",
    "build_agent_card_from_yaml",
    "ConfiguredA2aExecutor",
    "AgentBackendType",
    "AgentDescriptor",
    "AgentHealthStatus",
    "AgentInvocationResult",
    "AgentResolver",
    "BaseAgentHandle",
    "DynamicAgentRegistry",
    "HostOrchestrator",
    "InferenceProviderLlmAdapter",
    "LocalA2AHandle",
    "OrchestrationMode",
    "RemoteA2AHandle",
    "SkillDescriptor",
    "build_get_task_request",
    "build_local_a2a_agent",
    "build_message_payload",
    "build_post_request",
    "extract_task_id",
    "run_local_a2a_orchestration",
    "UserRoleNormalizedLlmAgent",
]
