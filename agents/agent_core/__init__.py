"""Shared agent core abstractions."""

from agents.agent_core.a2a import (
    A2AFlowResult,
    OrchestrationMode,
    build_agent_card_from_yaml,
    build_get_task_request,
    build_local_a2a_agent,
    build_message_payload,
    build_post_request,
    extract_task_id,
    local_a2a_orchestration_mode,
    run_local_a2a_orchestration,
    set_local_a2a_orchestration_mode,
)
from agents.agent_core.adk import (
    AdkA2aExecutionWrapper,
    ConfiguredA2aExecutor,
)
from agents.agent_core.inference import (
    InferenceProviderLlmAdapter,
    default_instruction_prompt_name,
    load_agent_instruction,
)
from agents.agent_core.routing import (
    AgentBackendType,
    AgentDescriptor,
    AgentHealthStatus,
    AgentInvocationResult,
    AgentResolver,
    BaseAgentHandle,
    DynamicAgentRegistry,
    HostOrchestrator,
    LocalA2AHandle,
    RemoteA2AHandle,
    SkillDescriptor,
)

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
    "local_a2a_orchestration_mode",
    "run_local_a2a_orchestration",
    "set_local_a2a_orchestration_mode",
    "default_instruction_prompt_name",
    "load_agent_instruction",
]
