# SPDX-License-Identifier: Apache-2.0
from imp_agent_core.agents.agent_core import (
    DEFAULT_SUB_AGENT_TOOL_INSTRUCTION,
    ConfiguredA2aExecutor,
    DynamicAgentRegistry,
    InferenceProviderLlmAdapter,
    SubAgentInvocationPolicy,
    SubAgentInvoker,
    SubAgentToolConfig,
    build_agent_card_from_yaml,
    get_global_agent_registry,
    make_sub_agent_tool,
    reset_global_agent_registry,
)
from imp_agent_core.agents.agent_core.a2a.agent_card_yaml import (
    build_agent_card_from_yaml as a2a_card_loader,
)
from imp_agent_core.agents.agent_core.adk.executor import ConfiguredA2aExecutor as adk_executor
from imp_agent_core.agents.agent_core.inference.llm_adapter import (
    InferenceProviderLlmAdapter as inference_adapter,
)
from imp_agent_core.agents.agent_core.routing import (
    get_global_agent_registry as routing_global_registry,
)
from imp_agent_core.agents.agent_core.routing import (
    reset_global_agent_registry as routing_reset_global_registry,
)
from imp_agent_core.agents.agent_core.routing.registry import (
    DynamicAgentRegistry as routing_registry,
)
from imp_agent_core.agents.agent_core.sub_agent_invoke import (
    DEFAULT_SUB_AGENT_TOOL_INSTRUCTION as sub_agent_instruction,
)
from imp_agent_core.agents.agent_core.sub_agent_invoke import (
    SubAgentInvocationPolicy as sub_agent_policy,
)
from imp_agent_core.agents.agent_core.sub_agent_invoke import (
    SubAgentInvoker as sub_agent_invoker,
)
from imp_agent_core.agents.agent_core.sub_agent_invoke import (
    SubAgentToolConfig as sub_agent_tool_config,
)
from imp_agent_core.agents.agent_core.sub_agent_invoke import (
    make_sub_agent_tool as sub_agent_tool_factory,
)


def test_agent_core_public_exports_match_new_subpackage_paths() -> None:
    assert build_agent_card_from_yaml is a2a_card_loader
    assert ConfiguredA2aExecutor is adk_executor
    assert InferenceProviderLlmAdapter is inference_adapter
    assert DynamicAgentRegistry is routing_registry
    assert get_global_agent_registry is routing_global_registry
    assert reset_global_agent_registry is routing_reset_global_registry
    assert DEFAULT_SUB_AGENT_TOOL_INSTRUCTION is sub_agent_instruction
    assert SubAgentInvocationPolicy is sub_agent_policy
    assert SubAgentInvoker is sub_agent_invoker
    assert SubAgentToolConfig is sub_agent_tool_config
    assert make_sub_agent_tool is sub_agent_tool_factory
