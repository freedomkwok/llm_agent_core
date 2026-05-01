from agents.agent_core import (
    ConfiguredA2aExecutor,
    DynamicAgentRegistry,
    InferenceProviderLlmAdapter,
    build_agent_card_from_yaml,
)
from agents.agent_core.a2a.agent_card_yaml import build_agent_card_from_yaml as a2a_card_loader
from agents.agent_core.adk.config_file_executor import ConfiguredA2aExecutor as adk_executor
from agents.agent_core.inference.llm_adapter import (
    InferenceProviderLlmAdapter as inference_adapter,
)
from agents.agent_core.routing.registry import DynamicAgentRegistry as routing_registry


def test_agent_core_public_exports_match_new_subpackage_paths() -> None:
    assert build_agent_card_from_yaml is a2a_card_loader
    assert ConfiguredA2aExecutor is adk_executor
    assert InferenceProviderLlmAdapter is inference_adapter
    assert DynamicAgentRegistry is routing_registry
