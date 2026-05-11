"""LlmAgent configuration for zep_agent."""

from __future__ import annotations

import os
from typing import Any

from google.adk.agents.llm_agent import LlmAgent

from agents.agent_core import SubAgentToolConfig
from agents.agent_core.inference import InferenceProviderLlmAdapter, load_agent_instruction
from agents.zep_agent._env import bootstrap_env
from agents.zep_agent.tools import (
    get_edges_for_node,
    get_node_by_id,
    search_around_node,
    search_edges,
    search_episodes,
    search_nodes,
)

bootstrap_env()

_AGENT_NAME = "zep_query_agent"
_PROJECT_NAME = "imp_agent_map.zep_agent"
_PROJECT_METADATA = {"component": "zep_agent"}
_SETTINGS_OVERRIDES = {"conversation_store_type": "lru"}
_FALLBACK_INSTRUCTION = (
    "You are a Zep Agent, you are given a user request and you need to use "
    "the tools to retrieve information base on use query"
)


def build_zep_llm_agent(
    *,
    langfuse_client: Any = None,
    instruction_prompt_name: str | None = None,
    instruction_prompt_label: str | None = None,
    fallback_instruction: str = _FALLBACK_INSTRUCTION,
    name: str = _AGENT_NAME,
    sub_agent_tool_config: SubAgentToolConfig | None = None,
) -> LlmAgent:
    """Build the LLM tool-calling agent for Zep-driven skill routing."""
    instruction = load_agent_instruction(
        agent_name=_AGENT_NAME,
        project_name=_PROJECT_NAME,
        project_metadata=_PROJECT_METADATA,
        settings_overrides=_SETTINGS_OVERRIDES,
        fallback_instruction=fallback_instruction,
        instruction_prompt_name=instruction_prompt_name,
        instruction_prompt_label=instruction_prompt_label,
    )
    tools = [
        search_nodes,
        get_edges_for_node,
        search_edges,
        search_episodes,
        get_node_by_id,
        search_around_node,
    ]
    if sub_agent_tool_config is not None:
        tools = sub_agent_tool_config.tools_for(tools)
        instruction = sub_agent_tool_config.instruction_for(instruction)

    return LlmAgent(
        model=InferenceProviderLlmAdapter(
            model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
            langfuse_client=langfuse_client,
            project_name=_PROJECT_NAME,
            project_metadata=_PROJECT_METADATA,
            settings_overrides=_SETTINGS_OVERRIDES,
        ),
        name=name,
        instruction=instruction,
        tools=tools,
    )
