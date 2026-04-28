"""LlmAgent configuration for zep_agent."""

from __future__ import annotations

import os
from typing import Any

from google.adk.agents.llm_agent import LlmAgent
from llm_inference_core import ProjectContext, make_prompt_provider

from agents.agent_core.inference_provider import build_default_inference_settings
from agents.agent_core.inference_provider_llm_adapter import InferenceProviderLlmAdapter
from agents.zep_agent._env import bootstrap_env
from agents.zep_agent.tools import (
    get_edges_for_node,
    get_node_by_id,
    search_around_node,
    search_edges,
    search_nodes,
)

bootstrap_env()

_AGENT_NAME = "zep_query_agent"
_PROJECT_NAME = "imp_agent_map.zep_agent"
_PROJECT_METADATA = {"component": "zep_agent"}
_SETTINGS_OVERRIDES = {"conversation_store_type": "lru"}
_DEFAULT_INSTRUCTION = (
    "You are a Zep Agent, you are given a user request and you need to use "
    "the tools to retrieve information base on use query"
)


def _default_instruction_prompt_name(agent_name: str) -> str:
    return f"agents/{agent_name}/instruction"


def _resolve_instruction(
    *,
    instruction_prompt_name: str | None,
    instruction_prompt_label: str | None,
) -> str:
    prompt_name = (instruction_prompt_name or _default_instruction_prompt_name(_AGENT_NAME)).strip()
    if not prompt_name:
        return _DEFAULT_INSTRUCTION

    prompt_label = instruction_prompt_label.strip() if isinstance(instruction_prompt_label, str) else None
    if prompt_label == "":
        prompt_label = None

    try:
        settings = build_default_inference_settings(overrides=_SETTINGS_OVERRIDES)
        project_context = ProjectContext(
            project_name=_PROJECT_NAME,
            metadata=dict(_PROJECT_METADATA),
        )
        prompt_provider = make_prompt_provider(settings=settings, project_context=project_context)
        prompt = prompt_provider.get(prompt_name, label=prompt_label)
    except Exception:  # noqa: BLE001
        return _DEFAULT_INSTRUCTION
    if isinstance(prompt, str):
        normalized_prompt = prompt.strip()
        if normalized_prompt:
            return normalized_prompt
    return _DEFAULT_INSTRUCTION


def build_zep_llm_agent(
    *,
    langfuse_client: Any = None,
    instruction_prompt_name: str | None = None,
    instruction_prompt_label: str | None = None,
) -> LlmAgent:
    """Build the LLM tool-calling agent for Zep-driven skill routing."""
    instruction = _resolve_instruction(
        instruction_prompt_name=instruction_prompt_name,
        instruction_prompt_label=instruction_prompt_label,
    )
    return LlmAgent(
        model=InferenceProviderLlmAdapter(
            model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
            langfuse_client=langfuse_client,
            project_name=_PROJECT_NAME,
            project_metadata=_PROJECT_METADATA,
            settings_overrides=_SETTINGS_OVERRIDES,
        ),
        name=_AGENT_NAME,
        description="Agent for Zep graph automated query operations.",
        instruction=instruction,
        tools=[
            search_nodes,
            get_edges_for_node,
            search_edges,
            get_node_by_id,
            search_around_node,
        ],
            #   tools=[zep_skill_toolset],
    )

