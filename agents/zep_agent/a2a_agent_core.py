"""LlmAgent configuration for zep_agent."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from google.adk.agents.llm_agent import LlmAgent
from google.adk.skills import load_skill_from_dir
from google.adk.tools import skill_toolset

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

_SKILL_DIR = (
    Path(__file__).resolve().parent
    / "skills"
    / "zep-graph-retrieval"
)


def build_zep_llm_agent(*, langfuse_client: Any = None) -> LlmAgent:
    """Build the LLM tool-calling agent for Zep-driven skill routing."""
    zep_retrieval_skill = load_skill_from_dir(_SKILL_DIR)
    zep_skill_toolset = skill_toolset.SkillToolset(
        skills=[zep_retrieval_skill],
        additional_tools=[
            search_nodes,
            get_edges_for_node,
            search_edges,
            get_node_by_id,
            search_around_node,
        ],
    )
    return LlmAgent(
        model=InferenceProviderLlmAdapter(
            model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
            langfuse_client=langfuse_client,
            project_name="imp_agent_map.zep_agent",
            project_metadata={"component": "zep_agent"},
            settings_overrides={"conversation_store_type": "lru"},
        ),
        name="zep_query_agent",
        description="Agent for Zep graph automated query operations.",
        instruction=(
            """
            You are a Zep Agent, you are given a user request and you need to use the tools to retrieve information base on use query
            """
        ),
        tools=[zep_skill_toolset],
        # tools=[
        #     search_nodes,
        #     get_edges_for_node,
        #     search_edges,
        #     get_node_by_id,
        #     search_around_node,
        # ],
    )

