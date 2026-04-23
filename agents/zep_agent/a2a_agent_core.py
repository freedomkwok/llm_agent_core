"""LlmAgent configuration for zep_agent."""

from __future__ import annotations

import os
from typing import AsyncGenerator
from typing import Any

from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.llm_agent import LlmAgent
from google.adk.events.event import Event
from google.genai import types
from typing_extensions import override

from agents.agent_core.inference_provider_llm_adapter import InferenceProviderLlmAdapter
from agents.zep_agent._env import bootstrap_env
from agents.zep_agent.tools import (
    get_edges_for_node,
    get_node_by_id,
    search_around_node,
    search_edges,
    search_skill_nodes,
)

bootstrap_env()


class ZepAgent(LlmAgent):
    """LlmAgent with explicit async execution hook."""

    @override
    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        if ctx.user_content and not ctx.user_content.role:
            ctx.user_content.role = "user"
        async for event in super()._run_async_impl(ctx):
            yield event


def build_zep_llm_agent(*, langfuse_client: Any = None) -> LlmAgent:
    """Build the LLM tool-calling agent for Zep-driven skill routing."""
    return ZepAgent(
        model=InferenceProviderLlmAdapter(
            model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
            langfuse_client=langfuse_client,
            project_name="imp_agent_map.zep_agent",
            project_metadata={"component": "zep_agent"},
            settings_overrides={"conversation_store_type": "lru"},
        ),
        name="skill_router",
        description="Tool-calling skill router powered by Zep graph operations.",
        instruction=(
            "You are a skill router.\n"
            "Use tools to decide the best skill.\n"
            "Do not guess early.\n"
            "You may search skill nodes first, inspect edges for promising nodes, "
            "or search edges directly if relation evidence is more important.\n"
            "When done, return:\n"
            "1. selected_skill_id\n"
            "2. selected_skill_name\n"
            "3. concise rationale\n"
            "4. evidence used"
        ),
        tools=[
            search_skill_nodes,
            get_edges_for_node,
            search_edges,
            get_node_by_id,
            search_around_node,
        ],
    )

