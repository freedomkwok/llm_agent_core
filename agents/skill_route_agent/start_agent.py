"""Build local skill route A2A agent and mode-based local flow."""

from __future__ import annotations

from typing import Any, Mapping

from vertexai.preview.reasoning_engines import A2aAgent

from agents.agent_core import A2AFlowResult, OrchestrationMode, run_local_a2a_orchestration
from agents.skill_route_agent._env import bootstrap_env
from agents.skill_route_agent.agent_card import agent_card
from agents.skill_route_agent.agent_executor import SkillRouteAgentExecutor

bootstrap_env()


def build_local_a2a_skill_route_agent(
    *,
    mode: OrchestrationMode = OrchestrationMode.AGENT_INTERNAL,
) -> A2aAgent:
    """Build local A2A skill route agent.

    Note: `mode` controls orchestration behavior in `run_local_skill_route_flow`.
    Agent construction is identical for both modes.
    """
    del mode
    agent = A2aAgent(
        agent_card=agent_card,
        agent_executor_builder=lambda: SkillRouteAgentExecutor(),
    )
    agent.set_up()
    return agent


async def run_local_skill_route_flow(
    *,
    message_text: str,
    mode: OrchestrationMode = OrchestrationMode.HOST_DRIVEN,
    metadata: Mapping[str, Any] | None = None,
    context: Any = None,
) -> A2AFlowResult:
    """Run skill route agent locally via host-driven or agent-internal orchestration."""
    a2a_agent = build_local_a2a_skill_route_agent(mode=mode)
    return await run_local_a2a_orchestration(
        a2a_agent=a2a_agent,
        message_text=message_text,
        mode=mode,
        metadata=metadata,
        context=context,
    )


if __name__ == "__main__":
    local_agent = build_local_a2a_skill_route_agent()
    print(local_agent)
