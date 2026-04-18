"""A2A execution wrapper for the skill route ADK agent."""

from typing import Any

from google.adk.agents.base_agent import BaseAgent
from langfuse import get_client

from agents.agent_core import AdkA2aExecutionWrapper
from agents.skill_route_agent._env import bootstrap_env
from agents.skill_route_agent.a2a_agent_core import SkillRouteAdkAgent

bootstrap_env()


class SkillRouteA2aExecutor(AdkA2aExecutionWrapper):
    """Skill-routing A2A wrapper with shared runner/session/trace orchestration."""

    def __init__(
        self,
        adk_agent: SkillRouteAdkAgent | None = None,
        *,
        langfuse_client: Any | None = None,
    ):
        self.langfuse_client = langfuse_client or get_client()
        self.adk_agent = adk_agent or SkillRouteAdkAgent(
            langfuse_client=self.langfuse_client
        )
        super().__init__(langfuse_client=self.langfuse_client)

    def build_adk_agent(self) -> BaseAgent:
        return self.adk_agent

    @property
    def trace_name(self) -> str:
        return "skill_route_executor_execute"

    @property
    def artifact_name(self) -> str:
        return "skill_route"

    @property
    def failed_text_message(self) -> str:
        return "Failed to generate a skill routing response with text content."


if __name__ == "__main__":
    executor = SkillRouteA2aExecutor()
    print(executor.__class__.__name__)
