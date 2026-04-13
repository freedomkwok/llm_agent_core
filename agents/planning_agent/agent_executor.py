"""AgentExecutor for planning agent."""

from typing import Any

from google.adk.agents.base_agent import BaseAgent
from langfuse import get_client

from agents.agent_core import AdkRunnerChainExecutor
from agents.planning_agent._env import bootstrap_env
from agents.planning_agent.planning_adk_agent import PlanningInferenceAdkAgent

bootstrap_env()


class PlanningAgentExecutor(AdkRunnerChainExecutor):
    """Planning executor with shared runner/session/trace orchestration."""

    def __init__(
        self,
        adk_agent: PlanningInferenceAdkAgent | None = None,
        *,
        langfuse_client: Any | None = None,
    ):
        self.langfuse_client = langfuse_client or get_client()
        self.adk_agent = adk_agent or PlanningInferenceAdkAgent(
            langfuse_client=self.langfuse_client
        )
        super().__init__(langfuse_client=self.langfuse_client)

    def build_adk_agent(self) -> BaseAgent:
        return self.adk_agent

    @property
    def trace_name(self) -> str:
        return "planning_executor_execute"

    @property
    def artifact_name(self) -> str:
        return "plan"

    @property
    def failed_text_message(self) -> str:
        return "Failed to generate a planning response with text content."


if __name__ == "__main__":
    executor = PlanningAgentExecutor()
    print(executor.__class__.__name__)
