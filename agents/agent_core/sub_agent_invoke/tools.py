"""ADK function-tool factory for sub-agent invocation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from google.adk.tools.tool_context import ToolContext

from agents.agent_core.routing import AgentResolver
from agents.agent_core.routing.registry import DynamicAgentRegistry
from agents.agent_core.sub_agent_invoke.invoker import SubAgentInvoker
from agents.agent_core.sub_agent_invoke.policy import SubAgentInvocationPolicy

DEFAULT_SUB_AGENT_TOOL_INSTRUCTION = (
    "\n\nWhen a request contains a narrow sub-question that should be delegated to "
    "another registered agent, use invoke_sub_agent. Prefer agent_id when known; "
    "otherwise use skill_id or tags from the registry. Keep delegated queries focused."
)


@dataclass(frozen=True)
class SubAgentToolConfig:
    """Reusable opt-in config for adding sub-agent delegation to an ADK agent."""

    registry: DynamicAgentRegistry
    policy: SubAgentInvocationPolicy | None = None
    resolver: AgentResolver | None = None
    instruction: str | None = DEFAULT_SUB_AGENT_TOOL_INSTRUCTION

    def tool(self):
        """Return the ADK function tool for this sub-agent config."""
        return make_sub_agent_tool(
            registry=self.registry,
            policy=self.policy,
            resolver=self.resolver,
        )

    def tools_for(self, tools: Sequence[Any]) -> list[Any]:
        """Return tools with the sub-agent invocation tool appended."""
        return [*tools, self.tool()]

    def instruction_for(self, instruction: str) -> str:
        """Return instruction text with the sub-agent guidance appended."""
        if not self.instruction:
            return instruction
        if self.instruction.startswith("\n"):
            return f"{instruction}{self.instruction}"
        return f"{instruction}\n\n{self.instruction.strip()}"


def make_sub_agent_tool(
    *,
    registry: DynamicAgentRegistry,
    policy: SubAgentInvocationPolicy | None = None,
    resolver: AgentResolver | None = None,
):
    """Return the generic ADK function tool for invoking registered sub-agents."""
    invoker = SubAgentInvoker(registry=registry, policy=policy, resolver=resolver)

    async def invoke_sub_agent(
        query: str,
        tool_context: ToolContext,
        agent_id: str | None = None,
        skill_id: str | None = None,
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        return await invoker.invoke(
            query=query,
            tool_context=tool_context,
            agent_id=agent_id,
            skill_id=skill_id,
            tags=tags,
        )

    return invoke_sub_agent
