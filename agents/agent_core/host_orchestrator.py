"""High-level capability-based host orchestration for dynamic agents."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from agents.agent_core.a2a_orchestration import OrchestrationMode
from agents.agent_core.agent_handle import AgentInvocationResult
from agents.agent_core.agent_registry import DynamicAgentRegistry
from agents.agent_core.agent_resolver import AgentResolver


class HostOrchestrator:
    """Resolve an agent by capability and invoke it through a unified handle."""

    def __init__(
        self,
        *,
        registry: DynamicAgentRegistry,
        resolver: AgentResolver | None = None,
    ) -> None:
        self.registry = registry
        self.resolver = resolver or AgentResolver(registry)

    async def invoke(
        self,
        *,
        skill_id: str | None = None,
        message_text: str,
        mode: OrchestrationMode = OrchestrationMode.HOST_DRIVEN,
        tags: Iterable[str] | None = None,
        name_contains: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        context: Any = None,
    ) -> AgentInvocationResult:
        """Resolve the best agent and invoke it through its unified handle."""
        handle = self.resolver.resolve_handle(
            skill_id=skill_id,
            tags=tags,
            name_contains=name_contains,
        )
        return await handle.run(
            message_text=message_text,
            mode=mode,
            metadata=metadata,
            context=context,
        )
