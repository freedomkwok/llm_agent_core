"""Unified handle abstractions for local and remote agent invocation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from agents.agent_core.a2a.local_orchestration import (
    A2AFlowResult,
    OrchestrationMode,
    build_get_task_request,
    build_message_payload,
    build_post_request,
    extract_task_id,
    run_local_a2a_orchestration,
)
from agents.agent_core.routing.descriptor import AgentBackendType, AgentDescriptor


@dataclass
class AgentInvocationResult:
    """Normalized result returned by unified agent handles and orchestrators."""

    descriptor: AgentDescriptor
    mode: OrchestrationMode
    backend_type: AgentBackendType
    send_response: Any
    card_response: Any | None = None
    task_response: Any | None = None
    task_id: str | None = None
    task_status: str | None = None
    final_text: str | None = None
    error: str | None = None


class BaseAgentHandle(ABC):
    """Common invocation interface for local and remote agent backends."""

    def __init__(self, descriptor: AgentDescriptor):
        self.descriptor = descriptor

    @abstractmethod
    async def get_agent_card(self, *, context: Any = None) -> Any:
        """Fetch the agent card for this handle."""

    @abstractmethod
    async def send_message(
        self,
        *,
        message_text: str,
        metadata: Mapping[str, Any] | None = None,
        context: Any = None,
        message_id: str | None = None,
    ) -> Any:
        """Send a user message to the agent and return the raw send response."""

    @abstractmethod
    async def get_task(self, *, task_id: str, context: Any = None) -> Any:
        """Fetch a task snapshot for the given task id."""

    async def run(
        self,
        *,
        message_text: str,
        mode: OrchestrationMode = OrchestrationMode.HOST_DRIVEN,
        metadata: Mapping[str, Any] | None = None,
        context: Any = None,
        include_authenticated_card: bool | None = None,
        fetch_task_response: bool | None = None,
    ) -> AgentInvocationResult:
        """Run the unified card/send/task flow through this handle."""
        include_card = (
            include_authenticated_card
            if include_authenticated_card is not None
            else mode == OrchestrationMode.HOST_DRIVEN
        )
        fetch_task = (
            fetch_task_response
            if fetch_task_response is not None
            else mode == OrchestrationMode.HOST_DRIVEN
        )

        card_response = None
        if include_card:
            card_response = await self.get_agent_card(context=context)

        send_response = await self.send_message(
            message_text=message_text,
            metadata=metadata,
            context=context,
        )
        task_id = extract_task_id(send_response)
        task_response = None
        if fetch_task:
            if not task_id:
                raise ValueError("Unified send response does not include task.id")
            task_response = await self.get_task(task_id=task_id, context=context)

        return AgentInvocationResult(
            descriptor=self.descriptor,
            mode=mode,
            backend_type=self.descriptor.backend_type,
            card_response=card_response,
            send_response=send_response,
            task_response=task_response,
            task_id=task_id,
        )


class LocalA2AHandle(BaseAgentHandle):
    """Handle for in-process A2A agents created from local Python builders."""

    def __init__(self, descriptor: AgentDescriptor):
        super().__init__(descriptor)
        self._a2a_agent: Any | None = None

    def _get_or_build_agent(self) -> Any:
        if self._a2a_agent is None:
            if self.descriptor.local_builder is None:
                raise ValueError(
                    f"Agent descriptor {self.descriptor.agent_id} does not define a local builder"
                )
            self._a2a_agent = self.descriptor.local_builder()
        return self._a2a_agent

    async def get_agent_card(self, *, context: Any = None) -> Any:
        if self.descriptor.cached_agent_card is not None:
            return self.descriptor.cached_agent_card
        a2a_agent = self._get_or_build_agent()
        return await a2a_agent.handle_authenticated_agent_card(request=None, context=context)

    async def send_message(
        self,
        *,
        message_text: str,
        metadata: Mapping[str, Any] | None = None,
        context: Any = None,
        message_id: str | None = None,
    ) -> Any:
        a2a_agent = self._get_or_build_agent()
        payload = build_message_payload(
            message_text=message_text,
            metadata=metadata,
            message_id=message_id,
        )
        request = build_post_request(payload)
        return await a2a_agent.on_message_send(request=request, context=context)

    async def get_task(self, *, task_id: str, context: Any = None) -> Any:
        a2a_agent = self._get_or_build_agent()
        request = build_get_task_request(task_id)
        return await a2a_agent.on_get_task(request=request, context=context)

    async def run(
        self,
        *,
        message_text: str,
        mode: OrchestrationMode = OrchestrationMode.HOST_DRIVEN,
        metadata: Mapping[str, Any] | None = None,
        context: Any = None,
        include_authenticated_card: bool | None = None,
        fetch_task_response: bool | None = None,
    ) -> AgentInvocationResult:
        flow_result: A2AFlowResult = await run_local_a2a_orchestration(
            a2a_agent=self._get_or_build_agent(),
            message_text=message_text,
            mode=mode,
            metadata=metadata,
            context=context,
            include_authenticated_card=include_authenticated_card,
            fetch_task_response=fetch_task_response,
        )
        return AgentInvocationResult(
            descriptor=self.descriptor,
            mode=flow_result.mode,
            backend_type=self.descriptor.backend_type,
            card_response=flow_result.card_response,
            send_response=flow_result.send_response,
            task_response=flow_result.task_response,
            task_id=flow_result.task_id,
            task_status=flow_result.task_status,
            final_text=flow_result.final_text,
        )


class RemoteA2AHandle(BaseAgentHandle):
    """Stub handle for future network-accessible A2A agents."""

    async def get_agent_card(self, *, context: Any = None) -> Any:
        del context
        if self.descriptor.cached_agent_card is not None:
            return self.descriptor.cached_agent_card
        return {
            "name": self.descriptor.agent_name,
            "description": self.descriptor.description,
            "skills": [skill.skill_id for skill in self.descriptor.skills],
            "endpoint": self.descriptor.endpoint,
        }

    async def send_message(
        self,
        *,
        message_text: str,
        metadata: Mapping[str, Any] | None = None,
        context: Any = None,
        message_id: str | None = None,
    ) -> Any:
        del message_text, metadata, context, message_id
        raise NotImplementedError(
            f"Remote A2A transport is not implemented yet for {self.descriptor.agent_id}"
        )

    async def get_task(self, *, task_id: str, context: Any = None) -> Any:
        del task_id, context
        raise NotImplementedError(
            f"Remote A2A task retrieval is not implemented yet for {self.descriptor.agent_id}"
        )


def build_agent_handle(descriptor: AgentDescriptor) -> BaseAgentHandle:
    """Create a handle instance for the given descriptor."""
    if descriptor.backend_type == AgentBackendType.LOCAL_A2A:
        return LocalA2AHandle(descriptor)
    if descriptor.backend_type == AgentBackendType.REMOTE_A2A:
        return RemoteA2AHandle(descriptor)
    raise ValueError(f"Unsupported backend type: {descriptor.backend_type}")
