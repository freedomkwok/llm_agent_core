# SPDX-License-Identifier: Apache-2.0
"""Local A2A runtime backed directly by a2a-sdk."""

from __future__ import annotations

import copy
from collections.abc import AsyncIterator, Callable, Mapping
from typing import Any

from a2a.types import TransportProtocol


class LocalA2aAgent:
    """A local A2A agent with the same handler surface used from Vertex's template."""

    agent_framework = "a2a"

    def __init__(
        self,
        *,
        agent_card: Any,
        task_store_builder: Callable[..., Any] | None = None,
        task_store_kwargs: Mapping[str, Any] | None = None,
        agent_executor_kwargs: Mapping[str, Any] | None = None,
        agent_executor_builder: Callable[..., Any] | None = None,
        request_handler_kwargs: Mapping[str, Any] | None = None,
        request_handler_builder: Callable[..., Any] | None = None,
        extended_agent_card: Any = None,
    ) -> None:
        if (
            agent_card.preferred_transport
            and agent_card.preferred_transport != TransportProtocol.http_json
        ):
            raise ValueError("Only HTTP+JSON is supported for preferred transport on agent card")

        self._attrs: dict[str, Any] = {
            "agent_card": agent_card,
            "agent_executor": None,
            "agent_executor_kwargs": dict(agent_executor_kwargs or {}),
            "agent_executor_builder": agent_executor_builder,
            "task_store": None,
            "task_store_kwargs": dict(task_store_kwargs or {}),
            "task_store_builder": task_store_builder,
            "request_handler": None,
            "request_handler_kwargs": dict(request_handler_kwargs or {}),
            "request_handler_builder": request_handler_builder,
            "extended_agent_card": extended_agent_card,
        }
        self.agent_card = agent_card
        self.a2a_rest_adapter = None
        self.request_handler = None
        self.rest_handler = None
        self.task_store = None
        self.agent_executor = None

    def clone(self) -> LocalA2aAgent:
        return LocalA2aAgent(
            agent_card=copy.deepcopy(self.agent_card),
            task_store_builder=self._attrs.get("task_store_builder"),
            task_store_kwargs=self._attrs.get("task_store_kwargs"),
            agent_executor_kwargs=self._attrs.get("agent_executor_kwargs"),
            agent_executor_builder=self._attrs.get("agent_executor_builder"),
            request_handler_kwargs=self._attrs.get("request_handler_kwargs"),
            request_handler_builder=self._attrs.get("request_handler_builder"),
            extended_agent_card=self._attrs.get("extended_agent_card"),
        )

    def set_up(self) -> None:
        from a2a.server.apps.rest.rest_adapter import RESTAdapter
        from a2a.server.request_handlers import DefaultRequestHandler
        from a2a.server.request_handlers.rest_handler import RESTHandler
        from a2a.server.tasks import InMemoryTaskStore

        agent_executor_builder = self._attrs.get("agent_executor_builder")
        if agent_executor_builder:
            self.agent_executor = agent_executor_builder(**self._attrs["agent_executor_kwargs"])
            self._attrs["agent_executor"] = self.agent_executor

        task_store_builder = self._attrs.get("task_store_builder")
        self.task_store = (
            task_store_builder(**self._attrs["task_store_kwargs"])
            if task_store_builder
            else InMemoryTaskStore()
        )
        self._attrs["task_store"] = self.task_store

        request_handler_builder = self._attrs.get("request_handler_builder")
        self.request_handler = (
            request_handler_builder(**self._attrs["request_handler_kwargs"])
            if request_handler_builder
            else DefaultRequestHandler(
                agent_executor=self._attrs.get("agent_executor"),
                task_store=self.task_store,
            )
        )
        self._attrs["request_handler"] = self.request_handler

        self.a2a_rest_adapter = RESTAdapter(
            agent_card=self.agent_card,
            http_handler=self.request_handler,
            extended_agent_card=self._attrs.get("extended_agent_card"),
        )
        self.rest_handler = RESTHandler(
            agent_card=self.agent_card,
            request_handler=self.request_handler,
        )

    async def on_message_send(self, request: Any, context: Any) -> dict[str, Any]:
        return await self.rest_handler.on_message_send(request, context)

    async def on_cancel_task(self, request: Any, context: Any) -> dict[str, Any]:
        return await self.rest_handler.on_cancel_task(request, context)

    async def on_get_task(self, request: Any, context: Any) -> dict[str, Any]:
        return await self.rest_handler.on_get_task(request, context)

    async def handle_authenticated_agent_card(self, request: Any, context: Any) -> dict[str, Any]:
        return await self.a2a_rest_adapter.handle_authenticated_agent_card(request, context)

    def register_operations(self) -> dict[str, list[str]]:
        routes = {
            "a2a_extension": [
                "on_message_send",
                "on_get_task",
                "on_cancel_task",
            ]
        }
        if self.agent_card.capabilities and self.agent_card.capabilities.streaming:
            routes["a2a_extension"].append("on_message_send_stream")
            routes["a2a_extension"].append("on_resubscribe_to_task")
        if self.agent_card.supports_authenticated_extended_card:
            routes["a2a_extension"].append("handle_authenticated_agent_card")
        return routes

    async def on_message_send_stream(self, request: Any, context: Any) -> AsyncIterator[str]:
        async for chunk in self.rest_handler.on_message_send_stream(request, context):
            yield chunk

    async def on_resubscribe_to_task(self, request: Any, context: Any) -> AsyncIterator[str]:
        async for chunk in self.rest_handler.on_resubscribe_to_task(request, context):
            yield chunk
