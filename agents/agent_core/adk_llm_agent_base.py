"""Shared ADK LlmAgent base behavior."""

from __future__ import annotations

from typing import AsyncGenerator

from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.llm_agent import LlmAgent
from google.adk.events.event import Event
from typing_extensions import override


class UserRoleNormalizedLlmAgent(LlmAgent):
    """Normalize missing user role before ADK execution."""

    @override
    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        if ctx.user_content and not ctx.user_content.role:
            ctx.user_content.role = "user"
        async for event in super()._run_async_impl(ctx):
            yield event
