"""ADK BaseAgent that routes skills via Zep-backed inference."""

from __future__ import annotations

from typing import Any, AsyncGenerator

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events.event import Event
from google.genai import types
from pydantic import Field
from typing_extensions import override

from agents.skill_route_agent.skill_route_agent import SkillRouteInferenceEngine


def _text_from_user_content(content: types.Content | None) -> str:
    if not content or not content.parts:
        return ""
    return "".join(part.text for part in content.parts if part.text)


class SkillRouteAdkAgent(BaseAgent):
    """Custom ADK agent: one turn = choose best next skill from Zep candidates."""

    name: str = "skill_route_agent"
    description: str = (
        "Routes an incoming request to the best matching skill by using Zep "
        "candidate retrieval plus llm_inference_core."
    )
    engine: Any = Field(..., exclude=True, repr=False)

    @override
    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        request_text = _text_from_user_content(ctx.user_content)
        engine: SkillRouteInferenceEngine = self.engine
        metadata = {
            "session_id": ctx.session.id,
            "user_id": ctx.user_id,
            "invocation_id": ctx.invocation_id,
        }
        route = await engine.route_request(request_text=request_text, metadata=metadata)
        rendered = engine.render_route(route)
        yield Event(
            invocation_id=ctx.invocation_id,
            author=self.name,
            content=types.Content(
                role="model",
                parts=[types.Part(text=rendered)],
            ),
        )
