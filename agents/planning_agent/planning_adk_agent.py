"""ADK BaseAgent that runs planning via llm_inference_core (custom agent)."""

from __future__ import annotations

from typing import Any, AsyncGenerator

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events.event import Event
from google.genai import types
from pydantic import Field
from typing_extensions import override

from agents.planning_agent.planning_agent import PlanningInferenceEngine


def _text_from_user_content(content: types.Content | None) -> str:
    if not content or not content.parts:
        return ""
    return "".join(part.text for part in content.parts if part.text)


class PlanningInferenceAdkAgent(BaseAgent):
    """Custom ADK agent: one turn = structured plan from PlanningInferenceEngine."""

    name: str = "planning_agent"
    description: str = (
        "Produces a structured implementation plan from the user request "
        "using llm_inference_core."
    )
    engine: Any = Field(..., exclude=True, repr=False)

    @override
    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        request_text = _text_from_user_content(ctx.user_content)
        engine: PlanningInferenceEngine = self.engine
        metadata = {
            "session_id": ctx.session.id,
            "user_id": ctx.user_id,
            "invocation_id": ctx.invocation_id,
        }
        plan = await engine.generate_plan(request_text=request_text, metadata=metadata)
        rendered = engine.render_plan(plan)
        yield Event(
            invocation_id=ctx.invocation_id,
            author=self.name,
            content=types.Content(
                role="model",
                parts=[types.Part(text=rendered)],
            ),
        )
