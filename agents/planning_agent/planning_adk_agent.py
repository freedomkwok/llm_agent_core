"""ADK BaseAgent that runs planning via llm_inference_core."""

from __future__ import annotations

import os
from typing import Any, AsyncGenerator

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events.event import Event
from google.genai import types
from llm_inference_core import (
    InferenceCoreSettings,
    InferenceProviderFactory,
    ProjectContext,
)
from pydantic import Field
from typing_extensions import override

from agents.planning_agent._env import bootstrap_env
from agents.planning_agent.schemas import PlanningSchema

bootstrap_env()

SYSTEM_PROMPT = (
    "You are a planning assistant. Given a user request, return an actionable plan.\n"
    "Produce output that strictly matches the requested schema.\n"
    "Rules:\n"
    "- Keep language concise and practical.\n"
    "- Steps must be executable and ordered.\n"
    "- If assumptions or risks are unknown, return an empty list."
)


def _text_from_user_content(content: types.Content | None) -> str:
    if not content or not content.parts:
        return ""
    return "".join(part.text for part in content.parts if part.text)


class PlanningInferenceAdkAgent(BaseAgent):
    """Custom ADK agent with built-in planning inference logic."""

    name: str = "planning_agent"
    description: str = (
        "Produces a structured implementation plan from the user request "
        "using llm_inference_core."
    )
    langfuse_client: Any = Field(default=None, exclude=True, repr=False)
    provider: Any = Field(default=None, exclude=True, repr=False)

    def _ensure_provider(self) -> Any:
        if self.provider is not None:
            return self.provider

        settings = InferenceCoreSettings(
            _env_file=None,
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
            inference_provider=os.getenv("INFERENCE_PROVIDER", "openai"),
            langfuse_public_key=os.getenv("LANGFUSE_PUBLIC_KEY", ""),
            langfuse_secret_key=os.getenv("LANGFUSE_SECRET_KEY", ""),
            langfuse_base_url=os.getenv("LANGFUSE_BASE_URL", "http://localhost:3000"),
        )
        project_context = ProjectContext(
            project_name="imp_agent_map.planning_agent",
            metadata={"component": "planning_agent"},
        )
        self.provider = InferenceProviderFactory.create(
            settings.inference_provider,
            settings=settings,
            project_context=project_context,
            langfuse=self.langfuse_client,
        )
        return self.provider

    async def generate_plan(
        self,
        *,
        request_text: str,
        metadata: dict[str, Any] | None = None,
    ) -> PlanningSchema:
        provider = self._ensure_provider()
        payload = {
            "trace_name": "planning_agent_generation",
            "langfuse_type": "generation",
            "system_prompt": SYSTEM_PROMPT,
            "inputs": [{"type": "text", "content": request_text}],
            "output_format": PlanningSchema,
            "metadata": metadata or {},
            "model_parameters": {"temperature": 0.2},
        }

        result = await provider.infer(payload)
        if isinstance(result, PlanningSchema):
            return result
        return PlanningSchema.model_validate(result)

    @staticmethod
    def render_plan(plan: PlanningSchema) -> str:
        assumptions = (
            "\n".join(f"- {item}" for item in plan.assumptions)
            if plan.assumptions
            else "- None"
        )
        steps = "\n".join(f"{idx + 1}. {step}" for idx, step in enumerate(plan.steps))
        risks = "\n".join(f"- {item}" for item in plan.risks) if plan.risks else "- None"
        return (
            f"Goal:\n{plan.goal}\n\n"
            f"Assumptions:\n{assumptions}\n\n"
            f"Steps:\n{steps}\n\n"
            f"Risks:\n{risks}\n\n"
            f"Next action:\n{plan.next_action}"
        )

    async def close(self) -> None:
        if self.provider is not None:
            await self.provider.close()

    @override
    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        request_text = _text_from_user_content(ctx.user_content)
        metadata = {
            "session_id": ctx.session.id,
            "user_id": ctx.user_id,
            "invocation_id": ctx.invocation_id,
        }
        plan = await self.generate_plan(request_text=request_text, metadata=metadata)
        rendered = self.render_plan(plan)
        yield Event(
            invocation_id=ctx.invocation_id,
            author=self.name,
            content=types.Content(
                role="model",
                parts=[types.Part(text=rendered)],
            ),
        )
