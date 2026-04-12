"""Planning inference wrapper using llm_inference_core."""

from __future__ import annotations

import os
from typing import Any

from llm_inference_core import (
    InferenceCoreSettings,
    InferenceProviderFactory,
    ProjectContext,
    read_langfuse_trace_context_from_env,
)

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


class PlanningInferenceEngine:
    """Generate structured planning output via llm_inference_core provider."""

    def __init__(self, *, langfuse_client: Any | None = None) -> None:
        # Avoid loading repo-wide .env into InferenceCoreSettings directly because
        # it may contain unrelated keys rejected by pydantic-settings.
        self.settings = InferenceCoreSettings(
            _env_file=None,
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
            inference_provider=os.getenv("INFERENCE_PROVIDER", "openai"),
            langfuse_public_key=os.getenv("LANGFUSE_PUBLIC_KEY", ""),
            langfuse_secret_key=os.getenv("LANGFUSE_SECRET_KEY", ""),
            langfuse_base_url=os.getenv("LANGFUSE_BASE_URL", "http://localhost:3000"),
        )
        self.project_context = ProjectContext(
            project_name="imp_agent_map.planning_agent",
            metadata={"component": "planning_agent"},
        )
        self.provider = InferenceProviderFactory.create(
            self.settings.inference_provider,
            settings=self.settings,
            project_context=self.project_context,
            langfuse=langfuse_client,
        )

    async def generate_plan(
        self,
        *,
        request_text: str,
        metadata: dict | None = None,
    ) -> PlanningSchema:
        trace_context = read_langfuse_trace_context_from_env()
        payload = {
            "trace_name": "planning_agent_generation",
            "langfuse_type": "generation",
            "system_prompt": SYSTEM_PROMPT,
            "inputs": [{"type": "text", "content": request_text}],
            "output_format": PlanningSchema,
            "metadata": metadata or {},
            "model_parameters": {"temperature": 0.2},
        }
        if trace_context.trace_id:
            payload["trace_id"] = trace_context.trace_id
        if trace_context.parent_observation_id:
            payload["parent_observation_id"] = trace_context.parent_observation_id

        result = await self.provider.infer(payload)
        if isinstance(result, PlanningSchema):
            return result
        return PlanningSchema.model_validate(result)

    @staticmethod
    def render_plan(plan: PlanningSchema) -> str:
        """Render a structured plan into a human-readable markdown text block."""
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
        await self.provider.close()
