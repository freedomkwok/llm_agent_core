"""Skill routing inference wrapper using llm_inference_core and Zep."""

from __future__ import annotations

import os
from typing import Any

from llm_inference_core import (
    InferenceCoreSettings,
    InferenceProviderFactory,
    ProjectContext,
    read_langfuse_trace_context_from_env,
)

from agents.skill_route_agent._env import bootstrap_env
from agents.skill_route_agent.schemas import RoutedSkillCandidate, SkillRouteSchema
from agents.utils.zep_helper import ZepSkillCandidate, ZepSkillSearchComponent

bootstrap_env()

SYSTEM_PROMPT = (
    "You are a skill routing assistant.\n"
    "Your job is to choose the best next skill from Zep-derived candidates.\n"
    "Return output that strictly matches the requested schema.\n"
    "Rules:\n"
    "- Only choose a skill from the provided candidates.\n"
    "- If no candidate clearly fits, leave selected_skill_id and selected_skill_name empty.\n"
    "- Keep rationale concise and operational.\n"
    "- planner_prompt should be the clean handoff text for the planning agent.\n"
    "- next_action should tell the host what to do immediately after routing."
)


class SkillRouteInferenceEngine:
    """Query Zep for candidate skills, then use LLM inference to choose one."""

    def __init__(
        self,
        *,
        langfuse_client: Any | None = None,
        zep_component: ZepSkillSearchComponent | None = None,
    ) -> None:
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
            project_name="imp_agent_map.skill_route_agent",
            metadata={"component": "skill_route_agent"},
        )
        self.provider = InferenceProviderFactory.create(
            self.settings.inference_provider,
            settings=self.settings,
            project_context=self.project_context,
            langfuse=langfuse_client,
        )
        self.zep_component = zep_component or ZepSkillSearchComponent()

    async def route_request(
        self,
        *,
        request_text: str,
        metadata: dict[str, Any] | None = None,
    ) -> SkillRouteSchema:
        zep_candidates = self.lookup_candidates(request_text=request_text, metadata=metadata)
        if not zep_candidates:
            return self._build_empty_route(request_text=request_text)

        trace_context = read_langfuse_trace_context_from_env()
        payload = {
            "trace_name": "skill_route_agent_generation",
            "langfuse_type": "generation",
            "system_prompt": SYSTEM_PROMPT,
            "inputs": [{"type": "text", "content": self._build_user_input(request_text, zep_candidates)}],
            "output_format": SkillRouteSchema,
            "metadata": {
                **(metadata or {}),
                "zep_candidate_count": len(zep_candidates),
            },
            "model_parameters": {"temperature": 0.1},
        }
        if trace_context.trace_id:
            payload["trace_id"] = trace_context.trace_id
        if trace_context.parent_observation_id:
            payload["parent_observation_id"] = trace_context.parent_observation_id

        result = await self.provider.infer(payload)
        route = result if isinstance(result, SkillRouteSchema) else SkillRouteSchema.model_validate(result)
        if route.candidate_skills:
            return route
        return route.model_copy(
            update={
                "candidate_skills": [
                    RoutedSkillCandidate(
                        skill_id=candidate.skill_id,
                        name=candidate.name,
                        description=candidate.description,
                    )
                    for candidate in zep_candidates
                ]
            }
        )

    def lookup_candidates(
        self,
        *,
        request_text: str,
        metadata: dict[str, Any] | None = None,
    ) -> list[ZepSkillCandidate]:
        user_id = str((metadata or {}).get("user_id") or "").strip() or None
        return self.zep_component.search_skills(
            query=request_text,
            user_id=user_id,
            limit=5,
        )

    def _build_empty_route(self, *, request_text: str) -> SkillRouteSchema:
        if not self.zep_component.is_configured:
            rationale = (
                "Zep is not configured, so no skill candidates could be retrieved."
            )
            next_action = "Configure ZEP_API_KEY and ZEP_USER_ID before routing again."
        else:
            rationale = "No Zep skill candidates matched the request strongly enough."
            next_action = "Ask for more detail or add the missing skill description to Zep."

        return SkillRouteSchema(
            request_summary=request_text.strip(),
            selected_skill_id="",
            selected_skill_name="",
            rationale=rationale,
            candidate_skills=[],
            planner_prompt=request_text.strip(),
            next_action=next_action,
        )

    @staticmethod
    def _build_user_input(
        request_text: str,
        zep_candidates: list[ZepSkillCandidate],
    ) -> str:
        formatted_candidates = "\n".join(
            [
                (
                    f"- skill_id: {candidate.skill_id}\n"
                    f"  name: {candidate.name}\n"
                    f"  description: {candidate.description or 'No description provided.'}"
                )
                for candidate in zep_candidates
            ]
        )
        return (
            f"User request:\n{request_text.strip()}\n\n"
            f"Zep skill candidates:\n{formatted_candidates}"
        )

    @staticmethod
    def render_route(route: SkillRouteSchema) -> str:
        """Render routing output into human-readable markdown text."""
        selected_skill = route.selected_skill_name or route.selected_skill_id or "No skill selected"
        candidates = (
            "\n".join(
                f"- {candidate.name} ({candidate.skill_id}): {candidate.description or 'No description provided.'}"
                for candidate in route.candidate_skills
            )
            if route.candidate_skills
            else "- None"
        )
        return (
            f"Request summary:\n{route.request_summary}\n\n"
            f"Selected skill:\n{selected_skill}\n\n"
            f"Rationale:\n{route.rationale}\n\n"
            f"Candidates:\n{candidates}\n\n"
            f"Planner prompt:\n{route.planner_prompt}\n\n"
            f"Next action:\n{route.next_action}"
        )

    async def close(self) -> None:
        await self.provider.close()
