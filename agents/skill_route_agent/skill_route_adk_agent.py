"""ADK BaseAgent that routes skills via Zep-backed inference."""

from __future__ import annotations

import os
import logging
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
from llm_inference_core.providers import InferenceProvider
from pydantic import Field, SkipValidation
from typing_extensions import override

from agents.skill_route_agent._env import bootstrap_env
from agents.skill_route_agent.schemas import (
    RoutedSkillCandidate,
    SkillRouteSchema,
    ZepQueryParams,
    ZepSearchScope,
)
from agents.skill_route_agent.utils.zep_helper import (
    ZepQueryRequest,
    ZepSkillCandidate,
    ZepSkillSearchComponent,
)

bootstrap_env()
logger = logging.getLogger(__name__)

ROUTE_SYSTEM_PROMPT = (
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

QUERY_PARAMS_SYSTEM_PROMPT = (
    "You are a routing pre-processor.\n"
    "Convert a user request into one concrete Zep graph search plan.\n"
    "Return output that strictly matches the requested schema.\n"
    "Output format fields:\n"
    "- query: concise search text for Zep.\n"
    "- scope: one of nodes or edges.\n"
    "- limit: integer from 1 to 20.\n"
    "- user_id: optional user id override, empty string if unknown.\n"
    "- rationale: short reason these parameters fit the request.\n"
    "Rules:\n"
    "- query should be compact and keyword-rich.\n"
    "- scope should be nodes unless edges are clearly better.\n"
    "- limit should be 3-10 for focused retrieval.\n"
    "- user_id can be empty when unknown.\n"
)


def _text_from_user_content(content: types.Content | None) -> str:
    if not content or not content.parts:
        return ""
    return "".join(part.text for part in content.parts if part.text)


class SkillRouteAdkAgent(BaseAgent):
    """Custom ADK agent with built-in skill routing inference logic."""

    name: str = "skill_route_agent"
    description: str = (
        "Routes an incoming request to the best matching skill by using Zep "
        "candidate retrieval plus llm_inference_core."
    )
    langfuse_client: Any = Field(default=None, exclude=True, repr=False)
    zep_component: SkipValidation[ZepSkillSearchComponent | None] = Field(
        default=None, exclude=True, repr=False
    )
    provider: SkipValidation[InferenceProvider | None] = Field(
        default=None, exclude=True, repr=False
    )
    max_loop_rounds: int = 2

    def _ensure_provider(self) -> InferenceProvider:
        if self.provider is not None:
            return self.provider

        settings = InferenceCoreSettings(
            _env_file=None,
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
            openai_base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            inference_provider=os.getenv("INFERENCE_PROVIDER", "openai"),
            langfuse_public_key=os.getenv("LANGFUSE_PUBLIC_KEY", ""),
            langfuse_secret_key=os.getenv("LANGFUSE_SECRET_KEY", ""),
            langfuse_base_url=os.getenv("LANGFUSE_BASE_URL", "http://localhost:3000"),
            prompt_base_dir=os.getenv("PROMPT_BASE_DIR", "app/prompts"),
            prompt_label=os.getenv("PROMPT_LABEL", "production"),
            prompt_cache_ttl_seconds=os.getenv("PROMPT_CACHE_TTL_SECONDS", 60),
            prompt_project_miss_ttl_seconds=os.getenv("PROMPT_PROJECT_MISS_TTL_SECONDS", 1800),
            prompt_tag_source=os.getenv("PROMPT_TAG_SOURCE", "local"),
            prompt_tag_file=os.getenv("PROMPT_TAG_FILE", "local_prompt_tags.json"),
            prompt_backend=os.getenv("PROMPT_BACKEND", "file"),
            example_capture_inputs=os.getenv("EXAMPLE_CAPTURE_INPUTS", False),
        )
        project_context = ProjectContext(
            project_name="imp_agent_map.skill_route_agent",
            metadata={"component": "skill_route_agent"},
        )
        self.provider = InferenceProviderFactory.create(
            settings.inference_provider,
            settings=settings,
            project_context=project_context,
            langfuse=self.langfuse_client,
        )
        return self.provider

    def _ensure_zep_component(self) -> ZepSkillSearchComponent:
        if self.zep_component is None:
            self.zep_component = ZepSkillSearchComponent()
        return self.zep_component

    async def route_request(
        self,
        *,
        request_text: str,
        metadata: dict[str, Any] | None = None,
    ) -> SkillRouteSchema:
        if not self._ensure_zep_component().is_configured:
            return self._build_empty_route(request_text=request_text)

        try:
            provider = self._ensure_provider()
            query_params = await self._infer_query_params(
                provider=provider,
                request_text=request_text,
                metadata=metadata,
            )
            zep_candidates = self.lookup_candidates(query_params=query_params, metadata=metadata)
        except Exception:  # noqa: BLE001
            logger.exception(
                "skill route pre-processing failed during query inference or Zep lookup",
                extra={
                    "request_preview": request_text[:120],
                    "metadata_keys": sorted((metadata or {}).keys()),
                },
            )
            raise
        loop_notes = [
            f"round=1 scope={query_params.scope.value} count={len(zep_candidates)} query={query_params.query}"
        ]
        if not zep_candidates and self.max_loop_rounds > 1:
            fallback_params = self._build_fallback_query_params(query_params)
            if fallback_params is not None:
                fallback_candidates = self.lookup_candidates(
                    query_params=fallback_params,
                    metadata=metadata,
                )
                zep_candidates.extend(fallback_candidates)
                loop_notes.append(
                    f"round=2 scope={fallback_params.scope.value} count={len(fallback_candidates)} "
                    f"query={fallback_params.query}"
                )
        if not zep_candidates:
            return self._build_empty_route(request_text=request_text)

        payload = {
            "trace_name": "skill_route_agent_generation",
            "langfuse_type": "generation",
            "system_prompt": ROUTE_SYSTEM_PROMPT,
            "inputs": [{"type": "text", "content": self._build_user_input(request_text, zep_candidates)}],
            "output_format": SkillRouteSchema,
            "metadata": {
                **(metadata or {}),
                "zep_candidate_count": len(zep_candidates),
                "zep_loop_notes": loop_notes,
            },
            "model_parameters": {"temperature": 0.1},
        }

        try:
            result = await provider.infer(payload)
        except Exception:  # noqa: BLE001
            logger.exception(
                "skill route inference failed",
                extra={
                    "trace_name": payload["trace_name"],
                    "request_preview": request_text[:120],
                    "candidate_count": len(zep_candidates),
                },
            )
            raise
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

    async def _infer_query_params(
        self,
        *,
        provider: InferenceProvider,
        request_text: str,
        metadata: dict[str, Any] | None,
    ) -> ZepQueryParams:
        payload = {
            "trace_name": "skill_route_query_params_generation",
            "langfuse_type": "generation",
            "system_prompt": QUERY_PARAMS_SYSTEM_PROMPT,
            "inputs": [{"type": "text", "content": request_text}],
            "output_format": ZepQueryParams,
            "metadata": metadata or {},
            "model_parameters": {"temperature": 0.0},
        }

        try:
            result = await provider.infer(payload)
        except Exception:  # noqa: BLE001
            logger.exception(
                "skill route query-params inference failed",
                extra={
                    "trace_name": payload["trace_name"],
                    "request_preview": request_text[:120],
                },
            )
            raise
        params = result if isinstance(result, ZepQueryParams) else ZepQueryParams.model_validate(result)
        if not params.query.strip():
            return params.model_copy(update={"query": request_text.strip()})
        return params

    @staticmethod
    def _build_fallback_query_params(params: ZepQueryParams) -> ZepQueryParams | None:
        if params.scope == ZepSearchScope.NODES:
            return params.model_copy(update={"scope": ZepSearchScope.EDGES})
        if params.scope == ZepSearchScope.EDGES:
            return params.model_copy(update={"scope": ZepSearchScope.NODES})
        return None

    def lookup_candidates(
        self,
        *,
        query_params: ZepQueryParams,
        metadata: dict[str, Any] | None = None,
    ) -> list[ZepSkillCandidate]:
        zep_component = self._ensure_zep_component()
        metadata_user_id = str((metadata or {}).get("user_id") or "").strip()
        explicit_user_id = query_params.user_id.strip()
        request = ZepQueryRequest(
            query=query_params.query.strip(),
            scope=query_params.scope.value,
            limit=query_params.limit,
            user_id=explicit_user_id or metadata_user_id or None,
        )
        return zep_component.execute_query(request)

    def _build_empty_route(self, *, request_text: str) -> SkillRouteSchema:
        zep_component = self._ensure_zep_component()
        if not zep_component.is_configured:
            rationale = "Zep is not configured, so no skill candidates could be retrieved."
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
        if self.provider is not None:
            await self.provider.close()

    # Entry point called by ADK Runner when an A2A message is received via
    # `on_message_send` -> executor -> `runner.run_async(...)`.
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
        try:
            route = await self.route_request(request_text=request_text, metadata=metadata)
        except Exception:  # noqa: BLE001
            logger.exception(
                "skill route request failed",
                extra={
                    "request_preview": request_text[:120],
                    "metadata_keys": sorted((metadata or {}).keys()),
                },
            )
            raise
        rendered = self.render_route(route)
        yield Event(
            invocation_id=ctx.invocation_id,
            author=self.name,
            content=types.Content(
                role="model",
                parts=[types.Part(text=rendered)],
            ),
        )
