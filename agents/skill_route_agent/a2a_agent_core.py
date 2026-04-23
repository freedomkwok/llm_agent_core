"""ADK BaseAgent that routes skills via Zep-backed inference."""

from __future__ import annotations

import logging
import os
from typing import Any, AsyncGenerator, Mapping
from uuid import uuid4

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events.event import Event
from google.genai import types
from llm_inference_core.providers import InferenceProvider
from pydantic import Field, SkipValidation
from typing_extensions import override

from agents.agent_core.inference_provider import create_inference_provider
from agents.skill_route_agent._env import bootstrap_env
from agents.skill_route_agent.schemas import (
    RoutedSkillCandidate,
    SkillRouteSchema,
    ZepQueryParams,
)
from agents.skill_route_agent.utils.zep_helper import (
    ZepQueryRequest,
    ZepSkillCandidate,
    ZepSkillSearchComponent,
    format_candidate_for_prompt,
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
    "Convert a user request into one concrete Zep graph.search query plan.\n"
    "Return output that strictly matches the requested schema.\n"
    "This call uses: graph.search(query, graph_id, scope, limit).\n"
    "Output format fields:\n"
    "- query: search text for Zep graph.search (max 400 chars).\n"
    "- scope: one of nodes or edges.\n"
    "- limit: integer from 1 to 20.\n"
    "- graph_id: optional Zep graph id override, empty string to use host GRAPH_ID.\n"
    "- rationale: short reason these parameters fit graph.search behavior.\n"
    "Rules:\n"
    "- Keep query compact, keyword-rich, and <= 400 characters.\n"
    "- Remove filler words; keep entities, actions, constraints, and domain terms.\n"
    "- Use scope='nodes' for entity/topic lookup and broad concept retrieval.\n"
    "- Use scope='edges' for specific facts, relations, events, or who-did-what details.\n"
    "- Prefer limit 5 for typical requests; use 3 for narrow lookup, 8-10 for broader recall.\n"
    "- Do not invent unsupported parameters (no reranker, filters, bfs, or episodes scope).\n"
    "- graph_id can be empty to rely on the configured GRAPH_ID environment variable.\n"
    "- If user explicitly provides graph_id, copy it exactly; otherwise leave graph_id empty.\n"
    "- Always output valid schema fields, never prose outside the schema.\n"
)


def _text_from_user_content(content: types.Content | None) -> str:
    if not content or not content.parts:
        return ""
    return "".join(part.text for part in content.parts if part.text)


def _route_invocation_metadata(ctx: InvocationContext) -> dict[str, Any]:
    """Metadata for routing/inference: session ids plus Zep graph_id.

    ``graph_id`` comes from session state when the A2A client sent
    ``message.metadata.graph_id`` (see ``AdkA2aExecutionWrapper``), else
    ``GRAPH_ID`` env.
    """
    md: dict[str, Any] = {
        "session_id": ctx.session.id,
        "user_id": ctx.user_id,
        "invocation_id": ctx.invocation_id,
    }
    graph_id = str(ctx.session.state.get("graph_id") or "").strip()
    if not graph_id:
        graph_id = os.getenv("GRAPH_ID", "").strip()
    if graph_id:
        md["graph_id"] = graph_id
    return md


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
    max_loop_rounds: int = 5

    def provider_settings_overrides(self) -> Mapping[str, Any]:
        """Override default inference settings for this agent if needed."""
        return {"conversation_store_type": "lru"}

    def provider_project_name(self) -> str:
        """Override project name used by inference tracing/context."""
        return "imp_agent_map.skill_route_agent"

    def provider_project_metadata(self) -> Mapping[str, Any]:
        """Override project metadata used by inference tracing/context."""
        return {"component": "skill_route_agent"}

    def _ensure_provider(self) -> InferenceProvider:
        if self.provider is not None:
            return self.provider

        self.provider = create_inference_provider(
            langfuse_client=self.langfuse_client,
            project_name=self.provider_project_name(),
            project_metadata=self.provider_project_metadata(),
            settings_overrides=self.provider_settings_overrides(),
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

 
        provider = self._ensure_provider()
        loop_notes: list[str] = []
        zep_candidates: list[ZepSkillCandidate] = []
        zep_feedback: str | None = None
        conversation_id = f"skill_route_query_params_{uuid4().hex}"

        for round_index in range(1, max(1, self.max_loop_rounds) + 1):
            llm_generated_zep_params = await self._infer_query_params(
                provider=provider,
                request_text=request_text,
                metadata=metadata,
                conversation_id=conversation_id,
                zep_feedback=zep_feedback,
            )
            try:
                round_candidates = self.lookup_candidates(
                    llm_generated_zep_params=llm_generated_zep_params,
                    metadata=metadata,
                )
            except Exception as exc:  # noqa: BLE001
                zep_error = f"{type(exc).__name__}: {exc}"
                zep_feedback = zep_error
                continue


            if round_candidates:
                zep_candidates = round_candidates
                break

            zep_feedback = "Zep returned zero candidates."

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
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "skill route inference failed",
                extra={
                    "trace_name": payload["trace_name"],
                    "request_preview": request_text[:120],
                    "candidate_count": len(zep_candidates),
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
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
        conversation_id: str,
        zep_feedback: str | None = None,
    ) -> ZepQueryParams:
        user_message = request_text.strip()
        if zep_feedback:
            user_message = zep_feedback.strip()
        payload = {
            "trace_name": "skill_route_query_params_generation",
            "langfuse_type": "generation",
            "system_prompt": QUERY_PARAMS_SYSTEM_PROMPT,
            "conversation_id": conversation_id,
            "user_message": user_message,
            "output_format": ZepQueryParams,
            "metadata": metadata or {},
            "model_parameters": {"temperature": 0.0},
        }

        result = await provider.infer(payload)

        params = result if isinstance(result, ZepQueryParams) else ZepQueryParams.model_validate(result)
        return params

    def lookup_candidates(
        self,
        *,
        llm_generated_zep_params: ZepQueryParams,
        metadata: dict[str, Any] | None = None,
    ) -> list[ZepSkillCandidate]:
        zep_component = self._ensure_zep_component()
        metadata_graph_id = str((metadata or {}).get("graph_id") or "").strip()
        explicit_graph_id = llm_generated_zep_params.graph_id.strip()
        request = ZepQueryRequest(
            query=llm_generated_zep_params.query.strip(),
            scope=llm_generated_zep_params.scope.value,
            limit=llm_generated_zep_params.limit,
            graph_id=explicit_graph_id or metadata_graph_id or None,
        )
        return zep_component.execute_query(request)

    def _build_empty_route(self, *, request_text: str) -> SkillRouteSchema:
        zep_component = self._ensure_zep_component()
        if not zep_component.is_configured:
            rationale = "Zep is not configured, so no skill candidates could be retrieved."
            next_action = "Configure ZEP_API_KEY and GRAPH_ID before routing again."
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
            format_candidate_for_prompt(candidate) for candidate in zep_candidates
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
        metadata = _route_invocation_metadata(ctx)
        try:
            route = await self.route_request(request_text=request_text, metadata=metadata)
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "skill route request failed",
                extra={
                    "request_preview": request_text[:120],
                    "metadata_keys": sorted((metadata or {}).keys()),
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
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
