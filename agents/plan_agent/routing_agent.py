"""Registry and router ADK steps for the plan agent."""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from typing import Any

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events.event import Event
from pydantic import Field
from typing_extensions import override

from agents.plan_agent.deps import (
    SESSION_BASE_INSTRUCTION,
    SESSION_REGISTRY_ATTEMPTS,
    SESSION_REGISTRY_LAST_QUERY,
    SESSION_SELECTED_SKILL,
    SESSION_SKILL_CANDIDATES,
    PlanAgentDeps,
    SkillCandidate,
)
from agents.utils.zep_retrievaler import retrieve_candidates_until_success

logger = logging.getLogger(__name__)


DEFAULT_REGISTRY_PROMPT = (
    "You generate JSON for Zep retrieval. Return strict JSON only.\n"
    "Schema:\n"
    "{\n"
    '  "request": {\n'
    '    "query": "...",\n'
    '    "top_k": 8,\n'
    '    "scope": null,\n'
    '    "filters": {},\n'
    '    "metadata": {}\n'
    "  },\n"
    '  "reason": "optional short reason"\n'
    "}"
)


def _user_message_from_ctx(ctx: InvocationContext) -> str:
    content = ctx.user_content
    if not content or not content.parts:
        return ""
    chunks: list[str] = []
    for part in content.parts:
        text = getattr(part, "text", None)
        if isinstance(text, str) and text:
            chunks.append(text)
    return "\n".join(chunks).strip()


class ZepSkillRegistryAgent(BaseAgent):
    """
    Runs before the router and performs:

    1) registry-core JSON generation (`ZepRegistryPlan` via pydantic parsing),
    2) repeated Zep retrieval attempts until success or max attempts,
    3) writes retrieved candidates into session state for router consumption.
    """

    deps: Any = Field(..., description="Concrete PlanAgentDeps from your app/library.")
    base_instruction: str = Field(
        default="",
        description="Langfuse prompt text; copied into session state for observability.",
    )
    retrieval_prompt: str = Field(
        default=DEFAULT_REGISTRY_PROMPT,
        description="Prompt sent to your registry LLM core for request JSON generation.",
    )
    max_retrieval_attempts: int = Field(
        default=4,
        ge=1,
        le=50,
        description="How many LLM->Zep retrieval attempts before giving up.",
    )

    @override
    async def _run_async_impl(
        self,
        ctx: InvocationContext,
    ) -> AsyncGenerator[Event, None]:
        if self.base_instruction:
            ctx.session.state[SESSION_BASE_INSTRUCTION] = self.base_instruction
        # Prevent stale selection from previous turns until router runs again.
        ctx.session.state.pop(SESSION_SELECTED_SKILL, None)
        deps: PlanAgentDeps = self.deps
        user_message = _user_message_from_ctx(ctx)

        retrieval_result = await retrieve_candidates_until_success(
            user_message=user_message,
            generation_prompt=self.retrieval_prompt,
            registry_core=deps.registry_core,
            zep_catalog=deps.zep,
            max_attempts=self.max_retrieval_attempts,
        )
        ctx.session.state[SESSION_REGISTRY_ATTEMPTS] = [
            {
                "attempt": a.attempt,
                "success": a.success,
                "request": a.request,
                "candidate_count": a.candidate_count,
                "error": a.error,
            }
            for a in retrieval_result.attempts
        ]
        if retrieval_result.last_request:
            ctx.session.state[SESSION_REGISTRY_LAST_QUERY] = (
                retrieval_result.last_request.model_dump()
            )
        else:
            ctx.session.state.pop(SESSION_REGISTRY_LAST_QUERY, None)

        candidates = retrieval_result.candidates
        ctx.session.state[SESSION_SKILL_CANDIDATES] = [c.as_state_dict() for c in candidates]

        if not candidates:
            logger.warning(
                (
                    "ZepSkillRegistryAgent: no candidates after %s attempt(s); "
                    "leaving selection empty."
                ),
                len(retrieval_result.attempts),
            )
            if False:
                yield  # pragma: no cover
            return

        if False:
            yield  # pragma: no cover


def _candidates_from_session_state(raw: Any) -> list[SkillCandidate]:
    if not isinstance(raw, list):
        return []
    out: list[SkillCandidate] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        skill_id = str(item.get("id", "")).strip()
        if not skill_id:
            continue
        metadata = item.get("metadata")
        out.append(
            SkillCandidate(
                id=skill_id,
                title=str(item.get("title", "")),
                summary=str(item.get("summary", "")),
                body=str(item.get("body", "")),
                metadata=metadata if isinstance(metadata, dict) else None,
            )
        )
    return out


class ZepSkillSelectionRouterAgent(BaseAgent):
    """Router step: pick one selected skill from retrieved candidates."""

    deps: Any = Field(..., description="Concrete PlanAgentDeps from your app/library.")

    @override
    async def _run_async_impl(
        self,
        ctx: InvocationContext,
    ) -> AsyncGenerator[Event, None]:
        deps: PlanAgentDeps = self.deps
        user_message = _user_message_from_ctx(ctx)
        candidates = _candidates_from_session_state(
            ctx.session.state.get(SESSION_SKILL_CANDIDATES)
        )
        if not candidates:
            logger.warning(
                "ZepSkillSelectionRouterAgent: no skill candidates in session state."
            )
            ctx.session.state.pop(SESSION_SELECTED_SKILL, None)
            if False:
                yield  # pragma: no cover
            return

        picked_id = await deps.router.pick_skill_id(user_message, candidates)
        selected: SkillCandidate | None = next(
            (c for c in candidates if c.id == picked_id),
            None,
        )
        if selected is None:
            logger.error(
                "Router returned unknown skill id %r; ids=%r",
                picked_id,
                [c.id for c in candidates],
            )
            ctx.session.state.pop(SESSION_SELECTED_SKILL, None)
            if False:
                yield  # pragma: no cover
            return

        ctx.session.state[SESSION_SELECTED_SKILL] = selected.as_state_dict()
        if False:
            yield  # pragma: no cover
