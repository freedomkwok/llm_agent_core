# SPDX-License-Identifier: Apache-2.0
import asyncio

from imp_agent_core.agents.agent_core import DynamicAgentRegistry
from imp_agent_core.agents.skill_route_agent.a2a_agent_core import SkillRouteAdkAgent
from imp_agent_core.agents.skill_route_agent.registry import register_local_skill_route_agent
from imp_agent_core.agents.skill_route_agent.schemas import (
    SkillRouteSchema,
    ZepQueryParams,
    ZepSearchScope,
)
from imp_agent_core.agents.skill_route_agent.utils.zep_helper import (
    ZepQueryRequest,
    ZepSkillCandidate,
)


class FakeLocalSkillRouteA2AAgent:
    async def handle_authenticated_agent_card(self, request=None, context=None):
        del request, context
        return {"name": "Skill Route Agent", "skills": ["route_skill_request"]}


def _build_fake_local_skill_route_agent() -> FakeLocalSkillRouteA2AAgent:
    return FakeLocalSkillRouteA2AAgent()


class DisabledZepComponent:
    is_configured = False

    def execute_query(self, request: ZepQueryRequest):
        del request
        return []


class FakeStructuredQueryProvider:
    async def infer(self, payload):
        del payload
        return ZepQueryParams(
            query="",
            scope=ZepSearchScope.NODES,
            limit=5,
            rationale="Use the raw request when the model leaves query empty.",
        )


class RetryFeedbackProvider:
    def __init__(self) -> None:
        self.query_payloads: list[dict] = []

    async def infer(self, payload):
        output_format = payload.get("output_format")
        if output_format is ZepQueryParams:
            self.query_payloads.append(dict(payload))
            return ZepQueryParams(
                query="redis session",
                scope=ZepSearchScope.NODES,
                limit=5,
                rationale="Try entity-centric lookup first.",
            )
        if output_format is SkillRouteSchema:
            return SkillRouteSchema(
                request_summary="find redis session skill",
                selected_skill_id="",
                selected_skill_name="",
                rationale="Use first candidate.",
                candidate_skills=[],
                planner_prompt="find redis session skill",
                next_action="invoke selected skill",
            )
        raise AssertionError(f"Unexpected output_format: {output_format!r}")


class ZepFailsThenReturnsCandidate:
    def __init__(self) -> None:
        self.calls = 0

    @property
    def is_configured(self) -> bool:
        return True

    def execute_query(self, request: ZepQueryRequest):
        del request
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("temporary zep timeout")
        return [
            ZepSkillCandidate(
                skill_id="skill.redis.lookup",
                name="Redis Lookup",
                description="Find Redis session helpers.",
                raw={},
            )
        ]


def test_register_local_skill_route_agent() -> None:
    registry = DynamicAgentRegistry()
    descriptor = register_local_skill_route_agent(
        registry,
        local_builder=_build_fake_local_skill_route_agent,
    )

    assert descriptor.agent_id == "skill_route_agent.local"
    assert descriptor.supports_skill("route_skill_request")


def test_skill_route_engine_returns_empty_route_when_zep_unavailable() -> None:
    agent = SkillRouteAdkAgent(zep_component=DisabledZepComponent())

    async def run():
        return await agent.route_request(
            request_text="Choose the best skill for building Redis sessions."
        )

    route = asyncio.run(run())

    assert route.selected_skill_id == ""
    assert route.candidate_skills == []
    assert "Zep is not configured" in route.rationale


def test_infer_query_params_accepts_structured_provider_result() -> None:
    agent = SkillRouteAdkAgent(provider=FakeStructuredQueryProvider())

    async def run():
        return await agent._infer_query_params(
            provider=agent.provider,
            request_text="find redis session skill",
            metadata=None,
            conversation_id="test-conversation",
        )

    params = asyncio.run(run())

    assert params.query == ""
    assert params.scope == ZepSearchScope.NODES


def test_route_request_retries_with_feedback_after_zep_exception() -> None:
    provider = RetryFeedbackProvider()
    agent = SkillRouteAdkAgent(
        provider=provider,
        zep_component=ZepFailsThenReturnsCandidate(),
        max_loop_rounds=3,
    )

    async def run():
        return await agent.route_request(request_text="find redis session skill")

    route = asyncio.run(run())

    assert route.candidate_skills
    assert len(provider.query_payloads) == 2
    first_conversation_id = str(provider.query_payloads[0].get("conversation_id"))
    second_conversation_id = str(provider.query_payloads[1].get("conversation_id"))
    assert first_conversation_id
    assert first_conversation_id == second_conversation_id
    second_round_message = str(provider.query_payloads[1].get("user_message", ""))
    assert "temporary zep timeout" in second_round_message
