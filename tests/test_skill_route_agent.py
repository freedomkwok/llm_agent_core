import asyncio

from agents.agent_core import DynamicAgentRegistry
from agents.skill_route_agent.registry import register_local_skill_route_agent
from agents.skill_route_agent.skill_route_adk_agent import SkillRouteAdkAgent
from agents.skill_route_agent.utils.zep_helper import ZepQueryRequest


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
