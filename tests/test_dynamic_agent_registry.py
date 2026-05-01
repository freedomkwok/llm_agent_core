import asyncio
from typing import Any

from agents.agent_core import DynamicAgentRegistry, HostOrchestrator, OrchestrationMode
from agents.agent_core.routing.descriptor import normalize_skill_descriptors
from agents.agent_core.routing.resolver import AgentResolver
from agents.planning_agent.registry import register_local_planning_agent


class FakeLocalPlanningA2AAgent:
    async def handle_authenticated_agent_card(self, request=None, context=None):
        del request, context
        return {"name": "Planning Agent", "skills": ["plan_request"]}

    async def on_message_send(self, request, context=None):
        del context
        payload = await request.json()
        return {
            "task": {
                "id": f"task-{payload['message']['messageId']}",
                "status": {"state": "submitted"},
            }
        }

    async def on_get_task(self, request, context=None):
        del context
        task_id = request.path_params["id"]
        return {
            "task": {
                "id": task_id,
                "status": {"state": "completed"},
                "artifacts": [
                    {
                        "name": "plan",
                        "parts": [{"text": "Goal:\nTest plan"}],
                    }
                ],
            }
        }


def _build_fake_local_planning_agent() -> FakeLocalPlanningA2AAgent:
    return FakeLocalPlanningA2AAgent()


def test_register_and_resolve_local_planning_agent() -> None:
    registry = DynamicAgentRegistry()
    descriptor = register_local_planning_agent(
        registry,
        local_builder=_build_fake_local_planning_agent,
    )

    resolver = AgentResolver(registry)
    resolved = resolver.resolve_descriptor(skill_id="plan_request")

    assert resolved.agent_id == descriptor.agent_id
    assert resolved.supports_skill("plan_request")


def test_register_remote_stub_agent() -> None:
    registry = DynamicAgentRegistry()
    descriptor = registry.register_remote_agent(
        agent_id="weather_agent.remote",
        agent_name="Weather Agent",
        description="Remote weather service",
        skills=normalize_skill_descriptors(
            [{"id": "weather_query", "name": "Weather Query", "tags": ["weather"]}]
        ),
        endpoint="http://weather-agent.internal:8080",
        metadata={"headers": {"x-service": "weather"}},
    )

    assert descriptor.endpoint == "http://weather-agent.internal:8080"
    assert descriptor.backend_type.value == "remote_a2a"


def test_resolver_prefers_local_when_same_skill_exists_remotely() -> None:
    registry = DynamicAgentRegistry()
    local_descriptor = register_local_planning_agent(
        registry,
        local_builder=_build_fake_local_planning_agent,
    )
    registry.register_remote_agent(
        agent_id="planning_agent.remote",
        agent_name="Planning Agent Remote",
        description="Remote planning service",
        skills=normalize_skill_descriptors(
            [{"id": "plan_request", "name": "Plan Request", "tags": ["planning"]}]
        ),
        endpoint="http://planning-agent.internal:8080",
    )

    resolver = AgentResolver(registry, prefer_local=True)
    resolved = resolver.resolve_descriptor(skill_id="plan_request")

    assert resolved.agent_id == local_descriptor.agent_id


def test_orchestrator_invokes_local_handle_through_unified_interface() -> None:
    registry = DynamicAgentRegistry()
    register_local_planning_agent(
        registry,
        local_builder=_build_fake_local_planning_agent,
    )

    orchestrator = HostOrchestrator(registry=registry)

    async def run() -> Any:
        return await orchestrator.invoke(
            skill_id="plan_request",
            message_text="Plan the next task.",
            mode=OrchestrationMode.HOST_DRIVEN,
            metadata={"user_id": "test-user"},
        )

    result = asyncio.run(run())
    assert result.descriptor.agent_name == "Planning Agent"
    assert result.task_status == "completed"
    assert result.final_text == "Goal:\nTest plan"
