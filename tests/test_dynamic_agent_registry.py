# SPDX-License-Identifier: Apache-2.0
import asyncio
import sys
from typing import Any

from imp_agent_core.agents.agent_core.routing.descriptor import normalize_skill_descriptors
from imp_agent_core.agents.agent_core.routing.resolver import AgentResolver

from imp_agent_core.agents.agent_core import (
    DynamicAgentRegistry,
    HostOrchestrator,
    OrchestrationMode,
    register_agent_package,
)


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


def _register_fake_local_planning_agent(registry: DynamicAgentRegistry):
    return registry.register_local_agent(
        agent_id="planning_agent.local",
        agent_name="Planning Agent",
        description="Local planning test agent",
        skills=normalize_skill_descriptors(
            [{"id": "plan_request", "name": "Plan Request", "tags": ["planning"]}]
        ),
        local_builder=_build_fake_local_planning_agent,
    )


def test_register_and_resolve_local_planning_agent() -> None:
    registry = DynamicAgentRegistry()
    descriptor = _register_fake_local_planning_agent(registry)

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


def test_register_agent_package_calls_worker_registrars(tmp_path, monkeypatch) -> None:
    package_dir = tmp_path / "sample_agents"
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    agent_dir = package_dir / "worker_agent"
    agent_dir.mkdir()
    (agent_dir / "__init__.py").write_text("", encoding="utf-8")
    (agent_dir / "registry.py").write_text(
        """
from imp_agent_core.agents.agent_core.routing.descriptor import normalize_skill_descriptors


def register_worker_agent(registry, *, replace=True):
    return registry.register_local_agent(
        agent_id="worker_agent.local",
        agent_name="Worker Agent",
        description="Temporary worker agent",
        skills=normalize_skill_descriptors([{"id": "work", "name": "Work"}]),
        local_builder=lambda: object(),
        replace=replace,
    )
""",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    registry = DynamicAgentRegistry()
    registered = register_agent_package(registry, package_name="sample_agents")

    assert [descriptor.agent_id for descriptor in registered] == ["worker_agent.local"]
    assert registry.get_descriptor("worker_agent.local").supports_skill("work")
    sys.modules.pop("sample_agents.worker_agent.registry", None)
    sys.modules.pop("sample_agents.worker_agent", None)
    sys.modules.pop("sample_agents", None)


def test_register_agent_package_includes_zep_agent_by_default(tmp_path, monkeypatch) -> None:
    package_dir = tmp_path / "sample_agents"
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    zep_dir = package_dir / "zep_agent"
    zep_dir.mkdir()
    (zep_dir / "__init__.py").write_text("", encoding="utf-8")
    (zep_dir / "registry.py").write_text(
        """
from imp_agent_core.agents.agent_core.routing.descriptor import normalize_skill_descriptors


def register_zep_worker_agent(registry, *, replace=True):
    return registry.register_local_agent(
        agent_id="zep_agent.worker",
        agent_name="Zep Agent",
        description="Temporary zep worker",
        skills=normalize_skill_descriptors([{"id": "zep", "name": "Zep"}]),
        local_builder=lambda: object(),
        replace=replace,
    )
""",
        encoding="utf-8",
    )
    monkeypatch.delenv("AGENT_CORE_A2A_RUNTIME", raising=False)
    monkeypatch.syspath_prepend(str(tmp_path))

    registry = DynamicAgentRegistry()
    registered = register_agent_package(registry, package_name="sample_agents")

    assert [descriptor.agent_id for descriptor in registered] == ["zep_agent.worker"]
    assert registry.get_descriptor("zep_agent.worker").supports_skill("zep")
    sys.modules.pop("sample_agents.zep_agent.registry", None)
    sys.modules.pop("sample_agents.zep_agent", None)
    sys.modules.pop("sample_agents", None)


def test_resolver_prefers_local_when_same_skill_exists_remotely() -> None:
    registry = DynamicAgentRegistry()
    local_descriptor = _register_fake_local_planning_agent(registry)
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
    _register_fake_local_planning_agent(registry)

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
