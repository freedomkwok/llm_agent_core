# SPDX-License-Identifier: Apache-2.0
from imp_agent_core.agents.agent_core import DynamicAgentRegistry, reset_global_agent_registry
from imp_agent_core.agents.zep_agent.registry import (
    register_local_zep_agent,
    register_zep_worker_agent,
)


class FakeLocalZepA2AAgent:
    async def handle_authenticated_agent_card(self, request=None, context=None):
        del request, context
        return {"name": "Zep Tool Agent", "skills": ["route_with_zep_tools"]}


def _build_fake_local_zep_agent() -> FakeLocalZepA2AAgent:
    return FakeLocalZepA2AAgent()


def test_register_local_zep_agent() -> None:
    registry = DynamicAgentRegistry()
    descriptor = register_local_zep_agent(
        registry,
        local_builder=_build_fake_local_zep_agent,
    )

    assert descriptor.agent_id == "zep_agent.local"
    assert descriptor.supports_skill("route_with_zep_tools")


def test_register_zep_worker_agent_registers_worker_descriptor() -> None:
    registry = reset_global_agent_registry()
    register_zep_worker_agent()
    descriptor = registry.get_descriptor("zep_agent.worker")

    assert descriptor.supports_skill("route_with_zep_tools")

