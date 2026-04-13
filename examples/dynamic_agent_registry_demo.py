"""Demo for dynamic agent registry and unified host orchestration."""

from __future__ import annotations

import asyncio

from agents.agent_core import DynamicAgentRegistry, HostOrchestrator, OrchestrationMode
from agents.agent_core.agent_descriptor import normalize_skill_descriptors
from agents.planning_agent import register_local_planning_agent


async def main() -> None:
    registry = DynamicAgentRegistry()

    register_local_planning_agent(registry)

    registry.register_remote_agent(
        agent_id="weather_agent.remote",
        agent_name="Weather Agent",
        description="Stub remote weather agent reachable over the network.",
        skills=normalize_skill_descriptors(
            [
                {
                    "id": "weather_query",
                    "name": "Weather Query",
                    "description": "Fetch weather for a city.",
                    "tags": ["weather", "forecast"],
                }
            ]
        ),
        endpoint="http://weather-agent.internal:8080",
        metadata={"auth": {"type": "bearer"}, "headers": {"x-service": "weather"}},
    )

    orchestrator = HostOrchestrator(registry=registry)
    result = await orchestrator.invoke(
        skill_id="plan_request",
        message_text="Plan how to add a remote weather agent to the registry.",
        mode=OrchestrationMode.HOST_DRIVEN,
    )

    print("Resolved agent:", result.descriptor.agent_id)
    print("Task status:", result.task_status)
    print("Final text:")
    print(result.final_text)


if __name__ == "__main__":
    asyncio.run(main())
