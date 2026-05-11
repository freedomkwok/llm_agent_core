# SPDX-License-Identifier: Apache-2.0
import asyncio

from agents.agent_core.a2a import (
    OrchestrationMode,
    local_a2a_orchestration_mode,
    run_local_a2a_orchestration,
    set_local_a2a_orchestration_mode,
)


class _FakeLocalA2AAgent:
    def __init__(self) -> None:
        self.card_calls = 0
        self.task_calls = 0
        self.return_direct_task = False

    async def handle_authenticated_agent_card(self, request=None, context=None):
        del request, context
        self.card_calls += 1
        return {"name": "fake"}

    async def on_message_send(self, request, context=None):
        del request, context
        return {"task": {"id": "task-1", "status": {"state": "submitted"}}}

    async def on_get_task(self, request, context=None):
        del request, context
        self.task_calls += 1
        task = {
            "id": "task-1",
            "status": {"state": "completed"},
            "artifacts": [{"name": "result", "parts": [{"text": "final answer"}]}],
        }
        if self.return_direct_task:
            return task
        return {"task": task}


def test_local_a2a_orchestration_uses_mode_stored_on_agent() -> None:
    agent = _FakeLocalA2AAgent()
    set_local_a2a_orchestration_mode(agent, OrchestrationMode.AGENT_INTERNAL)

    result = asyncio.run(
        run_local_a2a_orchestration(
            a2a_agent=agent,
            message_text="hello",
        )
    )

    assert local_a2a_orchestration_mode(agent) == OrchestrationMode.AGENT_INTERNAL
    assert result.mode == OrchestrationMode.AGENT_INTERNAL
    assert agent.card_calls == 0
    assert agent.task_calls == 0


def test_explicit_orchestration_mode_overrides_agent_mode() -> None:
    agent = _FakeLocalA2AAgent()
    set_local_a2a_orchestration_mode(agent, OrchestrationMode.AGENT_INTERNAL)

    result = asyncio.run(
        run_local_a2a_orchestration(
            a2a_agent=agent,
            message_text="hello",
            mode=OrchestrationMode.HOST_DRIVEN,
        )
    )

    assert result.mode == OrchestrationMode.HOST_DRIVEN
    assert agent.card_calls == 1
    assert agent.task_calls == 1


def test_host_driven_orchestration_extracts_final_text_from_direct_task() -> None:
    agent = _FakeLocalA2AAgent()
    agent.return_direct_task = True

    result = asyncio.run(
        run_local_a2a_orchestration(
            a2a_agent=agent,
            message_text="hello",
            mode=OrchestrationMode.HOST_DRIVEN,
        )
    )

    assert result.task_status == "completed"
    assert result.final_text == "final answer"
