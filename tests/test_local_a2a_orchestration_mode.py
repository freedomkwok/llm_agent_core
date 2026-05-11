# SPDX-License-Identifier: Apache-2.0
import asyncio

import pytest
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import AgentCapabilities, AgentCard, AgentSkill, TextPart

from agents.agent_core.a2a import (
    A2aRuntime,
    LocalA2aAgent,
    OrchestrationMode,
    configured_a2a_runtime,
    local_a2a_orchestration_mode,
    require_vertex_a2a_runtime,
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


class _EchoExecutor(AgentExecutor):
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        if not context.current_task:
            await updater.submit()
        await updater.start_work()
        await updater.add_artifact(
            [TextPart(text=f"echo: {context.get_user_input()}")],
            name="result",
        )
        await updater.complete()

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        del context, event_queue
        raise RuntimeError("cancel not supported")


def _echo_agent_card() -> AgentCard:
    return AgentCard(
        name="Echo",
        description="Echo test agent",
        skills=[AgentSkill(id="echo", name="Echo", description="Echo", tags=[])],
        capabilities=AgentCapabilities(streaming=False),
        defaultInputModes=["text/plain"],
        defaultOutputModes=["application/json"],
        preferredTransport="HTTP+JSON",
        supportsAuthenticatedExtendedCard=True,
        url="http://localhost:9999/",
        version="1.0.0",
    )


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


def test_local_a2a_agent_runs_without_vertex_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AGENT_CORE_A2A_RUNTIME", raising=False)
    agent = LocalA2aAgent(
        agent_card=_echo_agent_card(),
        agent_executor_builder=_EchoExecutor,
    )
    agent.set_up()

    result = asyncio.run(
        run_local_a2a_orchestration(
            a2a_agent=agent,
            message_text="hello",
            mode=OrchestrationMode.HOST_DRIVEN,
            task_poll_interval_sec=0.01,
            task_poll_timeout_sec=1,
        )
    )

    assert result.final_text == "echo: hello"
    assert result.task_status == "TASK_STATE_COMPLETED"


def test_a2a_runtime_defaults_to_local(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AGENT_CORE_A2A_RUNTIME", raising=False)

    assert configured_a2a_runtime() == A2aRuntime.LOCAL


def test_vertex_runtime_requires_explicit_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AGENT_CORE_A2A_RUNTIME", raising=False)

    with pytest.raises(RuntimeError, match="AGENT_CORE_A2A_RUNTIME=vertexai"):
        require_vertex_a2a_runtime()

    monkeypatch.setenv("AGENT_CORE_A2A_RUNTIME", "vertexai")

    require_vertex_a2a_runtime()
