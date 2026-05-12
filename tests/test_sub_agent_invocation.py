# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

from imp_agent_core.agents.agent_core import DynamicAgentRegistry, OrchestrationMode
from imp_agent_core.agents.agent_core.adk.executor import AdkA2aExecutor
from imp_agent_core.agents.agent_core.routing.descriptor import normalize_skill_descriptors
from imp_agent_core.agents.agent_core.sub_agent_invoke import (
    SubAgentInvocationPolicy,
    SubAgentInvoker,
    SubAgentToolConfig,
    make_sub_agent_tool,
)


class FakeLocalA2AAgent:
    def __init__(self) -> None:
        self.messages: list[dict[str, Any]] = []

    async def handle_authenticated_agent_card(self, request=None, context=None):
        del request, context
        return {"name": "Fake Agent", "skills": ["fake_skill"]}

    async def on_message_send(self, request, context=None):
        del context
        payload = await request.json()
        self.messages.append(payload["message"])
        return {
            "task": {
                "id": f"task-{payload['message']['messageId']}",
                "status": {"state": "submitted"},
            }
        }

    async def on_get_task(self, request, context=None):
        del context
        return {
            "task": {
                "id": request.path_params["id"],
                "status": {"state": "completed"},
                "artifacts": [{"name": "result", "parts": [{"text": "child answer"}]}],
            }
        }


def _register_fake_agent(
    registry: DynamicAgentRegistry,
    *,
    agent_id: str = "fake.local",
    agent_name: str = "Fake Agent",
    skill_id: str = "fake_skill",
    local_agent: FakeLocalA2AAgent | None = None,
) -> FakeLocalA2AAgent:
    agent = local_agent or FakeLocalA2AAgent()
    registry.register_local_agent(
        agent_id=agent_id,
        agent_name=agent_name,
        description="Fake local A2A agent.",
        skills=normalize_skill_descriptors(
            [{"id": skill_id, "name": skill_id.replace("_", " ").title()}]
        ),
        local_builder=lambda: agent,
    )
    return agent


def test_invoker_forwards_policy_selected_state_as_child_metadata() -> None:
    registry = DynamicAgentRegistry()
    child_agent = _register_fake_agent(registry)
    invoker = SubAgentInvoker(
        registry=registry,
        policy=SubAgentInvocationPolicy(
            forwarded_state_keys=(
                "agent_core.user_id",
                "agent_core.a2a_context_id",
                "graph_id",
            ),
            metadata_aliases={
                "agent_core.user_id": "user_id",
                "agent_core.a2a_context_id": "parent_context_id",
            },
            static_metadata={"source": "parent"},
        ),
    )
    tool_context = SimpleNamespace(
        state={
            "agent_core.user_id": "alice",
            "agent_core.a2a_context_id": "ctx-parent",
            "graph_id": "graph-1",
            "ignored": "nope",
        }
    )

    result = asyncio.run(
        invoker.invoke(
            query="Ask child",
            tool_context=tool_context,
            agent_id="fake.local",
        )
    )

    assert result["status"] == "completed"
    assert result["final_text"] == "child answer"
    assert child_agent.messages[0]["metadata"] == {
        "source": "parent",
        "user_id": "alice",
        "parent_context_id": "ctx-parent",
        "graph_id": "graph-1",
        "agent_core.subquery_depth": 1,
    }


def test_invoker_default_policy_forwards_graph_id() -> None:
    registry = DynamicAgentRegistry()
    child_agent = _register_fake_agent(registry)
    invoker = SubAgentInvoker(registry=registry)

    result = asyncio.run(
        invoker.invoke(
            query="Ask child",
            tool_context=SimpleNamespace(
                state={
                    "agent_core.user_id": "alice",
                    "agent_core.a2a_context_id": "ctx-parent",
                    "graph_id": "graph-1",
                }
            ),
            agent_id="fake.local",
        )
    )

    assert result["status"] == "completed"
    assert child_agent.messages[0]["metadata"] == {
        "user_id": "alice",
        "parent_context_id": "ctx-parent",
        "graph_id": "graph-1",
        "agent_core.subquery_depth": 1,
    }


def test_invoker_resolves_by_skill_when_one_candidate_matches() -> None:
    registry = DynamicAgentRegistry()
    _register_fake_agent(registry, agent_id="fake.local", skill_id="fake_skill")
    invoker = SubAgentInvoker(registry=registry)

    result = asyncio.run(
        invoker.invoke(
            query="Ask child",
            tool_context=SimpleNamespace(state={}),
            skill_id="fake_skill",
        )
    )

    assert result["agent_id"] == "fake.local"
    assert result["mode"] == OrchestrationMode.AGENT_INTERNAL.value
    assert result["final_text"] == "child answer"


def test_invoker_returns_candidates_when_resolution_is_ambiguous() -> None:
    registry = DynamicAgentRegistry()
    _register_fake_agent(registry, agent_id="alpha.local", agent_name="Alpha")
    _register_fake_agent(registry, agent_id="beta.local", agent_name="Beta")
    invoker = SubAgentInvoker(registry=registry)

    result = asyncio.run(
        invoker.invoke(
            query="Ask child",
            tool_context=SimpleNamespace(state={}),
            skill_id="fake_skill",
        )
    )

    assert result["status"] == "ambiguous"
    assert result["candidates"] == [
        {"agent_id": "alpha.local", "agent_name": "Alpha", "skills": ["fake_skill"]},
        {"agent_id": "beta.local", "agent_name": "Beta", "skills": ["fake_skill"]},
    ]


def test_invoker_enforces_max_depth() -> None:
    registry = DynamicAgentRegistry()
    _register_fake_agent(registry)
    invoker = SubAgentInvoker(
        registry=registry,
        policy=SubAgentInvocationPolicy(max_depth=1),
    )

    result = asyncio.run(
        invoker.invoke(
            query="Ask child",
            tool_context=SimpleNamespace(state={"agent_core.subquery_depth": 1}),
            agent_id="fake.local",
        )
    )

    assert result == {
        "status": "error",
        "error": "sub_agent_depth_exceeded",
        "message": "Sub-agent invocation depth 1 reached the configured maximum of 1.",
    }


def test_make_sub_agent_tool_exposes_invoke_sub_agent_name() -> None:
    registry = DynamicAgentRegistry()
    _register_fake_agent(registry)
    tool = make_sub_agent_tool(registry=registry)

    result = asyncio.run(
        tool(
            query="Ask child",
            tool_context=SimpleNamespace(state={}),
            agent_id="fake.local",
        )
    )

    assert tool.__name__ == "invoke_sub_agent"
    assert result["final_text"] == "child answer"


def test_sub_agent_tool_config_appends_instruction_with_spacing() -> None:
    config = SubAgentToolConfig(
        registry=DynamicAgentRegistry(),
        instruction="Delegate narrow searches with invoke_sub_agent.",
    )

    assert (
        config.instruction_for("Base instruction.")
        == "Base instruction.\n\nDelegate narrow searches with invoke_sub_agent."
    )


def test_a2a_executor_seeds_agent_core_state_from_request_context() -> None:
    context = SimpleNamespace(task_id="task-parent", context_id="ctx-parent")
    state_delta = AdkA2aExecutor.build_agent_state_delta(
        context=context,
        user_id="alice",
        trace_id="trace-1",
        parent_span_id="span-1",
        incoming_metadata={
            "graph_id": "graph-1",
            "graph_backend": "oracle",
            "agent_core.subquery_depth": 2,
            "custom": "kept out",
        },
    )

    assert state_delta == {
        "agent_core.user_id": "alice",
        "agent_core.a2a_context_id": "ctx-parent",
        "agent_core.a2a_task_id": "task-parent",
        "agent_core.trace_id": "trace-1",
        "agent_core.parent_span_id": "span-1",
        "agent_core.subquery_depth": 2,
        "graph_id": "graph-1",
        "graph_backend": "oracle",
    }
