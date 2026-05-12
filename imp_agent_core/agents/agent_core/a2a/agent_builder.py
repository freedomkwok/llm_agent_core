# SPDX-License-Identifier: Apache-2.0
"""Shared helper for creating local A2A agents from executor modules."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from imp_agent_core.agents.agent_core.a2a.local_agent import LocalA2aAgent
from imp_agent_core.agents.agent_core.a2a.runtime import A2aRuntime, configured_a2a_runtime


def build_local_a2a_agent(
    *,
    agent_card: Any,
    executor_module_path: str,
    executor_class_name: str,
) -> Any:
    """Build and set up a local A2A agent from an executor class path."""
    module = import_module(executor_module_path)
    executor_class = getattr(module, executor_class_name)
    agent_class: type[Any]
    if configured_a2a_runtime() == A2aRuntime.VERTEXAI:
        from vertexai.preview.reasoning_engines import A2aAgent

        agent_class = A2aAgent
    else:
        agent_class = LocalA2aAgent
    agent = agent_class(
        agent_card=agent_card,
        agent_executor_builder=lambda: executor_class(),
    )
    agent.set_up()
    return agent
