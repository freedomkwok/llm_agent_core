"""Shared helper for creating local A2A agents from executor modules."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from vertexai.preview.reasoning_engines import A2aAgent


def build_local_a2a_agent(
    *,
    agent_card: Any,
    executor_module_path: str,
    executor_class_name: str,
) -> A2aAgent:
    """Build and set up a local A2A agent from an executor class path."""
    module = import_module(executor_module_path)
    executor_class = getattr(module, executor_class_name)
    agent = A2aAgent(
        agent_card=agent_card,
        agent_executor_builder=lambda: executor_class(),
    )
    agent.set_up()
    return agent

