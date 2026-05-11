# SPDX-License-Identifier: Apache-2.0
"""Step 4: create local A2aAgent instance."""

import importlib.util
from pathlib import Path
from types import ModuleType

from vertexai.preview.reasoning_engines import A2aAgent

from langfuse_map_env import bootstrap_langfuse_from_repo_env

# Project root: imp_agent_map/.env; MAP_LANGFUSE_* → LANGFUSE_* for Langfuse SDK
bootstrap_langfuse_from_repo_env()


def _load_module(filename: str) -> ModuleType:
    base = Path(__file__).parent
    path = base / filename
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_local_a2a_agent() -> A2aAgent:
    card_mod = _load_module("01_agent_card.py")
    llm_mod = _load_module("02_llm_agent.py")
    exec_mod = _load_module("03_agent_executor.py")

    agent = A2aAgent(
        agent_card=card_mod.agent_card,
        agent_executor_builder=lambda: exec_mod.CurrencyAgentExecutorWithRunner(
            agent=llm_mod.my_llm_agent
        ),
    )
    agent.set_up()
    return agent


if __name__ == "__main__":
    local_agent = build_local_a2a_agent()
    print(local_agent)
