import types

import pytest

pytest.importorskip("google.adk")

import agents.agent_core.configured_a2a_executor as configured_executor_module
from agents.agent_core.configured_a2a_executor import ConfiguredA2aExecutor


def test_build_agent_from_config_passes_instruction_prompt_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def _builder(
        *,
        langfuse_client=None,
        instruction_prompt_name=None,
        instruction_prompt_label=None,
    ):
        captured["langfuse_client"] = langfuse_client
        captured["instruction_prompt_name"] = instruction_prompt_name
        captured["instruction_prompt_label"] = instruction_prompt_label
        return object()

    monkeypatch.setattr(
        configured_executor_module,
        "import_module",
        lambda _: types.SimpleNamespace(build_agent=_builder),
    )
    monkeypatch.setattr(configured_executor_module, "BaseAgent", object)

    executor = ConfiguredA2aExecutor.__new__(ConfiguredA2aExecutor)
    executor.langfuse_client = "lf-client"
    executor._config = {
        "adk_agent_builder": "dummy.module.build_agent",
        "instruction_prompt_name": " agents/zep_query_agent/instruction ",
        "instruction_prompt_label": " staging ",
    }

    ConfiguredA2aExecutor._build_agent_from_config(executor)

    assert captured == {
        "langfuse_client": "lf-client",
        "instruction_prompt_name": "agents/zep_query_agent/instruction",
        "instruction_prompt_label": "staging",
    }


def test_build_agent_from_config_ignores_prompt_overrides_when_not_supported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def _builder(*, langfuse_client=None):
        captured["langfuse_client"] = langfuse_client
        return object()

    monkeypatch.setattr(
        configured_executor_module,
        "import_module",
        lambda _: types.SimpleNamespace(build_agent=_builder),
    )
    monkeypatch.setattr(configured_executor_module, "BaseAgent", object)

    executor = ConfiguredA2aExecutor.__new__(ConfiguredA2aExecutor)
    executor.langfuse_client = "lf-client"
    executor._config = {
        "adk_agent_builder": "dummy.module.build_agent",
        "instruction_prompt_name": "agents/zep_query_agent/instruction",
        "instruction_prompt_label": "production",
    }

    ConfiguredA2aExecutor._build_agent_from_config(executor)

    assert captured == {"langfuse_client": "lf-client"}
