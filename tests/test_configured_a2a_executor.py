# SPDX-License-Identifier: Apache-2.0
import types
from typing import Any

import pytest

pytest.importorskip("google.adk")

import imp_agent_core.agents.agent_core.adk.executor as configured_executor_module
from imp_agent_core.agents.agent_core import (
    DynamicAgentRegistry,
    SubAgentToolConfig,
    get_global_agent_registry,
    reset_global_agent_registry,
)
from imp_agent_core.agents.agent_core.adk.executor import ConfiguredA2aExecutor


def test_build_agent_from_config_passes_instruction_prompt_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def _builder(
        *,
        langfuse_client=None,
        instruction_prompt_name=None,
        instruction_prompt_label=None,
        fallback_instruction=None,
    ):
        captured["langfuse_client"] = langfuse_client
        captured["instruction_prompt_name"] = instruction_prompt_name
        captured["instruction_prompt_label"] = instruction_prompt_label
        captured["fallback_instruction"] = fallback_instruction
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
        "fallback_instruction": " Use Zep tools. ",
    }

    ConfiguredA2aExecutor._build_agent_from_config(executor)

    assert captured == {
        "langfuse_client": "lf-client",
        "instruction_prompt_name": "agents/zep_query_agent/instruction",
        "instruction_prompt_label": "staging",
        "fallback_instruction": "Use Zep tools.",
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
        "fallback_instruction": "Use Zep tools.",
    }

    ConfiguredA2aExecutor._build_agent_from_config(executor)

    assert captured == {"langfuse_client": "lf-client"}


def test_sub_agent_instruction_accepts_instruction_prompt_alias() -> None:
    executor = ConfiguredA2aExecutor.__new__(ConfiguredA2aExecutor)
    text = executor._sub_agent_instruction(
        {"instruction_prompt": "Delegate with invoke_sub_agent when helpful."}
    )
    assert text == "Delegate with invoke_sub_agent when helpful."


def test_sub_agent_instruction_loads_prompt_when_name_or_label_is_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def _load_instruction(**kwargs: Any) -> str:
        captured.update(kwargs)
        return "Loaded sub-agent instruction."

    monkeypatch.setattr(configured_executor_module, "load_agent_instruction", _load_instruction)

    executor = ConfiguredA2aExecutor.__new__(ConfiguredA2aExecutor)
    executor._config = {}
    text = executor._sub_agent_instruction(
        {
            "instruction_prompt": "Fallback sub-agent instruction.",
            "instruction_prompt_name": "agents/zep_query_agent/instruction_subagent",
            "instruction_prompt_label": "production",
        }
    )

    assert text == "Loaded sub-agent instruction."
    assert captured["fallback_instruction"] == "Fallback sub-agent instruction."
    assert captured["instruction_prompt_name"] == "agents/zep_query_agent/instruction_subagent"
    assert captured["instruction_prompt_label"] == "production"


def test_build_agent_from_config_passes_constructor_sub_agent_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    registry = DynamicAgentRegistry()

    def _builder(*, sub_agent_tool_config=None):
        captured["sub_agent_tool_config"] = sub_agent_tool_config
        return object()

    monkeypatch.setattr(
        configured_executor_module,
        "import_module",
        lambda _: types.SimpleNamespace(build_agent=_builder),
    )
    monkeypatch.setattr(configured_executor_module, "BaseAgent", object)

    executor = ConfiguredA2aExecutor.__new__(ConfiguredA2aExecutor)
    executor.langfuse_client = "lf-client"
    executor._constructor_sub_agent_registry = registry
    executor._constructor_sub_agent_policy = None
    executor._constructor_sub_agent_resolver = None
    executor._constructor_sub_agent_instruction = "\nDelegate when useful."
    executor._config = {"adk_agent_builder": "dummy.module.build_agent"}

    ConfiguredA2aExecutor._build_agent_from_config(executor)

    sub_agent_tool_config = captured["sub_agent_tool_config"]
    assert isinstance(sub_agent_tool_config, SubAgentToolConfig)
    assert sub_agent_tool_config.registry is registry
    assert sub_agent_tool_config.instruction == "\nDelegate when useful."


def test_build_agent_from_config_passes_yaml_sub_agent_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    registry = DynamicAgentRegistry()

    def _builder(*, sub_agent_tool_config=None):
        captured["sub_agent_tool_config"] = sub_agent_tool_config
        return object()

    def _registry_builder():
        return registry

    def _import_module(module_path: str):
        if module_path == "dummy.module":
            return types.SimpleNamespace(build_agent=_builder)
        if module_path == "dummy.registry":
            return types.SimpleNamespace(build_registry=_registry_builder)
        raise AssertionError(module_path)

    monkeypatch.setattr(configured_executor_module, "import_module", _import_module)
    monkeypatch.setattr(configured_executor_module, "BaseAgent", object)

    executor = ConfiguredA2aExecutor.__new__(ConfiguredA2aExecutor)
    executor.langfuse_client = "lf-client"
    executor._constructor_sub_agent_registry = None
    executor._constructor_sub_agent_policy = None
    executor._constructor_sub_agent_resolver = None
    executor._constructor_sub_agent_instruction = None
    executor._config = {
        "adk_agent_builder": "dummy.module.build_agent",
        "sub_agent_tool": {
            "enabled": True,
            "registry_builder": "dummy.registry.build_registry",
            "instruction": "\nUse a sub-agent.",
            "max_depth": 2,
            "allowed_agent_ids": ["zep_agent.worker"],
            "forwarded_state_keys": ["graph_id"],
            "metadata_aliases": {"graph_id": "graph_id"},
            "static_metadata": {"source": "yaml"},
        },
    }

    ConfiguredA2aExecutor._build_agent_from_config(executor)

    sub_agent_tool_config = captured["sub_agent_tool_config"]
    assert isinstance(sub_agent_tool_config, SubAgentToolConfig)
    assert sub_agent_tool_config.registry is registry
    assert sub_agent_tool_config.instruction == "\nUse a sub-agent."
    assert sub_agent_tool_config.policy is not None
    assert sub_agent_tool_config.policy.max_depth == 2
    assert sub_agent_tool_config.policy.allowed_agent_ids == ("zep_agent.worker",)
    assert sub_agent_tool_config.policy.forwarded_state_keys == ("graph_id",)
    assert sub_agent_tool_config.policy.metadata_aliases == {"graph_id": "graph_id"}
    assert sub_agent_tool_config.policy.static_metadata == {"source": "yaml"}


def test_build_agent_from_config_can_use_global_sub_agent_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    registry = reset_global_agent_registry()
    auto_registered: list[object] = []

    def _builder(*, sub_agent_tool_config=None):
        captured["sub_agent_tool_config"] = sub_agent_tool_config
        return object()

    monkeypatch.setattr(
        configured_executor_module,
        "import_module",
        lambda _: types.SimpleNamespace(build_agent=_builder),
    )
    monkeypatch.setattr(configured_executor_module, "BaseAgent", object)
    monkeypatch.setattr(
        configured_executor_module,
        "register_agent_package",
        lambda registry: auto_registered.append(registry),
    )

    executor = ConfiguredA2aExecutor.__new__(ConfiguredA2aExecutor)
    executor.langfuse_client = "lf-client"
    executor._constructor_sub_agent_registry = None
    executor._constructor_sub_agent_policy = None
    executor._constructor_sub_agent_resolver = None
    executor._constructor_sub_agent_instruction = None
    executor._config = {
        "adk_agent_builder": "dummy.module.build_agent",
        "sub_agent_tool": {"enabled": True, "use_global_registry": True},
    }

    ConfiguredA2aExecutor._build_agent_from_config(executor)

    sub_agent_tool_config = captured["sub_agent_tool_config"]
    assert isinstance(sub_agent_tool_config, SubAgentToolConfig)
    assert sub_agent_tool_config.registry is registry
    assert sub_agent_tool_config.registry is get_global_agent_registry()
    assert auto_registered == [registry]


def test_build_agent_from_config_requires_registry_when_sub_agent_tool_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _builder(*, sub_agent_tool_config=None):
        del sub_agent_tool_config
        return object()

    monkeypatch.setattr(
        configured_executor_module,
        "import_module",
        lambda _: types.SimpleNamespace(build_agent=_builder),
    )

    executor = ConfiguredA2aExecutor.__new__(ConfiguredA2aExecutor)
    executor.langfuse_client = "lf-client"
    executor._constructor_sub_agent_registry = None
    executor._constructor_sub_agent_policy = None
    executor._constructor_sub_agent_resolver = None
    executor._constructor_sub_agent_instruction = None
    executor._config = {
        "adk_agent_builder": "dummy.module.build_agent",
        "sub_agent_tool": {"enabled": True},
    }

    with pytest.raises(ValueError, match="use_global_registry"):
        ConfiguredA2aExecutor._build_agent_from_config(executor)
