# SPDX-License-Identifier: Apache-2.0
import asyncio

import pytest

pytest.importorskip("google.adk")

from google.adk.models.llm_request import LlmRequest
from google.genai import types
from pydantic import BaseModel

import agents.agent_core.inference.prompt as prompt_module
import agents.zep_agent.a2a_agent_core as zep_agent_core
from agents.agent_core import DynamicAgentRegistry, SubAgentToolConfig
from agents.agent_core.adk.executor import ConfiguredA2aExecutor
from agents.agent_core.inference.llm_adapter import InferenceProviderLlmAdapter
from agents.zep_agent.a2a_agent_core import build_zep_llm_agent
from agents.zep_agent.registry import config_path


class _FakeZepResult(BaseModel):
    text: str


class _FakeZepProvider:
    async def infer(self, payload):
        del payload
        return _FakeZepResult(text="zep adapter response")


def test_build_zep_llm_agent_wires_inference_provider_adapter() -> None:
    agent = build_zep_llm_agent()

    assert isinstance(agent.model, InferenceProviderLlmAdapter)
    adapter = agent.model
    adapter.provider = _FakeZepProvider()
    request = LlmRequest(
        model=adapter.model,
        contents=[types.Content(role="user", parts=[types.Part(text="route with zep")])],
    )

    async def run():
        responses = []
        async for response in adapter.generate_content_async(request, stream=False):
            responses.append(response)
        return responses

    responses = asyncio.run(run())
    assert len(responses) == 1
    assert responses[0].content is not None
    assert responses[0].content.parts[0].text == "zep adapter response"


def test_build_zep_llm_agent_accepts_sub_agent_tool_config_instruction_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        zep_agent_core,
        "load_agent_instruction",
        lambda **kwargs: kwargs["fallback_instruction"],
    )

    agent = build_zep_llm_agent(
        fallback_instruction="Base instruction.",
        sub_agent_tool_config=SubAgentToolConfig(
            registry=DynamicAgentRegistry(),
            instruction="\nCustom delegation guidance.",
        ),
    )

    assert "invoke_sub_agent" in {
        getattr(tool, "__name__", getattr(tool, "name", "")) for tool in agent.tools
    }
    assert agent.instruction == "Base instruction.\nCustom delegation guidance."


def test_executor_config_adds_sub_agent_tool() -> None:
    executor = ConfiguredA2aExecutor.__new__(ConfiguredA2aExecutor)
    executor.langfuse_client = None
    executor._constructor_sub_agent_registry = DynamicAgentRegistry()
    executor._constructor_sub_agent_policy = None
    executor._constructor_sub_agent_resolver = None
    executor._constructor_sub_agent_instruction = "\nParent can delegate."
    executor._config = ConfiguredA2aExecutor._load_executor_config(
        config_path,
        config_section="executor_config",
    )

    agent = ConfiguredA2aExecutor._build_agent_from_config(executor)

    assert "invoke_sub_agent" in {
        getattr(tool, "__name__", getattr(tool, "name", "")) for tool in agent.tools
    }
    assert agent.instruction.endswith("\nParent can delegate.")


def test_load_agent_instruction_uses_default_prompt_path_and_yaml_label(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested: dict[str, str | None] = {"name": None, "label": None}

    class _FakePromptProvider:
        def get(self, name: str, *, label: str | None = None) -> str:
            requested["name"] = name
            requested["label"] = label
            return "Instruction from prompt provider."

    monkeypatch.setattr(
        prompt_module,
        "build_default_inference_settings",
        lambda overrides=None: object(),
    )
    monkeypatch.setattr(
        prompt_module,
        "make_prompt_provider",
        lambda **_: _FakePromptProvider(),
    )

    instruction = prompt_module.load_agent_instruction(
        agent_name="zep_query_agent",
        project_name="imp_agent_map.zep_agent",
        project_metadata={"component": "zep_agent"},
        settings_overrides={"conversation_store_type": "lru"},
        fallback_instruction="Fallback instruction.",
        instruction_prompt_name=None,
        instruction_prompt_label="staging",
    )

    assert instruction == "Instruction from prompt provider."
    assert requested == {
        "name": "agents/zep_query_agent/instruction",
        "label": "staging",
    }


def test_load_agent_instruction_falls_back_to_default_text_when_fetch_fails(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    class _FailingPromptProvider:
        def get(self, name: str, *, label: str | None = None) -> str:
            del name, label
            raise RuntimeError("prompt lookup failed")

    monkeypatch.setattr(
        prompt_module,
        "build_default_inference_settings",
        lambda overrides=None: object(),
    )
    monkeypatch.setattr(
        prompt_module,
        "make_prompt_provider",
        lambda **_: _FailingPromptProvider(),
    )

    with caplog.at_level("WARNING", logger=prompt_module.__name__):
        instruction = prompt_module.load_agent_instruction(
            agent_name="zep_query_agent",
            project_name="imp_agent_map.zep_agent",
            fallback_instruction="Fallback instruction.",
            instruction_prompt_name="agents/custom/instruction",
            instruction_prompt_label="production",
        )

    assert instruction == "Fallback instruction."
    assert "Failed to load instruction prompt agents/custom/instruction" in caplog.text
    assert "prompt lookup failed" in caplog.text
