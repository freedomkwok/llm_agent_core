import asyncio

import pytest

pytest.importorskip("google.adk")

from google.adk.models.llm_request import LlmRequest
from google.genai import types
from pydantic import BaseModel

from agents.agent_core.inference_provider_llm_adapter import InferenceProviderLlmAdapter
from agents.zep_agent import a2a_agent_core
from agents.zep_agent.a2a_agent_core import build_zep_llm_agent


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


def test_resolve_instruction_uses_default_prompt_path_and_yaml_label(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested: dict[str, str | None] = {"name": None, "label": None}

    class _FakePromptProvider:
        def get(self, name: str, *, label: str | None = None) -> str:
            requested["name"] = name
            requested["label"] = label
            return "Instruction from prompt provider."

    monkeypatch.setattr(
        a2a_agent_core,
        "build_default_inference_settings",
        lambda overrides=None: object(),
    )
    monkeypatch.setattr(
        a2a_agent_core,
        "make_prompt_provider",
        lambda **_: _FakePromptProvider(),
    )

    instruction = a2a_agent_core._resolve_instruction(
        instruction_prompt_name=None,
        instruction_prompt_label="staging",
    )

    assert instruction == "Instruction from prompt provider."
    assert requested == {
        "name": "agents/zep_query_agent/instruction",
        "label": "staging",
    }


def test_resolve_instruction_falls_back_to_default_text_when_fetch_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FailingPromptProvider:
        def get(self, name: str, *, label: str | None = None) -> str:
            del name, label
            raise RuntimeError("prompt lookup failed")

    monkeypatch.setattr(
        a2a_agent_core,
        "build_default_inference_settings",
        lambda overrides=None: object(),
    )
    monkeypatch.setattr(
        a2a_agent_core,
        "make_prompt_provider",
        lambda **_: _FailingPromptProvider(),
    )

    instruction = a2a_agent_core._resolve_instruction(
        instruction_prompt_name="agents/custom/instruction",
        instruction_prompt_label="production",
    )

    assert instruction == a2a_agent_core._DEFAULT_INSTRUCTION
