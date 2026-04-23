import asyncio

import pytest

pytest.importorskip("google.adk")

from google.adk.models.llm_request import LlmRequest
from google.genai import types
from pydantic import BaseModel

from agents.agent_core.inference_provider_llm_adapter import InferenceProviderLlmAdapter
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
