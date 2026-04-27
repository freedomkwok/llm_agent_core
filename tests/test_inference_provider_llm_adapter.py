import asyncio

import pytest

pytest.importorskip("google.adk")

from google.adk.models.llm_request import LlmRequest
from google.genai import types
from pydantic import BaseModel

from agents.agent_core.inference_provider_llm_adapter import InferenceProviderLlmAdapter


class FakeInferResult(BaseModel):
    text: str


class FakeInferToolCallResult(BaseModel):
    text: str = ""
    tool_calls: list[dict] = []


class FakeProvider:
    def __init__(self) -> None:
        self.payloads: list[dict] = []

    async def infer(self, payload):
        self.payloads.append(dict(payload))
        return FakeInferResult(text="adapter output")


class FailingProvider:
    async def infer(self, payload):
        del payload
        raise RuntimeError("provider infer failed")


def _request_with_text(*, user_text: str, system_prompt: str | None = None) -> LlmRequest:
    request = LlmRequest(
        model="gpt-4.1-mini",
        contents=[types.Content(role="user", parts=[types.Part(text=user_text)])],
    )
    request.config.temperature = 0.2
    if system_prompt is not None:
        request.config.system_instruction = system_prompt
    return request


def _run_collect(adapter: InferenceProviderLlmAdapter, request: LlmRequest, *, stream: bool) -> list:
    async def run():
        responses = []
        async for response in adapter.generate_content_async(request, stream=stream):
            responses.append(response)
        return responses

    return asyncio.run(run())


def test_adapter_maps_request_and_returns_text_response() -> None:
    provider = FakeProvider()
    adapter = InferenceProviderLlmAdapter(model="gpt-4.1-mini", provider=provider)
    request = _request_with_text(
        user_text="route this request",
        system_prompt="You are a routing agent.",
    )

    responses = _run_collect(adapter, request, stream=False)

    assert len(responses) == 1
    assert responses[0].partial is False
    assert responses[0].turn_complete is True
    assert responses[0].content is not None
    assert responses[0].content.parts[0].text == "adapter output"

    payload = provider.payloads[0]
    assert payload["model_name"] == "gpt-4.1-mini"
    assert payload["common_input"] == [
        {
            "role": "system",
            "content": [{"type": "input_text", "text": "You are a routing agent."}],
        },
        {
            "role": "user",
            "content": [{"type": "input_text", "text": "route this request"}],
        },
    ]
    assert payload["model_parameters"]["temperature"] == 0.2


def test_adapter_stream_mode_emits_single_final_response() -> None:
    provider = FakeProvider()
    adapter = InferenceProviderLlmAdapter(model="gpt-4.1-mini", provider=provider)
    request = _request_with_text(user_text="stream this")

    responses = _run_collect(adapter, request, stream=True)

    assert len(responses) == 1
    assert responses[0].partial is False
    assert responses[0].turn_complete is True
    assert responses[0].content is not None
    assert responses[0].content.parts[0].text == "adapter output"


def test_adapter_maps_provider_error_to_llm_response() -> None:
    adapter = InferenceProviderLlmAdapter(model="gpt-4.1-mini", provider=FailingProvider())
    request = _request_with_text(user_text="trigger error")

    responses = _run_collect(adapter, request, stream=False)

    assert len(responses) == 1
    assert responses[0].error_code == "RuntimeError"
    assert responses[0].error_message == "provider infer failed"
    assert responses[0].turn_complete is True


def test_adapter_maps_provider_tool_calls_to_function_call_parts() -> None:
    class ToolCallProvider:
        async def infer(self, payload):
            del payload
            return FakeInferToolCallResult(
                text="",
                tool_calls=[
                    {
                        "id": "call_123",
                        "name": "search_nodes",
                        "arguments": {"query": "payments"},
                    }
                ],
            )

    adapter = InferenceProviderLlmAdapter(model="gpt-4.1-mini", provider=ToolCallProvider())
    request = _request_with_text(user_text="find payment skills")
    responses = _run_collect(adapter, request, stream=False)

    assert len(responses) == 1
    assert responses[0].content is not None
    parts = responses[0].content.parts
    assert len(parts) == 1
    assert parts[0].function_call is not None
    assert parts[0].function_call.name == "search_nodes"
    assert parts[0].function_call.id == "call_123"
    assert parts[0].function_call.args == {"query": "payments"}


def test_adapter_skips_malformed_tool_calls_and_normalizes_string_arguments() -> None:
    class ToolCallProvider:
        async def infer(self, payload):
            del payload
            return FakeInferToolCallResult(
                text="",
                tool_calls=[
                    {"id": "call_invalid_blank", "name": "   ", "arguments": {"query": "ignore"}},
                    {"id": "call_invalid_chars", "name": "bad name!", "arguments": {"query": "ignore"}},
                    {"id": "call_123", "name": "search_nodes", "arguments": '{"query":"payments"}'},
                ],
            )

    adapter = InferenceProviderLlmAdapter(model="gpt-4.1-mini", provider=ToolCallProvider())
    request = _request_with_text(user_text="find payment skills")
    responses = _run_collect(adapter, request, stream=False)

    assert len(responses) == 1
    assert responses[0].content is not None
    parts = responses[0].content.parts
    assert len(parts) == 1
    assert parts[0].function_call is not None
    assert parts[0].function_call.name == "search_nodes"
    assert parts[0].function_call.id == "call_123"
    assert parts[0].function_call.args == {"query": "payments"}


def test_adapter_normalizes_google_schema_type_enums_for_openai_tools() -> None:
    provider = FakeProvider()
    adapter = InferenceProviderLlmAdapter(model="gpt-4.1-mini", provider=provider)
    request = _request_with_text(user_text="find payment skills")
    request.config.tools = [
        types.Tool(
            function_declarations=[
                types.FunctionDeclaration(
                    name="search_nodes",
                    description="Search skills by query and limit",
                    parameters={
                        "type": "OBJECT",
                        "properties": {
                            "query": {"type": "STRING"},
                            "limit": {"type": "INTEGER"},
                        },
                        "required": ["query"],
                    },
                )
            ]
        )
    ]

    _run_collect(adapter, request, stream=False)

    payload = provider.payloads[0]
    assert payload["tools"][0]["parameters"]["type"] == "object"
    assert payload["tools"][0]["parameters"]["properties"]["query"]["type"] == "string"
    assert payload["tools"][0]["parameters"]["properties"]["limit"]["type"] == "integer"
    assert payload["tools"][0]["parameters"]["required"] == ["query"]


def test_adapter_includes_trace_ids_from_labels() -> None:
    provider = FakeProvider()
    adapter = InferenceProviderLlmAdapter(model="gpt-4.1-mini", provider=provider)
    request = _request_with_text(user_text="trace from labels")
    request.config.labels = {
        "trace_id": "trace_from_labels",
        "parent_observation_id": "parent_from_labels",
    }

    _run_collect(adapter, request, stream=False)

    payload = provider.payloads[0]
    assert payload["trace_id"] == "trace_from_labels"
    assert payload["parent_span_id"] == "parent_from_labels"
    assert payload["parent_observation_id"] == "parent_from_labels"


def test_adapter_omits_trace_payload_when_labels_missing() -> None:
    provider = FakeProvider()
    adapter = InferenceProviderLlmAdapter(model="gpt-4.1-mini", provider=provider)
    request = _request_with_text(user_text="trace missing labels")

    _run_collect(adapter, request, stream=False)

    payload = provider.payloads[0]
    assert "trace_id" not in payload
    assert "parent_span_id" not in payload
    assert "parent_observation_id" not in payload
