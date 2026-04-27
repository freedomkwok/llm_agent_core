"""ADK BaseLlm adapter backed by llm_inference_core providers."""

from __future__ import annotations

import json
from enum import Enum
import re
from typing import Any, AsyncGenerator, Mapping

from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types
from pydantic import BaseModel, Field

from agents.agent_core.inference_provider import create_inference_provider
from llm_inference_core.providers import InferenceProvider

_MODEL_PARAM_NAMES = (
    "temperature",
    "top_p",
    "max_output_tokens",
    "presence_penalty",
    "frequency_penalty",
    "seed",
)
_JSON_SCHEMA_TYPE_MAP: dict[str, str] = {
    "OBJECT": "object",
    "ARRAY": "array",
    "STRING": "string",
    "NUMBER": "number",
    "INTEGER": "integer",
    "BOOLEAN": "boolean",
    "NULL": "null",
}


class InferenceProviderLlmAdapter(BaseLlm):
    """Expose InferenceProvider through ADK BaseLlm interface."""

    model: str = "gpt-4.1-mini"
    provider: InferenceProvider | None = Field(default=None, exclude=True, repr=False)
    langfuse_client: Any = Field(default=None, exclude=True, repr=False)
    project_name: str = "imp_agent_map.zep_agent"
    project_metadata: Mapping[str, Any] = Field(default_factory=dict)
    settings_overrides: Mapping[str, Any] = Field(default_factory=dict)

    def _ensure_provider(self) -> InferenceProvider:
        if self.provider is not None:
            return self.provider
        self.provider = create_inference_provider(
            langfuse_client=self.langfuse_client,
            project_name=self.project_name,
            project_metadata=self.project_metadata,
            settings_overrides=self.settings_overrides,
        )
        return self.provider

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        del stream
        provider = self._ensure_provider()
        payload = self._build_provider_payload(llm_request)
        try:
            parsed = await provider.infer(payload)
        except Exception as exc:  # noqa: BLE001
            yield LlmResponse(
                error_code=type(exc).__name__,
                error_message=str(exc),
                partial=False,
                turn_complete=True,
            )
            return

        response_text = self._parsed_output_text(parsed)
        function_call_parts = self._parsed_function_call_parts(parsed)
        content_parts: list[types.Part] = []
        if response_text:
            content_parts.append(types.Part(text=response_text))
        content_parts.extend(function_call_parts)
        if not content_parts:
            content_parts.append(
                types.Part(text=json.dumps(parsed.model_dump(mode="json"), ensure_ascii=True))
            )
        yield LlmResponse(
            content=types.Content(role="model", parts=content_parts),
            partial=False,
            turn_complete=True,
        )

    def _build_provider_payload(self, llm_request: LlmRequest) -> dict[str, Any]:
        common_input = self._common_input_from_contents(llm_request)
        instructions = self._system_prompt_text(llm_request)
        openai_tools = self._openai_tools_from_request(llm_request)
        trace_payload = self._preset_downstream_parent_span_payload(llm_request)
        metadata: dict[str, Any] = {}
        if openai_tools:
            metadata["tool_calling_mode"] = "enabled"
        payload: dict[str, Any] = {
            "trace_name": "adk_inference_provider_llm_adapter",
            "langfuse_type": "generation",
            "model_name": llm_request.model or self.model,
            "common_input": common_input,
            "model_parameters": self._model_parameters(llm_request),
            "metadata": metadata,
            **trace_payload,
        }
        if openai_tools:
            payload["tools"] = openai_tools
        if instructions:
            payload["instructions"] = instructions
        return payload

    @staticmethod
    def _preset_downstream_parent_span_payload(llm_request: LlmRequest) -> dict[str, str]:
        labels = getattr(llm_request.config, "labels", None)
        trace_id: str | None = None
        parent_span_id: str | None = None
        if isinstance(labels, Mapping):
            raw_tid = labels.get("trace_id")
            if isinstance(raw_tid, str) and raw_tid.strip():
                trace_id = raw_tid.strip()
            raw_pid = labels.get("parent_span_id") or labels.get("parent_observation_id")
            if isinstance(raw_pid, str) and raw_pid.strip():
                parent_span_id = raw_pid.strip()

        trace_payload: dict[str, str] = {}
        if trace_id:
            trace_payload["trace_id"] = trace_id
        if parent_span_id:
            trace_payload["parent_span_id"] = parent_span_id
            # Keep compatibility with payload readers still expecting old key.
            trace_payload["parent_observation_id"] = parent_span_id
        return trace_payload

    @staticmethod
    def _system_prompt_text(llm_request: LlmRequest) -> str | None:
        instruction = llm_request.config.system_instruction
        if isinstance(instruction, str):
            text = instruction.strip()
            return text or None
        if isinstance(instruction, types.Content):
            merged = "".join(part.text for part in instruction.parts or [] if part.text)
            merged = merged.strip()
            return merged or None
        return None

    @staticmethod
    def _common_input_from_contents(llm_request: LlmRequest) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for content in llm_request.contents:
            role = content.role or "user"
            message_parts: list[dict[str, Any]] = []
            for part in content.parts or []:
                if part.text:
                    text = part.text.strip()
                    if text:
                        message_parts.append({"type": "input_text", "text": text})
                    continue
                if part.function_call:
                    if message_parts:
                        items.append({"role": role, "content": message_parts})
                        message_parts = []
                    function_call = InferenceProviderLlmAdapter._function_call_input_item(
                        part.function_call
                    )
                    if function_call is not None:
                        items.append(function_call)
                    continue
                if part.function_response:
                    if message_parts:
                        items.append({"role": role, "content": message_parts})
                        message_parts = []
                    call_id = InferenceProviderLlmAdapter._resolved_call_id(
                        part.function_response.id,
                        part.function_response.name,
                    )
                    if call_id is None:
                        continue
                    items.append(
                        {
                            "type": "function_call_output",
                            "call_id": call_id,
                            "output": InferenceProviderLlmAdapter._function_call_output_items(
                                part.function_response.response
                            ),
                        }
                    )
            if message_parts:
                items.append({"role": role, "content": message_parts})
        return items

    @staticmethod
    def _resolved_call_id(raw_id: Any, raw_name: Any) -> str | None:
        if isinstance(raw_id, str):
            call_id = raw_id.strip()
            if call_id:
                return call_id
        if isinstance(raw_name, str):
            candidate = raw_name.strip()
            if candidate.startswith("call_"):
                return candidate
        return None

    @staticmethod
    def _function_call_input_item(function_call: Any) -> dict[str, Any] | None:
        name = getattr(function_call, "name", None)
        if not isinstance(name, str) or not name.strip():
            return None
        call_id = InferenceProviderLlmAdapter._resolved_call_id(
            getattr(function_call, "id", None),
            name,
        )
        if call_id is None:
            return None
        arguments = InferenceProviderLlmAdapter._normalized_function_call_arguments(
            getattr(function_call, "args", {})
        )
        return {
            "type": "function_call",
            "call_id": call_id,
            "name": name.strip(),
            "arguments": json.dumps(arguments, ensure_ascii=True),
        }

    @staticmethod
    def _normalized_function_call_arguments(raw_args: Any) -> dict[str, Any]:
        if hasattr(raw_args, "model_dump"):
            raw_args = raw_args.model_dump(mode="json")
        if isinstance(raw_args, Mapping):
            return dict(raw_args)
        if isinstance(raw_args, str):
            stripped = raw_args.strip()
            if not stripped:
                return {}
            try:
                loaded = json.loads(stripped)
            except json.JSONDecodeError:
                return {}
            if isinstance(loaded, dict):
                return loaded
        return {}

    @staticmethod
    def _function_call_output_items(response_payload: Any) -> list[dict[str, Any]]:
        if isinstance(response_payload, list):
            normalized_items: list[dict[str, Any]] = []
            for item in response_payload:
                if not isinstance(item, dict):
                    continue
                item_type = item.get("type")
                if isinstance(item_type, str) and item_type.strip():
                    normalized_items.append(dict(item))
            if normalized_items:
                return normalized_items

        if isinstance(response_payload, str):
            text = response_payload
        else:
            text = json.dumps(response_payload, ensure_ascii=True)
        return [{"type": "output_text", "text": text}]

    @staticmethod
    def _openai_tools_from_request(llm_request: LlmRequest) -> list[dict[str, Any]]:
        tools: list[dict[str, Any]] = []
        for tool in llm_request.config.tools or []:
            for declaration in tool.function_declarations or []:
                parameters = declaration.parameters
                if hasattr(parameters, "model_dump"):
                    parameters = parameters.model_dump(by_alias=True, exclude_none=True)
                if isinstance(parameters, Mapping):
                    parameters = InferenceProviderLlmAdapter._normalize_openai_schema_types(
                        dict(parameters)
                    )
                else:
                    parameters = {}
                tools.append(
                    {
                        "type": "function",
                        "name": declaration.name,
                        "description": declaration.description or "",
                        "parameters": parameters or {"type": "object", "properties": {}},
                    }
                )
        return tools

    @staticmethod
    def _normalize_openai_schema_types(
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        normalized: dict[str, Any] = {}
        for key, raw_value in payload.items():
            normalized[key] = InferenceProviderLlmAdapter._normalize_openai_schema_value(
                raw_value,
                current_key=key,
            )
        return normalized

    @staticmethod
    def _normalize_openai_schema_value(
        value: Any,
        *,
        current_key: str | None = None,
    ) -> Any:
        if isinstance(value, Enum):
            value = value.value

        if isinstance(value, Mapping):
            return InferenceProviderLlmAdapter._normalize_openai_schema_types(dict(value))

        if isinstance(value, list):
            return [
                InferenceProviderLlmAdapter._normalize_openai_schema_value(
                    item,
                    current_key=current_key,
                )
                for item in value
            ]

        if current_key == "type" and isinstance(value, str):
            return _JSON_SCHEMA_TYPE_MAP.get(value, value.lower())
        return value

    @staticmethod
    def _model_parameters(llm_request: LlmRequest) -> dict[str, Any]:
        params: dict[str, Any] = {}
        for key in _MODEL_PARAM_NAMES:
            value = getattr(llm_request.config, key, None)
            if value is not None:
                params[key] = value
        return params

    @staticmethod
    def _parsed_output_text(parsed: BaseModel) -> str | None:
        text = getattr(parsed, "text", None)
        if isinstance(text, str):
            stripped = text.strip()
            if stripped:
                return stripped
        return None

    @staticmethod
    def _parsed_function_call_parts(parsed: BaseModel) -> list[types.Part]:
        dumped = parsed.model_dump(mode="json")
        tool_calls = dumped.get("tool_calls", []) if isinstance(dumped, dict) else []
        if not isinstance(tool_calls, list):
            return []
        parts: list[types.Part] = []
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue
            name = tool_call.get("name")
            if not isinstance(name, str):
                continue
            normalized_name = name.strip()
            if not normalized_name:
                continue
            if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", normalized_name):
                continue
            raw_args = tool_call.get("arguments", {})
            if isinstance(raw_args, str):
                stripped_args = raw_args.strip()
                if stripped_args:
                    try:
                        loaded = json.loads(stripped_args)
                    except json.JSONDecodeError:
                        loaded = {}
                    args = loaded if isinstance(loaded, dict) else {}
                else:
                    args = {}
            else:
                args = raw_args if isinstance(raw_args, dict) else {}
            call_id = tool_call.get("id")
            function_call = types.FunctionCall(
                name=normalized_name,
                args=args,
            )
            if isinstance(call_id, str) and call_id:
                function_call.id = call_id
            parts.append(types.Part(function_call=function_call))
        return parts
