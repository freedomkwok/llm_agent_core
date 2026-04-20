"""Shared helpers for constructing llm_inference_core providers."""

from __future__ import annotations

import os
from typing import Any, Mapping

from llm_inference_core import (
    InferenceCoreSettings,
    InferenceProviderFactory,
    ProjectContext,
)
from llm_inference_core.providers import InferenceProvider

_LEGACY_OVERRIDE_MAP: dict[str, tuple[str, str]] = {
    "openai_api_key": ("inference_config", "openai_api_key"),
    "openai_model": ("inference_config", "openai_model"),
    "openai_base_url": ("inference_config", "openai_base_url"),
    "inference_provider": ("inference_config", "inference_provider"),
    "langfuse_public_key": ("langfuse_config", "langfuse_public_key"),
    "langfuse_secret_key": ("langfuse_config", "langfuse_secret_key"),
    "langfuse_base_url": ("langfuse_config", "langfuse_base_url"),
    "prompt_base_dir": ("prompt_label_config", "prompt_base_dir"),
    "prompt_label": ("prompt_label_config", "prompt_label"),
    "prompt_cache_ttl_seconds": ("prompt_label_config", "prompt_cache_ttl_seconds"),
    "prompt_project_miss_ttl_seconds": ("prompt_label_config", "prompt_project_miss_ttl_seconds"),
    "prompt_tag_source": ("prompt_label_config", "prompt_tag_source"),
    "prompt_tag_file": ("prompt_label_config", "prompt_tag_file"),
    "prompt_backend": ("prompt_label_config", "prompt_backend"),
    "example_capture_inputs": ("log_config", "example_capture_inputs"),
    "capture_conversation_examples_path": ("log_config", "capture_conversation_examples_path"),
    "conversation_store_type": ("conversation_store_config", "type"),
    "conversation_store_api_key": ("conversation_store_config", "api_key"),
    "conversation_store_thread_id": ("conversation_store_config", "thread_id"),
    "conversation_store_max_conversations": ("conversation_store_config", "max_conversations"),
    "conversation_store_max_messages_per_conversation": (
        "conversation_store_config",
        "max_messages_per_conversation",
    ),
}


def _parse_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
    return default


def _parse_int(value: Any, default: int) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if stripped:
            try:
                return int(stripped)
            except ValueError:
                return default
    return default


def _apply_overrides(settings_data: dict[str, Any], overrides: Mapping[str, Any]) -> None:
    for key, value in overrides.items():
        if key in _LEGACY_OVERRIDE_MAP:
            section, field = _LEGACY_OVERRIDE_MAP[key]
            section_payload = settings_data.setdefault(section, {})
            if isinstance(section_payload, dict):
                section_payload[field] = value
            continue

        current_section = settings_data.get(key)
        if isinstance(current_section, dict) and isinstance(value, Mapping):
            current_section.update(dict(value))
            continue

        settings_data[key] = value


def build_default_inference_settings(
    *,
    overrides: Mapping[str, Any] | None = None,
) -> InferenceCoreSettings:
    """Build baseline inference settings from env with optional overrides."""
    example_capture_inputs = _parse_bool(
        os.getenv("EXAMPLE_CAPTURE_INPUTS"),
        False,
    )
    settings_data: dict[str, Any] = {
        "_env_file": None,
        "inference_config": {
            "openai_api_key": os.getenv("OPENAI_API_KEY", ""),
            "openai_model": os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
            "openai_base_url": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            "inference_provider": os.getenv("INFERENCE_PROVIDER", "openai"),
        },
        "langfuse_config": {
            "langfuse_public_key": os.getenv("LANGFUSE_PUBLIC_KEY", ""),
            "langfuse_secret_key": os.getenv("LANGFUSE_SECRET_KEY", ""),
            "langfuse_base_url": os.getenv("LANGFUSE_BASE_URL", "http://localhost:3000"),
        },
        "prompt_label_config": {
            "prompt_base_dir": os.getenv("PROMPT_BASE_DIR", "app/prompts"),
            "prompt_label": os.getenv("PROMPT_LABEL", "production"),
            "prompt_cache_ttl_seconds": _parse_int(os.getenv("PROMPT_CACHE_TTL_SECONDS"), 60),
            "prompt_project_miss_ttl_seconds": _parse_int(
                os.getenv("PROMPT_PROJECT_MISS_TTL_SECONDS"),
                1800,
            ),
            "prompt_tag_source": os.getenv("PROMPT_TAG_SOURCE", "local"),
            "prompt_tag_file": os.getenv("PROMPT_TAG_FILE", "local_prompt_tags.json"),
            "prompt_backend": os.getenv("PROMPT_BACKEND", "file"),
        },
        "log_config": {
            "example_capture_inputs": example_capture_inputs,
            "capture_conversation_examples_path": os.getenv(
                "CAPTURE_CONVERSATION_EXAMPLES_PATH",
                "examples/conversations",
            ),
        },
        "conversation_store_config": {
            "type": os.getenv("CONVERSATION_STORE_TYPE", "lru"),
            "api_key": os.getenv("CONVERSATION_STORE_API_KEY", ""),
            "thread_id": os.getenv("CONVERSATION_STORE_THREAD_ID", ""),
            "max_conversations": _parse_int(
                os.getenv("CONVERSATION_STORE_MAX_CONVERSATIONS"),
                256,
            ),
            "max_messages_per_conversation": _parse_int(
                os.getenv("CONVERSATION_STORE_MAX_MESSAGES_PER_CONVERSATION"),
                50,
            ),
        },
    }
    if overrides:
        _apply_overrides(settings_data, overrides)
    return InferenceCoreSettings(**settings_data)


def create_inference_provider(
    *,
    langfuse_client: Any,
    project_name: str,
    project_metadata: Mapping[str, Any] | None = None,
    settings_overrides: Mapping[str, Any] | None = None,
) -> InferenceProvider:
    """Create an inference provider using shared defaults plus overrides."""
    settings = build_default_inference_settings(overrides=settings_overrides)
    project_context = ProjectContext(
        project_name=project_name,
        metadata=dict(project_metadata or {}),
    )
    return InferenceProviderFactory.create(
        settings.inference_config.inference_provider,
        settings=settings,
        project_context=project_context,
        langfuse=langfuse_client,
    )
