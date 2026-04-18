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


def build_default_inference_settings(
    *,
    overrides: Mapping[str, Any] | None = None,
) -> InferenceCoreSettings:
    """Build baseline inference settings from env with optional overrides."""
    settings_data: dict[str, Any] = {
        "_env_file": None,
        "openai_api_key": os.getenv("OPENAI_API_KEY", ""),
        "openai_model": os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        "openai_base_url": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        "inference_provider": os.getenv("INFERENCE_PROVIDER", "openai"),
        "langfuse_public_key": os.getenv("LANGFUSE_PUBLIC_KEY", ""),
        "langfuse_secret_key": os.getenv("LANGFUSE_SECRET_KEY", ""),
        "langfuse_base_url": os.getenv("LANGFUSE_BASE_URL", "http://localhost:3000"),
        "prompt_base_dir": os.getenv("PROMPT_BASE_DIR", "app/prompts"),
        "prompt_label": os.getenv("PROMPT_LABEL", "production"),
        "prompt_cache_ttl_seconds": os.getenv("PROMPT_CACHE_TTL_SECONDS", 60),
        "prompt_project_miss_ttl_seconds": os.getenv("PROMPT_PROJECT_MISS_TTL_SECONDS", 1800),
        "prompt_tag_source": os.getenv("PROMPT_TAG_SOURCE", "local"),
        "prompt_tag_file": os.getenv("PROMPT_TAG_FILE", "local_prompt_tags.json"),
        "prompt_backend": os.getenv("PROMPT_BACKEND", "file"),
        "example_capture_inputs": os.getenv("EXAMPLE_CAPTURE_INPUTS", False),
    }
    if overrides:
        settings_data.update(dict(overrides))
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
        settings.inference_provider,
        settings=settings,
        project_context=project_context,
        langfuse=langfuse_client,
    )
