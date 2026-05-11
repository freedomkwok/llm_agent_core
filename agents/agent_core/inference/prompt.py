"""Shared prompt helpers for agent instructions."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from llm_inference_core import ProjectContext, make_prompt_provider

from agents.agent_core.inference.settings import build_default_inference_settings

logger = logging.getLogger(__name__)


def default_instruction_prompt_name(agent_name: str) -> str:
    return f"agents/{agent_name}/instruction"


def load_agent_instruction(
    *,
    agent_name: str,
    project_name: str,
    fallback_instruction: str,
    project_metadata: Mapping[str, Any] | None = None,
    settings_overrides: Mapping[str, Any] | None = None,
    instruction_prompt_name: str | None = None,
    instruction_prompt_label: str | None = None,
) -> str:
    prompt_name = (instruction_prompt_name or default_instruction_prompt_name(agent_name)).strip()
    if not prompt_name:
        return fallback_instruction

    prompt_label = (
        instruction_prompt_label.strip() if isinstance(instruction_prompt_label, str) else None
    )
    if prompt_label == "":
        prompt_label = None

    try:
        settings = build_default_inference_settings(overrides=settings_overrides)
        project_context = ProjectContext(
            project_name=project_name,
            metadata=dict(project_metadata or {}),
        )
        prompt_provider = make_prompt_provider(settings=settings, project_context=project_context)
        prompt = prompt_provider.get(prompt_name, label=prompt_label)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Failed to load instruction prompt %s (label=%s); using fallback instruction.",
            prompt_name,
            prompt_label,
            exc_info=exc,
        )
        return fallback_instruction

    if isinstance(prompt, str):
        instruction = prompt.strip()
        if instruction:
            return instruction
    return fallback_instruction


__all__ = ["default_instruction_prompt_name", "load_agent_instruction"]
