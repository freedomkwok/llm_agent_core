# SPDX-License-Identifier: Apache-2.0
from agents.agent_core.inference.settings import build_default_inference_settings


def test_prompt_template_path_is_prompt_base_dir_alias(monkeypatch) -> None:
    monkeypatch.delenv("PROMPT_BASE_DIR", raising=False)
    monkeypatch.setenv("PROMPT_TEMPLATE_PATH", "prompts")

    settings = build_default_inference_settings()

    assert settings.prompt_label_config.prompt_base_dir == "prompts"


def test_prompt_base_dir_takes_precedence_over_prompt_template_path(monkeypatch) -> None:
    monkeypatch.setenv("PROMPT_BASE_DIR", "custom/prompts")
    monkeypatch.setenv("PROMPT_TEMPLATE_PATH", "prompts")

    settings = build_default_inference_settings()

    assert settings.prompt_label_config.prompt_base_dir == "custom/prompts"
