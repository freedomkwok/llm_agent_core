from imp_agent_core.agents.agent_core.inference.settings import build_default_inference_settings


def test_build_default_inference_settings_uses_llm_base_url(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setenv("LLM_BASE_URL", "https://llm-gateway.test/v1")

    settings = build_default_inference_settings()

    assert settings.inference_config.openai_base_url == "https://llm-gateway.test/v1"


def test_build_default_inference_settings_prefers_openai_base_url(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_BASE_URL", "https://openai-compatible.test/v1")
    monkeypatch.setenv("LLM_BASE_URL", "https://llm-gateway.test/v1")

    settings = build_default_inference_settings()

    assert settings.inference_config.openai_base_url == "https://openai-compatible.test/v1"
# SPDX-License-Identifier: Apache-2.0


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
