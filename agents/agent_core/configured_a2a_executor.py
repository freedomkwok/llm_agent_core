"""Config-driven A2A executor base."""

from __future__ import annotations

from inspect import getfile
from inspect import signature
from importlib import import_module
from pathlib import Path
from typing import Any

import yaml
from google.adk.agents.base_agent import BaseAgent
from langfuse import get_client

from agents.agent_core.adk_a2a_execution_wrapper import AdkA2aExecutionWrapper


class ConfiguredA2aExecutor(AdkA2aExecutionWrapper):
    """Shared executor that loads agent wiring from a YAML config file."""

    def __init__(
        self,
        *,
        config_path: str | Path | None = None,
        config_section: str = "executor_config",
        adk_agent: BaseAgent | None = None,
        langfuse_client: Any | None = None,
    ):
        self._config = self._load_config(
            self._resolve_config_path(config_path), config_section=config_section
        )
        self.langfuse_client = langfuse_client or get_client()
        self.adk_agent = adk_agent or self._build_agent_from_config()
        super().__init__(langfuse_client=self.langfuse_client)

    def _resolve_config_path(self, config_path: str | Path | None) -> Path:
        if config_path is not None:
            return Path(config_path)
        default_path = Path(getfile(self.__class__)).with_name("config.yaml")
        return default_path

    @staticmethod
    def _load_config(config_path: str | Path, *, config_section: str) -> dict[str, Any]:
        path = Path(config_path)
        with path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"Executor config must be a mapping: {path}")
        section = raw.get(config_section)
        if section is None:
            return raw
        if not isinstance(section, dict):
            raise ValueError(f"{config_section} must be a mapping in {path}")
        return section

    def _build_agent_from_config(self) -> BaseAgent:
        builder_path = str(self._config.get("adk_agent_builder") or "").strip()
        if not builder_path or "." not in builder_path:
            raise ValueError("adk_agent_builder must be module_path.function_name")
        module_path, func_name = builder_path.rsplit(".", 1)
        module = import_module(module_path)
        builder = getattr(module, func_name)
        builder_signature = signature(builder)
        if "langfuse_client" in builder_signature.parameters:
            agent = builder(langfuse_client=self.langfuse_client)
        else:
            agent = builder()
        if not isinstance(agent, BaseAgent):
            raise TypeError(f"{builder_path} must return BaseAgent")
        return agent

    def build_adk_agent(self) -> BaseAgent:
        return self.adk_agent

    @property
    def trace_name(self) -> str:
        return str(self._config.get("trace_name") or "a2a_executor_execute")

    @property
    def artifact_name(self) -> str:
        return str(self._config.get("artifact_name") or "result")

    @property
    def failed_text_message(self) -> str:
        return str(
            self._config.get("failed_text_message")
            or "Failed to generate a final response with text content."
        )

