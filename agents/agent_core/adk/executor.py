# SPDX-License-Identifier: Apache-2.0
"""ADK-backed A2A task executors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from importlib import import_module
from inspect import getfile, signature
from pathlib import Path
from typing import Any

import yaml
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import TaskState, TextPart, UnsupportedOperationError
from a2a.utils import new_agent_text_message
from a2a.utils.errors import ServerError
from google.adk import Runner
from google.adk.agents.base_agent import BaseAgent
from google.adk.artifacts import InMemoryArtifactService
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.sessions import InMemorySessionService
from google.genai import types
from langfuse import get_client
from llm_inference_core import run_with_langfuse_trace

from agents.agent_core.inference import load_agent_instruction
from agents.agent_core.routing import (
    AgentResolver,
    DynamicAgentRegistry,
    get_global_agent_registry,
    register_agent_package,
)
from agents.agent_core.sub_agent_invoke import (
    DEFAULT_SUB_AGENT_TOOL_INSTRUCTION,
    SubAgentInvocationPolicy,
    SubAgentToolConfig,
)

_AGENT_CORE_STATE_PREFIX = "agent_core."
_SUBQUERY_DEPTH_KEY = f"{_AGENT_CORE_STATE_PREFIX}subquery_depth"


class AdkA2aExecutor(AgentExecutor, ABC):
    """Base A2A executor for agents that run through ADK Runner."""

    def __init__(self, *, langfuse_client: Any, adk_agent: BaseAgent | None = None):
        self.langfuse_client = langfuse_client
        self.adk_agent = adk_agent or self.build_adk_agent()
        self.runner: Runner | None = None

    @abstractmethod
    def build_adk_agent(self) -> BaseAgent:
        """Return the ADK BaseAgent instance to run."""

    @property
    def trace_name(self) -> str:
        return "a2a_executor_execute"

    @property
    def artifact_name(self) -> str:
        return "result"

    @property
    def failed_text_message(self) -> str:
        return "Failed to generate a final response with text content."

    def build_trace_input(self, *, request_text: str, context: RequestContext) -> dict[str, Any]:
        return {"request": request_text, "task_id": context.task_id}

    def build_trace_metadata(self, *, context: RequestContext) -> dict[str, Any]:
        return {"context_id": context.context_id}

    def build_engine_metadata(self, *, context: RequestContext, user_id: str) -> dict[str, Any]:
        return {
            "context_id": context.context_id,
            "task_id": context.task_id,
            "user_id": user_id,
        }

    @staticmethod
    def build_agent_state_delta(
        *,
        context: RequestContext,
        user_id: str,
        trace_id: str | None = None,
        parent_span_id: str | None = None,
        incoming_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build session state shared with ADK tools during this A2A invocation."""
        metadata = incoming_metadata or {}
        state_delta: dict[str, Any] = {
            "agent_core.user_id": user_id,
            "agent_core.a2a_context_id": context.context_id,
            "agent_core.a2a_task_id": context.task_id,
            _SUBQUERY_DEPTH_KEY: 0,
        }
        if trace_id:
            state_delta["agent_core.trace_id"] = trace_id
        if parent_span_id:
            state_delta["agent_core.parent_span_id"] = parent_span_id

        raw_depth = metadata.get(_SUBQUERY_DEPTH_KEY)
        if raw_depth is not None:
            try:
                state_delta[_SUBQUERY_DEPTH_KEY] = max(0, int(raw_depth))
            except (TypeError, ValueError):
                state_delta[_SUBQUERY_DEPTH_KEY] = 0

        if "graph_id" in metadata:
            raw_graph_id = metadata.get("graph_id")
            state_delta["graph_id"] = str(raw_graph_id).strip() if raw_graph_id is not None else ""
        return state_delta

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        del context, event_queue
        raise ServerError(error=UnsupportedOperationError())

    def _init_adk(self) -> None:
        if self.runner is None:
            self.runner = Runner(
                app_name=self.adk_agent.name,
                agent=self.adk_agent,
                artifact_service=InMemoryArtifactService(),
                session_service=InMemorySessionService(),
                memory_service=InMemoryMemoryService(),
            )

    async def _run_runner_once(
        self,
        *,
        user_id: str,
        context: RequestContext,
        request_text: str,
        state_delta: dict[str, Any] | None = None,
    ) -> str:
        if self.runner is None:
            return ""

        session = await self.runner.session_service.get_session(
            app_name=self.runner.app_name,
            user_id=user_id,
            session_id=context.context_id,
        )
        if session is None:
            session = await self.runner.session_service.create_session(
                app_name=self.runner.app_name,
                user_id=user_id,
                session_id=context.context_id,
            )

        content = types.Content(role="user", parts=[types.Part(text=request_text)])
        final_event = None
        async for event in self.runner.run_async(
            session_id=session.id,
            user_id=user_id,
            new_message=content,
            state_delta=state_delta,
        ):
            if event.is_final_response():
                final_event = event

        if final_event and final_event.content and final_event.content.parts:
            return "".join(
                part.text
                for part in final_event.content.parts
                if hasattr(part, "text") and part.text
            )
        return ""

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        if not context.message:
            return

        self._init_adk()
        if self.runner is None:
            return

        incoming_metadata = (
            context.message.metadata if isinstance(context.message.metadata, dict) else {}
        )
        trace_id: str | None = None
        parent_span_id: str | None = None
        if incoming_metadata:
            raw_tid = incoming_metadata.get("trace_id")
            if isinstance(raw_tid, str) and raw_tid.strip():
                trace_id = raw_tid.strip()
            raw_pid = incoming_metadata.get("parent_span_id") or incoming_metadata.get(
                "parent_observation_id"
            )
            if isinstance(raw_pid, str) and raw_pid.strip():
                parent_span_id = raw_pid.strip()
        raw_user_id = incoming_metadata.get("user_id") if incoming_metadata else None
        user_id = str(raw_user_id).strip() if raw_user_id is not None else ""
        if not user_id:
            user_id = "a2a_user"
        agent_state_delta = self.build_agent_state_delta(
            context=context,
            user_id=user_id,
            trace_id=trace_id,
            parent_span_id=parent_span_id,
            incoming_metadata=incoming_metadata,
        )

        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        if not context.current_task:
            await updater.submit()
        await updater.start_work()

        request_text = context.get_user_input()
        try:

            async def _run_chain() -> str:
                return await self._run_runner_once(
                    user_id=user_id,
                    context=context,
                    request_text=request_text,
                    state_delta=agent_state_delta,
                )

            def _on_chain_success(result: Any, observation: Any) -> None:
                status = "completed" if isinstance(result, str) and result else "failed"
                observation.update(output={"status": status, "task_id": context.task_id})

            response_text = await run_with_langfuse_trace(
                langfuse=self.langfuse_client,
                langfuse_type="chain",
                trace_name=self.trace_name,
                trace_input=self.build_trace_input(request_text=request_text, context=context),
                runner=_run_chain,
                on_success=_on_chain_success,
                metadata=self.build_trace_metadata(context=context),
                trace_id=trace_id,
                parent_span_id=parent_span_id,
                flush_on_exit=True,
            )
            if response_text:
                await updater.add_artifact([TextPart(text=response_text)], name=self.artifact_name)
                await updater.complete()
                return

            await updater.update_status(
                TaskState.failed,
                message=new_agent_text_message(self.failed_text_message),
                final=True,
            )
        except Exception as exc:  # noqa: BLE001
            await updater.update_status(
                TaskState.failed,
                message=new_agent_text_message(f"An error occurred: {str(exc)}"),
                final=True,
            )


class ConfiguredA2aExecutor(AdkA2aExecutor):
    """Shared executor that loads agent wiring from a YAML config file."""

    def __init__(
        self,
        *,
        config_path: str | Path | None = None,
        config_section: str = "executor_config",
        adk_agent: BaseAgent | None = None,
        langfuse_client: Any | None = None,
        sub_agent_registry: DynamicAgentRegistry | None = None,
        sub_agent_policy: SubAgentInvocationPolicy | None = None,
        sub_agent_resolver: AgentResolver | None = None,
        sub_agent_instruction: str | None = None,
    ):
        self._config = self._load_executor_config(
            self._config_path(config_path),
            config_section=config_section,
        )
        self._constructor_sub_agent_registry = sub_agent_registry
        self._constructor_sub_agent_policy = sub_agent_policy
        self._constructor_sub_agent_resolver = sub_agent_resolver
        self._constructor_sub_agent_instruction = sub_agent_instruction
        super().__init__(
            langfuse_client=langfuse_client or get_client(),
            adk_agent=adk_agent,
        )

    def _config_path(self, config_path: str | Path | None) -> Path:
        if config_path is not None:
            return Path(config_path)
        return Path(getfile(self.__class__)).with_name("config.yaml")

    @staticmethod
    def _load_executor_config(config_path: str | Path, *, config_section: str) -> dict[str, Any]:
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

    def build_adk_agent(self) -> BaseAgent:
        return self._build_agent_from_config()

    def _build_agent_from_config(self) -> BaseAgent:
        builder_path, builder = self._adk_agent_builder()
        agent = builder(**self._builder_kwargs(builder))
        if not isinstance(agent, BaseAgent):
            raise TypeError(f"{builder_path} must return BaseAgent")
        return agent

    def _adk_agent_builder(self) -> tuple[str, Callable[..., Any]]:
        builder_path = str(self._config.get("adk_agent_builder") or "").strip()
        if not builder_path or "." not in builder_path:
            raise ValueError("adk_agent_builder must be module_path.function_name")
        builder = self._import_callable(builder_path, config_key="adk_agent_builder")
        if not callable(builder):
            raise TypeError(f"{builder_path} must be callable")
        return builder_path, builder

    def _builder_kwargs(self, builder: Callable[..., Any]) -> dict[str, Any]:
        builder_signature = signature(builder)
        builder_kwargs: dict[str, Any] = {}
        if "langfuse_client" in builder_signature.parameters:
            builder_kwargs["langfuse_client"] = self.langfuse_client
        for key in ("instruction_prompt_name", "instruction_prompt_label", "fallback_instruction"):
            if key not in builder_signature.parameters:
                continue
            raw_value = self._config.get(key)
            if not isinstance(raw_value, str):
                continue
            value = raw_value.strip()
            if value:
                builder_kwargs[key] = value
        if "sub_agent_tool_config" in builder_signature.parameters:
            sub_agent_tool_config = self._sub_agent_tool_config()
            if sub_agent_tool_config is not None:
                builder_kwargs["sub_agent_tool_config"] = sub_agent_tool_config
        return builder_kwargs

    def _sub_agent_tool_config(self) -> SubAgentToolConfig | None:
        raw_config = self._config.get("sub_agent_tool")
        tool_config = raw_config if isinstance(raw_config, dict) else {}
        has_constructor_override = any(
            value is not None
            for value in (
                self._constructor_sub_agent_registry,
                self._constructor_sub_agent_policy,
                self._constructor_sub_agent_resolver,
                self._constructor_sub_agent_instruction,
            )
        )
        enabled = bool(tool_config.get("enabled")) if tool_config else False
        if not enabled and not has_constructor_override:
            return None

        registry = self._constructor_sub_agent_registry or self._sub_agent_registry(tool_config)
        if registry is None:
            raise ValueError(
                "sub_agent_tool.enabled requires sub_agent_registry, use_global_registry, "
                "or registry_builder"
            )
        policy = (
            self._constructor_sub_agent_policy
            if self._constructor_sub_agent_policy is not None
            else self._sub_agent_policy(tool_config)
        )
        instruction = (
            self._constructor_sub_agent_instruction
            if self._constructor_sub_agent_instruction is not None
            else self._sub_agent_instruction(tool_config)
        )

        return SubAgentToolConfig(
            registry=registry,
            policy=policy,
            resolver=self._constructor_sub_agent_resolver,
            instruction=instruction,
        )

    def _sub_agent_registry(self, tool_config: dict[str, Any]) -> DynamicAgentRegistry | None:
        if bool(tool_config.get("use_global_registry")):
            registry = get_global_agent_registry()
            register_agent_package(registry)
            return registry
        builder_path = str(tool_config.get("registry_builder") or "").strip()
        if not builder_path:
            return None
        builder = self._import_callable(builder_path, config_key="registry_builder")
        registry = builder()
        if not isinstance(registry, DynamicAgentRegistry):
            raise TypeError(f"{builder_path} must return DynamicAgentRegistry")
        return registry

    def _sub_agent_policy(self, tool_config: dict[str, Any]) -> SubAgentInvocationPolicy | None:
        raw_policy = tool_config.get("policy")
        policy_config = raw_policy if isinstance(raw_policy, dict) else tool_config
        policy_kwargs: dict[str, Any] = {}
        for key in ("allowed_agent_ids", "allowed_skill_ids", "forwarded_state_keys"):
            raw_value = policy_config.get(key)
            if isinstance(raw_value, (list, tuple)):
                policy_kwargs[key] = tuple(str(item) for item in raw_value if item)
        if "max_depth" in policy_config:
            policy_kwargs["max_depth"] = int(policy_config["max_depth"])
        for key in ("metadata_aliases", "static_metadata"):
            raw_value = policy_config.get(key)
            if isinstance(raw_value, dict):
                policy_kwargs[key] = dict(raw_value)
        return SubAgentInvocationPolicy(**policy_kwargs) if policy_kwargs else None

    def _sub_agent_instruction(self, tool_config: dict[str, Any]) -> str | None:
        if "instruction" in tool_config:
            raw_instruction = tool_config.get("instruction")
            return str(raw_instruction) if raw_instruction is not None else None

        raw_prompt = tool_config.get("instruction_prompt")
        fallback_instruction = (
            str(raw_prompt).strip()
            if raw_prompt is not None and str(raw_prompt).strip()
            else DEFAULT_SUB_AGENT_TOOL_INSTRUCTION.strip()
        )
        raw_prompt_name = tool_config.get("instruction_prompt_name")
        raw_prompt_label = tool_config.get("instruction_prompt_label")
        prompt_name = raw_prompt_name.strip() if isinstance(raw_prompt_name, str) else ""
        prompt_label = raw_prompt_label.strip() if isinstance(raw_prompt_label, str) else None
        if prompt_label == "":
            prompt_label = None
        if prompt_name or prompt_label:
            raw_project_name = tool_config.get("instruction_project_name") or self._config.get(
                "instruction_project_name"
            )
            project_name = (
                raw_project_name.strip()
                if isinstance(raw_project_name, str) and raw_project_name.strip()
                else "imp_agent_map.agent_core"
            )
            raw_agent_name = tool_config.get("instruction_agent_name")
            agent_name = (
                raw_agent_name.strip()
                if isinstance(raw_agent_name, str) and raw_agent_name.strip()
                else "sub_agent_invoke"
            )
            return load_agent_instruction(
                agent_name=agent_name,
                project_name=project_name,
                fallback_instruction=fallback_instruction,
                instruction_prompt_name=prompt_name or None,
                instruction_prompt_label=prompt_label,
            )

        if "instruction_prompt" in tool_config:
            return fallback_instruction
        return DEFAULT_SUB_AGENT_TOOL_INSTRUCTION

    @staticmethod
    def _import_callable(import_path: str, *, config_key: str) -> Any:
        if not import_path or "." not in import_path:
            raise ValueError(f"{config_key} must be module_path.function_name")
        module_path, func_name = import_path.rsplit(".", 1)
        module = import_module(module_path)
        return getattr(module, func_name)

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
