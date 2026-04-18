"""Reusable A2A execution wrapper backed by ADK Runner."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

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
from llm_inference_core import extract_langfuse_trace_context, run_with_langfuse_chain_trace


class AdkA2aExecutionWrapper(AgentExecutor, ABC):
    """Base A2A wrapper for agents that run through ADK Runner."""

    def __init__(self, *, langfuse_client: Any):
        self.langfuse_client = langfuse_client
        self.adk_agent = self.build_adk_agent()
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

    def build_engine_metadata(
        self, *, context: RequestContext, user_id: str
    ) -> dict[str, Any]:
        return {
            "context_id": context.context_id,
            "task_id": context.task_id,
            "user_id": user_id,
        }

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
        trace_context = extract_langfuse_trace_context(incoming_metadata)
        user_id = incoming_metadata.get("user_id") if incoming_metadata else "a2a_user"
        graph_state_delta: dict[str, Any] | None = None
        if "graph_id" in incoming_metadata:
            raw_gid = incoming_metadata.get("graph_id")
            graph_state_delta = {
                "graph_id": (str(raw_gid).strip() if raw_gid is not None else "")
            }

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
                    state_delta=graph_state_delta,
                )

            def _on_chain_success(result: Any, observation: Any) -> None:
                status = "completed" if isinstance(result, str) and result else "failed"
                observation.update(output={"status": status, "task_id": context.task_id})

            response_text = await run_with_langfuse_chain_trace(
                langfuse=self.langfuse_client,
                trace_name=self.trace_name,
                trace_input=self.build_trace_input(request_text=request_text, context=context),
                runner=_run_chain,
                trace_context=trace_context,
                metadata=self.build_trace_metadata(context=context),
                on_success=_on_chain_success,
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
