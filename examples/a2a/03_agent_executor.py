"""Step 3: define AgentExecutor that runs an ADK agent."""

import os
from contextlib import nullcontext

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import TaskState, TextPart, UnsupportedOperationError
from a2a.utils import new_agent_text_message
from a2a.utils.errors import ServerError
from google.adk import Runner
from google.adk.agents import LlmAgent
from google.adk.artifacts import InMemoryArtifactService
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.sessions import InMemorySessionService
from google.genai import types
from langfuse import get_client


class CurrencyAgentExecutorWithRunner(AgentExecutor):
    """Executor that initializes ADK Runner from an LlmAgent."""

    def __init__(self, agent: LlmAgent):
        self.agent = agent
        self.runner: Runner | None = None

    def _init_adk(self) -> None:
        if self.runner is None:
            self.runner = Runner(
                app_name=self.agent.name,
                agent=self.agent,
                artifact_service=InMemoryArtifactService(),
                session_service=InMemorySessionService(),
                memory_service=InMemoryMemoryService(),
            )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        del context, event_queue
        raise ServerError(error=UnsupportedOperationError())

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        self._init_adk()
        if self.runner is None or not context.message:
            return

        incoming_metadata = context.message.metadata if context.message.metadata else {}
        incoming_trace_id = incoming_metadata.get("trace_id") if isinstance(incoming_metadata, dict) else None
        incoming_parent_obs = (
            incoming_metadata.get("parent_observation_id")
            if isinstance(incoming_metadata, dict)
            else None
        )

        user_id = (
            context.message.metadata.get("user_id")
            if context.message and context.message.metadata
            else "a2a_user"
        )
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        if not context.current_task:
            await updater.submit()
        await updater.start_work()

        query = context.get_user_input()
        content = types.Content(role="user", parts=[types.Part(text=query)])

        obs_kwargs = {
            "name": "a2a_executor_execute",
            "as_type": "chain",
            "input": {"query": query, "task_id": context.task_id},
            "metadata": {"context_id": context.context_id},
        }
        if isinstance(incoming_trace_id, str) and incoming_trace_id:
            obs_kwargs["trace_id"] = incoming_trace_id
        if isinstance(incoming_parent_obs, str) and incoming_parent_obs:
            obs_kwargs["parent_observation_id"] = incoming_parent_obs

        try:
            exec_obs_cm = get_client().start_as_current_observation(**obs_kwargs)
        except Exception:
            exec_obs_cm = nullcontext()

        with exec_obs_cm as exec_obs:
            if hasattr(exec_obs, "trace_id") and getattr(exec_obs, "trace_id", None):
                os.environ["A2A_TRACE_ID"] = str(exec_obs.trace_id)
            elif isinstance(incoming_trace_id, str) and incoming_trace_id:
                os.environ["A2A_TRACE_ID"] = incoming_trace_id
            if hasattr(exec_obs, "id") and getattr(exec_obs, "id", None):
                os.environ["A2A_PARENT_OBSERVATION_ID"] = str(exec_obs.id)
            elif isinstance(incoming_parent_obs, str) and incoming_parent_obs:
                os.environ["A2A_PARENT_OBSERVATION_ID"] = incoming_parent_obs

            try:
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

                final_event = None
                async for event in self.runner.run_async(
                    session_id=session.id,
                    user_id=user_id,
                    new_message=content,
                ):
                    if event.is_final_response():
                        final_event = event

                if final_event and final_event.content and final_event.content.parts:
                    response_text = "".join(
                        part.text
                        for part in final_event.content.parts
                        if hasattr(part, "text") and part.text
                    )
                    if response_text:
                        await updater.add_artifact([TextPart(text=response_text)], name="result")
                        await updater.complete()
                        if hasattr(exec_obs, "update"):
                            exec_obs.update(output={"status": "completed", "task_id": context.task_id})
                        return

                await updater.update_status(
                    TaskState.failed,
                    message=new_agent_text_message(
                        "Failed to generate a final response with text content."
                    ),
                    final=True,
                )
                if hasattr(exec_obs, "update"):
                    exec_obs.update(output={"status": "failed", "task_id": context.task_id})
            except Exception as exc:  # noqa: BLE001
                await updater.update_status(
                    TaskState.failed,
                    message=new_agent_text_message(f"An error occurred: {str(exc)}"),
                    final=True,
                )
                if hasattr(exec_obs, "update"):
                    exec_obs.update(output={"status": "error", "error": str(exc)})
            finally:
                os.environ.pop("A2A_PARENT_OBSERVATION_ID", None)
                os.environ.pop("A2A_TRACE_ID", None)
