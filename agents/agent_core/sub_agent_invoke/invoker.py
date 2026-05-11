# SPDX-License-Identifier: Apache-2.0
"""Registry-backed sub-agent invocation."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from google.adk.tools.tool_context import ToolContext

from agents.agent_core.a2a.local_orchestration import OrchestrationMode
from agents.agent_core.routing import AgentBackendType, AgentDescriptor, AgentResolver
from agents.agent_core.routing.registry import DynamicAgentRegistry
from agents.agent_core.sub_agent_invoke.policy import (
    DEPTH_STATE_KEY,
    SubAgentInvocationPolicy,
)


class SubAgentInvoker:
    """Resolve registered agents and invoke them from an ADK tool context."""

    def __init__(
        self,
        *,
        registry: DynamicAgentRegistry,
        policy: SubAgentInvocationPolicy | None = None,
        resolver: AgentResolver | None = None,
    ) -> None:
        self.registry = registry
        self.policy = policy or SubAgentInvocationPolicy()
        self.resolver = resolver or AgentResolver(registry)

    async def invoke(
        self,
        *,
        query: str,
        tool_context: ToolContext,
        agent_id: str | None = None,
        skill_id: str | None = None,
        tags: Iterable[str] | None = None,
        name_contains: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        backend_type: AgentBackendType | None = None,
    ) -> dict[str, Any]:
        """Invoke a registered agent by id or resolver filters."""
        query_text = query.strip()
        if not query_text:
            return self._error("empty_query", "Sub-agent query must not be empty.")

        state = self._tool_state(tool_context)
        current_depth = self._current_depth(state)
        if current_depth >= self.policy.max_depth:
            return self._error(
                "sub_agent_depth_exceeded",
                (
                    f"Sub-agent invocation depth {current_depth} reached the configured "
                    f"maximum of {self.policy.max_depth}."
                ),
            )

        descriptor_result = self._resolve_descriptor(
            agent_id=agent_id,
            skill_id=skill_id,
            tags=tags,
            name_contains=name_contains,
            metadata=metadata,
            backend_type=backend_type,
        )
        if isinstance(descriptor_result, dict):
            return descriptor_result

        child_metadata = self._child_metadata(
            state=state,
            depth=current_depth + 1,
            extra_metadata=metadata,
        )
        try:
            handle = self.registry.get_handle(descriptor_result.agent_id)
            result = await handle.run(
                message_text=query_text,
                mode=OrchestrationMode.AGENT_INTERNAL,
                metadata=child_metadata,
                include_authenticated_card=False,
                fetch_task_response=True,
            )
        except Exception as exc:  # noqa: BLE001
            return self._error("sub_agent_invocation_failed", str(exc))

        return {
            "status": result.task_status or "unknown",
            "agent_id": descriptor_result.agent_id,
            "agent_name": descriptor_result.agent_name,
            "mode": result.mode.value,
            "task_id": result.task_id,
            "final_text": result.final_text,
        }

    def _resolve_descriptor(
        self,
        *,
        agent_id: str | None,
        skill_id: str | None,
        tags: Iterable[str] | None,
        name_contains: str | None,
        metadata: Mapping[str, Any] | None,
        backend_type: AgentBackendType | None,
    ) -> AgentDescriptor | dict[str, Any]:
        requested_agent_id = (agent_id or "").strip()
        requested_skill_id = (skill_id or "").strip()
        if requested_agent_id:
            if not self._agent_id_allowed(requested_agent_id):
                return self._error(
                    "agent_not_allowed",
                    f"Agent id is not allowed for sub-agent invocation: {requested_agent_id}",
                )
            try:
                return self.registry.get_descriptor(requested_agent_id)
            except KeyError as exc:
                return self._error("agent_not_found", str(exc))

        requested_tags = tuple(tag for tag in (tags or ()) if tag)
        if not requested_skill_id and not requested_tags and not name_contains:
            return self._error(
                "missing_agent_selector",
                "Provide agent_id or at least one resolver filter.",
            )
        if requested_skill_id and not self._skill_id_allowed(requested_skill_id):
            return self._error(
                "skill_not_allowed",
                f"Skill id is not allowed for sub-agent invocation: {requested_skill_id}",
            )

        candidates = self.resolver.resolve_candidates(
            skill_id=requested_skill_id or None,
            tags=requested_tags,
            name_contains=name_contains,
            metadata=metadata,
            backend_type=backend_type,
        )
        if not candidates:
            return self._error("agent_not_found", "No registered agent matched the request.")
        if len(candidates) > 1:
            return {
                "status": "ambiguous",
                "message": "Multiple registered agents matched the request.",
                "candidates": [self._candidate_summary(candidate) for candidate in candidates],
            }
        return candidates[0]

    def _child_metadata(
        self,
        *,
        state: Mapping[str, Any],
        depth: int,
        extra_metadata: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        metadata = dict(self.policy.static_metadata)
        for key in self.policy.forwarded_state_keys:
            if key not in state:
                continue
            value = state[key]
            if value is None:
                continue
            metadata[self.policy.metadata_aliases.get(key, key)] = value
        if extra_metadata:
            metadata.update(dict(extra_metadata))
        metadata[DEPTH_STATE_KEY] = depth
        return metadata

    def _agent_id_allowed(self, agent_id: str) -> bool:
        return not self.policy.allowed_agent_ids or agent_id in self.policy.allowed_agent_ids

    def _skill_id_allowed(self, skill_id: str) -> bool:
        return not self.policy.allowed_skill_ids or skill_id in self.policy.allowed_skill_ids

    @staticmethod
    def _tool_state(tool_context: ToolContext) -> Mapping[str, Any]:
        state = getattr(tool_context, "state", None)
        return state if isinstance(state, Mapping) else {}

    @staticmethod
    def _current_depth(state: Mapping[str, Any]) -> int:
        raw_depth = state.get(DEPTH_STATE_KEY, 0)
        try:
            return max(0, int(raw_depth))
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _candidate_summary(descriptor: AgentDescriptor) -> dict[str, Any]:
        return {
            "agent_id": descriptor.agent_id,
            "agent_name": descriptor.agent_name,
            "skills": [skill.skill_id for skill in descriptor.skills],
        }

    @staticmethod
    def _error(error: str, message: str) -> dict[str, Any]:
        return {"status": "error", "error": error, "message": message}
