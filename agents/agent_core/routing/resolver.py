# SPDX-License-Identifier: Apache-2.0
"""Resolver logic for selecting agents from the dynamic registry."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from imp_agent_core.agents.agent_core.routing.descriptor import (
    AgentBackendType,
    AgentDescriptor,
    AgentHealthStatus,
)
from imp_agent_core.agents.agent_core.routing.handle import BaseAgentHandle
from imp_agent_core.agents.agent_core.routing.registry import DynamicAgentRegistry


class AgentResolver:
    """Select the best matching agent descriptor or handle for a capability request."""

    def __init__(self, registry: DynamicAgentRegistry, *, prefer_local: bool = True) -> None:
        self.registry = registry
        self.prefer_local = prefer_local

    def resolve_candidates(
        self,
        *,
        skill_id: str | None = None,
        tags: Iterable[str] | None = None,
        name_contains: str | None = None,
        metadata: Mapping[str, object] | None = None,
        backend_type: AgentBackendType | None = None,
    ) -> list[AgentDescriptor]:
        """Return candidate descriptors ordered by preference."""
        descriptors = self.registry.filter_descriptors(
            skill_id=skill_id,
            tags=tags,
            name_contains=name_contains,
            metadata=metadata,
            backend_type=backend_type,
        )
        prefer_local = self.prefer_local

        def sort_key(descriptor: AgentDescriptor) -> tuple[int, int, str]:
            backend_rank = 0
            if prefer_local and descriptor.backend_type != AgentBackendType.LOCAL_A2A:
                backend_rank = 1
            health_rank = 0
            if descriptor.health_status == AgentHealthStatus.UNKNOWN:
                health_rank = 1
            elif descriptor.health_status == AgentHealthStatus.UNHEALTHY:
                health_rank = 2
            return (backend_rank, health_rank, descriptor.agent_name.lower())

        return sorted(descriptors, key=sort_key)

    def resolve_descriptor(
        self,
        *,
        skill_id: str | None = None,
        tags: Iterable[str] | None = None,
        name_contains: str | None = None,
        metadata: Mapping[str, object] | None = None,
        backend_type: AgentBackendType | None = None,
    ) -> AgentDescriptor:
        """Resolve one best-match descriptor."""
        candidates = self.resolve_candidates(
            skill_id=skill_id,
            tags=tags,
            name_contains=name_contains,
            metadata=metadata,
            backend_type=backend_type,
        )
        if not candidates:
            filters = {
                "skill_id": skill_id,
                "tags": tuple(tags or ()),
                "name_contains": name_contains,
                "metadata": dict(metadata or {}),
                "backend_type": backend_type.value if backend_type else None,
            }
            raise LookupError(f"No registered agent matched filters: {filters}")
        return candidates[0]

    def resolve_handle(
        self,
        *,
        skill_id: str | None = None,
        tags: Iterable[str] | None = None,
        name_contains: str | None = None,
        metadata: Mapping[str, object] | None = None,
        backend_type: AgentBackendType | None = None,
    ) -> BaseAgentHandle:
        """Resolve a ready-to-use handle for the best matching descriptor."""
        descriptor = self.resolve_descriptor(
            skill_id=skill_id,
            tags=tags,
            name_contains=name_contains,
            metadata=metadata,
            backend_type=backend_type,
        )
        return self.registry.get_handle(descriptor.agent_id)
