"""Dynamic runtime registry for local and remote agent descriptors."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from importlib import import_module
from inspect import signature
from pkgutil import iter_modules
from typing import Any

from agents.agent_core.routing.descriptor import (
    AgentBackendType,
    AgentDescriptor,
    AgentHealthStatus,
    SkillDescriptor,
    aggregate_tags,
)
from agents.agent_core.routing.handle import BaseAgentHandle, build_agent_handle


class DynamicAgentRegistry:
    """Registry for runtime agent registration, lookup, and handle creation."""

    def __init__(self) -> None:
        self._descriptors: dict[str, AgentDescriptor] = {}
        self._handles: dict[str, BaseAgentHandle] = {}

    def register_descriptor(
        self,
        descriptor: AgentDescriptor,
        *,
        replace: bool = False,
    ) -> AgentDescriptor:
        """Register a descriptor and clear any stale cached handle."""
        if not replace and descriptor.agent_id in self._descriptors:
            raise ValueError(f"Agent id already registered: {descriptor.agent_id}")
        self._descriptors[descriptor.agent_id] = descriptor
        self._handles.pop(descriptor.agent_id, None)
        return descriptor

    def register_local_agent(
        self,
        *,
        agent_id: str,
        agent_name: str,
        description: str,
        skills: Iterable[SkillDescriptor],
        local_builder: Any,
        tags: Iterable[str] | None = None,
        cached_agent_card: Any | None = None,
        metadata: Mapping[str, Any] | None = None,
        health_status: AgentHealthStatus = AgentHealthStatus.UNKNOWN,
        available: bool = True,
        replace: bool = False,
    ) -> AgentDescriptor:
        """Register a local builder-backed A2A agent."""
        normalized_skills = tuple(skills)
        descriptor = AgentDescriptor(
            agent_id=agent_id,
            agent_name=agent_name,
            description=description,
            skills=normalized_skills,
            tags=aggregate_tags(normalized_skills, tags),
            backend_type=AgentBackendType.LOCAL_A2A,
            local_builder=local_builder,
            cached_agent_card=cached_agent_card,
            health_status=health_status,
            available=available,
            metadata=dict(metadata or {}),
        )
        return self.register_descriptor(descriptor, replace=replace)

    def register_remote_agent(
        self,
        *,
        agent_id: str,
        agent_name: str,
        description: str,
        skills: Iterable[SkillDescriptor],
        endpoint: str,
        tags: Iterable[str] | None = None,
        cached_agent_card: Any | None = None,
        metadata: Mapping[str, Any] | None = None,
        health_status: AgentHealthStatus = AgentHealthStatus.UNKNOWN,
        available: bool = True,
        replace: bool = False,
    ) -> AgentDescriptor:
        """Register a remote/network A2A agent endpoint."""
        normalized_skills = tuple(skills)
        descriptor = AgentDescriptor(
            agent_id=agent_id,
            agent_name=agent_name,
            description=description,
            skills=normalized_skills,
            tags=aggregate_tags(normalized_skills, tags),
            backend_type=AgentBackendType.REMOTE_A2A,
            endpoint=endpoint,
            cached_agent_card=cached_agent_card,
            health_status=health_status,
            available=available,
            metadata=dict(metadata or {}),
        )
        return self.register_descriptor(descriptor, replace=replace)

    def get_descriptor(self, agent_id: str) -> AgentDescriptor:
        """Return a descriptor by id."""
        try:
            return self._descriptors[agent_id]
        except KeyError as exc:
            raise KeyError(f"Unknown agent id: {agent_id}") from exc

    def list_descriptors(self) -> list[AgentDescriptor]:
        """Return all registered descriptors."""
        return list(self._descriptors.values())

    def filter_descriptors(
        self,
        *,
        skill_id: str | None = None,
        tags: Iterable[str] | None = None,
        name_contains: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        backend_type: AgentBackendType | None = None,
        available_only: bool = True,
    ) -> list[AgentDescriptor]:
        """Filter descriptors by capability metadata."""
        candidates = self.list_descriptors()
        normalized_name = (name_contains or "").strip().lower()
        required_tags = tuple(tag for tag in (tags or ()) if tag)
        required_metadata = dict(metadata or {})

        results: list[AgentDescriptor] = []
        for descriptor in candidates:
            if available_only and not descriptor.available:
                continue
            if backend_type is not None and descriptor.backend_type != backend_type:
                continue
            if skill_id and not descriptor.supports_skill(skill_id):
                continue
            if required_tags and not descriptor.matches_tags(required_tags):
                continue
            if normalized_name and normalized_name not in descriptor.agent_name.lower():
                continue
            if required_metadata:
                if any(
                    descriptor.metadata.get(key) != value
                    for key, value in required_metadata.items()
                ):
                    continue
            results.append(descriptor)
        return results

    def get_handle(self, agent_id: str) -> BaseAgentHandle:
        """Resolve or create a handle for the given agent id."""
        if agent_id not in self._handles:
            descriptor = self.get_descriptor(agent_id)
            self._handles[agent_id] = build_agent_handle(descriptor)
        return self._handles[agent_id]


_GLOBAL_AGENT_REGISTRY = DynamicAgentRegistry()


def get_global_agent_registry() -> DynamicAgentRegistry:
    """Return the process-wide agent registry."""
    return _GLOBAL_AGENT_REGISTRY


def register_agent_package(
    registry: DynamicAgentRegistry | None = None,
    *,
    package_name: str = "agents",
) -> list[AgentDescriptor]:
    """Register worker agents exposed by registry modules under an agent package."""
    target_registry = registry or get_global_agent_registry()
    package = import_module(package_name)
    package_paths = getattr(package, "__path__", None)
    if package_paths is None:
        return []

    registered: list[AgentDescriptor] = []
    for module_info in iter_modules(package_paths, prefix=f"{package_name}."):
        if not module_info.ispkg:
            continue
        if module_info.name.rsplit(".", 1)[-1] == "agent_core":
            continue
        registry_module_name = f"{module_info.name}.registry"
        try:
            registry_module = import_module(registry_module_name)
        except ModuleNotFoundError as exc:
            if exc.name == registry_module_name:
                continue
            raise
        for registrar in _agent_registrars(registry_module):
            registered.append(_call_agent_registrar(registrar, target_registry))
    return registered


def _agent_registrars(registry_module: Any) -> list[Any]:
    exact_registrar = getattr(registry_module, "register_worker_agent", None)
    if callable(exact_registrar):
        return [exact_registrar]
    return [
        registrar
        for name, registrar in vars(registry_module).items()
        if name.startswith("register_") and name.endswith("_worker_agent") and callable(registrar)
    ]


def _call_agent_registrar(registrar: Any, registry: DynamicAgentRegistry) -> AgentDescriptor:
    registrar_signature = signature(registrar)
    if "registry" in registrar_signature.parameters:
        return registrar(registry=registry, replace=True)
    return registrar(replace=True)


def reset_global_agent_registry() -> DynamicAgentRegistry:
    """Reset and return the process-wide agent registry."""
    global _GLOBAL_AGENT_REGISTRY
    _GLOBAL_AGENT_REGISTRY = DynamicAgentRegistry()
    return _GLOBAL_AGENT_REGISTRY
