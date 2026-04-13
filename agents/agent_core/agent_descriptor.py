"""Descriptor models for dynamically registered agent backends."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterable, Mapping


class AgentBackendType(str, Enum):
    """Supported backend implementations for an agent capability endpoint."""

    LOCAL_A2A = "local_a2a"
    REMOTE_A2A = "remote_a2a"


class AgentHealthStatus(str, Enum):
    """Best-effort availability status for an agent endpoint."""

    UNKNOWN = "unknown"
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"


@dataclass(frozen=True)
class SkillDescriptor:
    """Normalized description of one skill exposed by an agent."""

    skill_id: str
    name: str
    description: str = ""
    tags: tuple[str, ...] = ()
    examples: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentDescriptor:
    """Normalized runtime description of one resolvable agent endpoint."""

    agent_id: str
    agent_name: str
    description: str
    skills: tuple[SkillDescriptor, ...]
    tags: tuple[str, ...] = ()
    backend_type: AgentBackendType = AgentBackendType.LOCAL_A2A
    endpoint: str | None = None
    local_builder: Callable[[], Any] | None = field(default=None, repr=False, compare=False)
    cached_agent_card: Any | None = field(default=None, repr=False, compare=False)
    health_status: AgentHealthStatus = AgentHealthStatus.UNKNOWN
    available: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def supports_skill(self, skill_id: str) -> bool:
        """Return whether this descriptor exposes the requested skill id."""
        return any(skill.skill_id == skill_id for skill in self.skills)

    def matches_tags(self, required_tags: Iterable[str]) -> bool:
        """Return whether all requested tags are present on the descriptor or skills."""
        required = {tag for tag in required_tags if tag}
        if not required:
            return True
        available = set(self.tags)
        for skill in self.skills:
            available.update(skill.tags)
        return required.issubset(available)


def normalize_skill_descriptors(raw_skills: Iterable[Any] | None) -> tuple[SkillDescriptor, ...]:
    """Normalize skill-like objects or dicts into `SkillDescriptor` values."""
    if raw_skills is None:
        return ()

    normalized: list[SkillDescriptor] = []
    for raw_skill in raw_skills:
        if hasattr(raw_skill, "model_dump"):
            skill_data = raw_skill.model_dump(mode="json")
        elif isinstance(raw_skill, Mapping):
            skill_data = dict(raw_skill)
        else:
            skill_data = {
                "id": getattr(raw_skill, "id", ""),
                "name": getattr(raw_skill, "name", ""),
                "description": getattr(raw_skill, "description", ""),
                "tags": getattr(raw_skill, "tags", ()) or (),
                "examples": getattr(raw_skill, "examples", ()) or (),
            }

        skill_id = str(skill_data.get("id") or "").strip()
        name = str(skill_data.get("name") or skill_id).strip()
        if not skill_id:
            continue
        normalized.append(
            SkillDescriptor(
                skill_id=skill_id,
                name=name or skill_id,
                description=str(skill_data.get("description") or "").strip(),
                tags=tuple(str(tag) for tag in (skill_data.get("tags") or []) if tag),
                examples=tuple(
                    str(example) for example in (skill_data.get("examples") or []) if example
                ),
            )
        )
    return tuple(normalized)


def aggregate_tags(skills: Iterable[SkillDescriptor], tags: Iterable[str] | None = None) -> tuple[str, ...]:
    """Merge descriptor-level and skill-level tags into a stable tuple."""
    merged: list[str] = []
    seen: set[str] = set()
    for tag in tags or ():
        if tag and tag not in seen:
            merged.append(tag)
            seen.add(tag)
    for skill in skills:
        for tag in skill.tags:
            if tag and tag not in seen:
                merged.append(tag)
                seen.add(tag)
    return tuple(merged)
