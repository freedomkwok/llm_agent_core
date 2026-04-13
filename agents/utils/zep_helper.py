"""Shared helpers for retrieving skill candidates from Zep."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Iterable

from zep_cloud import Zep


@dataclass(frozen=True)
class ZepSkillCandidate:
    """Normalized skill-like record returned from Zep graph search."""

    skill_id: str
    name: str
    description: str
    raw: dict[str, Any]


def _to_plain_dict(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        dumped = value.model_dump(mode="json")
        if isinstance(dumped, dict):
            return dumped
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "__dict__"):
        return {
            key: raw_value
            for key, raw_value in vars(value).items()
            if not key.startswith("_")
        }
    return {}


def _first_non_empty(values: Iterable[Any]) -> str:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _compact_text(value: Any) -> str:
    if isinstance(value, list):
        parts = [str(item).strip() for item in value if str(item).strip()]
        return ", ".join(parts)
    if value is None:
        return ""
    return str(value).strip()


class ZepSkillSearchComponent:
    """Thin Zep graph wrapper that normalizes search results into skill candidates."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        default_user_id: str | None = None,
    ) -> None:
        resolved_api_key = (api_key or os.getenv("ZEP_API_KEY", "")).strip()
        self.default_user_id = (default_user_id or os.getenv("ZEP_USER_ID", "")).strip()
        self.client = Zep(api_key=resolved_api_key) if resolved_api_key else None

    @property
    def is_configured(self) -> bool:
        return self.client is not None

    def search_skills(
        self,
        *,
        query: str,
        user_id: str | None = None,
        limit: int = 5,
    ) -> list[ZepSkillCandidate]:
        """Search Zep graph nodes and normalize them into skill candidates."""
        if not self.client:
            return []

        resolved_user_id = (user_id or self.default_user_id).strip()
        if not resolved_user_id or not query.strip():
            return []

        response = self.client.graph.search(
            query=query,
            user_id=resolved_user_id,
            scope="nodes",
            limit=limit,
        )
        nodes = getattr(response, "nodes", None) or []
        return [candidate for candidate in (self._normalize_node(node) for node in nodes) if candidate]

    def _normalize_node(self, node: Any) -> ZepSkillCandidate | None:
        raw = _to_plain_dict(node)
        if not raw:
            return None

        metadata = raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {}
        attributes = raw.get("attributes") if isinstance(raw.get("attributes"), dict) else {}

        skill_id = _first_non_empty(
            [
                metadata.get("skill_id"),
                raw.get("uuid"),
                raw.get("id"),
                attributes.get("skill_id"),
                raw.get("name"),
            ]
        )
        if not skill_id:
            return None

        name = _first_non_empty(
            [
                metadata.get("skill_name"),
                raw.get("name"),
                raw.get("label"),
                attributes.get("name"),
                skill_id,
            ]
        )
        description = _first_non_empty(
            [
                metadata.get("description"),
                raw.get("summary"),
                raw.get("content"),
                attributes.get("description"),
                _compact_text(raw.get("labels")),
            ]
        )

        return ZepSkillCandidate(
            skill_id=skill_id,
            name=name,
            description=description,
            raw=raw,
        )