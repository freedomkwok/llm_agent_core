"""Shared Zep helper for zep_agent tools."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from zep_cloud import Zep


def to_plain_dict(value: Any) -> dict[str, Any]:
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


def extract_items(container: Any, *, preferred_keys: tuple[str, ...]) -> list[Any]:
    if isinstance(container, list):
        return list(container)
    payload = to_plain_dict(container)
    for key in preferred_keys:
        value = payload.get(key)
        if isinstance(value, list):
            return value
    return []


def trim_node_fields(raw_node: Any) -> dict[str, Any]:
    node_payload = to_plain_dict(raw_node)
    summary = node_payload.get("summary")
    score = node_payload.get("score")
    return {
        "name": node_payload.get("name"),
        "attributes": node_payload.get("attributes")
        if isinstance(node_payload.get("attributes"), dict)
        else {},
        "metadata": node_payload.get("metadata")
        if isinstance(node_payload.get("metadata"), dict)
        else {},
        "summary": summary if isinstance(summary, str) else "",
        "score": score if isinstance(score, (int, float)) else None,
    }


@dataclass(frozen=True)
class ZepToolRequest:
    query: str
    limit: int = 10
    graph_id: str | None = None


class ZepToolClient:
    """Small, focused client wrapper used by LLM tools."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        default_graph_id: str | None = None,
    ) -> None:
        resolved_api_key = (api_key or os.getenv("ZEP_API_KEY", "")).strip()
        if not resolved_api_key:
            raise ValueError("ZEP_API_KEY is not set")
        self.client = Zep(api_key=resolved_api_key)
        self.default_graph_id = (default_graph_id or os.getenv("GRAPH_ID", "")).strip()

    def resolve_graph_id(self, explicit_graph_id: str | None = None) -> str:
        return (explicit_graph_id or self.default_graph_id).strip()

