"""Shared helpers for retrieving skill candidates from Zep."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Iterable

from zep_cloud import Zep
from agents.skill_route_agent.utils import zep_tools


@dataclass(frozen=True)
class ZepSkillCandidate:
    """Normalized skill-like record returned from Zep graph search."""

    skill_id: str
    name: str
    description: str
    raw: dict[str, Any]


@dataclass(frozen=True)
class ZepQueryRequest:
    """Typed request payload used for executing Zep graph searches."""

    query: str
    scope: str = "nodes"
    limit: int = 5
    graph_id: str | None = None


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


def _extract_items(container: Any, *, preferred_keys: tuple[str, ...]) -> list[Any]:
    if isinstance(container, list):
        return list(container)
    payload = _to_plain_dict(container)
    for key in preferred_keys:
        value = payload.get(key)
        if isinstance(value, list):
            return value
    return []


def _edge_fact(raw_edge: dict[str, Any]) -> str:
    attributes = raw_edge.get("attributes") if isinstance(raw_edge.get("attributes"), dict) else {}
    return _first_non_empty(
        [
            raw_edge.get("fact"),
            attributes.get("fact"),
            raw_edge.get("name"),
            attributes.get("edge_type"),
        ]
    )


def _episode_excerpt(raw_episode: dict[str, Any], *, max_chars: int = 140) -> str:
    content = str(raw_episode.get("content") or "").strip()
    if not content:
        return ""
    compact = " ".join(content.split())
    if len(compact) <= max_chars:
        return compact
    return f"{compact[:max_chars].rstrip()}..."


async def fetch_edges_by_node_uuids(
    *,
    client: Any,
    node_uuids: list[str],
    concurrency: int = 10,
) -> dict[str, list[dict[str, Any]]]:
    """Fetch node edges concurrently for the provided UUID list."""
    return await zep_tools.expand_node_edges(
        client=client,
        node_uuids=node_uuids,
        concurrency=concurrency,
    )


def format_candidate_for_prompt(candidate: ZepSkillCandidate) -> str:
    """Render a Zep candidate as prompt-ready text for routing LLM input."""
    raw = candidate.raw if isinstance(candidate.raw, dict) else {}
    node_count = int(raw.get("node_count") or 0)
    edge_facts = raw.get("edge_facts") if isinstance(raw.get("edge_facts"), list) else []
    episode_previews = raw.get("episode_previews") if isinstance(raw.get("episode_previews"), list) else []

    lines = [
        f"- skill_id: {candidate.skill_id}",
        f"  name: {candidate.name}",
        f"  description: {candidate.description or 'No description provided.'}",
    ]
    if node_count:
        lines.append(f"  node_count: {node_count}")
    if edge_facts:
        facts_preview = "; ".join(str(fact).strip() for fact in edge_facts[:5] if str(fact).strip())
        if facts_preview:
            lines.append(f"  edge_facts: {facts_preview}")
    if episode_previews:
        episode_preview = " | ".join(
            str(preview).strip() for preview in episode_previews[:2] if str(preview).strip()
        )
        if episode_preview:
            lines.append(f"  episode_previews: {episode_preview}")
    return "\n".join(lines)


class ZepSkillSearchComponent:
    """Thin Zep graph wrapper that normalizes search results into skill candidates."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        default_graph_id: str | None = None,
    ) -> None:
        resolved_api_key = (api_key or os.getenv("ZEP_API_KEY", "")).strip()
        self.default_graph_id = (default_graph_id or os.getenv("GRAPH_ID", "")).strip()
        
        if resolved_api_key:
            self.client = Zep(api_key=resolved_api_key)
        else: 
            raise ValueError("ZEP_API_KEY is not set")
        self.api_key = resolved_api_key

    @property
    def is_configured(self) -> bool:
        return self.client is not None

    def search_nodes(
        self,
        *,
        query: str,
        graph_id: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Search graph nodes and return normalized node payloads."""
        if not self.client:
            return []
        return zep_tools.search_nodes(
            client=self.client,
            query=query,
            graph_id=(graph_id or self.default_graph_id).strip(),
            limit=limit,
        )

    def search_edges(
        self,
        *,
        query: str,
        graph_id: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Search graph edges and return normalized edge payloads."""
        if not self.client:
            return []
        return zep_tools.search_edges(
            client=self.client,
            query=query,
            graph_id=(graph_id or self.default_graph_id).strip(),
            limit=limit,
        )

    async def expand_node_edges(
        self,
        *,
        node_uuids: list[str],
        concurrency: int = 10,
    ) -> dict[str, list[dict[str, Any]]]:
        """Fetch edges for many node UUIDs concurrently."""
        if not self.client:
            return {}
        return await zep_tools.expand_node_edges(
            client=self.client,
            node_uuids=node_uuids,
            concurrency=concurrency,
        )

    def get_node_by_id(self, node_uuid: str) -> dict[str, Any] | None:
        """Get one node by UUID and normalize the response."""
        if not self.client:
            return None
        return zep_tools.get_node_by_id(client=self.client, node_uuid=node_uuid)

    def search_around_node(
        self,
        *,
        node_uuid: str,
        query: str | None = None,
        graph_id: str | None = None,
        limit: int = 10,
    ) -> dict[str, Any]:
        """Collect a node plus local graph context for downstream reasoning."""
        if not self.client:
            return {
                "node": None,
                "edges": [],
                "episodes": [],
                "related_nodes": [],
                "related_edges": [],
            }
        return zep_tools.search_around_node(
            client=self.client,
            node_uuid=node_uuid,
            graph_id=(graph_id or self.default_graph_id).strip(),
            query=query,
            limit=limit,
        )

    def execute_query(self, request: ZepQueryRequest) -> list[ZepSkillCandidate]:
        """Execute a typed Zep query request and return normalized candidates."""
        if not self.client:
            return []

        resolved_graph_id = (request.graph_id or self.default_graph_id).strip()
        if not resolved_graph_id or not request.query.strip():
            return []

        normalized_scope = "edges" if str(request.scope).strip().lower() == "edges" else "nodes"
        response = self.client.graph.search(
            query=request.query,
            graph_id=resolved_graph_id,
            scope=normalized_scope,
            limit=request.limit,
        )
        raw_records = (
            getattr(response, "edges", None)
            if normalized_scope == "edges"
            else getattr(response, "nodes", None)
        ) or []
        if normalized_scope == "nodes":
            return self._normalize_grouped_nodes(raw_records, limit=request.limit)
        return [
            candidate
            for candidate in (self._normalize_record(record) for record in raw_records)
            if candidate
        ]

    def _normalize_grouped_nodes(
        self,
        raw_nodes: list[Any],
        *,
        limit: int,
    ) -> list[ZepSkillCandidate]:
        grouped: dict[str, dict[str, Any]] = {}
        ordered_keys: list[str] = []
        for node in raw_nodes:
            raw = _to_plain_dict(node)
            if not raw:
                continue
            metadata = raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {}
            attributes = raw.get("attributes") if isinstance(raw.get("attributes"), dict) else {}
            node_uuid = _first_non_empty([raw.get("uuid"), raw.get("uuid_"), raw.get("id")])
            skill_id = _first_non_empty(
                [
                    metadata.get("skill_id"),
                    attributes.get("skill_id"),
                    raw.get("skill_id"),
                    node_uuid,
                    raw.get("name"),
                ]
            )
            if not skill_id:
                continue
            bucket = grouped.get(skill_id)
            if bucket is None:
                bucket = {
                    "nodes": [],
                    "edges": [],
                    "episodes": [],
                }
                grouped[skill_id] = bucket
                ordered_keys.append(skill_id)
            bucket["nodes"].append(raw)
            if node_uuid:
                bucket["edges"].extend(self._fetch_node_edges(node_uuid))
                bucket["episodes"].extend(self._fetch_node_episodes(node_uuid))

        candidates: list[ZepSkillCandidate] = []
        for skill_id in ordered_keys:
            bucket = grouped[skill_id]
            candidate = self._build_grouped_node_candidate(
                skill_id=skill_id,
                grouped_nodes=bucket["nodes"],
                grouped_edges=bucket["edges"],
                grouped_episodes=bucket["episodes"],
            )
            if candidate is not None:
                candidates.append(candidate)
            if len(candidates) >= limit:
                break
        return candidates

    def _fetch_node_edges(self, node_uuid: str) -> list[dict[str, Any]]:
        try:
            response = self.client.graph.node.get_edges(node_uuid=node_uuid)
        except Exception:
            return []
        return [_to_plain_dict(edge) for edge in _extract_items(response, preferred_keys=("edges",)) if edge]

    def _fetch_node_episodes(self, node_uuid: str) -> list[dict[str, Any]]:
        try:
            response = self.client.graph.node.get_episodes(node_uuid=node_uuid)
        except Exception:
            return []
        return [
            _to_plain_dict(episode)
            for episode in _extract_items(response, preferred_keys=("episodes",))
            if episode
        ]

    def _build_grouped_node_candidate(
        self,
        *,
        skill_id: str,
        grouped_nodes: list[dict[str, Any]],
        grouped_edges: list[dict[str, Any]],
        grouped_episodes: list[dict[str, Any]],
    ) -> ZepSkillCandidate | None:
        if not grouped_nodes:
            return None

        first_node = grouped_nodes[0]
        first_metadata = (
            first_node.get("metadata") if isinstance(first_node.get("metadata"), dict) else {}
        )
        first_attributes = (
            first_node.get("attributes") if isinstance(first_node.get("attributes"), dict) else {}
        )
        name = _first_non_empty(
            [
                first_metadata.get("skill_name"),
                first_node.get("name"),
                first_node.get("label"),
                first_attributes.get("name"),
                skill_id,
            ]
        )
        base_description = _first_non_empty(
            [
                first_metadata.get("description"),
                first_node.get("summary"),
                first_node.get("content"),
                first_attributes.get("description"),
                _compact_text(first_node.get("labels")),
            ]
        )
        edge_facts = [_edge_fact(edge) for edge in grouped_edges]
        edge_facts = [fact for fact in edge_facts if fact]
        unique_edge_facts: list[str] = []
        seen_facts: set[str] = set()
        for fact in edge_facts:
            if fact in seen_facts:
                continue
            unique_edge_facts.append(fact)
            seen_facts.add(fact)
        fact_preview = "; ".join(unique_edge_facts[:3])
        description = base_description
        if fact_preview:
            description = f"{base_description} Key facts: {fact_preview}".strip()

        node_uuids = [
            uuid
            for uuid in (_first_non_empty([node.get("uuid"), node.get("uuid_"), node.get("id")]) for node in grouped_nodes)
            if uuid
        ]
        episode_previews = [_episode_excerpt(episode) for episode in grouped_episodes]
        episode_previews = [preview for preview in episode_previews if preview]

        raw = dict(first_node)
        raw["skill_metadata"] = first_metadata
        raw["node_count"] = len(grouped_nodes)
        raw["node_uuids"] = node_uuids
        raw["edge_facts"] = unique_edge_facts
        raw["episode_previews"] = episode_previews[:5]
        raw["edges"] = grouped_edges
        raw["episodes"] = grouped_episodes
        return ZepSkillCandidate(
            skill_id=skill_id,
            name=name,
            description=description,
            raw=raw,
        )

    def _normalize_record(self, record: Any) -> ZepSkillCandidate | None:
        raw = _to_plain_dict(record)
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
        