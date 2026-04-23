"""Zep graph tool operations used by the skill route agent."""

from __future__ import annotations

import asyncio
from typing import Any, Callable


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


def _extract_items(container: Any, *, preferred_keys: tuple[str, ...]) -> list[Any]:
    if isinstance(container, list):
        return list(container)
    payload = _to_plain_dict(container)
    for key in preferred_keys:
        value = payload.get(key)
        if isinstance(value, list):
            return value
    return []


def _first_non_empty(values: list[Any]) -> str:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def search_nodes(
    *,
    client: Any,
    query: str,
    graph_id: str,
    limit: int = 10,
) -> list[dict[str, Any]]:
    if not graph_id.strip() or not query.strip():
        return []
    response = client.graph.search(
        query=query,
        graph_id=graph_id,
        scope="nodes",
        limit=limit,
    )
    raw_nodes = getattr(response, "nodes", None) or []
    return [_to_plain_dict(node) for node in raw_nodes if node]


def search_edges(
    *,
    client: Any,
    query: str,
    graph_id: str,
    limit: int = 10,
) -> list[dict[str, Any]]:
    if not graph_id.strip() or not query.strip():
        return []
    response = client.graph.search(
        query=query,
        graph_id=graph_id,
        scope="edges",
        limit=limit,
    )
    raw_edges = getattr(response, "edges", None) or []
    return [_to_plain_dict(edge) for edge in raw_edges if edge]


def get_node_by_id(*, client: Any, node_uuid: str) -> dict[str, Any] | None:
    normalized_uuid = str(node_uuid).strip()
    if not normalized_uuid:
        return None
    try:
        node = client.graph.node.get(uuid_=normalized_uuid)
    except Exception:
        return None
    payload = _to_plain_dict(node)
    return payload or None


def _fetch_node_edges(*, client: Any, node_uuid: str) -> list[dict[str, Any]]:
    try:
        response = client.graph.node.get_edges(node_uuid=node_uuid)
    except Exception:
        return []
    return [_to_plain_dict(edge) for edge in _extract_items(response, preferred_keys=("edges",)) if edge]


def _fetch_node_episodes(*, client: Any, node_uuid: str) -> list[dict[str, Any]]:
    try:
        response = client.graph.node.get_episodes(node_uuid=node_uuid)
    except Exception:
        return []
    return [
        _to_plain_dict(episode)
        for episode in _extract_items(response, preferred_keys=("episodes",))
        if episode
    ]


async def expand_node_edges(
    *,
    client: Any,
    node_uuids: list[str],
    concurrency: int = 10,
) -> dict[str, list[dict[str, Any]]]:
    normalized_uuids = [str(uuid).strip() for uuid in node_uuids if str(uuid).strip()]
    if not normalized_uuids:
        return {}
    semaphore = asyncio.Semaphore(max(1, int(concurrency)))

    async def fetch_one(node_uuid: str) -> tuple[str, list[dict[str, Any]]]:
        async with semaphore:
            edges = await asyncio.to_thread(_fetch_node_edges, client=client, node_uuid=node_uuid)
            return node_uuid, edges

    rows = await asyncio.gather(*(fetch_one(node_uuid) for node_uuid in normalized_uuids))
    return {node_uuid: edges for node_uuid, edges in rows}


def search_around_node(
    *,
    client: Any,
    node_uuid: str,
    graph_id: str,
    query: str | None = None,
    limit: int = 10,
) -> dict[str, Any]:
    node = get_node_by_id(client=client, node_uuid=node_uuid)
    if node is None:
        return {
            "node": None,
            "edges": [],
            "episodes": [],
            "related_nodes": [],
            "related_edges": [],
        }
    edges = _fetch_node_edges(client=client, node_uuid=node_uuid)
    episodes = _fetch_node_episodes(client=client, node_uuid=node_uuid)
    resolved_query = str(query or "").strip() or _first_non_empty(
        [node.get("name"), node.get("summary"), node.get("uuid"), node_uuid]
    )
    related_nodes = search_nodes(client=client, query=resolved_query, graph_id=graph_id, limit=limit)
    related_edges = search_edges(client=client, query=resolved_query, graph_id=graph_id, limit=limit)
    return {
        "node": node,
        "edges": edges,
        "episodes": episodes,
        "related_nodes": related_nodes,
        "related_edges": related_edges,
    }

