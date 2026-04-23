"""LLM tool functions for Zep graph operations."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from agents.zep_agent.tools.zep_helper import ZepToolClient, extract_items, to_plain_dict


@lru_cache(maxsize=1)
def _client() -> ZepToolClient:
    return ZepToolClient()


def search_skill_nodes(query: str, limit: int = 10, graph_id: str = "") -> dict[str, Any]:
    """Search skill nodes in Zep by semantic+keyword query text."""
    client = _client()
    resolved_graph_id = client.resolve_graph_id(graph_id)
    if not query.strip() or not resolved_graph_id:
        return {"graph_id": resolved_graph_id, "nodes": [], "count": 0}
    response = client.client.graph.search(
        query=query.strip(),
        graph_id=resolved_graph_id,
        scope="nodes",
        limit=max(1, limit),
    )
    nodes = [to_plain_dict(node) for node in (getattr(response, "nodes", None) or []) if node]
    return {"graph_id": resolved_graph_id, "nodes": nodes, "count": len(nodes)}


def search_edges(query: str, limit: int = 10, graph_id: str = "") -> dict[str, Any]:
    """Search relation edges in Zep by query text."""
    client = _client()
    resolved_graph_id = client.resolve_graph_id(graph_id)
    if not query.strip() or not resolved_graph_id:
        return {"graph_id": resolved_graph_id, "edges": [], "count": 0}
    response = client.client.graph.search(
        query=query.strip(),
        graph_id=resolved_graph_id,
        scope="edges",
        limit=max(1, limit),
    )
    edges = [to_plain_dict(edge) for edge in (getattr(response, "edges", None) or []) if edge]
    return {"graph_id": resolved_graph_id, "edges": edges, "count": len(edges)}


def get_edges_for_node(node_uuid: str) -> dict[str, Any]:
    """Get all edges connected to a specific node UUID."""
    client = _client()
    normalized_uuid = str(node_uuid).strip()
    if not normalized_uuid:
        return {"node_uuid": normalized_uuid, "edges": [], "count": 0}
    response = client.client.graph.node.get_edges(node_uuid=normalized_uuid)
    edges = [to_plain_dict(edge) for edge in extract_items(response, preferred_keys=("edges",)) if edge]
    return {"node_uuid": normalized_uuid, "edges": edges, "count": len(edges)}


def get_node_by_id(node_uuid: str) -> dict[str, Any]:
    """Get node details by node UUID."""
    client = _client()
    normalized_uuid = str(node_uuid).strip()
    if not normalized_uuid:
        return {"node_uuid": normalized_uuid, "node": None}
    try:
        node = client.client.graph.node.get(uuid_=normalized_uuid)
    except Exception as exc:  # noqa: BLE001
        return {"node_uuid": normalized_uuid, "node": None, "error": str(exc)}
    return {"node_uuid": normalized_uuid, "node": to_plain_dict(node)}


def search_around_node(node_uuid: str, query: str = "", limit: int = 10, graph_id: str = "") -> dict[str, Any]:
    """Collect local context around a node: node, connected edges, and related searches."""
    node_result = get_node_by_id(node_uuid=node_uuid)
    node = node_result.get("node")
    if not isinstance(node, dict):
        return {
            "node_uuid": str(node_uuid).strip(),
            "node": None,
            "edges": [],
            "related_nodes": [],
            "related_edges": [],
        }
    edge_result = get_edges_for_node(node_uuid=node_uuid)
    fallback_query = query.strip() or str(node.get("name") or node.get("summary") or node_uuid)
    related_nodes = search_skill_nodes(query=fallback_query, limit=limit, graph_id=graph_id)
    related_edges = search_edges(query=fallback_query, limit=limit, graph_id=graph_id)
    return {
        "node_uuid": str(node_uuid).strip(),
        "node": node,
        "edges": edge_result.get("edges", []),
        "related_nodes": related_nodes.get("nodes", []),
        "related_edges": related_edges.get("edges", []),
    }

