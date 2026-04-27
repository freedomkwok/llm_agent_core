"""LLM tool functions for Zep graph operations."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from agents.zep_agent.tools.zep_helper import (
    ZepToolClient,
    extract_items,
    to_plain_dict,
    trim_node_fields,
)


@lru_cache(maxsize=1)
def _client() -> ZepToolClient:
    return ZepToolClient()


def search_nodes(query: str, limit: int = 10, graph_id: str = "") -> dict[str, Any]:
    """Find graph entities relevant to a natural-language query.

    Best for entity discovery when the model needs candidate skills/concepts
    before taking deeper actions.

    Args:
        query: Natural-language search text (semantic + keyword matching).
        limit: Maximum number of node hits to return (minimum 1).
        graph_id: Target graph ID; falls back to default configured graph when empty.

    Returns:
        A compact node result payload:
        {
          "graph_id": str,
          "nodes": [
            {"name": str | None, "attributes": dict, "metadata": dict, "summary": str, "score": float | None}
          ],
          "count": int
        }
    """
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
    nodes = [trim_node_fields(node) for node in (getattr(response, "nodes", None) or []) if node]
    return {"graph_id": resolved_graph_id, "nodes": nodes, "count": len(nodes)}


def search_edges(query: str, limit: int = 10, graph_id: str = "") -> dict[str, Any]:
    """Find relationship facts relevant to a natural-language query.

    Best when the model needs relation/evidence-level context (facts and links)
    instead of only entity candidates.

    Args:
        query: Natural-language search text.
        limit: Maximum number of edge hits to return (minimum 1).
        graph_id: Target graph ID; falls back to default configured graph when empty.

    Returns:
        {
          "graph_id": str,
          "edges": [edge_dict, ...],
          "count": int
        }
    """
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
    """Fetch all edges directly connected to one node.

    Best for local graph expansion around a known entity node.

    Args:
        node_uuid: UUID of the anchor node.

    Returns:
        {
          "node_uuid": str,
          "edges": [edge_dict, ...],
          "count": int
        }
    """
    client = _client()
    normalized_uuid = str(node_uuid).strip()
    if not normalized_uuid:
        return {"node_uuid": normalized_uuid, "edges": [], "count": 0}
    response = client.client.graph.node.get_edges(node_uuid=normalized_uuid)
    edges = [to_plain_dict(edge) for edge in extract_items(response, preferred_keys=("edges",)) if edge]
    return {"node_uuid": normalized_uuid, "edges": edges, "count": len(edges)}


def get_node_by_id(node_uuid: str) -> dict[str, Any]:
    """Fetch one node by UUID and return a compact, model-safe shape.

    Best when the model already has a node identifier and needs stable node
    identity/context fields without heavy internals.

    Args:
        node_uuid: UUID of the node to retrieve.

    Returns:
        {
          "node_uuid": str,
          "node": {"name", "attributes", "metadata", "summary", "score"} | None,
          "error": str (optional)
        }
    """
    client = _client()
    normalized_uuid = str(node_uuid).strip()
    if not normalized_uuid:
        return {"node_uuid": normalized_uuid, "node": None}
    try:
        node = client.client.graph.node.get(uuid_=normalized_uuid)
    except Exception as exc:  # noqa: BLE001
        return {"node_uuid": normalized_uuid, "node": None, "error": str(exc)}
    return {"node_uuid": normalized_uuid, "node": trim_node_fields(node)}


def search_around_node(node_uuid: str, query: str = "", limit: int = 10, graph_id: str = "") -> dict[str, Any]:
    """Build a neighborhood context bundle around a node.

    Combines direct node lookup, connected edges, and related node/edge search
    into one response for routing/reasoning steps.

    Args:
        node_uuid: Anchor node UUID.
        query: Optional override query for related search. If empty, uses node-derived text.
        limit: Maximum results for related node/edge searches.
        graph_id: Target graph ID for related searches.

    Returns:
        {
          "node_uuid": str,
          "node": compact_node | None,
          "edges": [...],
          "related_nodes": [compact_node, ...],
          "related_edges": [...]
        }
    """
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
    related_nodes = search_nodes(query=fallback_query, limit=limit, graph_id=graph_id)
    related_edges = search_edges(query=fallback_query, limit=limit, graph_id=graph_id)
    return {
        "node_uuid": str(node_uuid).strip(),
        "node": node,
        "edges": edge_result.get("edges", []),
        "related_nodes": related_nodes.get("nodes", []),
        "related_edges": related_edges.get("edges", []),
    }

