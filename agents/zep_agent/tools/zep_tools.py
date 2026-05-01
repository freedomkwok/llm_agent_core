"""LLM tool functions for Zep graph operations."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from zep_cloud.types import SearchFilters

from agents.zep_agent.tools.zep_helper import (
    ZepToolClient,
    extract_items,
    to_plain_dict,
    trim_node_fields,
)

_RERANKERS = {"rrf", "mmr", "node_distance", "episode_mentions", "cross_encoder"}
_REQUEST_OPTION_KEYS = {"timeout_in_seconds", "max_retries"}
_MAX_ZEP_SEARCH_LIMIT = 10


@lru_cache(maxsize=1)
def _client() -> ZepToolClient:
    return ZepToolClient()


def _zep_search_filters(raw_filters: dict[str, Any] | None) -> SearchFilters | None:
    if not raw_filters:
        return None
    return SearchFilters(**raw_filters)


def _zep_reranker(raw_reranker: str) -> str | None:
    reranker = raw_reranker.strip()
    if not reranker:
        return None
    if reranker not in _RERANKERS:
        raise ValueError(f"reranker must be one of {sorted(_RERANKERS)}")
    return reranker


def _zep_mmr_lambda(raw_mmr_lambda: float | None) -> float | None:
    if raw_mmr_lambda is None:
        return None
    mmr_lambda = float(raw_mmr_lambda)
    if not 0 <= mmr_lambda <= 1:
        raise ValueError("mmr_lambda must be between 0 and 1.")
    return mmr_lambda


def _zep_request_options(raw_options: dict[str, Any] | None) -> dict[str, Any] | None:
    if not raw_options:
        return None
    return {key: value for key, value in raw_options.items() if key in _REQUEST_OPTION_KEYS}


def _zep_search_limit(raw_limit: int) -> int:
    return min(_MAX_ZEP_SEARCH_LIMIT, max(1, int(raw_limit)))


def _graph_result_score(item: dict[str, Any]) -> float:
    score = item.get("score")
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        return float("-inf")
    return float(score)


def _trim_episode_fields(raw_episode: Any) -> dict[str, Any]:
    episode = to_plain_dict(raw_episode)
    return {
        "uuid_": episode.get("uuid_") or episode.get("uuid") or "",
        "content": episode.get("content") or "",
        "created_at": episode.get("created_at") or "",
        "metadata": episode.get("metadata") if isinstance(episode.get("metadata"), dict) else {},
        "role": episode.get("role"),
        "role_type": episode.get("role_type"),
        "source": episode.get("source"),
        "source_description": episode.get("source_description"),
        "score": episode.get("score") if isinstance(episode.get("score"), (int, float)) else None,
        "relevance": episode.get("relevance")
        if isinstance(episode.get("relevance"), (int, float))
        else None,
    }


def search_nodes(
    query: str,
    limit: int = 10,
    graph_id: str = "",
    search_filters: dict[str, Any] | None = None,
    bfs_origin_node_uuids: list[str] | None = None,
    center_node_uuid: str = "",
    mmr_lambda: float | None = None,
    reranker: str = "",
    request_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Find graph entities relevant to a natural-language query.

    Best for entity discovery when the model needs candidate skills/concepts
    before taking deeper actions. Use search_filters when the request needs
    a specific check instead of broad retrieval.

    Limit: request 1-10 results only; higher values are capped to 10.

    Useful filters:
      node_labels: include only nodes with these labels.
      exclude_node_labels: remove nodes with these labels.
      property_filters: check node/edge attributes by property_name,
        comparison_operator, and property_value. Operators include =, <>, >,
        <, >=, <=, IS NULL, IS NOT NULL, and CONTAINS.
      created_at, valid_at, invalid_at, expired_at: 2D date filter arrays.
        Inner lists are AND; outer lists are OR.
      edge_types, exclude_edge_types, edge_uuids: mainly for edge/fact searches.
      episode_metadata_filters: AND/OR metadata predicates for source episodes.

    Search tuning priority:
      High: search_filters. Use for explicit checks like label/type/date/property.
      High: bfs_origin_node_uuids. Passing this list is BFS graph search:
        Zep starts breadth-first traversal from these known node UUIDs instead
        of relying only on semantic/BM25 seeds. Example ["node-123", "node-456"].
      Medium: center_node_uuid with reranker="node_distance". Use when ranking
        results around a known entity node. Example center_node_uuid="node-123".
      Low: reranker. Leave empty unless the task needs a specific ranking
        strategy. Examples: "rrf" (default), "mmr", "node_distance",
        "episode_mentions", "cross_encoder".
      Low: mmr_lambda. Only use with reranker="mmr"; values are 0.0-1.0.
        Example 0.7 balances relevance with diversity.
      Operational only: request_options. Use for SDK behavior, not search
        meaning. Example {"timeout_in_seconds": 30, "max_retries": 2}.

    Args:
        query: Natural-language search text (semantic + keyword matching).
        limit: Maximum number of node hits to return. Zep calls are capped to 10.
        graph_id: Target graph ID; falls back to default configured graph when empty.
        search_filters: Optional Zep SearchFilters dict. Examples:
          {"node_labels": ["Person"]}
          {"property_filters": [{"property_name": "status",
            "comparison_operator": "=", "property_value": "active"}]}
          {"created_at": [[{"comparison_operator": ">=",
            "date": "2026-01-01T00:00:00Z"}]]}
        bfs_origin_node_uuids: Optional origin node UUIDs for BFS search.
            Supplying this parameter alone is enough to seed BFS.
        center_node_uuid: Optional node UUID used by node_distance reranking.
        mmr_lambda: Optional MMR weight from 0.0 to 1.0; use with reranker="mmr".
        reranker: Optional ranking strategy: rrf, mmr, node_distance,
            episode_mentions, or cross_encoder.
        request_options: Optional SDK request settings. Supported keys:
            timeout_in_seconds and max_retries.

    Returns:
        A compact node result payload:
        {
          "graph_id": str,
          "nodes": [  # sorted by score descending
            {
              "uuid_": str,
              "name": str | None,
              "attributes": dict,
              "metadata": dict,
              "summary": str,
              "score": float | None,
            }
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
        limit=_zep_search_limit(limit),
        search_filters=_zep_search_filters(search_filters),
        bfs_origin_node_uuids=bfs_origin_node_uuids,
        center_node_uuid=center_node_uuid.strip() or None,
        mmr_lambda=_zep_mmr_lambda(mmr_lambda),
        reranker=_zep_reranker(reranker),
        request_options=_zep_request_options(request_options),
    )
    nodes = sorted(
        [trim_node_fields(node) for node in (getattr(response, "nodes", None) or []) if node],
        key=_graph_result_score,
        reverse=True,
    )
    return {"graph_id": resolved_graph_id, "nodes": nodes, "count": len(nodes)}


def search_edges(
    query: str,
    limit: int = 10,
    graph_id: str = "",
    search_filters: dict[str, Any] | None = None,
    bfs_origin_node_uuids: list[str] | None = None,
    center_node_uuid: str = "",
    mmr_lambda: float | None = None,
    reranker: str = "",
    request_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Find relationship facts relevant to a natural-language query.

    Best when the model needs relation/evidence-level context (facts and links)
    instead of only entity candidates. Use search_filters for targeted checks,
    such as only a certain relationship type, time range, attribute value, or
    source episode metadata.

    Limit: request 1-10 results only; higher values are capped to 10.

    Useful filters:
      edge_types: include only these relationship/fact types.
      exclude_edge_types: remove these relationship/fact types.
      edge_uuids: restrict to known edge UUIDs.
      property_filters: check edge/node attributes by property_name,
        comparison_operator, and property_value. Operators include =, <>, >,
        <, >=, <=, IS NULL, IS NOT NULL, and CONTAINS.
      created_at, valid_at, invalid_at, expired_at: 2D date filter arrays.
        Inner lists are AND; outer lists are OR.
      node_labels, exclude_node_labels: constrain the attached entity labels.
      episode_metadata_filters: AND/OR metadata predicates for source episodes.

    Search tuning priority:
      High: search_filters. Use for explicit checks like relationship type,
        source metadata, date validity, or property values.
      High: bfs_origin_node_uuids. Passing this list is BFS graph search:
        Zep starts breadth-first traversal from these known node UUIDs instead
        of relying only on semantic/BM25 seeds. Example ["node-123", "node-456"].
      Medium: center_node_uuid with reranker="node_distance". Use when facts
        should be ranked by closeness to a known entity node.
      Low: reranker. Leave empty unless the task needs a specific ranking
        strategy. Examples: "rrf" (default), "mmr", "node_distance",
        "episode_mentions", "cross_encoder".
      Low: mmr_lambda. Only use with reranker="mmr"; values are 0.0-1.0.
        Example 0.7 balances relevance with diversity.
      Operational only: request_options. Use for SDK behavior, not search
        meaning. Example {"timeout_in_seconds": 30, "max_retries": 2}.

    Args:
        query: Natural-language search text.
        limit: Maximum number of edge hits to return. Zep calls are capped to 10.
        graph_id: Target graph ID; falls back to default configured graph when empty.
        search_filters: Optional Zep SearchFilters dict. Examples:
          {"edge_types": ["WORKS_AT"]}
          {"property_filters": [{"property_name": "confidence",
            "comparison_operator": ">=", "property_value": 0.8}]}
          {"valid_at": [[{"comparison_operator": "<=",
            "date": "2026-04-01T00:00:00Z"}]]}
        bfs_origin_node_uuids: Optional origin node UUIDs for BFS search.
            Supplying this parameter alone is enough to seed BFS.
        center_node_uuid: Optional node UUID used by node_distance reranking.
        mmr_lambda: Optional MMR weight from 0.0 to 1.0; use with reranker="mmr".
        reranker: Optional ranking strategy: rrf, mmr, node_distance,
            episode_mentions, or cross_encoder.
        request_options: Optional SDK request settings. Supported keys:
            timeout_in_seconds and max_retries.

    Returns:
        {
          "graph_id": str,
          "edges": [edge_dict, ...],  # sorted by score descending
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
        limit=_zep_search_limit(limit),
        search_filters=_zep_search_filters(search_filters),
        bfs_origin_node_uuids=bfs_origin_node_uuids,
        center_node_uuid=center_node_uuid.strip() or None,
        mmr_lambda=_zep_mmr_lambda(mmr_lambda),
        reranker=_zep_reranker(reranker),
        request_options=_zep_request_options(request_options),
    )
    edges = sorted(
        [to_plain_dict(edge) for edge in (getattr(response, "edges", None) or []) if edge],
        key=_graph_result_score,
        reverse=True,
    )
    return {"graph_id": resolved_graph_id, "edges": edges, "count": len(edges)}


def search_episodes(
    query: str,
    limit: int = 10,
    graph_id: str = "",
    search_filters: dict[str, Any] | None = None,
    bfs_origin_node_uuids: list[str] | None = None,
    center_node_uuid: str = "",
    mmr_lambda: float | None = None,
    reranker: str = "",
    request_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Find source episodes relevant to a natural-language query.

    Best when the model needs the original memory/source text instead of only
    extracted entities or relationship facts.

    Limit: request 1-10 results only; higher values are capped to 10.

    Search tuning priority:
      High: search_filters. Use episode_metadata_filters when the request
        needs source metadata checks.
      High: bfs_origin_node_uuids. Passing this list is BFS graph search:
        Zep starts breadth-first traversal from these known node UUIDs instead
        of relying only on semantic/BM25 seeds. Example ["node-123", "node-456"].
      Medium: center_node_uuid with reranker="node_distance". Use when episodes
        should be ranked by closeness to a known entity node.
      Low: reranker. Leave empty unless the task needs a specific ranking
        strategy. Examples: "rrf" (default), "mmr", "node_distance",
        "episode_mentions", "cross_encoder".
      Low: mmr_lambda. Only use with reranker="mmr"; values are 0.0-1.0.
        Example 0.7 balances relevance with diversity.
      Operational only: request_options. Use for SDK behavior, not search
        meaning. Example {"timeout_in_seconds": 30, "max_retries": 2}.

    Args:
        query: Natural-language search text.
        limit: Maximum number of episode hits to return. Zep calls are capped to 10.
        graph_id: Target graph ID; falls back to default configured graph when empty.
        search_filters: Optional Zep SearchFilters dict. Example:
          {"episode_metadata_filters": {"type": "and", "filters": [
            {"property_name": "source", "comparison_operator": "=", "property_value": "chat"}
          ]}}
        bfs_origin_node_uuids: Optional origin node UUIDs for BFS search.
            Supplying this parameter alone is enough to seed BFS.
        center_node_uuid: Optional node UUID used by node_distance reranking.
        mmr_lambda: Optional MMR weight from 0.0 to 1.0; use with reranker="mmr".
        reranker: Optional ranking strategy: rrf, mmr, node_distance,
            episode_mentions, or cross_encoder.
        request_options: Optional SDK request settings. Supported keys:
            timeout_in_seconds and max_retries.

    Returns:
        {
          "graph_id": str,
          "episodes": [episode_dict, ...],  # sorted by score descending
          "count": int
        }
    """
    client = _client()
    resolved_graph_id = client.resolve_graph_id(graph_id)
    if not query.strip() or not resolved_graph_id:
        return {"graph_id": resolved_graph_id, "episodes": [], "count": 0}
    response = client.client.graph.search(
        query=query.strip(),
        graph_id=resolved_graph_id,
        scope="episodes",
        limit=_zep_search_limit(limit),
        search_filters=_zep_search_filters(search_filters),
        bfs_origin_node_uuids=bfs_origin_node_uuids,
        center_node_uuid=center_node_uuid.strip() or None,
        mmr_lambda=_zep_mmr_lambda(mmr_lambda),
        reranker=_zep_reranker(reranker),
        request_options=_zep_request_options(request_options),
    )
    episodes = sorted(
        [
            _trim_episode_fields(episode)
            for episode in (getattr(response, "episodes", None) or [])
            if episode
        ],
        key=_graph_result_score,
        reverse=True,
    )
    return {"graph_id": resolved_graph_id, "episodes": episodes, "count": len(episodes)}


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
    edges = sorted(
        [
            to_plain_dict(edge)
            for edge in extract_items(response, preferred_keys=("edges",))
            if edge
        ],
        key=_graph_result_score,
        reverse=True,
    )
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
          "node": {"uuid_", "name", "attributes", "metadata", "summary", "score"} | None,
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


def search_around_node(
    node_uuid: str,
    query: str = "",
    limit: int = 10,
    graph_id: str = "",
) -> dict[str, Any]:
    """Build a neighborhood context bundle around a node.

    Combines direct node lookup, connected edges, and related node/edge search
    into one response for routing/reasoning steps.

    Limit: request 1-10 related results only; higher values are capped to 10.

    Args:
        node_uuid: Anchor node UUID.
        query: Optional override query for related search. If empty, uses node-derived text.
        limit: Maximum results for related node/edge searches. Zep calls are capped to 10.
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

