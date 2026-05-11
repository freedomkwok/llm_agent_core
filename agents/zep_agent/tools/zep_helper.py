# SPDX-License-Identifier: Apache-2.0
"""Shared Zep helper for zep_agent tools."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from zep_cloud import Zep

ZEP_CLIENT_BACKEND_CLOUD = "zep_cloud"
ZEP_CLIENT_BACKEND_ORACLE_PG = "oraclepg"


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
    uuid_ = node_payload.get("uuid_") or node_payload.get("uuid")
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
        "uuid_": uuid_ if isinstance(uuid_, str) else "",
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
        backend: str | None = None,
    ) -> None:
        self.backend = _zep_client_backend(backend)
        self.default_graph_id = (default_graph_id or os.getenv("GRAPH_ID", "")).strip()
        self.client = (
            _oracle_pg_client(self.default_graph_id)
            if self.backend == ZEP_CLIENT_BACKEND_ORACLE_PG
            else _zep_cloud_client(api_key)
        )

    def resolve_graph_id(self, explicit_graph_id: str | None = None) -> str:
        return (explicit_graph_id or self.default_graph_id).strip()

    def search_graph(
        self,
        *,
        query: str,
        graph_id: str,
        scope: str,
        limit: int,
        search_filters: dict[str, Any] | None = None,
        bfs_origin_node_uuids: list[str] | None = None,
        center_node_uuid: str | None = None,
        mmr_lambda: float | None = None,
        reranker: str | None = None,
        request_options: dict[str, Any] | None = None,
    ) -> Any:
        if self.backend == ZEP_CLIENT_BACKEND_ORACLE_PG:
            del mmr_lambda, request_options
            kwargs: dict[str, Any] = {
                "query": query,
                "graph_id": graph_id,
                "scope": scope,
                "limit": limit,
            }
            if bfs_origin_node_uuids:
                kwargs["bfs_origin_node_uuids"] = bfs_origin_node_uuids
            if center_node_uuid:
                kwargs["center_node_uuid"] = center_node_uuid
            if reranker:
                kwargs["reranker"] = reranker
            if search_filters:
                kwargs["search_filter"] = _graphiti_search_filter(search_filters)
            return self.client.graph.search(**kwargs)

        return self.client.graph.search(
            query=query,
            graph_id=graph_id,
            scope=scope,
            limit=limit,
            search_filters=_zep_search_filters(search_filters),
            bfs_origin_node_uuids=bfs_origin_node_uuids,
            center_node_uuid=center_node_uuid,
            mmr_lambda=mmr_lambda,
            reranker=reranker,
            request_options=request_options,
        )


def _zep_client_backend(value: str | None = None) -> str:
    default_backend = (
        ZEP_CLIENT_BACKEND_ORACLE_PG if _oracle_pg_connection_configured() else ZEP_CLIENT_BACKEND_CLOUD
    )
    normalized = str(value or os.getenv("ZEP_CLIENT_BACKEND", "") or default_backend)
    normalized = normalized.strip().lower().replace("_", "").replace("-", "")
    if normalized in {"zep", "zepcloud", "cloud"}:
        return ZEP_CLIENT_BACKEND_CLOUD
    if normalized in {"oraclepg", "oracle"}:
        return ZEP_CLIENT_BACKEND_ORACLE_PG
    raise ValueError("ZEP_CLIENT_BACKEND must be one of: zep_cloud, OraclePG")


def _zep_cloud_client(api_key: str | None = None) -> Zep:
    resolved_api_key = (api_key or os.getenv("ZEP_API_KEY", "")).strip()
    if not resolved_api_key:
        raise ValueError("ZEP_API_KEY is not set")
    return Zep(api_key=resolved_api_key)


def _oracle_pg_connection_configured() -> bool:
    return bool(
        _env_first("ORACLEPG_DSN", "GRAPHDB_DSN", "ORACLE_DSN")
        and _env_first("ORACLEPG_USER", "GRAPHDB_USER", "ORACLE_USER")
        and _env_first("ORACLEPG_PASSWORD", "GRAPHDB_PASSWORD", "ORACLE_PASSWORD")
    )


def _env_first(*names: str) -> str:
    for name in names:
        value = os.getenv(name)
        if value and value.strip():
            return value.strip()
    return ""


def _optional_positive_int(*names: str) -> int | None:
    value = _env_first(*names)
    if not value:
        return None
    try:
        parsed = int(value)
    except ValueError:
        return None
    return parsed if parsed > 0 else None


def _env_bool(*names: str) -> bool:
    value = _env_first(*names).lower()
    return value in {"1", "true", "yes", "on"}


def _oracle_pg_client(default_graph_id: str) -> Any:
    dsn = _env_first("ORACLEPG_DSN", "GRAPHDB_DSN", "ORACLE_DSN")
    user = _env_first("ORACLEPG_USER", "GRAPHDB_USER", "ORACLE_USER")
    password = _env_first("ORACLEPG_PASSWORD", "GRAPHDB_PASSWORD", "ORACLE_PASSWORD")
    graph_id = _env_first("ORACLEPG_GRAPH_ID", "GRAPH_ID") or default_graph_id or "GRAPHITI"
    if not dsn or not user or not password:
        raise ValueError(
            "OraclePG backend requires ORACLEPG_DSN, ORACLEPG_USER, and ORACLEPG_PASSWORD "
            "(or GRAPHDB_DSN, GRAPHDB_USER, and GRAPHDB_PASSWORD)."
        )

    return _graphiti_oracle_pg_client_class().from_connection(
        dsn=dsn,
        user=user,
        password=password,
        graph_id=graph_id,
        max_coroutines=_optional_positive_int("ORACLEPG_MAX_COROUTINES", "ORACLE_MAX_COROUTINES"),
        log_queries=_env_bool("ORACLEPG_LOG_QUERIES", "ORACLE_LOG_QUERIES"),
    )


def _graphiti_oracle_pg_client_class() -> Any:
    from graphiti_client import GraphitiOraclePGClient

    return GraphitiOraclePGClient


def _zep_search_filters(raw_filters: dict[str, Any] | None) -> Any | None:
    if not raw_filters:
        return None
    from zep_cloud.types import SearchFilters

    return SearchFilters(**raw_filters)


def _graphiti_search_filter(raw_filters: dict[str, Any]) -> Any:
    from graphiti_core.search.search_filters import SearchFilters

    allowed_keys = {
        "node_labels",
        "edge_types",
        "valid_at",
        "invalid_at",
        "created_at",
        "expired_at",
        "edge_uuids",
        "property_filters",
    }
    return SearchFilters(
        **{key: value for key, value in raw_filters.items() if key in allowed_keys}
    )

