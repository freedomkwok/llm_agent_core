# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

pytest.importorskip("a2a")
pytest.importorskip("google.adk")

from agents.zep_agent import _env as zep_env
from agents.zep_agent.tools import zep_helper, zep_tools


class _FakeGraphNodeApi:
    def get(self, *, uuid_: str):
        return {
            "uuid": uuid_,
            "name": "Node Alpha",
            "attributes": {"kind": "skill"},
            "metadata": {"skill_id": "skill.alpha"},
            "embedding": [0.1, 0.2, 0.3],
            "summary": "long summary",
            "score": 0.87,
        }


class _FakeGraphApi:
    def __init__(self) -> None:
        self.node = _FakeGraphNodeApi()
        self.search_calls: list[dict] = []

    def search(
        self,
        *,
        query: str,
        graph_id: str,
        scope: str,
        limit: int,
        search_filters=None,
        bfs_origin_node_uuids=None,
        center_node_uuid=None,
        mmr_lambda=None,
        reranker=None,
        request_options=None,
    ):
        self.search_calls.append(
            {
                "query": query,
                "graph_id": graph_id,
                "scope": scope,
                "limit": limit,
                "search_filters": search_filters,
                "bfs_origin_node_uuids": bfs_origin_node_uuids,
                "center_node_uuid": center_node_uuid,
                "mmr_lambda": mmr_lambda,
                "reranker": reranker,
                "request_options": request_options,
            }
        )
        return SimpleNamespace(
            nodes=[
                {
                    "uuid": "node-2",
                    "name": "Node Beta",
                    "attributes": {"kind": "skill"},
                    "metadata": {"skill_id": "skill.beta"},
                    "embedding": [0.3, 0.2, 0.1],
                    "summary": "lower score summary",
                    "score": 0.51,
                },
                {
                    "uuid": "node-1",
                    "name": "Node Alpha",
                    "attributes": {"kind": "skill"},
                    "metadata": {"skill_id": "skill.alpha"},
                    "embedding": [0.1, 0.2, 0.3],
                    "summary": "long summary",
                    "score": 0.92,
                }
            ],
            edges=[],
            episodes=[
                {
                    "uuid": "episode-2",
                    "content": "less relevant source",
                    "created_at": "2026-01-02T00:00:00Z",
                    "metadata": {"source": "chat"},
                    "score": 0.2,
                },
                {
                    "uuid": "episode-1",
                    "content": "most relevant source",
                    "created_at": "2026-01-01T00:00:00Z",
                    "metadata": {"source": "chat"},
                    "score": 0.9,
                },
            ],
        )


class _FakeClient:
    def __init__(self) -> None:
        self.client = SimpleNamespace(graph=_FakeGraphApi())

    def resolve_graph_id(self, explicit_graph_id: str | None = None) -> str:
        return (explicit_graph_id or "graph-1").strip()

    def search_graph(self, **kwargs):
        return self.client.graph.search(**kwargs)


class _FakeZepCloud:
    def __init__(self, *, api_key: str) -> None:
        self.api_key = api_key


class _FakeOracleGraphApi:
    def __init__(self) -> None:
        self.search_calls: list[dict] = []

    def search(self, **kwargs):
        self.search_calls.append(kwargs)
        return SimpleNamespace(nodes=[], edges=[])


class _FakeOraclePGClient:
    calls: list[dict] = []
    last_client = None

    @classmethod
    def from_connection(cls, **kwargs):
        cls.calls.append(kwargs)
        cls.last_client = SimpleNamespace(graph=_FakeOracleGraphApi())
        return cls.last_client


def test_zep_tool_client_uses_zep_cloud_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ZEP_API_KEY", "cloud-key")
    monkeypatch.delenv("ZEP_CLIENT_BACKEND", raising=False)
    monkeypatch.delenv("GRAPHDB_DSN", raising=False)
    monkeypatch.delenv("GRAPHDB_USER", raising=False)
    monkeypatch.delenv("GRAPHDB_PASSWORD", raising=False)
    monkeypatch.setattr(zep_helper, "Zep", _FakeZepCloud)

    client = zep_helper.ZepToolClient(default_graph_id="graph-1")

    assert client.backend == "zep_cloud"
    assert isinstance(client.client, _FakeZepCloud)
    assert client.client.api_key == "cloud-key"


def test_zep_tool_client_defaults_to_oracle_pg_when_graphdb_env_is_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _FakeOraclePGClient.calls.clear()
    monkeypatch.delenv("ZEP_CLIENT_BACKEND", raising=False)
    monkeypatch.setenv("GRAPHDB_DSN", "graphdb-dsn")
    monkeypatch.setenv("GRAPHDB_USER", "graphdb-user")
    monkeypatch.setenv("GRAPHDB_PASSWORD", "graphdb-password")
    monkeypatch.delenv("GRAPH_ID", raising=False)
    monkeypatch.delenv("ORACLEPG_GRAPH_ID", raising=False)
    monkeypatch.setattr(
        zep_helper,
        "_graphiti_oracle_pg_client_class",
        lambda: _FakeOraclePGClient,
    )

    client = zep_helper.ZepToolClient()

    assert client.backend == "oraclepg"
    assert _FakeOraclePGClient.calls[-1]["dsn"] == "graphdb-dsn"
    assert _FakeOraclePGClient.calls[-1]["user"] == "graphdb-user"
    assert _FakeOraclePGClient.calls[-1]["password"] == "graphdb-password"
    assert _FakeOraclePGClient.calls[-1]["graph_id"] == "GRAPHITI"


def test_zep_bootstrap_defaults_backend_to_oracle_pg_for_graphdb_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ZEP_CLIENT_BACKEND", raising=False)
    monkeypatch.setenv("GRAPHDB_DSN", "graphdb-dsn")
    monkeypatch.setenv("GRAPHDB_USER", "graphdb-user")
    monkeypatch.setenv("GRAPHDB_PASSWORD", "graphdb-password")

    zep_env.bootstrap_env()

    assert os.environ["ZEP_CLIENT_BACKEND"] == "OraclePG"


def test_zep_tool_client_uses_oracle_pg_env(monkeypatch: pytest.MonkeyPatch) -> None:
    _FakeOraclePGClient.calls.clear()
    monkeypatch.setenv("ZEP_CLIENT_BACKEND", "OraclePG")
    monkeypatch.setenv("ORACLEPG_DSN", "oracle-dsn")
    monkeypatch.setenv("ORACLEPG_USER", "oracle-user")
    monkeypatch.setenv("ORACLEPG_PASSWORD", "oracle-password")
    monkeypatch.setenv("GRAPH_ID", "graph-1")
    monkeypatch.setenv("ORACLEPG_MAX_COROUTINES", "8")
    monkeypatch.setenv("ORACLEPG_LOG_QUERIES", "true")
    monkeypatch.setattr(
        zep_helper,
        "_graphiti_oracle_pg_client_class",
        lambda: _FakeOraclePGClient,
    )

    client = zep_helper.ZepToolClient()

    assert client.backend == "oraclepg"
    assert _FakeOraclePGClient.calls == [
        {
            "dsn": "oracle-dsn",
            "user": "oracle-user",
            "password": "oracle-password",
            "graph_id": "graph-1",
            "max_coroutines": 8,
            "log_queries": True,
        }
    ]


def test_zep_tool_client_oracle_pg_accepts_graphdb_env_fallbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _FakeOraclePGClient.calls.clear()
    monkeypatch.setenv("ZEP_CLIENT_BACKEND", "OraclePG")
    monkeypatch.setenv("GRAPHDB_DSN", "graphdb-dsn")
    monkeypatch.setenv("GRAPHDB_USER", "graphdb-user")
    monkeypatch.setenv("GRAPHDB_PASSWORD", "graphdb-password")
    monkeypatch.setenv("GRAPH_ID", "graph-1")
    monkeypatch.delenv("ORACLEPG_DSN", raising=False)
    monkeypatch.delenv("ORACLEPG_USER", raising=False)
    monkeypatch.delenv("ORACLEPG_PASSWORD", raising=False)
    monkeypatch.setattr(
        zep_helper,
        "_graphiti_oracle_pg_client_class",
        lambda: _FakeOraclePGClient,
    )

    zep_helper.ZepToolClient()

    assert _FakeOraclePGClient.calls[-1]["dsn"] == "graphdb-dsn"
    assert _FakeOraclePGClient.calls[-1]["user"] == "graphdb-user"
    assert _FakeOraclePGClient.calls[-1]["password"] == "graphdb-password"


def test_zep_tool_client_oracle_pg_search_uses_graphiti_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _FakeOraclePGClient.calls.clear()
    monkeypatch.setenv("ZEP_CLIENT_BACKEND", "OraclePG")
    monkeypatch.setenv("ORACLEPG_DSN", "oracle-dsn")
    monkeypatch.setenv("ORACLEPG_USER", "oracle-user")
    monkeypatch.setenv("ORACLEPG_PASSWORD", "oracle-password")
    monkeypatch.setenv("GRAPH_ID", "graph-1")
    monkeypatch.setattr(
        zep_helper,
        "_graphiti_oracle_pg_client_class",
        lambda: _FakeOraclePGClient,
    )

    client = zep_helper.ZepToolClient()
    client.search_graph(
        query="alpha",
        graph_id="graph-1",
        scope="nodes",
        limit=3,
        search_filters={"node_labels": ["Entity"]},
        bfs_origin_node_uuids=["node-1"],
        center_node_uuid="node-2",
        mmr_lambda=0.5,
        reranker="cross_encoder",
        request_options={"timeout_in_seconds": 5},
    )

    assert _FakeOraclePGClient.last_client.graph.search_calls == [
        {
            "query": "alpha",
            "graph_id": "graph-1",
            "scope": "nodes",
            "limit": 3,
            "bfs_origin_node_uuids": ["node-1"],
            "center_node_uuid": "node-2",
            "reranker": "cross_encoder",
            "search_filter": zep_helper._graphiti_search_filter({"node_labels": ["Entity"]}),
        }
    ]


def test_search_nodes_and_get_node_by_id_trim_heavy_node_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(zep_tools, "_client", lambda: _FakeClient())

    search_result = zep_tools.search_nodes(query="alpha", graph_id="graph-1", limit=2)
    assert search_result["count"] == 2
    assert search_result["nodes"][0] == {
        "uuid_": "node-1",
        "name": "Node Alpha",
        "attributes": {"kind": "skill"},
        "metadata": {"skill_id": "skill.alpha"},
        "summary": "long summary",
        "score": 0.92,
    }
    assert search_result["nodes"][1]["uuid_"] == "node-2"
    assert "embedding" not in search_result["nodes"][0]

    node_result = zep_tools.get_node_by_id("node-1")
    assert node_result["node"] == {
        "uuid_": "node-1",
        "name": "Node Alpha",
        "attributes": {"kind": "skill"},
        "metadata": {"skill_id": "skill.alpha"},
        "summary": "long summary",
        "score": 0.87,
    }
    assert "embedding" not in node_result["node"]


def test_search_episodes_uses_episode_scope_and_sorts_by_score(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = _FakeClient()
    monkeypatch.setattr(zep_tools, "_client", lambda: fake_client)

    search_result = zep_tools.search_episodes(
        query="source",
        graph_id="graph-1",
        limit=2,
        bfs_origin_node_uuids=["node-1"],
        reranker="mmr",
        mmr_lambda=0.7,
        request_options={"timeout_in_seconds": 30, "additional_headers": {"skip": "me"}},
    )

    assert search_result["count"] == 2
    assert search_result["episodes"][0]["uuid_"] == "episode-1"
    assert search_result["episodes"][1]["uuid_"] == "episode-2"
    assert fake_client.client.graph.search_calls[-1]["scope"] == "episodes"
    assert fake_client.client.graph.search_calls[-1]["bfs_origin_node_uuids"] == ["node-1"]
    assert fake_client.client.graph.search_calls[-1]["request_options"] == {
        "timeout_in_seconds": 30
    }


def test_search_limit_is_capped_to_zep_maximum(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_client = _FakeClient()
    monkeypatch.setattr(zep_tools, "_client", lambda: fake_client)

    zep_tools.search_nodes(query="alpha", graph_id="graph-1", limit=100)
    zep_tools.search_edges(query="alpha", graph_id="graph-1", limit=0)

    assert fake_client.client.graph.search_calls[-2]["limit"] == 10
    assert fake_client.client.graph.search_calls[-1]["limit"] == 1
