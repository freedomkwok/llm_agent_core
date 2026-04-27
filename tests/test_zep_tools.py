from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("a2a")
pytest.importorskip("google.adk")

from agents.zep_agent.tools import zep_tools


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

    def search(self, *, query: str, graph_id: str, scope: str, limit: int):
        del query, graph_id, scope, limit
        return SimpleNamespace(
            nodes=[
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
        )


class _FakeClient:
    def __init__(self) -> None:
        self.client = SimpleNamespace(graph=_FakeGraphApi())

    def resolve_graph_id(self, explicit_graph_id: str | None = None) -> str:
        return (explicit_graph_id or "graph-1").strip()


def test_search_nodes_and_get_node_by_id_trim_heavy_node_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(zep_tools, "_client", lambda: _FakeClient())

    search_result = zep_tools.search_nodes(query="alpha", graph_id="graph-1", limit=1)
    assert search_result["count"] == 1
    assert search_result["nodes"][0] == {
        "name": "Node Alpha",
        "attributes": {"kind": "skill"},
        "metadata": {"skill_id": "skill.alpha"},
        "summary": "long summary",
        "score": 0.92,
    }
    assert "embedding" not in search_result["nodes"][0]

    node_result = zep_tools.get_node_by_id("node-1")
    assert node_result["node"] == {
        "name": "Node Alpha",
        "attributes": {"kind": "skill"},
        "metadata": {"skill_id": "skill.alpha"},
        "summary": "long summary",
        "score": 0.87,
    }
    assert "embedding" not in node_result["node"]
