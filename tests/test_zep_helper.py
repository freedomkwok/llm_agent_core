# SPDX-License-Identifier: Apache-2.0
import asyncio
from types import SimpleNamespace

from imp_agent_core.agents.skill_route_agent.utils.zep_helper import (
    ZepQueryRequest,
    ZepSkillSearchComponent,
    fetch_edges_by_node_uuids,
)


class _FakeNodeApi:
    def __init__(self) -> None:
        self.edge_calls: list[str] = []
        self.episode_calls: list[str] = []

    def get_edges(self, *, node_uuid: str):
        self.edge_calls.append(node_uuid)
        edge = {
            "fact": f"{node_uuid} has capability",
            "attributes": {"edge_type": "HAS_CAPABILITY"},
        }
        return SimpleNamespace(edges=[edge])

    def get_episodes(self, *, node_uuid: str):
        self.episode_calls.append(node_uuid)
        episode = {"content": f"{node_uuid} episode details for routing context"}
        return SimpleNamespace(episodes=[episode])

    def get(self, *, uuid_: str):
        return {
            "uuid": uuid_,
            "name": f"Node {uuid_}",
            "summary": f"Summary {uuid_}",
            "metadata": {"skill_id": "skill.alpha", "skill_name": "Skill Alpha"},
        }


class _FakeGraphApi:
    def __init__(self, node_api: _FakeNodeApi) -> None:
        self.node = node_api

    def search(self, *, query: str, graph_id: str, scope: str, limit: int):
        del graph_id, limit
        if scope == "edges":
            return SimpleNamespace(
                edges=[
                    {
                        "fact": f"{query} edge fact",
                        "name": "RELATED_TO",
                    }
                ],
                nodes=[],
            )
        nodes = [
            {
                "uuid": "node-1",
                "name": "Skill node 1",
                "summary": "summary one",
                "metadata": {"skill_id": "skill.alpha", "skill_name": "Skill Alpha"},
            },
            {
                "uuid": "node-2",
                "name": "Skill node 2",
                "summary": "summary two",
                "metadata": {"skill_id": "skill.alpha", "skill_name": "Skill Alpha"},
            },
        ]
        return SimpleNamespace(nodes=nodes, edges=[])


class _FakeZepClient:
    def __init__(self) -> None:
        self.node_api = _FakeNodeApi()
        self.graph = _FakeGraphApi(node_api=self.node_api)


def _build_component() -> ZepSkillSearchComponent:
    component = object.__new__(ZepSkillSearchComponent)
    component.default_graph_id = "graph-1"
    component.client = _FakeZepClient()
    component.api_key = "fake"
    return component


def test_execute_query_groups_node_results_by_skill_id() -> None:
    component = _build_component()
    request = ZepQueryRequest(query="alpha", scope="nodes", limit=10, graph_id="graph-1")

    candidates = component.execute_query(request)

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.skill_id == "skill.alpha"
    assert candidate.name == "Skill Alpha"
    assert candidate.raw.get("node_count") == 2
    edge_facts = candidate.raw.get("edge_facts") or []
    assert len(edge_facts) == 2
    assert "node-1 has capability" in edge_facts
    assert "node-2 has capability" in edge_facts
    episode_previews = candidate.raw.get("episode_previews") or []
    assert len(episode_previews) == 2
    assert component.client.node_api.edge_calls == ["node-1", "node-2"]
    assert component.client.node_api.episode_calls == ["node-1", "node-2"]


def test_fetch_edges_by_node_uuids_runs_concurrently() -> None:
    component = _build_component()

    async def run():
        return await fetch_edges_by_node_uuids(
            client=component.client,
            node_uuids=["node-1", "node-2", ""],
            concurrency=2,
        )

    edges_by_uuid = asyncio.run(run())

    assert sorted(edges_by_uuid.keys()) == ["node-1", "node-2"]
    assert edges_by_uuid["node-1"][0]["fact"] == "node-1 has capability"
    assert edges_by_uuid["node-2"][0]["fact"] == "node-2 has capability"


def test_component_high_level_search_helpers() -> None:
    component = _build_component()

    nodes = component.search_nodes(query="alpha", graph_id="graph-1", limit=5)
    edges = component.search_edges(query="alpha", graph_id="graph-1", limit=5)
    around = component.search_around_node(node_uuid="node-1", graph_id="graph-1", limit=3)

    assert len(nodes) == 2
    assert len(edges) == 1
    assert edges[0]["fact"] == "alpha edge fact"
    assert around["node"]["uuid"] == "node-1"
    assert len(around["edges"]) == 1
    assert len(around["episodes"]) == 1
    assert len(around["related_nodes"]) == 2
    assert len(around["related_edges"]) == 1

