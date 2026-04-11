"""Zep Cloud graph examples: load credentials from .env, then fetch nodes and edges."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from zep_cloud import Zep


def _load_env() -> None:
    repo_root = Path(__file__).resolve().parent
    load_dotenv(repo_root / ".env")


def main() -> None:
    _load_env()
    api_key = os.environ.get("ZEP_API_KEY")
    if not api_key:
        raise SystemExit("Set ZEP_API_KEY in .env")

    client = Zep(api_key=api_key)

    # Set these to real ids from your Zep project.
    user_id = os.environ.get("ZEP_EXAMPLE_USER_ID", "replace-with-user-id")
    node_uuid = os.environ.get("ZEP_EXAMPLE_NODE_UUID", "replace-with-node-uuid")
    edge_uuid = os.environ.get("ZEP_EXAMPLE_EDGE_UUID", "replace-with-edge-uuid")

    # --- Node retrieval ---
    if user_id != "replace-with-user-id":
        node_search = client.graph.search(
            query="example topic",
            user_id=user_id,
            scope="nodes",
            limit=10,
        )
        print("node search results:", node_search.nodes)

        nodes_for_user = client.graph.node.get_by_user_id(user_id, limit=50)
        print("nodes for user (count):", len(nodes_for_user))

        edge_search = client.graph.search(
            query="example topic",
            user_id=user_id,
            scope="edges",
            limit=10,
        )
        print("edge search results:", edge_search.edges)

        edges_for_user = client.graph.edge.get_by_user_id(user_id, limit=50)
        print("edges for user (count):", len(edges_for_user))
    else:
        print("Set ZEP_EXAMPLE_USER_ID or edit user_id to run search / list-by-user examples.")

    if node_uuid != "replace-with-node-uuid":
        node = client.graph.node.get(uuid_=node_uuid)
        print("node by uuid:", node)
        entity_edges = client.graph.node.get_edges(node_uuid=node_uuid)
        print("entity edges for node:", entity_edges)
    else:
        print("Set ZEP_EXAMPLE_NODE_UUID or edit node_uuid for node.get / get_edges.")

    if edge_uuid != "replace-with-edge-uuid":
        edge = client.graph.edge.get(uuid_=edge_uuid)
        print("edge by uuid:", edge)
    else:
        print("Set ZEP_EXAMPLE_EDGE_UUID or edit edge_uuid for edge.get.")


if __name__ == "__main__":
    main()
