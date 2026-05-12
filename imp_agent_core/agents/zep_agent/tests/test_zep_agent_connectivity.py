# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
from pathlib import Path

import httpx
import pytest
from dotenv import load_dotenv
from imp_agent_core.agents.zep_agent.tools.zep_tools import search_nodes


def _load_zep_env() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    load_dotenv(repo_root / "agents" / "zep_agent" / ".env")


def test_zep_connectivity_simple_search() -> None:
    _load_zep_env()
    api_key = (os.getenv("ZEP_API_KEY") or "").strip()
    graph_id = (os.getenv("GRAPH_ID") or "").strip()
    if not api_key or not graph_id:
        pytest.skip("ZEP_API_KEY or GRAPH_ID is not configured for connectivity test.")

    try:
        result = search_nodes(query="韩立", limit=1, graph_id=graph_id)
    except (httpx.ProxyError, httpx.ConnectError, httpx.TimeoutException) as exc:
        pytest.skip(f"Network-restricted environment prevented Zep connectivity check: {exc}")
    nodes = result.get("nodes")
    assert isinstance(nodes, list)


if __name__ == "__main__":
    test_zep_connectivity_simple_search()