# SPDX-License-Identifier: Apache-2.0
"""Environment bootstrap utilities for zep agent."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


def _clean(raw: str) -> str:
    return raw.strip().strip('"')


def bootstrap_env() -> None:
    """Load zep-agent .env first, then repo root .env as fallback."""
    agent_dir = Path(__file__).resolve().parent
    repo_root = agent_dir.parents[1]

    # Prefer zep-agent-local environment values when present.
    load_dotenv(agent_dir / ".env")
    load_dotenv(repo_root / ".env", override=False)

    secret_key = os.getenv("MAP_LANGFUSE_SECRET_KEY")
    public_key = os.getenv("MAP_LANGFUSE_PUBLIC_KEY")
    base_url = os.getenv("MAP_LANGFUSE_BASE_URL")

    if secret_key:
        os.environ["LANGFUSE_SECRET_KEY"] = _clean(secret_key)
    if public_key:
        os.environ["LANGFUSE_PUBLIC_KEY"] = _clean(public_key)
    if base_url:
        os.environ["LANGFUSE_BASE_URL"] = _clean(base_url)

    graphdb_ready = all(
        (os.getenv(name) or "").strip()
        for name in ("GRAPHDB_DSN", "GRAPHDB_USER", "GRAPHDB_PASSWORD")
    )
    oraclepg_ready = all(
        (os.getenv(name) or "").strip()
        for name in ("ORACLEPG_DSN", "ORACLEPG_USER", "ORACLEPG_PASSWORD")
    )
    if not (os.getenv("ZEP_CLIENT_BACKEND") or "").strip() and (graphdb_ready or oraclepg_ready):
        os.environ["ZEP_CLIENT_BACKEND"] = "OraclePG"


__all__ = ["bootstrap_env"]

