"""Load repo .env and map MAP_LANGFUSE_* → LANGFUSE_* for Langfuse SDK."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


def _strip_val(raw: str) -> str:
    return raw.strip().strip('"')


def apply_map_langfuse_env() -> None:
    """Use imp_agent_map project Langfuse keys when MAP_LANGFUSE_* are set."""
    env = os.environ
    sk = env.get("MAP_LANGFUSE_SECRET_KEY")
    pk = env.get("MAP_LANGFUSE_PUBLIC_KEY")
    base = env.get("MAP_LANGFUSE_BASE_URL")
    if sk:
        env["LANGFUSE_SECRET_KEY"] = _strip_val(sk)
    if pk:
        env["LANGFUSE_PUBLIC_KEY"] = _strip_val(pk)
    if base:
        env["LANGFUSE_BASE_URL"] = _strip_val(base)


def bootstrap_langfuse_from_repo_env() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    load_dotenv(repo_root / ".env")
    apply_map_langfuse_env()
