"""Environment bootstrap utilities for planning agent examples."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


def _clean(raw: str) -> str:
    return raw.strip().strip('"')


def bootstrap_env() -> None:
    """Load root .env and map MAP_LANGFUSE_* to Langfuse defaults."""
    repo_root = Path(__file__).resolve().parents[2]
    load_dotenv(repo_root / ".env")

    secret_key = os.getenv("MAP_LANGFUSE_SECRET_KEY")
    public_key = os.getenv("MAP_LANGFUSE_PUBLIC_KEY")
    base_url = os.getenv("MAP_LANGFUSE_BASE_URL")

    if secret_key:
        os.environ["LANGFUSE_SECRET_KEY"] = _clean(secret_key)
    if public_key:
        os.environ["LANGFUSE_PUBLIC_KEY"] = _clean(public_key)
    if base_url:
        os.environ["LANGFUSE_BASE_URL"] = _clean(base_url)
