# SPDX-License-Identifier: Apache-2.0
"""Runtime selection for A2A integrations."""

from __future__ import annotations

import os
from enum import StrEnum

A2A_RUNTIME_ENV = "AGENT_CORE_A2A_RUNTIME"


class A2aRuntime(StrEnum):
    LOCAL = "local"
    VERTEXAI = "vertexai"


def configured_a2a_runtime() -> A2aRuntime:
    value = os.getenv(A2A_RUNTIME_ENV, A2aRuntime.LOCAL.value).strip().lower()
    if value in {"vertex", "vertex_ai", "vertexai"}:
        return A2aRuntime.VERTEXAI
    return A2aRuntime.LOCAL


def vertex_a2a_enabled() -> bool:
    return configured_a2a_runtime() == A2aRuntime.VERTEXAI


def require_vertex_a2a_runtime() -> None:
    if configured_a2a_runtime() == A2aRuntime.VERTEXAI:
        return
    raise RuntimeError(
        f"Vertex A2A runtime is disabled. Set {A2A_RUNTIME_ENV}=vertexai to use it."
    )
