"""Reusable local A2A orchestration helpers."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import Enum
import json
import time
from typing import Any, Mapping
from uuid import uuid4

from starlette.requests import Request
from vertexai.preview.reasoning_engines import A2aAgent


class OrchestrationMode(str, Enum):
    """High-level orchestration mode for local A2A calls."""

    HOST_DRIVEN = "host_driven"
    AGENT_INTERNAL = "agent_internal"


@dataclass
class A2AFlowResult:
    """Result object for local A2A orchestration."""

    mode: OrchestrationMode
    send_response: Any
    card_response: Any | None = None
    task_response: Any | None = None
    task_id: str | None = None
    task_status: str | None = None
    final_text: str | None = None


# Align with a2a TaskState terminal set used by DefaultRequestHandler.
_TERMINAL_TASK_STATES = frozenset({"completed", "canceled", "failed", "rejected"})


def _is_terminal_task_status(status: str | None) -> bool:
    return status is not None and status in _TERMINAL_TASK_STATES


def _extract_task_status(task_response: Any) -> str | None:
    if not isinstance(task_response, Mapping):
        return None
    task = task_response.get("task")
    if not isinstance(task, Mapping):
        return None
    raw_status = task.get("status")
    if isinstance(raw_status, Mapping):
        raw_status = raw_status.get("state")
    if isinstance(raw_status, str) and raw_status:
        return raw_status
    return None


def _extract_text_from_content_items(content: Any) -> list[str]:
    texts: list[str] = []
    if not isinstance(content, list):
        return texts
    for item in content:
        if isinstance(item, Mapping):
            text = item.get("text")
            if isinstance(text, str) and text:
                texts.append(text)
    return texts


def _extract_final_text(task_response: Any) -> str | None:
    if not isinstance(task_response, Mapping):
        return None
    task = task_response.get("task")
    if not isinstance(task, Mapping):
        return None
    artifacts = task.get("artifacts")
    if not isinstance(artifacts, list):
        return None
    parts: list[str] = []
    for artifact in artifacts:
        if not isinstance(artifact, Mapping):
            continue
        parts.extend(_extract_text_from_content_items(artifact.get("parts")))
    if not parts:
        return None
    return "\n".join(parts)


def build_message_payload(
    *,
    message_text: str,
    metadata: Mapping[str, Any] | None = None,
    message_id: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "message": {
            "messageId": message_id or f"local-message-{uuid4().hex}",
            "content": [{"text": message_text}],
            "role": "ROLE_USER",
        }
    }
    if metadata:
        payload["message"]["metadata"] = dict(metadata)
    return payload


def build_post_request(payload: Mapping[str, Any]) -> Request:
    scope = {
        "type": "http",
        "http_version": "1.1",
        "method": "POST",
        "headers": [(b"content-type", b"application/json")],
    }
    body = json.dumps(dict(payload)).encode("utf-8")

    async def receive() -> dict[str, Any]:
        # Mirror examples/a2a/05_test_local_calls.py local receive behavior.
        return {"type": "http.request", "body": body, "more_body": False}

    return Request(scope, receive=receive)


def build_get_task_request(task_id: str) -> Request:
    scope = {
        "type": "http",
        "http_version": "1.1",
        "method": "GET",
        "headers": [],
        "query_string": b"",
        "path_params": {"id": task_id},
    }

    async def receive() -> dict[str, Any]:
        return {"type": "http.disconnect"}

    return Request(scope, receive=receive)


def extract_task_id(send_response: Any) -> str | None:
    if not isinstance(send_response, Mapping):
        return None
    task = send_response.get("task")
    if not isinstance(task, Mapping):
        return None
    raw_id = task.get("id")
    if isinstance(raw_id, str) and raw_id:
        return raw_id
    return None


async def run_local_a2a_orchestration(
    *,
    a2a_agent: A2aAgent,
    message_text: str,
    mode: OrchestrationMode,
    metadata: Mapping[str, Any] | None = None,
    context: Any = None,
    include_authenticated_card: bool | None = None,
    fetch_task_response: bool | None = None,
    task_poll_timeout_sec: float = 300.0,
    task_poll_interval_sec: float = 5.0,
) -> A2AFlowResult:
    """Run local A2A flow in host-driven or agent-internal mode."""
    include_card = (
        include_authenticated_card
        if include_authenticated_card is not None
        else mode == OrchestrationMode.HOST_DRIVEN
    )
    fetch_task = (
        fetch_task_response
        if fetch_task_response is not None
        else mode == OrchestrationMode.HOST_DRIVEN
    )

    card_response = None
    if include_card:
        card_response = await a2a_agent.handle_authenticated_agent_card(
            request=None,
            context=context,
        )

    payload = build_message_payload(message_text=message_text, metadata=metadata)
    post_request = build_post_request(payload)
    send_response = await a2a_agent.on_message_send(request=post_request, context=context)

    task_response = None
    task_id = extract_task_id(send_response)
    if fetch_task:
        if not task_id:
            raise ValueError("A2A send response does not include task.id")
        deadline = time.monotonic() + task_poll_timeout_sec
        task_response = None
        while True:
            get_request = build_get_task_request(task_id)
            task_response = await a2a_agent.on_get_task(
                request=get_request,
                context=context,
            )
            status = _extract_task_status(task_response)
            if _is_terminal_task_status(status):
                break
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"A2A task {task_id!r} did not reach a terminal state within "
                    f"{task_poll_timeout_sec}s (last status: {status!r})"
                )
            await asyncio.sleep(task_poll_interval_sec)

    return A2AFlowResult(
        mode=mode,
        card_response=card_response,
        send_response=send_response,
        task_response=task_response,
        task_id=task_id,
        task_status=_extract_task_status(task_response),
        final_text=_extract_final_text(task_response),
    )
