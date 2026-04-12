"""Reusable local A2A orchestration helpers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
from typing import Any, Mapping
from uuid import uuid4

from starlette.requests import Request


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


def _build_message_payload(
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


def _build_post_request(payload: Mapping[str, Any]) -> Request:
    scope = {
        "type": "http",
        "http_version": "1.1",
        "method": "POST",
        "headers": [(b"content-type", b"application/json")],
    }
    body = json.dumps(dict(payload)).encode("utf-8")
    state = {"delivered": False}

    async def receive() -> dict[str, Any]:
        if state["delivered"]:
            return {"type": "http.disconnect"}
        state["delivered"] = True
        return {"type": "http.request", "body": body, "more_body": False}

    return Request(scope, receive=receive)


def _build_get_task_request(task_id: str) -> Request:
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


def _extract_task_id(send_response: Any) -> str | None:
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
    a2a_agent: Any,
    message_text: str,
    mode: OrchestrationMode,
    metadata: Mapping[str, Any] | None = None,
    context: Any = None,
    include_authenticated_card: bool | None = None,
    fetch_task_response: bool | None = None,
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

    payload = _build_message_payload(message_text=message_text, metadata=metadata)
    post_request = _build_post_request(payload)
    send_response = await a2a_agent.on_message_send(request=post_request, context=context)

    task_response = None
    if fetch_task:
        task_id = _extract_task_id(send_response)
        if not task_id:
            raise ValueError("A2A send response does not include task.id")
        get_request = _build_get_task_request(task_id)
        task_response = await a2a_agent.on_get_task(request=get_request, context=context)

    return A2AFlowResult(
        mode=mode,
        card_response=card_response,
        send_response=send_response,
        task_response=task_response,
    )
