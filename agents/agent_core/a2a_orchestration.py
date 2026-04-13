"""Reusable local A2A orchestration helpers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
import logging
import time
from typing import Any, Mapping
from uuid import uuid4

from starlette.requests import Request

logger = logging.getLogger(__name__)


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


def build_local_a2a_message_payload(
    *,
    message_text: str,
    metadata: Mapping[str, Any] | None = None,
    message_id: str | None = None,
) -> dict[str, Any]:
    """Build local JSON payload for an A2A message send call."""
    return _build_message_payload(
        message_text=message_text,
        metadata=metadata,
        message_id=message_id,
    )


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


def build_local_a2a_post_request(payload: Mapping[str, Any]) -> Request:
    """Build a Starlette POST request for local in-process A2A calls."""
    return _build_post_request(payload)


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


def build_local_a2a_get_task_request(task_id: str) -> Request:
    """Build a Starlette GET request for local task retrieval."""
    return _build_get_task_request(task_id)


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


def extract_local_a2a_task_id(send_response: Any) -> str | None:
    """Extract task id from local A2A send response."""
    return _extract_task_id(send_response)


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
    send_started = time.perf_counter()
    try:
        send_response = await a2a_agent.on_message_send(request=post_request, context=context)
    except Exception:  # noqa: BLE001
        logger.exception(
            "on_message_send failed mode=%s include_card=%s fetch_task=%s message_id=%s",
            mode.value,
            include_card,
            fetch_task,
            payload["message"].get("messageId"),
        )
        raise
    logger.debug(
        "on_message_send completed mode=%s include_card=%s fetch_task=%s elapsed_ms=%d has_task_id=%s",
        mode.value,
        include_card,
        fetch_task,
        int((time.perf_counter() - send_started) * 1000),
        bool(_extract_task_id(send_response)),
    )

    task_response = None
    task_id = _extract_task_id(send_response)
    if fetch_task:
        if not task_id:
            raise ValueError("A2A send response does not include task.id")
        get_request = _build_get_task_request(task_id)
        get_started = time.perf_counter()
        try:
            task_response = await a2a_agent.on_get_task(request=get_request, context=context)
        except Exception:  # noqa: BLE001
            logger.exception("on_get_task failed task_id=%s mode=%s", task_id, mode.value)
            raise
        logger.debug(
            "on_get_task completed task_id=%s mode=%s elapsed_ms=%d task_status=%s",
            task_id,
            mode.value,
            int((time.perf_counter() - get_started) * 1000),
            _extract_task_status(task_response),
        )

    return A2AFlowResult(
        mode=mode,
        card_response=card_response,
        send_response=send_response,
        task_response=task_response,
        task_id=task_id,
        task_status=_extract_task_status(task_response),
        final_text=_extract_final_text(task_response),
    )
