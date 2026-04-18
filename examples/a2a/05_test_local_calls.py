"""Step 5: local test calls for authenticated card, send, and get-task."""

from starlette.datastructures import State


import asyncio
import importlib.util
import json
from contextlib import nullcontext
from pathlib import Path
from uuid import uuid4

from langfuse import get_client
from starlette.requests import Request

from langfuse_map_env import bootstrap_langfuse_from_repo_env
from vertexai.preview.reasoning_engines import A2aAgent
# Project root: imp_agent_map/.env; MAP_LANGFUSE_* → LANGFUSE_* for Langfuse SDK
bootstrap_langfuse_from_repo_env()


def _load_local_agent_builder():
    path = Path(__file__).parent / "04_local_agent.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_local_a2a_agent


async def run_local_flow() -> None:
    # Create a root trace/observation before loading/bootstrapping the local A2A agent.
    client = get_client()
    try:
        root_cm = client.start_as_current_observation(
            name="run_local_flow",
            as_type="chain",
            input={"entrypoint": "examples/a2a/05_test_local_calls.py"},
        )
    except Exception:
        root_cm = nullcontext()

    with root_cm as root_obs:
        trace_id = str(getattr(root_obs, "trace_id", "") or uuid4().hex)
        parent_observation_id = str(getattr(root_obs, "id", "") or "")
        print(f"Langfuse trace_id: {trace_id}")
        if parent_observation_id:
            print(f"Langfuse root_observation_id: {parent_observation_id}")

        build_local_a2a_agent = _load_local_agent_builder()
        a2a_agent : A2aAgent = build_local_a2a_agent()

        # 1) authenticated agent card
        card_resp = await a2a_agent.handle_authenticated_agent_card(request=None, context=None)
        print("=== handle_authenticated_agent_card ===")
        print(card_resp)

        # 2) on_message_send
        message_data = {
            "message": {
                "messageId": "local-test-message-id",
                "content": [{"text": "What is the exchange rate from USD to EUR today?"}],
                "role": "ROLE_USER",
                "metadata": {
                    "trace_id": trace_id,
                    "parent_observation_id": parent_observation_id,
                    "user_id": "local-test-user",
                },
            },
        }
        post_scope = {
            "type": "http",
            "http_version": "1.1",
            "method": "POST",
            "headers": [(b"content-type", b"application/json")],
        }

        async def receive() -> dict:
            body = json.dumps(message_data).encode("utf-8")
            return {"type": "http.request", "body": body, "more_body": False}

        post_request = Request(post_scope, receive=receive)
        send_resp = await a2a_agent.on_message_send(request=post_request, context=None)
        print("=== on_message_send ===")
        print(send_resp)

        # 3) on_get_task
        task_id = send_resp["task"]["id"]
        get_scope = {
            "type": "http",
            "http_version": "1.1",
            "method": "GET",
            "headers": [],
            "query_string": b"",
            "path_params": {"id": task_id},
        }

        async def empty_receive() -> dict:
            return {"type": "http.disconnect"}

        get_request = Request[State](get_scope, empty_receive)
        task_resp = await a2a_agent.on_get_task(request=get_request, context=None)
        print("=== on_get_task ===")
        print(task_resp)

        if hasattr(root_obs, "update"):
            root_obs.update(
                output={
                    "task_id": task_id,
                    "status": task_resp.get("task", {}).get("status", "unknown")
                    if isinstance(task_resp, dict)
                    else "unknown",
                }
            )

    client.flush()


if __name__ == "__main__":
    asyncio.run(run_local_flow())
