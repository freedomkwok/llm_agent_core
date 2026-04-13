"""Debug runner for local skill route A2A flow."""

from __future__ import annotations

import argparse
import asyncio
import logging
from pprint import pprint
import sys
import traceback
from pathlib import Path

# Running `python examples/this_file.py` puts `examples/` on sys.path, not the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from agents.agent_core import OrchestrationMode
from agents.skill_route_agent.start_agent import run_local_skill_route_flow


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local skill route flow for debugging.")
    parser.add_argument(
        "--message",
        default="neeed for analysis a conversation.",
        help="User message to route.",
    )
    parser.add_argument(
        "--user-id",
        default="",
        help="Optional Zep user id override.",
    )
    parser.add_argument(
        "--mode",
        choices=["host_driven", "agent_internal"],
        default="host_driven",
        help="Local orchestration mode.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose debug logs and print raw send/task responses.",
    )
    return parser.parse_args()


async def _main() -> None:
    args = _parse_args()
    if args.debug:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        )
    mode = OrchestrationMode(args.mode)
    metadata = {"user_id": args.user_id} if args.user_id else None
    try:
        result = await run_local_skill_route_flow(
            message_text=args.message,
            mode=mode,
            metadata=metadata,
        )
    except Exception as exc:  # noqa: BLE001
        print("run_local_skill_route_flow failed:", repr(exc))
        if args.debug:
            traceback.print_exc()
        raise
    print("task_id:", result.task_id)
    print("task_status:", result.task_status)
    print("final_text:\n", result.final_text)
    if args.debug:
        print("\nsend_response:")
        pprint(result.send_response)
        print("\ntask_response:")
        pprint(result.task_response)


if __name__ == "__main__":
    asyncio.run(_main())
