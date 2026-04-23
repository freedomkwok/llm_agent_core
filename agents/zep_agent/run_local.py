"""Debug runner for local zep agent A2A flow."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from agents.agent_core import OrchestrationMode, run_local_a2a_orchestration
from agents.zep_agent.registry import build_local_a2a_zep_agent


async def _main() -> None:
    mode = OrchestrationMode.HOST_DRIVEN
    a2a_agent = build_local_a2a_zep_agent(mode=mode)
    result = await run_local_a2a_orchestration(
        a2a_agent=a2a_agent,
        message_text="route this request to the best skill",
        mode=mode,
        metadata={"graph_id": "proj_63edb3c4f72f"},
    )
    print("task_id:", result.task_id)
    print("task_status:", result.task_status)
    print("final_text:\n", result.final_text)


if __name__ == "__main__":
    asyncio.run(_main())

