# SPDX-License-Identifier: Apache-2.0
"""Debug runner for local zep agent A2A flow."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from agents.agent_core.a2a import OrchestrationMode, run_local_a2a_orchestration  # noqa: E402
from agents.zep_agent.registry import build_local_a2a_zep_agent  # noqa: E402


async def _main() -> None:
    mode = OrchestrationMode.HOST_DRIVEN
    a2a_agent = build_local_a2a_zep_agent(mode=mode)
    result = await run_local_a2a_orchestration(
        a2a_agent=a2a_agent,
        message_text="韩立喜欢的人喜欢韩立吗？on graph mirofish_53c089d117c649c7",
        metadata={"graph_id": "mirofish_53c089d117c649c7"},
    )
    print("task_id:", result.task_id)
    print("task_status:", result.task_status)
    print("final_text:\n", result.final_text)


if __name__ == "__main__":
    asyncio.run(_main())

