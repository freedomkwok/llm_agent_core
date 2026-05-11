# SPDX-License-Identifier: Apache-2.0
import os

from dotenv import load_dotenv
from google.adk.agents import Context, LlmAgent, LoopAgent

load_dotenv()


def health_check() -> str:
    """Simple local tool to verify the tool pipeline is active."""
    return "imp_chat_agent is running."


def finish_loop(note: str, tool_context: Context) -> str:
    """
    End the current loop once the agent has a final answer.

    Args:
      note: A short machine-readable completion note.
      tool_context: ADK context injected automatically.
    """
    tool_context.state["loop_finish_note"] = note
    tool_context.actions.escalate = True
    return f"Loop finished: {note}"


executor_agent = LlmAgent(
    name="executor_agent",
    model=os.getenv("OPENAI_MODEL", "openai/gpt-5"),
    description="Loop executor for chat tasks.",
    instruction=(
        "You are the executor inside a loop. "
        "Think step-by-step and use tools when helpful. "
        "When the final answer is ready, call finish_loop(note=...) exactly once "
        "to stop the LoopAgent."
    ),
    tools=[health_check, finish_loop],
)

root_agent = LoopAgent(
    name="imp_chat_agent",
    description="IMP ADK loop agent example.",
    max_iterations=8,
    sub_agents=[executor_agent],
)
