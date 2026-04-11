"""Assemble Langfuse + Zep registry + looping ADK executor."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from google.adk.agents import Context, LoopAgent
from google.adk.agents.llm_agent import InstructionProvider, LlmAgent
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.agents.sequential_agent import SequentialAgent

from agents.plan_agent.deps import (
    SESSION_SELECTED_SKILL,
    PlanAgentDeps,
    SkillCandidate,
)
from agents.plan_agent.routing_agent import (
    DEFAULT_REGISTRY_PROMPT,
    ZepSkillRegistryAgent,
    ZepSkillSelectionRouterAgent,
)


def plan_executor_finish_loop(note: str, tool_context: Context) -> str:
    """End the executor loop when the model has a final answer (same pattern as ADK LoopAgent)."""
    tool_context.state["loop_finish_note"] = note
    tool_context.actions.escalate = True
    return f"Loop finished: {note}"


def _skill_section(selected: dict[str, Any]) -> str:
    body = (selected.get("body") or "").strip()
    summary = (selected.get("summary") or "").strip()
    title = (selected.get("title") or "").strip()
    skill_id = (selected.get("id") or "").strip()
    text = body or summary
    header = f"Routed skill: {skill_id}" + (f" — {title}" if title else "")
    if not text:
        return f"{header}\n(No skill body; use summaries from state if needed.)"
    return f"{header}\n{text}"


def _make_executor_instruction(
    base_instruction: str,
    *,
    finish_loop_tool_name: str | None,
) -> InstructionProvider:
    finish_hint = ""
    if finish_loop_tool_name:
        finish_hint = (
            "\n\n## Loop control\n"
            "You run inside a LoopAgent. Use tools as needed. "
            f"When the final answer is ready, call {finish_loop_tool_name}(note=...) "
            "exactly once with a short completion note to stop the loop."
        )

    async def instruction(ctx: ReadonlyContext) -> str:
        selected = ctx.state.get(SESSION_SELECTED_SKILL)
        if not isinstance(selected, dict):
            return (
                f"{base_instruction}\n\n"
                "## Skill routing\n"
                "No skill was selected. Answer helpfully using the base policy only."
                f"{finish_hint}"
            )
        return (
            f"{base_instruction}\n\n## Active skill\n{_skill_section(selected)}"
            f"{finish_hint}"
        )

    return instruction


async def create_plan_agent_root(
    *,
    name: str,
    description: str,
    deps: PlanAgentDeps,
    langfuse_prompt_name: str,
    langfuse_label: str | None = None,
    executor_model: str,
    executor_tools: Sequence[Any] | None = None,
    registry_agent_name: str = "zep_skill_registry",
    router_agent_name: str = "zep_skill_router",
    registry_prompt: str = DEFAULT_REGISTRY_PROMPT,
    max_registry_retrieval_attempts: int = 4,
    executor_agent_name: str = "plan_executor",
    loop_agent_name: str = "plan_executor_loop",
    max_executor_iterations: int | None = 8,
    add_finish_loop_tool: bool = True,
) -> SequentialAgent:
    """
    Build a root agent:

    1. Load base instruction text from Langfuse (async, once).
    2. On each user invocation, `ZepSkillRegistryAgent` runs once:
       - registry LLM core generates typed retrieval JSON (pydantic),
       - Zep is retried until candidates are returned or attempts are exhausted.
    3. `ZepSkillSelectionRouterAgent` selects final skill id from retrieved candidates.
    4. `LoopAgent` wraps the executor `LlmAgent` so tool use can iterate until
       `plan_executor_finish_loop` escalates (same idea as `agent.py`).

    The root remains a `SequentialAgent` so the registry+router phase is not re-run
    on every loop tick. If you need all phases repeated each cycle, use a single
    `LoopAgent(sub_agents=[registry_agent, router_agent, executor])` instead.
    """
    base_instruction = await deps.langfuse.load_text_prompt(
        langfuse_prompt_name,
        label=langfuse_label,
    )
    tools = list(executor_tools or [])
    finish_name: str | None = plan_executor_finish_loop.__name__ if add_finish_loop_tool else None
    if add_finish_loop_tool:
        tools.append(plan_executor_finish_loop)

    registry_agent = ZepSkillRegistryAgent(
        name=registry_agent_name,
        description=(
            "Registry step: generate retrieval JSON and query Zep with retries."
        ),
        deps=deps,
        base_instruction=base_instruction,
        retrieval_prompt=registry_prompt,
        max_retrieval_attempts=max_registry_retrieval_attempts,
    )
    router_agent = ZepSkillSelectionRouterAgent(
        name=router_agent_name,
        description="Router step: choose one skill from retrieved Zep candidates.",
        deps=deps,
    )
    executor = LlmAgent(
        name=executor_agent_name,
        model=executor_model,
        description="Executes the user task under the routed skill.",
        instruction=_make_executor_instruction(
            base_instruction,
            finish_loop_tool_name=finish_name,
        ),
        tools=tools,
    )
    executor_loop = LoopAgent(
        name=loop_agent_name,
        description="Runs the executor until finish_loop escalates or max iterations.",
        max_iterations=max_executor_iterations,
        sub_agents=[executor],
    )
    return SequentialAgent(
        name=name,
        description=description,
        sub_agents=[registry_agent, router_agent, executor_loop],
    )


def skill_dicts(candidates: Sequence[SkillCandidate]) -> list[dict[str, Any]]:
    """Helper for tests or logging: serialize candidates."""
    return [c.as_state_dict() for c in candidates]
