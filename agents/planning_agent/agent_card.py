"""Planning agent card definition."""

from a2a.types import AgentSkill
from vertexai.preview.reasoning_engines.templates.a2a import create_agent_card

planning_skill = AgentSkill(
    id="plan_request",
    name="Plan Request",
    description="Turns a user request into a concise, ordered execution plan.",
    tags=["planning", "reasoning", "task-breakdown"],
    examples=[
        "Plan how to build an A2A weather agent.",
        "Create a step-by-step implementation plan for adding Redis sessions.",
    ],
)

agent_card = create_agent_card(
    agent_name="Planning Agent",
    description="An agent that converts requests into practical step-by-step plans.",
    skills=[planning_skill],
)

if __name__ == "__main__":
    print(agent_card)
