"""Skill route agent card definition."""

from a2a.types import AgentSkill
from vertexai.preview.reasoning_engines.templates.a2a import create_agent_card

skill_routing = AgentSkill(
    id="route_skill_request",
    name="Route Skill Request",
    description=(
        "Looks up relevant skills from Zep and selects the best next skill for a request."
    ),
    tags=["routing", "skill-selection", "zep", "orchestration"],
    examples=[
        "Figure out which skill should handle adding Redis-backed session state.",
        "Choose the best skill to use for building a weather API integration.",
    ],
)

agent_card = create_agent_card(
    agent_name="Skill Route Agent",
    description=(
        "An agent that queries Zep for candidate skills and routes the request "
        "to the best match."
    ),
    skills=[skill_routing],
)

if __name__ == "__main__":
    print(agent_card)
