"""Step 1: define an A2A AgentCard."""

from a2a.types import AgentSkill
from vertexai.preview.reasoning_engines.templates.a2a import create_agent_card


currency_skill = AgentSkill(
    id="get_exchange_rate",
    name="Get Currency Exchange Rate",
    description="Retrieves exchange rate between two currencies on a date.",
    tags=["finance", "currency", "exchange-rate"],
    examples=[
        "What is USD to EUR now?",
        "How much is 1 USD in JPY today?",
    ],
)

agent_card = create_agent_card(
    agent_name="Zep Skill Registry Agent",
    description="Zep Skill Registry Agent that can retrieve skills from Zep.",
    skills=[currency_skill],
)


if __name__ == "__main__":
    print(agent_card)
