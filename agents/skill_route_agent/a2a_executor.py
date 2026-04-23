"""A2A execution wrapper for the skill route ADK agent."""

from agents.agent_core import ConfiguredA2aExecutor
from agents.skill_route_agent._env import bootstrap_env

bootstrap_env()


class SkillRouteA2aExecutor(ConfiguredA2aExecutor):
    """Skill-routing A2A wrapper with shared runner/session/trace orchestration."""


if __name__ == "__main__":
    executor = SkillRouteA2aExecutor()
    print(executor.__class__.__name__)
