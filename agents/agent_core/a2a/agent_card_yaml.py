# SPDX-License-Identifier: Apache-2.0
"""Build A2A AgentCard values from YAML config."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from a2a.types import AgentSkill
from vertexai.preview.reasoning_engines.templates.a2a import create_agent_card


def build_agent_card_from_yaml(
    config_path: str | Path, *, config_section: str = "card_config"
) -> Any:
    """Load AgentCard fields from YAML and return `create_agent_card(...)`."""
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Agent card config must be a mapping: {path}")

    card_config = config.get(config_section)
    if card_config is None:
        card_config = config
    if not isinstance(card_config, dict):
        raise ValueError(f"{config_section} must be a mapping in {path}")

    agent_name = str(card_config.get("agent_name") or "").strip()
    if not agent_name:
        raise ValueError(f"agent_name is required in {path}")
    description = str(card_config.get("description") or "").strip()

    raw_skills = card_config.get("skills") or []
    if not isinstance(raw_skills, list):
        raise ValueError(f"skills must be a list in {path}")

    skills: list[AgentSkill] = []
    for raw_skill in raw_skills:
        if not isinstance(raw_skill, dict):
            raise ValueError(f"Each skill must be a mapping in {path}")
        skill_id = str(raw_skill.get("id") or "").strip()
        if not skill_id:
            raise ValueError(f"Skill id is required in {path}")
        skills.append(
            AgentSkill(
                id=skill_id,
                name=str(raw_skill.get("name") or skill_id).strip(),
                description=str(raw_skill.get("description") or "").strip(),
                tags=[str(tag) for tag in (raw_skill.get("tags") or []) if str(tag).strip()],
                examples=[
                    str(example)
                    for example in (raw_skill.get("examples") or [])
                    if str(example).strip()
                ],
            )
        )

    return create_agent_card(agent_name=agent_name, description=description, skills=skills)
