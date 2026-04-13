"""Structured schema for skill routing output."""

from __future__ import annotations

from pydantic import BaseModel, Field


class RoutedSkillCandidate(BaseModel):
    """A Zep-derived skill candidate considered during routing."""

    skill_id: str = Field(..., description="Stable identifier of the skill candidate.")
    name: str = Field(..., description="Display name of the skill candidate.")
    description: str = Field(
        default="",
        description="Short description or summary from Zep for this skill.",
    )


class SkillRouteSchema(BaseModel):
    """Planner-facing output describing which skill should be used next."""

    request_summary: str = Field(
        ...,
        description="Concise summary of the incoming user request.",
    )
    selected_skill_id: str = Field(
        default="",
        description="Best matching skill id chosen from Zep candidates, if any.",
    )
    selected_skill_name: str = Field(
        default="",
        description="Display name of the selected skill, if any.",
    )
    rationale: str = Field(
        ...,
        description="Why this skill was selected over the other candidates.",
    )
    candidate_skills: list[RoutedSkillCandidate] = Field(
        default_factory=list,
        description="Ordered Zep-derived skill candidates considered during routing.",
    )
    planner_prompt: str = Field(
        ...,
        description="Normalized handoff text that a planning or execution agent can use next.",
    )
    next_action: str = Field(
        ...,
        description="Immediate next action for the host after routing completes.",
    )
