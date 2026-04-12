"""Planning agent schema definitions."""

from pydantic import BaseModel, Field


class PlanningSchema(BaseModel):
    """Structured plan output for internal and cross-agent usage."""

    goal: str = Field(..., description="Primary objective inferred from the request.")
    assumptions: list[str] = Field(
        default_factory=list,
        description="Optional assumptions made to build the plan.",
    )
    steps: list[str] = Field(..., min_length=1, description="Ordered implementation steps.")
    risks: list[str] = Field(
        default_factory=list,
        description="Potential blockers or unknowns that may affect execution.",
    )
    next_action: str = Field(..., description="Single immediate next action.")
