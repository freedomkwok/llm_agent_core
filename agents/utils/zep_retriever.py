"""Helpers for LLM-driven Zep retrieval with typed JSON + retries."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from agents.plan_agent.deps import (
    SkillCandidate,
    ZepRegistryCore,
    ZepRegistryPlan,
    ZepRetrievalRequest,
    ZepSkillCatalog,
)


@dataclass(frozen=True)
class ZepRetrievalAttempt:
    """One retrieval attempt summary for observability/state logging."""

    attempt: int
    success: bool
    request: dict[str, Any] | None
    candidate_count: int
    error: str | None = None


@dataclass(frozen=True)
class ZepRetrievalResult:
    """Output of retry loop used by `ZepSkillRegistryAgent`."""

    candidates: list[SkillCandidate]
    attempts: list[ZepRetrievalAttempt]
    last_request: ZepRetrievalRequest | None
    last_error: str | None


def _strip_code_fence(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1]
    if cleaned.endswith("```"):
        cleaned = cleaned.rsplit("```", 1)[0]
    return cleaned.strip()


def _parse_plan(raw_json: str) -> ZepRegistryPlan:
    cleaned = _strip_code_fence(raw_json)
    try:
        return ZepRegistryPlan.model_validate_json(cleaned)
    except ValidationError:
        payload = json.loads(cleaned)
        if isinstance(payload, dict) and "request" not in payload:
            payload = {"request": payload}
        return ZepRegistryPlan.model_validate(payload)


async def retrieve_candidates_until_success(
    *,
    user_message: str,
    generation_prompt: str,
    registry_core: ZepRegistryCore,
    zep_catalog: ZepSkillCatalog,
    max_attempts: int = 4,
) -> ZepRetrievalResult:
    """
    Generate typed Zep requests with LLM and retry retrieval until success.

    A successful attempt means the Zep call completes and returns at least one
    candidate. The loop stops early on first success.
    """
    attempts: list[ZepRetrievalAttempt] = []
    last_request: ZepRetrievalRequest | None = None
    previous_error: str | None = None

    safe_attempts = max(1, max_attempts)
    for attempt in range(1, safe_attempts + 1):
        try:
            raw_json = await registry_core.generate_registry_json(
                prompt=generation_prompt,
                user_message=user_message,
                attempt=attempt,
                previous_error=previous_error,
            )
            plan = _parse_plan(raw_json)
            request = plan.request
            last_request = request
        except Exception as exc:  # noqa: BLE001
            previous_error = f"request generation failed: {exc}"
            attempts.append(
                ZepRetrievalAttempt(
                    attempt=attempt,
                    success=False,
                    request=last_request.model_dump() if last_request else None,
                    candidate_count=0,
                    error=previous_error,
                )
            )
            continue

        try:
            candidates = await zep_catalog.retrieve_candidates(request)
        except Exception as exc:  # noqa: BLE001
            previous_error = f"zep retrieval failed: {exc}"
            attempts.append(
                ZepRetrievalAttempt(
                    attempt=attempt,
                    success=False,
                    request=request.model_dump(),
                    candidate_count=0,
                    error=previous_error,
                )
            )
            continue

        if candidates:
            attempts.append(
                ZepRetrievalAttempt(
                    attempt=attempt,
                    success=True,
                    request=request.model_dump(),
                    candidate_count=len(candidates),
                    error=None,
                )
            )
            return ZepRetrievalResult(
                candidates=candidates,
                attempts=attempts,
                last_request=request,
                last_error=None,
            )

        previous_error = "zep returned no candidates"
        attempts.append(
            ZepRetrievalAttempt(
                attempt=attempt,
                success=False,
                request=request.model_dump(),
                candidate_count=0,
                error=previous_error,
            )
        )

    return ZepRetrievalResult(
        candidates=[],
        attempts=attempts,
        last_request=last_request,
        last_error=previous_error,
    )
