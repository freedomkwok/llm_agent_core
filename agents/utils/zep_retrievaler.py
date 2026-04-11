"""Compatibility shim for misspelled module path.

Prefer importing from `agents.utils.zep_retriever`.
"""

from agents.utils.zep_retriever import (  # noqa: F401
    ZepRetrievalAttempt,
    ZepRetrievalResult,
    retrieve_candidates_until_success,
)

__all__ = [
    "ZepRetrievalAttempt",
    "ZepRetrievalResult",
    "retrieve_candidates_until_success",
]
