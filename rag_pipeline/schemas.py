"""Schemas for the RAG pipeline."""

from typing import TypedDict, List


class ClauseJustification(TypedDict):
    """TypedDict for clause justification in the claim response.

    Args:
        TypedDict : A dictionary-like class that defines the structure of the clause justification.
    """

    clause_id: int
    text: str


class ClaimResponse(TypedDict):
    """TypedDict for the claim response.

    Args:
        TypedDict : A dictionary-like class that defines the structure of the claim response.
    """

    decision: str
    amount: int
    justification: List[ClauseJustification]
