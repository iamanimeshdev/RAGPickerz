from typing import TypedDict, List


class ClauseJustification(TypedDict):
    clause_id: int
    text: str


class ClaimResponse(TypedDict):
    decision: str
    amount: int
    justification: List[ClauseJustification]
