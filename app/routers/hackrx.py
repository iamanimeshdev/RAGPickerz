from fastapi import APIRouter
from pydantic import BaseModel, HttpUrl
from typing import List

router = APIRouter(
    prefix="/hackrx",
    tags=["HackRx"]
)

class RunRequest(BaseModel):
    documents: List[HttpUrl]  # Ensures valid URLs
    questions: List[str]

class QAResponse(BaseModel):
    question: str
    answer: str

@router.post("/run", response_model=List[QAResponse])
async def run_qa(request: RunRequest):
    # Placeholder logic: return mock answer
    responses = [
        QAResponse(question=q, answer="This is a placeholder answer.")
        for q in request.questions
    ]
    return responses
