from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(
    prefix="/query",
    tags=["Query"]
)

class QueryRequest(BaseModel):
    question: str

@router.post("/ask")
async def ask_question(query: QueryRequest):
    # Placeholder: Use previously uploaded documents
    return {"question": query.question, "answer": "This is a placeholder answer."}
