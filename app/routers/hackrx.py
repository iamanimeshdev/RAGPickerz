from fastapi import APIRouter, HTTPException, File, UploadFile
from typing import List, Dict
import tempfile
import os

from rag_pipeline.document_loader import load_documents
from rag_pipeline.embedder import build_vector_store
from rag_pipeline.query_pipeline import run_batch_query_pipeline

router = APIRouter(
    prefix="/api/v1/hackrx",
    tags=["HackRx"]
)

@router.post("/run", response_model=Dict[str, List[str]])
async def run_qa(
    questions: List[str],
    file: UploadFile = File(...),
):
    try:
        # Step 1: Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        # Step 2: Load documents from uploaded file
        all_docs = load_documents([tmp_path])
        os.remove(tmp_path)

        if not all_docs:
            raise HTTPException(status_code=400, detail="No readable content found in the uploaded PDF.")

        # Step 3: Build vector store
        build_vector_store(all_docs)

        # Step 4: Run queries
        answers = run_batch_query_pipeline(questions)
        return {"answers": answers}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
