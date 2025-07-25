from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, HttpUrl
from typing import List
import tempfile
import requests
import os

from langchain.docstore.document import Document
from langchain_community.document_loaders import PyMuPDFLoader

from rag_pipeline.embedder import build_vector_store
from rag_pipeline.query_pipeline import run_batch_query_pipeline

router = APIRouter(
    prefix="/hackrx",
    tags=["HackRx"]
)

class RunRequest(BaseModel):
    documents: List[HttpUrl]
    questions: List[str]

@router.post("/run", response_model=List[str])
async def run_qa(request: RunRequest):
    try:
        all_docs = []

        # Step 1: Download each PDF and load using PyMuPDFLoader
        for url in request.documents:
            response = requests.get(str(url))
            response.raise_for_status()

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(response.content)
                tmp_path = tmp.name

            loader = PyMuPDFLoader(tmp_path)
            docs = loader.load()
            all_docs.extend(docs)

            os.remove(tmp_path)  # Clean up temp file

        if not all_docs:
            raise HTTPException(status_code=400, detail="No readable content found in PDFs.")

        # Step 2: Build vector store
        build_vector_store(all_docs)

        # Step 3: Run queries
        answers = run_batch_query_pipeline(request.questions)
        return answers

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
