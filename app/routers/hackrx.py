from fastapi import APIRouter, Depends, HTTPException, Header
from pydantic import BaseModel, HttpUrl
from typing import List, Dict
import tempfile
import requests
import os

from langchain_community.document_loaders import PyMuPDFLoader

from rag_pipeline.embedder import build_vector_store
from rag_pipeline.query_pipeline import run_batch_query_pipeline

router = APIRouter(
    prefix="/api/v1/hackrx",
    tags=["HackRx"]
)

class RunRequest(BaseModel):
    documents: HttpUrl
    questions: List[str]

API_KEY = "4cddf75ac147708172d676dce84c367b3e9f55654166b361e654df27aa26f424"

def verify_token(authorization: str = Header(...)):
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header")
    token = authorization.split(" ")[1]
    if token != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")

@router.post("/run", response_model=Dict[str,List[str]], dependencies=[Depends((verify_token))])
async def run_qa(request: RunRequest):
    try:
        all_docs = []
        urls =  [request.documents]
        # Step 1: Download each PDF and load using PyMuPDFLoader
        for url in urls:
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
        return {"answers": answers}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
