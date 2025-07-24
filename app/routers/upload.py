from fastapi import APIRouter, UploadFile, File
from typing import List

router = APIRouter(
    prefix="/upload",
    tags=["Upload"]
)

@router.post("/documents")
async def upload_documents(files: List[UploadFile] = File(...)):
    # Placeholder: Save or process documents
    return {"filenames": [file.filename for file in files]}
