from typing import List
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_community.docstore.document import Document


def load_documents(file_paths: List[str]) -> List[Document]:
    docs = []
    for path in file_paths:
        if path.endswith(".pdf"):
            loader = PyMuPDFLoader(path)
        elif path.endswith(".docx"):
            loader = UnstructuredWordDocumentLoader(path)
        else:
            raise ValueError(f"Unsupported file type: {path}")
        docs.append(loader.load())
    return docs

