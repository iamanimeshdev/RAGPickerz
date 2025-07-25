"""Builds a vector store from the provided documents.

Returns:
    FAISS: The FAISS vector store containing the document embeddings.
"""

import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from dotenv import load_dotenv

load_dotenv()

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
VECTOR_DB_PATH = os.path.relpath(str(os.getenv("VECTOR_DB_PATH")))


def build_vector_store(documents: list[Document]) -> FAISS:
    """Builds a vector store from the provided documents.

    Args:
        documents (list[Document]): List of documents to be embedded.

    Returns:
        FAISS: The FAISS vector store containing the document embeddings.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(os.path.join(VECTOR_DB_PATH))
    return vector_store
