"""Builds a vector store from the provided documents.

Returns:
    FAISS: The FAISS vector store containing the document embeddings.
"""

import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import time

from dotenv import load_dotenv

load_dotenv()

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
VECTOR_DB_PATH = os.path.relpath(str(os.getenv("VECTOR_DB_PATH")))


embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

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
    start_split = time.time()
    chunks = splitter.split_documents(documents)
    print(f"🧩 Split into {len(chunks)} chunks in {time.time() - start_split:.2f}s")

    start_embed = time.time()
    vector_store = FAISS.from_documents(chunks, embeddings)
    print(f"📦 Embedded & indexed in {time.time() - start_embed:.2f}s")

    start_save = time.time()
    vector_store.save_local(os.path.join(VECTOR_DB_PATH))
    print(f"📦 saving in db in {time.time() - start_save:.2f}s")

    return vector_store
