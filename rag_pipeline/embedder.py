"""Builds a vector store from the provided documents.

Returns:
    FAISS: The FAISS vector store containing the document embeddings.
"""

import os
import time
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from rag_pipeline.config import VECTOR_DB_PATH, embeddings, splitter


def build_vector_store(documents: list[Document]) -> FAISS:
    """Builds a vector store from the provided documents.

    Args:
        documents (list[Document]): List of documents to be embedded.

    Returns:
        FAISS: The FAISS vector store containing the document embeddings.
    """
    if os.path.exists(os.path.join(VECTOR_DB_PATH, "index.faiss")):
        print("✅ Vector store already exists. Loading from disk.")
        return FAISS.load_local(VECTOR_DB_PATH, embeddings, allow_dangerous_deserialization=True)
    
    start_split = time.perf_counter()
    chunks = splitter.split_documents(documents)
    print(f"🧩 Split into {len(chunks)} chunks in {time.perf_counter() - start_split:.2f}s")

    start_embed = time.perf_counter()
    vector_store = FAISS.from_documents(chunks, embeddings)
    print(f"📦 Embedded & indexed in {time.perf_counter() - start_embed:.2f}s")

    start_save = time.perf_counter()
    vector_store.save_local(os.path.join(VECTOR_DB_PATH))
    print(f"📦 saving in db in {time.perf_counter() - start_save:.2f}s")

    return vector_store
