"""Builds a vector store from the provided documents.

Returns:
    FAISS: The FAISS vector store containing the document embeddings.
"""

import os
import time
import hashlib
from typing import List
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from rag_pipeline.config import VECTOR_DB_PATH, embeddings, splitter

def compute_hash(text: str) -> str:
    """Computes SHA256 hash of the given text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def build_vector_store(documents: List[Document]) -> FAISS:
    """Builds or updates the vector store from the provided documents, skipping duplicates."""
    vector_store = None
    existing_hashes = set()

    if os.path.exists(os.path.join(VECTOR_DB_PATH, "index.faiss")):
        print("✅ Vector store already exists. Loading from disk.")
        vector_store = FAISS.load_local(VECTOR_DB_PATH, embeddings, allow_dangerous_deserialization=True)
        all_docs = vector_store.similarity_search("warmup", k=10000)
        existing_hashes = {doc.metadata.get("hash") for doc in all_docs if "hash" in doc.metadata}

    start_split = time.perf_counter()
    chunks = splitter.create_documents([doc.page_content for doc in documents])
    print(f"🧩 Split into {len(chunks)} chunks in {time.perf_counter() - start_split:.2f}s")

    start_chunk = time.perf_counter()
    new_chunks = []
    for chunk in chunks:
        h = compute_hash(chunk.page_content)
        if h not in existing_hashes:
            chunk.metadata["hash"] = h
            new_chunks.append(chunk)
        
    print(f"🔍 Found {len(new_chunks)} new chunks to embed in {time.perf_counter() - start_chunk:.2f}s")

    if not new_chunks:
        return vector_store

    start_embed = time.perf_counter()
    if vector_store:
        vector_store.add_documents(new_chunks)
    else:
        vector_store = FAISS.from_documents(new_chunks, embeddings)
    print(f"📦 Embedded & indexed {len(new_chunks)} new chunks in {time.perf_counter() - start_embed:.2f}s")

    start_save = time.perf_counter()
    vector_store.save_local(os.path.join(VECTOR_DB_PATH))
    print(f"💾 Saved updated vector store in {time.perf_counter() - start_save:.2f}s")

    return vector_store
