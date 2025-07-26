"""Retriever module for the RAG pipeline.

Raises:
    ValueError: If the VECTOR_DB_PATH environment variable is not set.
"""

from typing import List
import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document
import time

from dotenv import load_dotenv

load_dotenv()

VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH")

embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

def load_vector_store() -> FAISS:
    """Loads the FAISS vector store from the specified path.

    Raises:
        ValueError: If the VECTOR_DB_PATH environment variable is not set.

    Returns:
        FAISS: The FAISS vector store loaded from the specified path.
    """
   
    if not VECTOR_DB_PATH:
        raise ValueError("VECTOR_DB_PATH environment variable is not set.")
    return FAISS.load_local(
        VECTOR_DB_PATH, embeddings, allow_dangerous_deserialization=True
    )


def retrieve_documents(query: str, top_k: int = 5) -> List[Document]:
    """Retrieves documents relevant to the query from the vector store.

    Args:
        query (str): The query string to search for.
        top_k (int, optional): The number of top documents to consider for retrieval. Defaults to 5.

    Returns:
        List[Docuemnt]: A list of documents retrieved from the vector store.
    """
    store = load_vector_store()
    start = time.time()
    result= store.similarity_search(query, k=top_k)
    print(f"🔍 Document retrieval time: {time.time() - start:.2f}s")
    return result
