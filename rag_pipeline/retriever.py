import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from dotenv import load_dotenv

load_dotenv()

VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH")


def load_vector_store() -> FAISS:
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    if not VECTOR_DB_PATH:
        raise ValueError("VECTOR_DB_PATH environment variable is not set.")
    return FAISS.load_local(
        VECTOR_DB_PATH, embeddings, allow_dangerous_deserialization=True
    )


def retrieve_documents(query: str, top_k: int = 5):
    store = load_vector_store()
    return store.similarity_search(query, k=top_k)
